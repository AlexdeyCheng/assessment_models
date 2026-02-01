import pandas as pd
import numpy as np
import warnings
import time
from tqdm import tqdm

# Import our modules
from structures import SeasonRules, Contestant
from solver import ConstraintBuilder
from mallows import MallowsModel
from sampler import MCMCSampler
from utils_output import StatisticsCalculator

# --- Plotting Placeholder ---
def plot_results(samples, season, week, output_dir="plots"):
    """
    TODO: Implement visualization logic.
    Ideas: Heatmap of rank probabilities, Violin plot of percentages.
    """
    pass

class SimulationRunner:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.raw_df = pd.read_csv(data_path)
        
        # Global State for Offsets: {Name: Accumulated_Value}
        self.current_offsets = {} 
        # Cache for Inheritance: {Name: {stats_dict}}
        self.previous_week_stats = {}

    def _get_season_rule(self, season):
        if season <= 2:
            return SeasonRules.RANK_SUM
        elif season <= 27:
            return SeasonRules.PERCENTAGE_SUM
        else:
            return SeasonRules.RANK_BOTTOM2

    def run(self):
        results_rows = []
        
        # Sort data to ensure chronological order
        df = self.raw_df.sort_values(by=['season', 'week'])
        seasons = df['season'].unique()
        
        print(f"Starting Simulation for {len(seasons)} Seasons...")
        
        for season in tqdm(seasons, desc="Seasons"):
            # Reset Offsets at start of season (Just in case, though logic handles it)
            self.current_offsets = {}
            self.previous_week_stats = {}
            
            season_data = df[df['season'] == season]
            weeks = season_data['week'].unique()
            season_rule = self._get_season_rule(season)
            
            for week in weeks:
                week_data = season_data[season_data['week'] == week]
                
                # 1. Identify Contestants and Exit Mode
                contestants = []
                eliminated_name = None
                exit_mode = "unknown"
                
                # Check for elimination in this week
                # We assume 'exit_mode' column exists and is consistent for the week's eliminated person
                # Logic: Find row where exit_mode == 'eliminated'
                elim_rows = week_data[week_data['exit_mode'] == 'eliminated']
                if not elim_rows.empty:
                    eliminated_name = elim_rows.iloc[0]['celebrity_name']
                    exit_mode = 'eliminated'
                
                # Check for Final
                final_rows = week_data[week_data['exit_mode'] == 'final']
                if not final_rows.empty:
                    exit_mode = 'final'
                
                # Build Contestant Objects
                # Note: We need to handle ALL contestants active in this week
                # The CSV assumes 1 row per contestant per week.
                for _, row in week_data.iterrows():
                    name = row['celebrity_name']
                    
                    # Determine Metric Type
                    # effective_score in CSV is likely the Judge Score we need
                    # But we need to distinguish Rank vs Percent input if judge_metric column exists
                    j_metric = row['judge_metric'] if 'judge_metric' in row else row['effective_score']
                    
                    # Get Offset
                    offset = self.current_offsets.get(name, 0.0)
                    
                    c = Contestant(
                        name=name,
                        judge_metric=float(j_metric),
                        fan_offset=offset,
                        is_accumulated=(offset != 0),
                        is_eliminated=(name == eliminated_name)
                    )
                    contestants.append(c)
                
                # 2. Logic Dispatch: Solve or Inherit?
                # Condition: Solve if Eliminated OR Final OR (Week 1 Non-Elim? No, just inherit/random)
                # Actually, Final requires solving specific rank constraints too.
                # But user said: "Non-elimination week data inherits from previous week."
                
                current_stats_map = {} # Name -> Stats
                
                is_scorable_week = (exit_mode == 'eliminated') or (exit_mode == 'final')
                
                # Special Case: Week 1 Non-Elimination (No history to inherit)
                # We treat this as "Scorable" but with Empty Constraints (Random Uniform)
                if not is_scorable_week and week == 1:
                     is_scorable_week = True # Force solver to run empty to generate initial seed
                
                if is_scorable_week:
                    # --- CORE ALGORITHM ---
                    try:
                        # A. Build Polytope
                        builder = ConstraintBuilder(season_rule)
                        poly = builder.build(contestants, eliminated_name)
                        
                        # B. Mallows (if needed)
                        mallows_model = None
                        mallows_params = None
                        if season_rule != SeasonRules.PERCENTAGE_SUM:
                            mallows_model = MallowsModel(poly)
                            # Run ICR
                            center, phi = mallows_model.icr_search(n_coarse=5000, n_fine=10000)
                            mallows_params = (center, phi)
                        
                        # C. Sample
                        sampler = MCMCSampler(poly)
                        # Hit-and-Run or Rejection
                        samples, metadata = sampler.generate_samples(
                            n_samples=25000,
                            mallows_model=mallows_model, 
                            mallows_params=mallows_params,
                            timeout=100.0
                        )
                        
                        # D. Stats
                        names = [c.name for c in contestants]
                        current_stats_map = StatisticsCalculator.compute_stats(samples, names, season_rule.value)
                        
                        # Inject Meta info
                        for name in names:
                            current_stats_map[name]['phi'] = mallows_params[1] if mallows_params else 0.0
                            current_stats_map[name]['mode'] = metadata.get('status', 'strict')
                            
                    except Exception as e:
                        warnings.warn(f"Simulation Failed S{season}W{week}: {e}")
                        # Fallback: Zero output
                        for c in contestants:
                            current_stats_map[c.name] = {k:0 for k in ['mean','std','entropy']}

                else:
                    # --- INHERITANCE LOGIC ---
                    # Non-elimination week (and not Week 1)
                    # Copy stats from previous week
                    for c in contestants:
                        prev = self.previous_week_stats.get(c.name)
                        if prev:
                            # Inherit everything
                            current_stats_map[c.name] = prev.copy()
                            current_stats_map[c.name]['mode'] = 'inherited'
                        else:
                            # New person in non-elim week? Unlikely, but fallback to 0
                            current_stats_map[c.name] = {'mean': 0, 'std': 0, 'mode': 'missing_history'}
                
                # 3. Output & Update State
                
                # Save Current to Previous for next week
                self.previous_week_stats = current_stats_map.copy()
                
                # Update Offsets
                if exit_mode == 'eliminated':
                    # Rule: "检测到淘汰发生时清空 Offsets"
                    # Reset ALL offsets for next week
                    self.current_offsets = {}
                else:
                    # Rule: Accumulate
                    # Offset += Mean
                    # Note: For Percentage, Mean is e.g. 0.20. Cumulative.
                    # For Rank, Mean is e.g. 5. Cumulative.
                    for name, stats in current_stats_map.items():
                        old_val = self.current_offsets.get(name, 0.0)
                        # Important: Should we normalize Rank accumulation?
                        # Rule says "Fan Vote Diff accumulates".
                        # Simplest interpretation: Accumulate the Score.
                        self.current_offsets[name] = old_val + stats.get('mean', 0)
                
                # 4. Generate CSV Rows
                for c in contestants:
                    st = current_stats_map.get(c.name, {})
                    row = {
                        'season': season,
                        'week': week,
                        'celebrity_name': c.name,
                        'phi': st.get('phi', 0),
                        'mean': st.get('mean', 0),
                        'center': st.get('mean', 0), # Center approx by mean for CSV output req
                        'std': st.get('std', 0),
                        'RQI_width': st.get('RQI_width', 0),
                        'entropy': st.get('entropy', 0),
                        'RQI_d': st.get('RQI_d', 0),
                        'RQI_u': st.get('RQI_u', 0),
                        'accumulated_fan_offset': c.is_accumulated,
                        'mode': st.get('mode', 'unknown'),
                        'sample_count': st.get('sample_count', 0)
                    }
                    results_rows.append(row)
                
                # 5. Plotting Hook
                if is_scorable_week: # Only plot if we actually calculated new stuff
                    plot_results(None, season, week) # Pass actual samples if needed

        # Save Final CSV
        out_df = pd.DataFrame(results_rows)
        out_df.to_csv(self.output_path, index=False)
        print(f"Simulation Complete. Results saved to {self.output_path}")

if __name__ == "__main__":
    # Example Usage
    sim = SimulationRunner("D:/Files/Study/code/DataProcessing/assessment_models/Reset/Raw_data/cleaned_dwts_data_final.csv", "simulation_results.csv")
    sim.run()
