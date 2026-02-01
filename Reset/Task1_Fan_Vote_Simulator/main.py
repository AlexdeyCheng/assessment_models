import pandas as pd
import numpy as np
import warnings
import time
from tqdm import tqdm

from structures import SeasonRules, Contestant
from solver import ConstraintBuilder
from mallows import MallowsModel
from sampler import MCMCSampler
from utils_output import StatisticsCalculator

def plot_results(samples, season, week, output_dir="plots"):
    pass

class SimulationRunner:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.raw_df = pd.read_csv(data_path)
        self.current_offsets = {} 
        self.previous_week_stats = {}

    def _get_season_rule(self, season):
        if season <= 2: return SeasonRules.RANK_SUM
        elif season <= 27: return SeasonRules.PERCENTAGE_SUM
        else: return SeasonRules.RANK_BOTTOM2

    def run(self):
        results_rows = []
        df = self.raw_df.sort_values(by=['season', 'week'])
        seasons = df['season'].unique()
        
        print(f"Starting Simulation for {len(seasons)} Seasons...")
        
        for season in tqdm(seasons, desc="Seasons"):
            self.current_offsets = {}
            self.previous_week_stats = {}
            season_data = df[df['season'] == season]
            weeks = season_data['week'].unique()
            season_rule = self._get_season_rule(season)
            
            for week in weeks:
                week_data = season_data[season_data['week'] == week]
                contestants = []
                eliminated_name = None
                exit_mode = "unknown"
                
                elim_rows = week_data[week_data['exit_mode'] == 'eliminated']
                if not elim_rows.empty:
                    eliminated_name = elim_rows.iloc[0]['celebrity_name']
                    exit_mode = 'eliminated'
                
                final_rows = week_data[week_data['exit_mode'] == 'final']
                if not final_rows.empty: exit_mode = 'final'
                
                for _, row in week_data.iterrows():
                    name = row['celebrity_name']
                    j_metric = row['judge_metric'] if 'judge_metric' in row else row['effective_score']
                    offset = self.current_offsets.get(name, 0.0)
                    c = Contestant(name=name, judge_metric=float(j_metric), fan_offset=offset,
                                   is_accumulated=(offset != 0), is_eliminated=(name == eliminated_name))
                    contestants.append(c)
                
                current_stats_map = {}
                is_scorable_week = (exit_mode == 'eliminated') or (exit_mode == 'final')
                if not is_scorable_week and week == 1: is_scorable_week = True
                
                if is_scorable_week:
                    try:
                        builder = ConstraintBuilder(season_rule)
                        poly = builder.build(contestants, eliminated_name)
                        
                        mallows_model = None
                        mallows_params = None
                        center_str = "N/A" # Default string for center
                        
                        if season_rule != SeasonRules.PERCENTAGE_SUM:
                            mallows_model = MallowsModel(poly)
                            center, phi = mallows_model.icr_search(n_coarse=5000, n_fine=10000)
                            mallows_params = (center, phi)
                            # Convert center (rank vector) to string for CSV
                            center_str = "-".join(map(str, center))
                        else:
                            # For Percentage, center is geometric center (float vector)
                            center_str = "Geometric"
                        
                        sampler = MCMCSampler(poly)
                        # TIMEOUT RELAXED TO 100s as requested
                        samples, metadata = sampler.generate_samples(n_samples=25000,
                                                                     mallows_model=mallows_model, 
                                                                     mallows_params=mallows_params,
                                                                     timeout=100.0)
                        
                        names = [c.name for c in contestants]
                        current_stats_map = StatisticsCalculator.compute_stats(samples, names, season_rule.value)
                        
                        for name in names:
                            current_stats_map[name]['phi'] = mallows_params[1] if mallows_params else 0.0
                            current_stats_map[name]['center'] = center_str
                            current_stats_map[name]['mode'] = metadata.get('status', 'strict')
                            
                    except Exception as e:
                        warnings.warn(f"Simulation Failed S{season}W{week}: {e}")
                        for c in contestants:
                            current_stats_map[c.name] = {'mean': 0, 'std': 0, 'entropy': 0, 'mode': 'failed', 'center': 'err'}
                else:
                    for c in contestants:
                        prev = self.previous_week_stats.get(c.name)
                        if prev:
                            current_stats_map[c.name] = prev.copy()
                            current_stats_map[c.name]['mode'] = 'inherited'
                        else:
                            current_stats_map[c.name] = {'mean': 0, 'std': 0, 'mode': 'missing_history', 'center': 'none'}
                
                self.previous_week_stats = current_stats_map.copy()
                
                if exit_mode == 'eliminated': self.current_offsets = {}
                else:
                    for name, stats in current_stats_map.items():
                        old_val = self.current_offsets.get(name, 0.0)
                        self.current_offsets[name] = old_val + stats.get('mean', 0)
                
                for c in contestants:
                    st = current_stats_map.get(c.name, {})
                    row = {
                        'season': season, 'week': week, 'celebrity_name': c.name,
                        'phi': st.get('phi', 0), 'mean': st.get('mean', 0),
                        'center': st.get('center', 'N/A'), # Fixed: now outputs the rank center string
                        'std': st.get('std', 0), 'RQI_width': st.get('RQI_width', 0),
                        'entropy': st.get('entropy', 0), 'RQI_d': st.get('RQI_d', 0),
                        'RQI_u': st.get('RQI_u', 0), 'accumulated_fan_offset': c.is_accumulated,
                        'mode': st.get('mode', 'unknown'), 'sample_count': st.get('sample_count', 0)
                    }
                    results_rows.append(row)
                
                if is_scorable_week: plot_results(None, season, week)

        out_df = pd.DataFrame(results_rows)
        out_df.to_csv(self.output_path, index=False)
        print(f"Simulation Complete. Results saved to {self.output_path}")

if __name__ == "__main__":
    # Example Usage
    sim = SimulationRunner("D:/Files/Study/code/DataProcessing/assessment_models/Reset/Raw_data/cleaned_dwts_data_final.csv", "simulation_results.csv")
    sim.run()
