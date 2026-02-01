import pandas as pd
import numpy as np
import re
import warnings
import os

# ==========================================
# PART 1: DATA CLEANING (Physical Layer)
# ==========================================
def clean_dwts_data(df_raw):
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    
    rename_map = {
        'celebrity_age_during_season': 'age',
        'celebrity_industry': 'industry',
        'ballroom_partner': 'partner'
    }
    df.rename(columns=rename_map, inplace=True)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    score_cols = [c for c in df.columns if 'judge' in c and 'score' in c]
    id_vars = [c for c in df.columns if c not in score_cols]

    df_long = pd.melt(df, id_vars=id_vars, value_vars=score_cols, var_name='week_judge_id', value_name='raw_score')
    df_long['raw_score'] = pd.to_numeric(df_long['raw_score'], errors='coerce')
    df_clean = df_long[df_long['raw_score'] > 0].copy()

    df_clean['week'] = df_clean['week_judge_id'].str.extract(r'week(\d+)_').astype(int)
    df_clean['judge_idx'] = df_clean['week_judge_id'].str.extract(r'judge(\d+)_').astype(int)

    # Standardize Withdrew text
    withdrew_mask = df_clean['results'].str.lower() == 'withdrew'
    withdrew_people = df_clean.loc[withdrew_mask, 'celebrity_name'].unique()

    for person in withdrew_people:
        person_seasons = df_clean.loc[df_clean['celebrity_name'] == person, 'season'].unique()
        for season in person_seasons:
            mask = (df_clean['celebrity_name'] == person) & (df_clean['season'] == season) & (df_clean['results'].str.lower() == 'withdrew')
            if mask.any():
                last_week = df_clean.loc[mask, 'week'].max()
                elim_week = 1 if last_week == 1 else last_week + 1
                new_status = f"Eliminated Week {elim_week} (Withdrew)"
                df_clean.loc[mask, 'results'] = new_status

    df_clean.sort_values(by=['season', 'week', 'celebrity_name', 'judge_idx'], inplace=True)
    return df_clean

# ==========================================
# PART 2: RANKING LOGIC
# ==========================================
class StatusParser:
    @staticmethod
    def parse(row, season_max_week, person_max_week):
        res = str(row['results']).strip()
        current_week = row['week']
        
        # 1. Extract Rank Value
        rank_val = np.nan
        rank_match = re.search(r'(\d+)(?:st|nd|rd|th)[\s\xa0]+Place', res, re.IGNORECASE)
        if rank_match:
            rank_val = float(rank_match.group(1))
        elif 'winner' in res.lower(): rank_val = 1.0
        elif 'runner-up' in res.lower(): rank_val = 2.0
        elif 'third' in res.lower(): rank_val = 3.0

        # 2. Active Check
        if current_week < person_max_week:
             return 'safe', np.nan, np.nan

        # 3. Withdrew Check
        if 'withdrew' in res.lower():
            week_match = re.search(r'Week\s+(\d+)', res, re.IGNORECASE)
            exit_w = int(week_match.group(1)) if week_match else current_week
            return 'withdrew', np.nan, exit_w

        # 4. Time Gate
        if current_week < season_max_week:
            # Eliminated, but PASS THE RANK VALUE
            return 'eliminated', rank_val, current_week

        # 5. Finale Logic
        return 'final', rank_val, current_week

class RankCalculator:
    @staticmethod
    def calculate_season_ranks(df_season):
        rank_map = {}
        
        # 1. Withdrew -> 0
        withdrew = df_season[df_season['exit_mode'] == 'withdrew']
        for name in withdrew['celebrity_name'].unique():
            rank_map[name] = 0.0
            
        # 2. Explicit Rank Holders
        rank_holders = df_season.dropna(subset=['final_rank_parsed']).copy()
        for _, row in rank_holders.iterrows():
            rank_map[row['celebrity_name']] = row['final_rank_parsed']
            
        if not rank_holders.empty:
            start_rank = rank_holders['final_rank_parsed'].max() + 1
        else:
            start_rank = 1.0

        # 3. Implicit Rankers (Finale Eliminated & Regular Eliminated)
        no_rankers = df_season[df_season['final_rank_parsed'].isna()].copy()
        no_rankers = no_rankers[no_rankers['exit_mode'] != 'withdrew']
        
        if not no_rankers.empty:
            exit_weeks = no_rankers['exit_week_parsed'].sort_values(ascending=False).unique()
            current_rank = start_rank
            
            for w in exit_weeks:
                people = no_rankers[no_rankers['exit_week_parsed'] == w]['celebrity_name'].unique()
                while current_rank in rank_holders['final_rank_parsed'].values:
                    current_rank += 1
                for p in people:
                    rank_map[p] = current_rank
                current_rank += len(people)
            
        return rank_map

def _accumulate_scores(df_weekly):
    df_out = df_weekly.copy()
    df_out['effective_score'] = df_out['weekly_raw_score']
    df_out['is_accumulated'] = False
    
    for s in df_out['season'].unique():
        s_data = df_out[df_out['season'] == s]
        weeks = sorted(s_data['week'].unique())
        
        for i in range(1, len(weeks)):
            curr_w = weeks[i]
            prev_w = weeks[i-1]
            mask_prev_elim = ((s_data['exit_mode'] == 'eliminated') & (s_data['exit_week_parsed'] == prev_w))
            
            if not mask_prev_elim.any():
                prev_scores = df_out[(df_out['season'] == s) & (df_out['week'] == prev_w)].set_index('celebrity_name')['effective_score']
                mask_curr = (df_out['season'] == s) & (df_out['week'] == curr_w)
                df_out.loc[mask_curr, 'effective_score'] += df_out.loc[mask_curr, 'celebrity_name'].map(prev_scores).fillna(0)
                df_out.loc[mask_curr, 'is_accumulated'] = True
    return df_out

def _calculate_metrics(df_slice, season_num):
    if (season_num <= 2) or (season_num >= 28):
        return df_slice.groupby('week')['effective_score'].rank(method='min', ascending=False)
    else:
        sums = df_slice.groupby('week')['effective_score'].transform('sum')
        return df_slice['effective_score'] / sums

def get_ordinal_text(n):
    """Converts 4.0 -> '4th Place'"""
    if pd.isna(n): return ""
    n = int(n)
    if 11 <= (n % 100) <= 13: suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix} Place"

def process_ranking_features(df_clean):
    # 1. Aggregation
    group_cols = ['season', 'week', 'celebrity_name', 'results', 'age', 'industry', 'partner']
    df_weekly = df_clean.groupby(group_cols)['raw_score'].sum().reset_index()
    df_weekly.rename(columns={'raw_score': 'weekly_raw_score'}, inplace=True)
    
    # 2. Parsing
    season_max_map = df_weekly.groupby('season')['week'].max().to_dict()
    person_max_map = df_weekly.groupby(['season', 'celebrity_name'])['week'].max().to_dict()
    
    def apply_parser(row):
        return pd.Series(StatusParser.parse(row, season_max_map[row['season']], person_max_map[(row['season'], row['celebrity_name'])]))

    parsed = df_weekly.apply(apply_parser, axis=1)
    parsed.columns = ['exit_mode', 'final_rank_parsed', 'exit_week_parsed']
    df_weekly = pd.concat([df_weekly, parsed], axis=1)

    # 3. Accumulation
    df_weekly = _accumulate_scores(df_weekly)
    
    # 4. Metrics
    df_weekly['judge_metric'] = np.nan
    for s in df_weekly['season'].unique():
        mask = df_weekly['season'] == s
        df_weekly.loc[mask, 'judge_metric'] = _calculate_metrics(df_weekly.loc[mask], s)

    # 5. Global Rank Calculation
    rank_frames = []
    for s in df_weekly['season'].unique():
        s_data = df_weekly[df_weekly['season'] == s].copy()
        last_rows = s_data.sort_values('week').groupby('celebrity_name').tail(1)
        ranks = RankCalculator.calculate_season_ranks(last_rows)
        s_data['final_rank'] = s_data['celebrity_name'].map(ranks)
        rank_frames.append(s_data)
    
    df_final = pd.concat(rank_frames)

    # 6. RESULTS STANDARDIZATION (CRITICAL FIX)
    print("Executing Final Results Standardization...")

    # CASE A: Fix ELIMINATED (e.g. Joey S1: "3rd Place" -> "Eliminated Week 5")
    mask_elim = (df_final['exit_mode'] == 'eliminated') & \
                (~df_final['results'].str.contains('Eliminated', case=False, na=False)) & \
                (~df_final['results'].str.contains('Withdrew', case=False, na=False))
    
    if mask_elim.any():
        print(f" -> Standardizing {mask_elim.sum()} rows for 'Eliminated' status.")
        def fix_elim(row):
            w = int(row['exit_week_parsed']) if pd.notna(row['exit_week_parsed']) else row['week']
            return f"Eliminated Week {w}"
        df_final.loc[mask_elim, 'results'] = df_final[mask_elim].apply(fix_elim, axis=1)

    # CASE B: Fix FINALISTS (e.g. Drew Scott S25: "Eliminated Week 10" -> "4th Place")
    # Rule: If they are FINAL, but text says "Eliminated", overwrite with Rank.
    mask_final = (df_final['exit_mode'] == 'final') & \
                 (df_final['results'].str.contains('Eliminated', case=False, na=False))
    
    if mask_final.any():
        print(f" -> Standardizing {mask_final.sum()} rows for 'Final' status (Drew Scott Case).")
        def fix_final(row):
            # Use the calculated Global Rank to generate text
            return get_ordinal_text(row['final_rank'])
        df_final.loc[mask_final, 'results'] = df_final[mask_final].apply(fix_final, axis=1)

    # 7. Final Polish
    keep_cols = ['season', 'week', 'celebrity_name', 'results', 'age', 'industry', 'partner', 
                 'weekly_raw_score', 'effective_score', 'is_accumulated',
                 'exit_mode', 'final_rank', 'exit_week_parsed', 'judge_metric']
    
    return df_final[keep_cols].sort_values(by=['season', 'celebrity_name', 'week'])



# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    DATA_PATH = r"D:/Files/Study/code/DataProcessing/assessment_models/Contest/Data_Processing/2026_MCM_Problem_C_Data.csv"
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
    else:
        print("Running Final Pipeline...")
        df_raw = pd.read_csv(DATA_PATH)
        df_clean = clean_dwts_data(df_raw)
        df_final = process_ranking_features(df_clean)
        
        # DEBUG CHECK: JOEY S1
        print("\n--- CHECK: JOEY S1 W5 ---")
        joey = df_final[(df_final['season']==1) & (df_final['celebrity_name']=='Joey McIntyre') & (df_final['week']==5)]
        if not joey.empty:
            print(joey[['season', 'week', 'results', 'exit_mode', 'final_rank']].to_string(index=False))
        
        output_csv = "dwts_model_ready_final.csv"
        df_final.to_csv(output_csv, index=False)
        print(f"\nSaved to {output_csv}")