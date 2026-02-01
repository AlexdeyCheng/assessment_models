import pandas as pd
import numpy as np
import re
import warnings

# --- 1. PARSER LAYER ---
class StatusParser:
    @staticmethod
    def parse(row, season_max_week, person_max_week):
        res = str(row['results']).strip()
        current_week = row['week']
        
        # Active Check
        if current_week < person_max_week:
             return 'safe', np.nan, np.nan

        # Parse Withdrew
        if 'withdrew' in res.lower():
            week_match = re.search(r'Week\s+(\d+)', res, re.IGNORECASE)
            exit_w = int(week_match.group(1)) if week_match else current_week
            return 'withdrew', np.nan, exit_w

        # Parse Ranks (1st Place...)
        rank_match = re.search(r'(\d+)(?:st|nd|rd|th)[\s\xa0]+Place', res, re.IGNORECASE)
        if rank_match:
            rank_val = float(rank_match.group(1))
            if rank_val <= 5.0:
                if rank_val <= 3.0 or current_week >= season_max_week:
                    return 'final', rank_val, current_week
                else:
                    return 'eliminated', np.nan, current_week

        # Parse Standard Elimination
        if 'eliminated' in res.lower():
            week_match = re.search(r'Week\s+(\d+)', res, re.IGNORECASE)
            exit_w = int(week_match.group(1)) if week_match else current_week
            return 'eliminated', np.nan, exit_w

        # Fallback
        if 'winner' in res.lower(): return 'final', 1.0, current_week
        if 'runner-up' in res.lower(): return 'final', 2.0, current_week
        return 'eliminated', np.nan, current_week

# --- 2. RANK CALCULATOR ---
class RankCalculator:
    @staticmethod
    def calculate_season_ranks(df_season):
        rank_map = {}
        # 1. Withdrew -> 0
        withdrew = df_season[df_season['exit_mode'] == 'withdrew']
        for name in withdrew['celebrity_name'].unique():
            rank_map[name] = 0.0
            
        # 2. Finalists
        finalists = df_season[df_season['exit_mode'] == 'final'].copy()
        for _, row in finalists.iterrows():
            rank_map[row['celebrity_name']] = row['final_rank_parsed']
            
        # 3. Eliminated (Sort by Exit Week Descending)
        if not finalists.empty:
            next_rank = finalists['final_rank_parsed'].max() + 1
        else:
            next_rank = 1.0
            
        eliminated = df_season[df_season['exit_mode'] == 'eliminated'].copy()
        exit_weeks = eliminated['exit_week_parsed'].sort_values(ascending=False).unique()
        
        current_rank = next_rank
        for w in exit_weeks:
            people = eliminated[eliminated['exit_week_parsed'] == w]['celebrity_name'].unique()
            for p in people:
                rank_map[p] = current_rank
            current_rank += len(people) # Skip numbers
            
        return rank_map

# --- 3. LOGIC LAYER ---
def _accumulate_scores(df_weekly):
    df_out = df_weekly.copy()
    df_out['effective_score'] = df_out['weekly_raw_score']
    df_out['is_accumulated'] = False
    
    for s in df_out['season'].unique():
        s_data = df_out[df_out['season'] == s]
        mask_w1_elim = (s_data['exit_mode'] == 'eliminated') & (s_data['exit_week_parsed'] == 1)
        
        if not mask_w1_elim.any(): # Only accumulate if NO elimination in W1
            w1_scores = s_data[s_data['week'] == 1].set_index('celebrity_name')['weekly_raw_score']
            mask_w2 = (df_out['season'] == s) & (df_out['week'] == 2)
            df_out.loc[mask_w2, 'effective_score'] += df_out.loc[mask_w2, 'celebrity_name'].map(w1_scores).fillna(0)
            df_out.loc[mask_w2, 'is_accumulated'] = True
    return df_out

def _calculate_metrics(df_slice, season_num):
    if (season_num <= 2) or (season_num >= 28):
        return df_slice.groupby('week')['effective_score'].rank(method='min', ascending=False)
    else:
        sums = df_slice.groupby('week')['effective_score'].transform('sum')
        return df_slice['effective_score'] / sums

def process_ranking_features(df_clean):
    # Aggregation
    group_cols = ['season', 'week', 'celebrity_name', 'results', 'age', 'industry', 'partner']
    df_weekly = df_clean.groupby(group_cols)['raw_score'].sum().reset_index()
    df_weekly.rename(columns={'raw_score': 'weekly_raw_score'}, inplace=True)
    
    # Parser
    season_max_map = df_weekly.groupby('season')['week'].max().to_dict()
    person_max_map = df_weekly.groupby(['season', 'celebrity_name'])['week'].max().to_dict()
    
    def apply_parser(row):
        return pd.Series(StatusParser.parse(row, season_max_map[row['season']], person_max_map[(row['season'], row['celebrity_name'])]))

    parsed = df_weekly.apply(apply_parser, axis=1)
    parsed.columns = ['exit_mode', 'final_rank_parsed', 'exit_week_parsed']
    df_weekly = pd.concat([df_weekly, parsed], axis=1)

    # Logic
    df_weekly = _accumulate_scores(df_weekly)
    
    # Metrics
    df_weekly['judge_metric'] = np.nan
    for s in df_weekly['season'].unique():
        mask = df_weekly['season'] == s
        df_weekly.loc[mask, 'judge_metric'] = _calculate_metrics(df_weekly.loc[mask], s)

    # Global Rank (The Missing Link!)
    rank_frames = []
    for s in df_weekly['season'].unique():
        s_data = df_weekly[df_weekly['season'] == s].copy()
        # Find last week for each person to determine their rank
        last_rows = s_data.sort_values('week').groupby('celebrity_name').tail(1)
        ranks = RankCalculator.calculate_season_ranks(last_rows)
        s_data['final_rank'] = s_data['celebrity_name'].map(ranks)
        rank_frames.append(s_data)
    
    df_final = pd.concat(rank_frames)

    # Column Filter (Strict Check)
    keep_cols = [
        'season', 'week', 'celebrity_name', 'results', 'age', 'industry', 'partner', 
        'weekly_raw_score', 'effective_score', 'is_accumulated',
        'exit_mode', 'final_rank', 'exit_week_parsed', 'judge_metric'
    ]
    
    # Check if 'final_rank' exists immediately
    if 'final_rank' not in df_final.columns:
         raise ValueError("CRITICAL ERROR: 'final_rank' was not created in data_ranking.py!")

    df_final = df_final[keep_cols]
    df_final.sort_values(by=['season', 'celebrity_name', 'week'], inplace=True)
    
    return df_final