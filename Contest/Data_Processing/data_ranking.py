import pandas as pd
import numpy as np
import re
import warnings

# --- CANONICAL REPRESENTATION LAYER ---

class StatusParser:
    """
    Decouples text parsing from logic.
    """
    
    @staticmethod
    def parse(row, season_max_week, person_max_week):
        res = str(row['results']).strip() # Ensure strip happens here too
        current_week = row['week']
        
        # 0. Active Check
        # If current week is strictly less than person's max week, they are SAFE.
        if current_week < person_max_week:
             return 'safe', 0.0, np.nan

        # --- LAST WEEK LOGIC ---

        # 1. Parse Withdrew
        if 'withdrew' in res.lower():
            week_match = re.search(r'Week\s+(\d+)', res, re.IGNORECASE)
            exit_w = int(week_match.group(1)) if week_match else current_week
            return 'withdrew', np.nan, exit_w

        # 2. Parse Specific Ranks (1st Place, 2nd Place...)
        # FIXED REGEX: Uses \s+ which covers \xa0 in Python 3, but we add robustness.
        # We also allow case insensitivity and loose spacing.
        rank_match = re.search(r'(\d+)(?:st|nd|rd|th)[\s\xa0]+Place', res, re.IGNORECASE)
        
        if rank_match:
            rank_val = float(rank_match.group(1))
            
            # Top 3 are always Final
            if rank_val <= 3.0:
                return 'final', rank_val, current_week
            
            # 4th/5th need to be at Season Finale
            if current_week >= season_max_week: 
                return 'final', rank_val, current_week
            else:
                return 'eliminated', np.nan, current_week

        # 3. Parse Standard Elimination
        if 'eliminated' in res.lower():
            week_match = re.search(r'Week\s+(\d+)', res, re.IGNORECASE)
            exit_w = int(week_match.group(1)) if week_match else current_week
            return 'eliminated', np.nan, exit_w

        # 4. Emergency Fallback for "Place" without Rank (Data Error?)
        if 'place' in res.lower():
            # If we see "Place" but regex failed, it's likely a weird formatting issue.
            # Assume Finalist if Logic 0 passed (it's their last week).
            warnings.warn(f"Warning: ambiguous 'Place' string found: '{res}'. Marking as potential final.")
            # Try to extract ANY digit
            digit_match = re.search(r'(\d+)', res)
            if digit_match:
                r_val = float(digit_match.group(1))
                return 'final', r_val, current_week

        # 5. True Fallback
        return 'eliminated', np.nan, current_week


# --- LOGIC LAYER ---

def _calculate_metrics(df_slice, season_num):
    if (season_num <= 2) or (season_num >= 28):
        return df_slice.groupby('week')['effective_score'].rank(method='min', ascending=False)
    else:
        week_sums = df_slice.groupby('week')['effective_score'].transform('sum')
        return df_slice['effective_score'] / week_sums

def _accumulate_scores(df_weekly):
    df_out = df_weekly.copy()
    df_out['effective_score'] = df_out['weekly_raw_score']
    df_out['is_cumulative'] = False
    
    seasons = df_out['season'].unique()
    
    for s in seasons:
        s_data = df_out[df_out['season'] == s]
        
        # LOGIC FIX:
        # A person breaks accumulation ONLY IF:
        # 1. Their mode is 'eliminated' or 'withdrew'
        # 2. AND their exit_week is ACTUALLY 1.
        
        mask_w1_exit = (
            (s_data['exit_mode'].isin(['eliminated', 'withdrew'])) & 
            (s_data['exit_week_parsed'] == 1)
        )
        
        has_w1_exit = mask_w1_exit.any()
        
        if not has_w1_exit:
            w1_scores = s_data[s_data['week'] == 1].set_index('celebrity_name')['weekly_raw_score']
            mask_w2 = (df_out['season'] == s) & (df_out['week'] == 2)
            df_out.loc[mask_w2, 'effective_score'] += df_out.loc[mask_w2, 'celebrity_name'].map(w1_scores).fillna(0)
            df_out.loc[mask_w2, 'is_cumulative'] = True
            
    return df_out

def process_ranking_features(df_clean):
    # 1. Aggregation
    group_cols = ['season', 'week', 'celebrity_name', 'results', 'age', 'industry', 'partner']
    df_weekly = df_clean.groupby(group_cols)['raw_score'].sum().reset_index()
    df_weekly.rename(columns={'raw_score': 'weekly_raw_score'}, inplace=True)
    
    # 2. Canonical Parsing
    season_max_map = df_weekly.groupby('season')['week'].max().to_dict()
    person_max_map = df_weekly.groupby(['season', 'celebrity_name'])['week'].max().to_dict()
    
    # [DEBUG PRINT INJECTION]
    # Check Kelly Monaco's Max Week
    k_max = person_max_map.get((1, 'Kelly Monaco'), -1)
    if k_max != -1 and k_max < 6:
        print(f"DEBUG WARNING: Kelly Monaco Max Week detected as {k_max}. Expecting 6.")

    def apply_parser(row):
        s_max = season_max_map[row['season']]
        p_max = person_max_map[(row['season'], row['celebrity_name'])]
        return pd.Series(StatusParser.parse(row, s_max, p_max))

    parsed_data = df_weekly.apply(apply_parser, axis=1)
    parsed_data.columns = ['exit_mode', 'final_rank_parsed', 'exit_week_parsed']
    
    df_weekly = pd.concat([df_weekly, parsed_data], axis=1)

    # 3. Accumulation
    df_weekly = _accumulate_scores(df_weekly)

    # 4. Metrics
    df_weekly['judge_metric'] = np.nan
    seasons = df_weekly['season'].unique()
    for s in seasons:
        mask = df_weekly['season'] == s
        df_weekly.loc[mask, 'judge_metric'] = _calculate_metrics(df_weekly.loc[mask], s)

    # 5. Target Generation
    df_weekly['audience_rank_proxy'] = 0.0 
    
    mask_final = df_weekly['exit_mode'] == 'final'
    df_weekly.loc[mask_final, 'audience_rank_proxy'] = df_weekly.loc[mask_final, 'final_rank_parsed']
    
    mask_gone = df_weekly['exit_mode'].isin(['eliminated', 'withdrew'])
    df_weekly.loc[mask_gone, 'audience_rank_proxy'] = np.nan
    
    df_weekly['is_eliminated'] = mask_gone
    
    df_weekly.sort_values(by=['season', 'celebrity_name', 'week'], inplace=True)
    
    return df_weekly