import pandas as pd
import numpy as np
import re
import warnings

def extract_placement_numeric(val):
    """Helper: Extract pure numbers from placement strings."""
    if pd.isna(val): return np.nan
    match = re.search(r'\d+', str(val))
    return int(match.group()) if match else np.nan

def clean_dwts_data(df_raw):
    """
    Physical Cleaning Layer.
    Converts raw CSV into a clean, long-format DataFrame with correct types.
    """
    df = df_raw.copy()

    # --- DEFENSE 1: Check Input Integrity ---
    df.columns = df.columns.str.strip() # Remove header whitespace
    
    # Required columns map
    rename_map = {
        'celebrity_age_during_season': 'age',
        'celebrity_industry': 'industry',
        'ballroom_partner': 'partner'
    }
    
    missing = [c for c in rename_map.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"CRITICAL ERROR: Input CSV missing required columns: {missing}")

    # 1. Rename
    df.rename(columns=rename_map, inplace=True)

    # 2. Type Validation (Age)
    # Coerce errors to NaN. If we lose too much data, warn the user.
    original_nans = df['age'].isna().sum()
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    new_nans = df['age'].isna().sum()
    
    if new_nans > original_nans:
         warnings.warn(f"Warning: {new_nans - original_nans} 'age' values were non-numeric and converted to NaN.")

    # 3. Global Text Strip
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())

    # 4. Wide to Long Transformation
    score_cols = [c for c in df.columns if 'judge' in c and 'score' in c]
    id_vars = [c for c in df.columns if c not in score_cols]

    df_long = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=score_cols,
        var_name='week_judge_id',
        value_name='raw_score'
    )

    # 5. Score Validation
    df_long['raw_score'] = pd.to_numeric(df_long['raw_score'], errors='coerce')
    
    # DEFENSE 2: Check for Impossible Scores
    if (df_long['raw_score'] < 0).any():
        raise ValueError("CRITICAL ERROR: Negative scores detected in raw data.")

    # Filter: Valid scores only (>0). 0 usually means didn't dance.
    df_clean = df_long[df_long['raw_score'] > 0].copy()

    # 6. Extract Metadata (Week & Judge ID)
    df_clean['week'] = df_clean['week_judge_id'].str.extract(r'week(\d+)_').astype(int)
    df_clean['judge_idx'] = df_clean['week_judge_id'].str.extract(r'judge(\d+)_').astype(int)

    # 7. Standardize 'Withdrew' timing in 'results' string
    # Logic: Inject "Eliminated Week X" into the string so Parsers can find the timing.
    # But KEEP the word 'Withdrew' so StatusParser can set exit_mode='withdrew'.
    withdrew_mask = df_clean['results'].str.lower() == 'withdrew'
    withdrew_people = df_clean.loc[withdrew_mask, 'celebrity_name'].unique()

    for person in withdrew_people:
        person_seasons = df_clean.loc[df_clean['celebrity_name'] == person, 'season'].unique()
        for season in person_seasons:
            mask = (df_clean['celebrity_name'] == person) & \
                   (df_clean['season'] == season) & \
                   (df_clean['results'].str.lower() == 'withdrew')
            
            if mask.any():
                last_week = df_clean.loc[mask, 'week'].max()
                # Rule: Withdrew at Week X -> Left at Week X (or X+1 depending on timing, assuming X here)
                # To be safe and consistent with elimination logic:
                # If they danced in Week X and withdrew, they are gone by Week X+1 results.
                # We mark the effective exit week.
                elim_week = last_week 
                new_status = f"Eliminated Week {elim_week} (Withdrew)"
                df_clean.loc[mask, 'results'] = new_status

    return df_clean.sort_values(by=['season', 'week', 'celebrity_name', 'judge_idx'])