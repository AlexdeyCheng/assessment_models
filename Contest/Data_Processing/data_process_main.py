import pandas as pd
import os
import warnings
from data_clear import clean_dwts_data
from data_ranking import process_ranking_features

def generate_data_dictionary(df, output_file="data_dictionary.txt"):
    """Generates data dictionary."""
    definitions = {
        "season": "Season number.",
        "week": "Week number.",
        "celebrity_name": "Contestant name.",
        "age": "Contestant age.",
        "industry": "Contestant profession.",
        "partner": "Professional partner.",
        "results": "Original raw status text.",
        "weekly_raw_score": "Sum of judge scores for the week.",
        "effective_score": "Score used for elimination. Includes Week 1 score if W1 had no eliminations.",
        "is_cumulative": "True if score includes previous week.",
        "exit_mode": "Canonical status: 'final', 'eliminated', 'withdrew', 'safe'.",
        "final_rank_parsed": "Parsed rank (1.0-5.0) for finalists.",
        "exit_week_parsed": "Week number when the contestant left.",
        "judge_metric": "Normalized Score. Rank (1=Best) for early/late seasons, Percentage for middle.",
        "audience_rank_proxy": "Target: 0.0=Safe, 1.0+=Rank, NaN=Eliminated.",
        "is_eliminated": "Boolean flag for elimination event."
    }

    with open(output_file, "w") as f:
        f.write("DWTS Processed Data Dictionary\n==============================\n\n")
        for col in df.columns:
            f.write(f"Column: {col}\nDesc:   {definitions.get(col, '')}\nType:   {df[col].dtype}\n{'-'*40}\n")
    print(f"Generated {output_file}")

def main(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print("Step 1: Physical Cleaning...")
    df_raw = pd.read_csv(data_path)
    if df_raw.empty: raise ValueError("Input CSV is empty.")
    
    df_clean = clean_dwts_data(df_raw)
    
    print("Step 2: Processing Logic (Canonical Parsing)...")
    df_final = process_ranking_features(df_clean)
    
    # --- FINAL VALIDATION CHECKS (Defensive) ---
    print("\nStep 3: Validation...")
    
    # Check 1: Data Volume
    if len(df_final) < 2000:
        raise ValueError(f"Output rows too low ({len(df_final)}). Aggregation error likely.")

    # Check 2: Exit Modes
    modes = df_final['exit_mode'].unique()
    print(f"Detected Exit Modes: {modes}")
    if 'final' not in modes or 'eliminated' not in modes:
        warnings.warn("Warning: Missing 'final' or 'eliminated' in exit_mode. Check parser.")

    # Check 3: Rank Targets
    ranks = df_final['audience_rank_proxy'].dropna().unique()
    print(f"Detected Ranks in Target: {sorted(ranks)}")
    if 1.0 not in ranks:
        raise ValueError("CRITICAL: No '1.0' (Winner) found in target variable.")

    # Check 4: Accumulation Logic
    # Verify Season 1 (Known: No elim in W1 -> Accumulate)
    s1_w2 = df_final[(df_final['season']==1) & (df_final['week']==2)]
    if not s1_w2['is_cumulative'].all():
        warnings.warn("Warning: Season 1 Week 2 should be cumulative but isn't. Check logic.")

    # Save
    output_csv = "dwts_model_ready.csv"
    df_final.to_csv(output_csv, index=False)
    generate_data_dictionary(df_final)
    print(f"\nSUCCESS. Saved to {output_csv}. Shape: {df_final.shape}")

if __name__ == "__main__":
    DATA_PATH = r"D:/Files/Study/code/DataProcessing/assessment_models/Contest/Data_Processing/2026_MCM_Problem_C_Data.csv"
    main(DATA_PATH)