import pandas as pd
import os
import warnings
import sys

# Ensure local modules are importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_clear import clean_dwts_data
from data_ranking import process_ranking_features

def generate_data_dictionary(df, output_file="data_dictionary.txt"):
    definitions = {
        "season": "Season number (1-33).",
        "week": "Week number (Time Index).",
        "celebrity_name": "Unique identifier for contestant.",
        "results": "Original text status (Withdrew annotated with exit week).",
        "age": "Contestant Age.",
        "industry": "Contestant Industry.",
        "partner": "Professional Partner Name.",
        "weekly_raw_score": "Sum of judges' scores for this week.",
        "effective_score": "Score used for elimination. Includes W1 score if W2 is cumulative.",
        "is_accumulated": "Boolean. True if effective_score = W1 + W2.",
        "exit_mode": "'safe' (active), 'final', 'eliminated', 'withdrew'.",
        "final_rank": "Global Rank. 0=Withdrew. 1+=Rank. Calculated based on text or exit timing.",
        "exit_week_parsed": "The week the contestant left the competition.",
        "judge_metric": "Standardized Judge Score. S1-2/S28+=Rank(Min), S3-27=Percentage."
    }

    with open(output_file, "w") as f:
        f.write("DWTS Processed Data Dictionary\n==============================\n\n")
        for col in df.columns:
            f.write(f"Column: {col}\n")
            f.write(f"Desc:   {definitions.get(col, 'No definition.')}\n")
            f.write(f"Type:   {df[col].dtype}\n")
            f.write("-" * 40 + "\n")
    print(f"Dictionary generated: {output_file}")

def main(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print("--- Step 1: Physical Cleaning ---")
    df_raw = pd.read_csv(data_path)
    if df_raw.empty: raise ValueError("Input CSV is empty.")
    
    df_clean = clean_dwts_data(df_raw)
    print(f"Cleaned Data Shape: {df_clean.shape}")

    print("--- Step 2: Logic Processing & Ranking ---")
    df_final = process_ranking_features(df_clean)
    
    # --- FINAL VALIDATION CHECKS ---
    print("\n--- Step 3: Validation & QA ---")
    
    # 1. Check Row Count
    if len(df_final) < 2000:
        warnings.warn("Warning: Resulting dataset has < 2000 rows. Check aggregation.")

    # 2. Check Withdrew Logic (Rank should be 0)
    withdrew_df = df_final[df_final['exit_mode'] == 'withdrew']
    if not withdrew_df.empty:
        w_ranks = withdrew_df['final_rank'].unique()
        print(f"Withdrew Ranks (Expect [0.]): {w_ranks}")
        if any(r != 0 for r in w_ranks):
            raise ValueError("CRITICAL: Some withdrew contestants have non-zero rank!")
    
    # 3. Check Accumulation Logic (Season 1 Test Case)
    # Season 1: No elimination in Week 1 -> Week 2 MUST be accumulated.
    s1_w2 = df_final[(df_final['season'] == 1) & (df_final['week'] == 2)]
    if not s1_w2.empty:
        accum_ratio = s1_w2['is_accumulated'].mean()
        print(f"Season 1 Week 2 Accumulation Rate: {accum_ratio*100:.1f}%")
        if accum_ratio < 1.0:
             warnings.warn("Warning: Season 1 Week 2 should be 100% accumulated. Logic check needed.")
    
    # 4. Check Final Rank Validity
    if df_final['final_rank'].isna().any():
        n_na = df_final['final_rank'].isna().sum()
        warnings.warn(f"Warning: {n_na} rows have NaN Final Rank (Should be filled for all).")

    # Save
    output_csv = "dwts_model_ready_final.csv"
    df_final.to_csv(output_csv, index=False)
    generate_data_dictionary(df_final)
    
    print(f"\nSUCCESS. Data saved to: {output_csv}")
    print(f"Final Shape: {df_final.shape}")
    print("Columns:", df_final.columns.tolist())

if __name__ == "__main__":
    # Update this path to your actual file location
    DATA_PATH = r"D:/Files/Study/code/DataProcessing/assessment_models/Contest/Data_Processing/2026_MCM_Problem_C_Data.csv"
    main(DATA_PATH)