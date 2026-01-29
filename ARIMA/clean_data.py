import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# ================= 配置区 =================
# 输入文件路径 (请确保与 main_sarimax.py 中的路径一致)
RAW_DATA_PATH = "D:/Files/Study/code/DataProcessing/assessment_models/Data_processsing/2023_MCM_Problem_C_Data.xlsx"
# 输出文件路径
OUTPUT_CLEAN_PATH = "W_clean.csv"
# =========================================

def load_and_preprocess_raw(filepath):
    """
    读取原始 Excel，执行列名映射和时间排序。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件未找到: {filepath}")

    print(f"  [IO] Loading raw data from: {filepath}")
    # 1. 读取 Excel
    df = pd.read_excel(filepath, header=1)
    
    # 2. 清洗列名
    df.columns = df.columns.map(lambda x: str(x).strip())
    
    # 3. 映射列名
    rename_map = {
        '1 try': 'A', '2 tries': 'B', '3 tries': 'C', '4 tries': 'D',
        '5 tries': 'E', '6 tries': 'F', '7 or more tries (X)': 'X'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # 4. 时间正序排序 (Past -> Future)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', ascending=True, inplace=True)
    else:
        # 假设原始数据是倒序的
        df = df.iloc[::-1]
    
    df.reset_index(drop=True, inplace=True)
    return df

def clean_dataset(df):
    """
    执行两步清洗逻辑：
    1. 严重坏行修复 (Sum Check) -> ffill
    2. 离群点修复 (Outlier Z-score) -> Interpolate (Option B)
    3. 全局归一化
    """
    df_clean = df.copy()
    cols = ['A', 'B', 'C', 'D', 'E', 'F', 'X']
    
    print("  [Clean] Starting data cleaning process...")

    # --- Step 1: 严重坏行修复 (Rule 2) ---
    # 判据: |Sum - 100| > 4
    row_sums = df_clean[cols].sum(axis=1)
    bad_rows_mask = np.abs(row_sums - 100) > 4
    
    bad_count = bad_rows_mask.sum()
    if bad_count > 0:
        print(f"    - Found {bad_count} rows with severe sum deviation (>4%). Replacing with neighbor data.")
        # 将坏行设为 NaN
        df_clean.loc[bad_rows_mask, cols] = np.nan
        # 前向填充 (优先用前一天)，如果是第一行则后向填充
        df_clean[cols] = df_clean[cols].fillna(method='ffill').fillna(method='bfill')
    
    # --- Step 2: 离群点修复 (Rule 1 - Option B) ---
    # 使用滚动窗口检测局部突变
    total_outliers = 0
    
    for col in cols:
        # 滚动窗口: 7天 (能够覆盖一周的波动)
        rolling_mean = df_clean[col].rolling(window=7, center=True, min_periods=1).mean()
        rolling_std = df_clean[col].rolling(window=7, center=True, min_periods=1).std()
        
        # 防止除以0
        rolling_std[rolling_std == 0] = 1e-6
        
        # 计算 Z-score
        z_scores = np.abs((df_clean[col] - rolling_mean) / rolling_std)
        
        # 判据: 偏离均值超过 3倍标准差
        outliers = z_scores > 3
        
        if outliers.any():
            n_out = outliers.sum()
            total_outliers += n_out
            # print(f"    - Column {col}: Fixing {n_out} outliers via Interpolation.")
            
            # 设为 NaN 并使用线性插值
            df_clean.loc[outliers, col] = np.nan
            df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
            
    print(f"    - Total outliers interpolated: {total_outliers}")

    # --- Step 3: 全局归一化 ---
    # 插值后总和可能不再是 100，需重新归一化
    print("    - Applying final normalization (Sum -> 100%).")
    current_sums = df_clean[cols].sum(axis=1)
    current_sums[current_sums == 0] = 1 # 防除0
    
    for col in cols:
        df_clean[col] = (df_clean[col] / current_sums) * 100
        
    return df_clean

if __name__ == "__main__":
    try:
        # 1. 加载
        df_raw = load_and_preprocess_raw(RAW_DATA_PATH)
        
        # 2. 清洗
        df_clean = clean_dataset(df_raw)
        
        # 3. 保存
        df_clean.to_csv(OUTPUT_CLEAN_PATH, index=False)
        print(f"\n[Success] Cleaned data saved to: {os.path.abspath(OUTPUT_CLEAN_PATH)}")
        
        # 4. 验证打印
        print("\n[Validation] First 5 rows of cleaned data:")
        print(df_clean[['Date'] + ['A','B','C','D','E','F','X']].head())
        
        print("\n[Validation] Row Sum Stats:")
        print(df_clean[['A','B','C','D','E','F','X']].sum(axis=1).describe())

    except Exception as e:
        print(f"\n[Error] {str(e)}")