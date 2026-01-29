import pandas as pd
import numpy as np
import feature_engineering_aux as aux
import sarimax_solver as solver
import visualization_module as vis 
import warnings
import os

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# ================= 全局配置 (Configuration) =================
CONFIG = {
    # 请修改为您真实的文件路径 (注意反斜杠转义或使用正斜杠)
    'DATA_PATH': "D:/Files/Study/code/DataProcessing/assessment_models/W_clean.csv",
    
    'FORECAST_DAYS': 59,          # 预测未来天数
    'MAX_P': 5,                   # Grid Search 最大 P
    'MAX_Q': 5,                   # Grid Search 最大 Q
    'CRITERION': 'aic',           # 选优标准
    'NORMALIZE_OUTPUT': False,    # 是否强制归一化到100% (默认False，保留原始误差)
    'OUTPUT_FORECAST': 'final_forecast.csv',
    'OUTPUT_DIAGNOSTICS': 'model_diagnostics.csv'
}
# ==========================================================

def load_data_simple(filepath):
    """
    [功能]: 智能读取数据，兼容 CSV (已清洗) 和 Excel (原始)。
    """
    print(f"  [IO] Loading data from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件未找到: {filepath}")

    try:
        # ================= 分支 A: 读取清洗后的 CSV =================
        if filepath.endswith('.csv'):
            # 清洗后的数据通常不需要 header=1，列名也已经是 A-X
            df = pd.read_csv(filepath)
            
            # 确保时间正序 (防止 CSV 保存时乱序)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.sort_values('Date', ascending=True, inplace=True)
            
            print("  [IO] Detected CSV format. Loaded directly.")

        # ================= 分支 B: 读取原始 Excel =================
        else:
            # 原始 Excel 逻辑: header=1, 需要列名映射, 需要反转
            df = pd.read_excel(filepath, header=1)
            
            # 清洗列名
            df.columns = df.columns.map(lambda x: str(x).strip())
            
            # 映射列名
            rename_map = {
                '1 try': 'A', '2 tries': 'B', '3 tries': 'C', '4 tries': 'D',
                '5 tries': 'E', '6 tries': 'F', '7 or more tries (X)': 'X'
            }
            df.rename(columns=rename_map, inplace=True)
            
            # 时间排序
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.sort_values('Date', ascending=True, inplace=True)
                print("  [IO] Sorted data by Date (Ascending).")
            else:
                df = df.iloc[::-1] # 盲猜倒序
                print("  [IO] Date column not found, assuming reverse chronological order.")

        # ================= 公共后处理 =================
        # 重置索引
        df.reset_index(drop=True, inplace=True)
        
        # 检查必要列
        required_cols = ['A', 'B', 'C', 'D', 'E', 'F', 'X']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        print(f"  [IO] Data loaded. Shape: {df.shape}")
        return df
        
    except Exception as e:
        raise ValueError(f"Failed to load data: {str(e)}")
def post_process_results(forecast_df, do_normalize=False):
    """
    [功能]: 后处理预测结果 (截断负值 + 可选归一化)。
    """
    processed_df = forecast_df.copy()
    
    # 1. 物理约束：截断负值 (Clip < 0 to 0)
    processed_df[processed_df < 0] = 0
    
    # 2. 计算总和
    cols = [c for c in processed_df.columns if c in ['A','B','C','D','E','F','X']]
    current_sum = processed_df[cols].sum(axis=1)
    
    # 3. 归一化 (可选)
    if do_normalize:
        print("  [Post-Process] Applying Normalization (Sum -> 100%)...")
        current_sum[current_sum == 0] = 1 
        for col in cols:
            processed_df[col] = (processed_df[col] / current_sum) * 100
        processed_df['Sum_Check'] = 100.0
    else:
        print("  [Post-Process] Skipping Normalization (Keeping raw model outputs).")
        processed_df['Sum_Check'] = current_sum
        
    return processed_df

def run_prediction_pipeline(filepath):
    """
    [主控函数]: 串联整个预测流程。
    """
    print("======== Starting MCM Wordle Prediction Pipeline ========")
    
    # 1. 加载数据
    df = load_data_simple(filepath)
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'X']
    
    # 容器初始化
    raw_forecasts = pd.DataFrame()     
    full_results_dict = {}             
    diagnostics_rows = []              
    
    # 2. 循环处理每一列
    for col in categories:
        # 这里的 df 已经是映射好 A-X 列名的了
        if col not in df.columns:
            print(f"  [Warning] Column {col} not found. Skipping.")
            continue
            
        print(f"\n>>> Processing Category: {col} <<<")
        series = df[col]
        
        # --- Step A: 特征工程 (Aux) ---
        trend_type, trend_conf, _ = aux.identify_trend_shape(series)
        period, period_conf = aux.analyze_periodicity(series, method='acf')
        d, stat_report = aux.check_stationarity_combined(series)
        
        print(f"  [Aux] Trend: {trend_type} | Period: {period} | Diff d={d}")
        
        # 构造外生变量
        exog_train = aux.construct_advanced_exog(series.index, trend_type=trend_type, period_list=[period])
        
        last_idx = series.index[-1]
        future_idx = pd.RangeIndex(start=last_idx + 1, stop=last_idx + 1 + CONFIG['FORECAST_DAYS'])
        exog_future = aux.construct_advanced_exog(future_idx, trend_type=trend_type, period_list=[period])
        
        # --- Step B: 核心求解 (Solver) ---
        search_cfg = {'max_p': CONFIG['MAX_P'], 'max_q': CONFIG['MAX_Q'], 'criterion': CONFIG['CRITERION']}
        solve_res = solver.solve_single_series(series, exog_train, exog_future, d, CONFIG['FORECAST_DAYS'], search_config=search_cfg)
        
        # --- Step C: 数据收集 ---
        full_results_dict[col] = solve_res
        raw_forecasts[col] = solve_res['forecast_data']['Forecast'].values
        full_results_dict[col]['diagnostics']['Detected_Period'] = period
        full_results_dict[col]['diagnostics']['Trend_Type'] = trend_type

        # 收集诊断信息
        diag_data = {
            'Category': col,
            'Trend_Type': trend_type,
            'Detected_Period': period,
            'Best_Order': str(solve_res['order']),
            'LjungBox_Pval': solve_res['diagnostics']['lb_pvalue'],
            'Mean_Conf_Score': solve_res['forecast_data']['Confidence_Score'].mean()
        }
        diagnostics_rows.append(diag_data)

    # 3. 后处理 (Post-Processing)
    future_days_idx = pd.RangeIndex(start=len(df), stop=len(df) + CONFIG['FORECAST_DAYS'])
    raw_forecasts.index = future_days_idx
    
    print("\n>>> Executing Post-Processing...")
    final_forecast_df = post_process_results(raw_forecasts, do_normalize=CONFIG['NORMALIZE_OUTPUT'])
    
    # 4. 导出结果
    print(f"\n>>> Exporting Results...")
    final_forecast_df.to_csv(CONFIG['OUTPUT_FORECAST'])
    pd.DataFrame(diagnostics_rows).to_csv(CONFIG['OUTPUT_DIAGNOSTICS'], index=False)
    print(f"  - Files saved: {CONFIG['OUTPUT_FORECAST']}, {CONFIG['OUTPUT_DIAGNOSTICS']}")
    
    # 5. 可视化 (Visualization)
    print(f"\n>>> Generating Plots (Theme: Blue-Purple)...")
    vis.run_all_plots(full_results_dict, final_forecast_df)
    
    print("\n======== Pipeline Completed Successfully ========")
    return final_forecast_df

if __name__ == "__main__":
    # 直接使用 CONFIG 中的路径
    try:
        run_prediction_pipeline(CONFIG['DATA_PATH'])
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")