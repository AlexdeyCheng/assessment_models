import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

# 屏蔽收敛警告，以免在Grid Search期间刷屏
warnings.filterwarnings("ignore")

def grid_search_sarimax(endog, exog=None, d=1, max_p=5, max_q=5, criterion='aic'):
    """
    [功能]: 遍历 (p, q) 寻找最优 SARIMAX 模型。
    
    [参数]:
        endog: 训练数据 (Series)
        exog: 外生变量矩阵 (DataFrame/None)
        d: 差分阶数 (int)
        max_p, max_q: 搜索上限 (默认5)
        criterion: 'aic' 或 'bic' (默认 'aic')
    
    [返回]:
        best_model_result: 拟合好的模型结果对象
        best_order: (p, d, q) 元组
        search_log: 搜索过程日志 (list of dict)
    """
    # 进度条描述
    print(f"  [Solver] Starting Grid Search (d={d}, criterion={criterion.upper()})...")
    
    best_score = float('inf')
    best_model_result = None
    best_order = None
    
    # 搜索空间：p, q in [0, max]
    # 使用 tqdm 显示进度
    p_values = range(0, max_p + 1)
    q_values = range(0, max_q + 1)
    total_iters = len(p_values) * len(q_values)
    
    with tqdm(total=total_iters, desc="  Fitting Models", leave=False) as pbar:
        for p in p_values:
            for q in q_values:
                # 跳过空模型 (如果 d=0 且 p=q=0)
                if p == 0 and q == 0 and d == 0:
                    pbar.update(1)
                    continue
                
                try:
                    # 构建模型
                    # 注意：seasonal_order 设为 (0,0,0,0)，因为长周期已通过 exog 处理
                    # enforce_stationarity=False: 允许非平稳，防止报错，依靠差分保证平稳
                    model = SARIMAX(endog, 
                                    exog=exog,
                                    order=(p, d, q),
                                    seasonal_order=(0, 0, 0, 0),
                                    trend='c' if exog is None else None, # 若有外生变量，通常移除内置趋势
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    
                    # 拟合 (最大迭代次数适当增加)
                    results = model.fit(disp=False, maxiter=200, method='lbfgs')
                    
                    # 获取评价指标
                    current_score = results.aic if criterion == 'aic' else results.bic
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_order = (p, d, q)
                        best_model_result = results
                        
                except Exception as e:
                    # 捕获矩阵不可逆 (LinAlgError) 或其他算术错误，跳过当前组合
                    pass
                
                pbar.update(1)
    
    if best_model_result is None:
        raise ValueError("All model combinations failed. Please check input data stability or reduce exog complexity.")
        
    print(f"  [Solver] Best Model Found: Order={best_order}, {criterion.upper()}={best_score:.2f}")
    return best_model_result, best_order


def diagnose_model(model_result):
    """
    [功能]: 对模型残差进行白噪声检验。
    
    [指标]:
        - Ljung-Box Test: H0 = 残差是白噪声 (无自相关)。
          如果 p-value > 0.05，无法拒绝H0 -> 模型良好。
          
    [返回]:
        dict: 包含 p-value, Q-stat, 残差序列
    """
    resid = model_result.resid
    
    # Ljung-Box 检验 (Lags取 10 或 长度的1/5)
    lags = [10]
    lb_test = acorr_ljungbox(resid, lags=lags, return_df=True)
    
    p_value = lb_test['lb_pvalue'].values[0]
    q_stat = lb_test['lb_stat'].values[0]
    
    return {
        'lb_pvalue': p_value,
        'lb_qstat': q_stat,
        'residuals': resid # 返回残差用于绘图
    }


def forecast_with_confidence(model_result, steps, exog_future=None):
    """
    [功能]: 生成预测值及全套置信度指标。
    
    [指标定义]:
        - Forecast: 点预测 (Point Forecast)
        - SE (Standard Error): 预测标准误
        - 95% CI: Lower/Upper Bound
        - RSE (Relative Standard Error): SE / Forecast
        - Confidence Score: 1 / (1 + RSE) (归一化置信分, 1为满分)
    
    [返回]:
        pd.DataFrame: 包含上述所有列，索引为未来的时间步
    """
    # 获取预测对象
    forecast_obj = model_result.get_forecast(steps=steps, exog=exog_future)
    
    # 提取基础数据
    pred_mean = forecast_obj.predicted_mean
    pred_se = forecast_obj.se_mean
    conf_int = forecast_obj.conf_int(alpha=0.05) # 95% CI
    
    # 构建 DataFrame
    df_res = pd.DataFrame(index=pred_mean.index)
    df_res['Forecast'] = pred_mean
    df_res['SE'] = pred_se
    df_res['CI_Lower'] = conf_int.iloc[:, 0]
    df_res['CI_Upper'] = conf_int.iloc[:, 1]
    df_res['CI_Width'] = df_res['CI_Upper'] - df_res['CI_Lower']
    
    # 计算衍生指标
    # 注意：防止除以0 (加一个极小值 eps)
    epsilon = 1e-6
    df_res['RSE'] = df_res['SE'] / (df_res['Forecast'].abs() + epsilon)
    
    # Confidence Score: 越接近1越好
    # 逻辑：当 SE=0 时 RSE=0 -> Score=1
    #       当 SE=Forecast 时 RSE=1 -> Score=0.5
    df_res['Confidence_Score'] = 1 / (1 + df_res['RSE'])
    
    return df_res


def solve_single_series(series, exog_train, exog_future, d, forecast_days, search_config=None):
    """
    [功能]: 求解器主入口。将搜索、诊断、预测串联起来。
    """
    if search_config is None:
        search_config = {'max_p': 5, 'max_q': 5, 'criterion': 'aic'}
        
    # 1. 网格搜索最佳模型
    best_model, order = grid_search_sarimax(
        endog=series,
        exog=exog_train,
        d=d,
        max_p=search_config['max_p'],
        max_q=search_config['max_q'],
        criterion=search_config['criterion']
    )
    
    # 2. 诊断
    diag_info = diagnose_model(best_model)
    
    # 3. 预测
    forecast_df = forecast_with_confidence(
        model_result=best_model,
        steps=forecast_days,
        exog_future=exog_future
    )
    
    return {
        'model_obj': best_model,
        'order': order,
        'diagnostics': diag_info,
        'forecast_data': forecast_df
    }