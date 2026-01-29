import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf
from scipy.signal import periodogram
import warnings

# 屏蔽统计测试中除以零可能产生的警告，保持日志整洁
warnings.filterwarnings("ignore")

def identify_trend_shape(series: pd.Series):
    """
    [功能]: 识别数据的最佳趋势形态（均值型、线性型、抛物线型）。
    
    [原理]: 对三种假设模型进行 OLS 回归，比较 BIC (Bayesian Information Criterion)。
           利用 Schwarz Weights 计算后验概率作为置信度。
    
    [置信度]: W_i = exp(-0.5 * Delta_BIC) / Sum(exp(-0.5 * Delta_BIC))
             代表该模型是真实模型的概率。
    
    Returns:
        best_shape (str): 'mean', 'linear', 'quadratic'
        confidence (float): 0.0 ~ 1.0
        details (dict): 包含各模型的 BIC 值
    """
    if not isinstance(series, pd.Series):
        raise ValueError(f"Input must be a pandas Series, got {type(series)}.")
    if len(series) < 5:
        raise ValueError("Series length is too short to fit quadratic trend (min 5 samples).")

    y = series.values
    t = np.arange(len(series))
    
    # 准备三种设计矩阵
    X_mean = np.ones_like(t).reshape(-1, 1)                  # y = c
    X_linear = sm.add_constant(t)                            # y = c + at
    X_quad = sm.add_constant(np.column_stack((t, t**2)))     # y = c + at + bt^2
    
    results = {}
    models = {
        'mean': X_mean,
        'linear': X_linear,
        'quadratic': X_quad
    }
    
    # 计算所有模型的 BIC
    min_bic = float('inf')
    bic_scores = {}
    
    for name, X in models.items():
        try:
            model = sm.OLS(y, X).fit()
            bic = model.bic
            bic_scores[name] = bic
            if bic < min_bic:
                min_bic = bic
        except Exception as e:
            raise ValueError(f"OLS fitting failed for shape '{name}': {str(e)}")

    # 计算 Schwarz Weights (置信度)
    # W_i = exp(-0.5 * (BIC_i - Min_BIC)) / Sum(...)
    weights = {}
    denom = 0
    for name, bic in bic_scores.items():
        delta = bic - min_bic
        # 防止 exp 溢出，delta >= 0
        w = np.exp(-0.5 * delta)
        weights[name] = w
        denom += w
        
    best_shape = min(bic_scores, key=bic_scores.get)
    confidence = weights[best_shape] / denom
    
    return best_shape, confidence, bic_scores


def check_stationarity_combined(series: pd.Series, max_diff=2):
    """
    [功能]: 结合 ADF 和 KPSS 进行保守的平稳性检验。
    
    [策略]: 
       - 只有当 ADF 拒绝单位根 (p<0.05) 且 KPSS 无法拒绝平稳性 (p>0.05) 时，才认为平稳 (d=0)。
       - 任何冲突情况均建议差分 (d=1)。
    
    Returns:
        suggested_d (int): 0 或 1
        report (dict): 详细统计量
    """
    clean_series = series.dropna()
    if clean_series.nunique() <= 1:
         # 常数序列，无需差分
        return 0, {'note': 'Constant series'}

    # 1. ADF Test (H0: Non-stationary)
    try:
        adf_res = adfuller(clean_series, autolag='AIC')
        adf_p = adf_res[1]
    except Exception:
        adf_p = 1.0 # 假设不平稳

    # 2. KPSS Test (H0: Stationary)
    try:
        kpss_res = kpss(clean_series, regression='c', nlags="auto")
        kpss_p = kpss_res[1]
    except Exception:
        kpss_p = 0.0 # 假设不平稳

    # 判定逻辑
    is_adf_stationary = adf_p < 0.05
    is_kpss_stationary = kpss_p >= 0.05
    
    if is_adf_stationary and is_kpss_stationary:
        d = 0
    else:
        d = 1 # 保守策略
        
    report = {
        'adf_p_value': adf_p,
        'kpss_p_value': kpss_p,
        'adf_stationary': is_adf_stationary,
        'kpss_stationary': is_kpss_stationary
    }
    
    return d, report


def analyze_periodicity(series: pd.Series, method='acf', fs=1):
    """
    [功能]: 检测主要周期。
    
    [接口]:
        - method='acf': 自相关法，寻找显著整数周期。
        - method='fft': 频谱法，寻找能量最强频率。
    
    [置信度]:
        - ACF: 峰值处的自相关系数 (Abs Correlation)。
        - FFT: 峰值能量占比 (Spectral Power Ratio)。
    
    Returns:
        period (int/float): 检测到的周期
        confidence (float): 置信度
    """
    clean_series = series.interpolate().dropna()
    # 去线性趋势以免干扰周期检测
    detrended = clean_series - np.polyval(np.polyfit(np.arange(len(clean_series)), clean_series, 1), np.arange(len(clean_series)))
    
    if method == 'acf':
        # 计算 ACF，最大滞后设为长度的 1/2
        nlags = min(len(series) // 2, 60)
        acf_vals = acf(detrended, nlags=nlags, fft=True)
        
        # 寻找峰值 (忽略 lag=0)
        # 简单算法：找局部最大值
        peaks = []
        for i in range(1, len(acf_vals)-1):
            if acf_vals[i] > acf_vals[i-1] and acf_vals[i] > acf_vals[i+1]:
                peaks.append((i, acf_vals[i]))
        
        if not peaks:
            return 1, 0.0 # 无明显周期
            
        # 按相关系数排序，取最大的
        best_lag, conf = sorted(peaks, key=lambda x: x[1], reverse=True)[0]
        
        # 阈值检查：如果最大相关系数都很小 (<0.2)，认为无周期
        if conf < 0.2:
             return 1, conf
             
        return int(best_lag), float(conf)
        
    elif method == 'fft':
        # 补零以提高分辨率 (Padding)
        pad_len = len(detrended) * 10 
        f, Pxx = periodogram(detrended, fs=fs, nfft=pad_len)
        
        # 忽略极低频 (Trend)
        # 假设最小周期为 3 天，即频率 < 1/3
        mask = f > (1/100) 
        f_search = f[mask]
        P_search = Pxx[mask]
        
        if len(P_search) == 0:
            return 1, 0.0
            
        idx = np.argmax(P_search)
        dominant_freq = f_search[idx]
        peak_power = P_search[idx]
        total_power = np.sum(P_search)
        
        period = 1 / dominant_freq
        confidence = peak_power / total_power # 能量占比
        
        return float(period), float(confidence)
    
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'acf' or 'fft'.")


def construct_advanced_exog(index: pd.Index, 
                            trend_type='linear', 
                            period_list=None, 
                            fourier_order=2):
    """
    [功能]: 构造用于 SARIMAX 的混合外生变量矩阵。
    
    Args:
        index: 时间索引
        trend_type: 'mean' (无Trend列), 'linear' (t), 'quadratic' (t, t^2)
        period_list: list of periods e.g. [45, 7]. None provided means no fourier terms.
        fourier_order: K value for Fourier series.
        
    Returns:
        pd.DataFrame or None
    """
    t = np.arange(len(index))
    exog_dict = {}
    
    # 1. 添加趋势项
    if trend_type == 'linear':
        exog_dict['trend_t'] = t
    elif trend_type == 'quadratic':
        exog_dict['trend_t'] = t
        exog_dict['trend_t2'] = t**2
    elif trend_type == 'mean':
        pass # 不添加列，依赖模型自身的 const
    else:
        raise ValueError(f"Unsupported trend_type: {trend_type}")
        
    # 2. 添加傅里叶周期项
    if period_list:
        if not isinstance(period_list, list):
            period_list = [period_list]
            
        for p in period_list:
            if p <= 1: continue # 忽略无效周期
            for k in range(1, fourier_order + 1):
                exog_dict[f'sin_{p}_{k}'] = np.sin(2 * np.pi * k * t / p)
                exog_dict[f'cos_{p}_{k}'] = np.cos(2 * np.pi * k * t / p)
    
    # 转化为 DataFrame
    if not exog_dict:
        return None
        
    return pd.DataFrame(exog_dict, index=index)