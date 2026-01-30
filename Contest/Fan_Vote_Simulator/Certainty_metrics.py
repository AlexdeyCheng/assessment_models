import numpy as np
import scipy.stats as stats
import pandas as pd

class CertaintyAnalyzer:
    """
    专门用于量化观众投票估算确定性的分析器。
    输入: 采样矩阵 (Samples Matrix), 形状为 (N_samples, N_candidates)
    输出: 每位选手的确定性指标 (DataFrame)
    """
    
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level

    def analyze(self, samples, candidate_names):
        """
        执行核心计算算法。
        """
        if samples is None or len(samples) == 0:
            return None
        
        # 确保 samples 是 numpy array
        samples = np.array(samples)
        n_samples, n_candidates = samples.shape
        
        # 1. 基础波动 (Standard Deviation)
        # 衡量指标: Sigma (越小越确信)
        std_devs = np.std(samples, axis=0)
        
        # 2. 可行域范围 (Feasible Range Width)
        # 衡量指标: Range (Max - Min). 
        # 物理含义: 在满足所有约束的前提下，该选手得票率的理论上下限距离。
        min_vals = np.min(samples, axis=0)
        max_vals = np.max(samples, axis=0)
        range_widths = max_vals - min_vals
        
        # 3. 置信区间 (Confidence Interval)
        # 衡量指标: CI Width (排除掉极端异常点后的核心分布宽度)
        # 默认 95% 置信度: 取 2.5% 分位点 和 97.5% 分位点
        lower_percentile = (1 - self.confidence_level) / 2 * 100
        upper_percentile = 100 - lower_percentile
        
        ci_lowers = np.percentile(samples, lower_percentile, axis=0)
        ci_uppers = np.percentile(samples, upper_percentile, axis=0)
        ci_widths = ci_uppers - ci_lowers
        
        # 4. 相对不确定性 (Coefficient of Variation, CV)
        # 衡量指标: Std / Mean. 用于跨选手比较(消除得票率基数影响)
        means = np.mean(samples, axis=0)
        # 防止除以0
        cvs = np.divide(std_devs, means, out=np.zeros_like(std_devs), where=means!=0)
        
        # 5. 系统级确定性: 熵 (Entropy)
        # 衡量指标: 单一数值，表示当周比赛整体的混沌程度
        # 归一化 Mean Vector 后计算香农熵
        normalized_means = means / np.sum(means)
        system_entropy = -np.sum(normalized_means * np.log(normalized_means + 1e-9))
        
        # 组装结果
        metrics_df = pd.DataFrame({
            'celebrity_name': candidate_names,
            'mean_vote_share': means,
            'uncertainty_std': std_devs,        # 核心指标 1
            'feasible_range_width': range_widths, # 核心指标 2
            'ci_width_95': ci_widths,           # 核心指标 3 (更稳健)
            'ci_lower': ci_lowers,
            'ci_upper': ci_uppers,
            'relative_uncertainty_cv': cvs
        })
        
        return metrics_df, system_entropy

# 在主 Pipeline._process_samples 函数中调用:
def _process_samples(self, season, week, names, samples, method):
    # 初始化分析器
    analyzer = CertaintyAnalyzer()
    
    # 计算所有指标
    metrics_df, system_entropy = analyzer.analyze(samples, names)
    
    # 将 metrics_df 的每一行合并到最终结果中
    for _, row in metrics_df.iterrows():
        self.results.append({
            'season': season,
            'week': week,
            'celebrity_name': row['celebrity_name'],
            'est_vote_share_mean': row['mean_vote_share'],
            
            # --- 新增的确定性度量 ---
            'est_vote_share_std': row['uncertainty_std'],
            'feasible_range': row['feasible_range_width'],
            'ci_width': row['ci_width_95'],
            # -----------------------
            
            'method': method,
            'entropy': system_entropy,
            'sample_count': len(samples)
        })
