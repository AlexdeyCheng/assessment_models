import pandas as pd 
from statsmodels.tsa.stattools import adfuller, kpss 
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning 
import statsmodels.api as sm 
import numpy as np
from scipy.signal import periodogram, find_peaks

df = pd.read_excel( 
    "D:/Files/Study/code/DataProcessing/assessment_models/Data_processsing/2023_MCM_Problem_C_Data.xlsx", 
    header=1 ) 

df.columns = df.columns.map(lambda x: str(x).strip()) 
time_col = "Date" 
df[time_col] = pd.to_datetime(df[time_col], errors="coerce") 
df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

# ===== 你的设定 =====
cols = df.columns[5:13]                  # 指定列范围
trend_cols = {"4 tries", "6 tries"}      # 仅这两列线性去趋势，其余只去均值
period_min, period_max = 5, 180          # 周期搜索范围（天）
fixed_periods = [7, 30]                  # 固定要报告的周期（天）

n_perm = 500                             # 置换次数：越大越稳（500~2000都行）
seed = 2026                              # 随机种子，保证可复现
alpha = 0.05                             # “显著”标记阈值（可改 0.01）

rng = np.random.default_rng(seed)

def detrend_series(y: pd.Series, mode: str) -> np.ndarray:
    """mode: 'linear' -> 线性去趋势残差; 'mean' -> 去均值"""
    y = pd.to_numeric(y, errors="coerce").dropna()
    if mode == "mean":
        return (y - y.mean()).values.astype(float)

    # linear detrend: y ~ a + b*t
    t = np.arange(len(y), dtype=float)
    X = sm.add_constant(t)
    model = sm.OLS(y.values.astype(float), X).fit()
    resid = y.values.astype(float) - model.predict(X)
    return resid.astype(float)

def periodogram_full(y: np.ndarray, fs: float = 1.0):
    """返回频率 f 和功率 pxx（去掉 f=0）"""
    y = np.asarray(y, dtype=float)
    y = y - np.mean(y)
    f, pxx = periodogram(y, fs=fs, detrend=False, scaling="density")
    # 去掉 f=0
    return f[1:], pxx[1:]

def idx_for_period(f: np.ndarray, target_period: float) -> int:
    """找到最接近 target_period 的频率索引（通过 target_freq=1/period）"""
    target_freq = 1.0 / float(target_period)
    return int(np.argmin(np.abs(f - target_freq)))

def top3_auto_peaks(f: np.ndarray, pxx: np.ndarray, pmin=5, pmax=180, topk=3):
    """
    在 period∈[pmin,pmax] 中找功率谱峰值 topk 个（按功率从大到小）
    返回：[(period_days, power, idx_in_f), ...]
    """
    # 周期范围转换到频率范围：period in [pmin,pmax] -> freq in [1/pmax, 1/pmin]
    f_lo, f_hi = 1.0 / pmax, 1.0 / pmin
    mask = (f >= f_lo) & (f <= f_hi)
    if mask.sum() < 5:
        return []

    f_m = f[mask]
    p_m = pxx[mask]

    peaks, _ = find_peaks(p_m)
    if len(peaks) == 0:
        # 没找到局部峰，退化为取最大
        k = int(np.argmax(p_m))
        idx_global = np.where(mask)[0][k]
        return [(1.0 / f[idx_global], float(pxx[idx_global]), int(idx_global))]

    # 取 topk 个峰（按功率排序）
    peak_powers = p_m[peaks]
    order = np.argsort(peak_powers)[::-1][:topk]
    res = []
    global_indices = np.where(mask)[0]
    for i in order:
        pk = peaks[i]
        idx_global = int(global_indices[pk])
        res.append((1.0 / float(f[idx_global]), float(pxx[idx_global]), idx_global))
    return res

def permutation_pvalues(y: np.ndarray, f: np.ndarray, target_indices: list[int], p_obs: list[float], n_perm=500):
    """
    置换检验：随机打乱 y，计算 periodogram，在 target_indices 上取功率
    p-value = mean(p_perm >= p_obs)
    """
    count = np.zeros(len(target_indices), dtype=int)
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        _, p_perm = periodogram_full(y_perm, fs=1.0)
        # 用同一频率网格 f；periodogram_full 的 f 只与长度有关，置换不改变 f
        for j, idx in enumerate(target_indices):
            if p_perm[idx] >= p_obs[j]:
                count[j] += 1
    pvals = count / float(n_perm)
    conf = 1.0 - pvals
    return pvals.tolist(), conf.tolist()

rows = []

for c in cols:
    mode = "linear" if c in trend_cols else "mean"
    y = detrend_series(df[c], mode=mode)
    n = len(y)
    if n < 30:
        rows.append({"column": c, "n": n, "detrend_mode": mode, "note": "too_short"})
        continue

    f, pxx = periodogram_full(y, fs=1.0)

    # 自动峰值 top3
    peaks = top3_auto_peaks(f, pxx, pmin=period_min, pmax=period_max, topk=3)

    # 固定 7d/30d（即使重复也照样报告）
    fixed = []
    for p in fixed_periods:
        idx = idx_for_period(f, p)
        fixed.append((float(1.0 / f[idx]), float(pxx[idx]), idx))  # 记录实际对齐到的周期

    # 组织要做置换检验的目标（peak1~3 + fixed7 + fixed30，允许重复）
    labels = []
    periods = []
    powers = []
    indices = []

    for i in range(3):
        if i < len(peaks):
            per, powv, idx = peaks[i]
            labels.append(f"peak{i+1}")
            periods.append(float(per))
            powers.append(float(powv))
            indices.append(int(idx))
        else:
            labels.append(f"peak{i+1}")
            periods.append(np.nan)
            powers.append(np.nan)
            indices.append(None)

    for p, (per_a, pow_a, idx_a) in zip(fixed_periods, fixed):
        labels.append(f"fixed_{p}d")
        periods.append(float(per_a))   # 实际对齐到的周期（可能不是精确 7/30）
        powers.append(float(pow_a))
        indices.append(int(idx_a))

    # 对存在的目标做置换检验
    valid_map = [(j, indices[j]) for j in range(len(indices)) if indices[j] is not None and np.isfinite(powers[j])]
    valid_js = [j for j, _ in valid_map]
    valid_idx = [idx for _, idx in valid_map]
    valid_pobs = [powers[j] for j in valid_js]

    pvals = [np.nan] * len(labels)
    confs = [np.nan] * len(labels)
    if valid_idx:
        pv, cf = permutation_pvalues(y, f, valid_idx, valid_pobs, n_perm=n_perm)
        for k, j in enumerate(valid_js):
            pvals[j] = pv[k]
            confs[j] = cf[k]

    # 生成一行 CSV（每个 column 一行）
    row = {
        "column": c,
        "n": n,
        "detrend_mode": mode,

        "best_period_days": periods[0],          # “最可能周期”按 peak1 给
        "best_confidence": confs[0],
        "best_pvalue": pvals[0],
    }

    # 写入 peak1~3
    for i in range(3):
        row[f"peak{i+1}_period_days"] = periods[i]
        row[f"peak{i+1}_power"] = powers[i]
        row[f"peak{i+1}_pvalue"] = pvals[i]
        row[f"peak{i+1}_confidence"] = confs[i]

    # 写入 fixed 7/30
    for k, p in enumerate(fixed_periods):
        j = 3 + k
        row[f"fixed_{p}d_period_days_actual"] = periods[j]
        row[f"fixed_{p}d_power"] = powers[j]
        row[f"fixed_{p}d_pvalue"] = pvals[j]
        row[f"fixed_{p}d_confidence"] = confs[j]
        row[f"fixed_{p}d_significant(alpha={alpha})"] = (pvals[j] < alpha) if np.isfinite(pvals[j]) else None

    rows.append(row)

out = pd.DataFrame(rows)

# 输出 CSV
csv_path = "periodicity_summary.csv"
out.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Saved: {csv_path}")
print(out.head(10).to_string(index=False))
