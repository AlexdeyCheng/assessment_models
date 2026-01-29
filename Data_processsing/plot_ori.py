import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scipy.signal import periodogram, find_peaks

df = pd.read_excel(
    "D:/Files/Study/code/DataProcessing/assessment_models/Data_processsing/2023_MCM_Problem_C_Data.xlsx",
    header=1
)
df.columns = df.columns.map(lambda x: str(x).strip())
time_col = "Date"
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

# ===== 你已有 df，且 Date 已作为 index，数据完整 =====
# df = ...

cols = df.columns[5:13]  # 你指定的列范围
trend_cols = {"4 tries", "6 tries"}  # 只对这两列做线性去趋势（ct 思路）
max_lag = 60                         # ACF/PACF、Ljung-Box 检查到 60
target_periods = [7, 30]             # 关心的周期（天）
peak_period_range = (2, 180)         # 自动找峰值的周期范围（天）；日采样最小周期=2天

def detrend_series(y: pd.Series, mode: str) -> pd.Series:
    """mode: 'linear' -> y ~ a + b*t 残差; 'mean' -> 去均值"""
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(y) < 20:
        return y  # 太短就不处理
    if mode == "mean":
        return y - y.mean()

    # linear detrend
    t = np.arange(len(y), dtype=float)
    X = sm.add_constant(t)
    model = sm.OLS(y.values.astype(float), X).fit()
    resid = y.values.astype(float) - model.predict(X)
    return pd.Series(resid, index=y.index, name=y.name)

def periodogram_peaks(y: np.ndarray, fs=1.0, prange=(2, 180), topk=3):
    """返回 periodogram 中最显著的 topk 个周期（天）及其功率"""
    y = np.asarray(y, dtype=float)
    y = y - np.nanmean(y)

    f, pxx = periodogram(y, fs=fs, detrend=False, scaling="density")
    # 去掉 f=0
    f = f[1:]
    pxx = pxx[1:]
    # freq -> period
    period = 1.0 / f

    lo, hi = prange
    mask = (period >= lo) & (period <= hi)
    if mask.sum() < 5:
        return [], (period, pxx)

    period_m = period[mask]
    pxx_m = pxx[mask]

    # 找峰（在功率谱上找局部最大）
    peaks, _ = find_peaks(pxx_m)
    if len(peaks) == 0:
        # 退化：直接取最大值
        idx = int(np.argmax(pxx_m))
        return [(float(period_m[idx]), float(pxx_m[idx]))], (period_m, pxx_m)

    # 取 topk 个最大峰
    peak_powers = pxx_m[peaks]
    order = np.argsort(peak_powers)[::-1][:topk]
    top = [(float(period_m[peaks[i]]), float(pxx_m[peaks[i]])) for i in order]

    return top, (period_m, pxx_m)

def power_at_period(period_arr, power_arr, target_period):
    """取最接近 target_period 的功率"""
    idx = int(np.argmin(np.abs(period_arr - target_period)))
    return float(power_arr[idx]), float(period_arr[idx])

summary_rows = []

# ===== A) 每列画图 + B) 汇总表 =====
for c in cols:
    # 1) 去趋势（仅对指定列线性去趋势，其余去均值）
    mode = "linear" if c in trend_cols else "mean"
    r = detrend_series(df[c], mode=mode)

    # 对齐为 numpy
    y = r.values.astype(float)
    n = len(y)
    if n < 30:
        summary_rows.append({
            "column": c, "n": n, "detrend_mode": mode,
            "note": "样本太短，跳过"
        })
        continue

    # 2) Ljung-Box：整体自相关性检验（原假设：无自相关）
    lb = acorr_ljungbox(y, lags=[7, 30, max_lag], return_df=True)
    lb_p7  = float(lb.loc[7, "lb_pvalue"])
    lb_p30 = float(lb.loc[30, "lb_pvalue"])
    lb_p60 = float(lb.loc[max_lag, "lb_pvalue"])

    # 3) ACF/PACF 数值：找显著 lag（粗略阈值 95%：±1.96/sqrt(n)）
    acf_vals = acf(y, nlags=max_lag, fft=True)
    pacf_vals = pacf(y, nlags=max_lag, method="ywm")

    ci = 1.96 / np.sqrt(n)
    sig_acf_lags = [i for i in range(1, len(acf_vals)) if abs(acf_vals[i]) > ci]
    # 只展示前 12 个，避免太长
    sig_acf_preview = sig_acf_lags[:12]

    # 4) 周期谱：自动峰值 + 特定 7/30 天强度
    top_peaks, (per_arr, pow_arr) = periodogram_peaks(y, fs=1.0, prange=peak_period_range, topk=3)

    p7, p7_at = power_at_period(per_arr, pow_arr, 7)
    p30, p30_at = power_at_period(per_arr, pow_arr, 30)

    # 记录汇总行（B）
    summary_rows.append({
        "column": c,
        "n": n,
        "detrend_mode": mode,
        "lb_p(7)": lb_p7,
        "lb_p(30)": lb_p30,
        "lb_p(60)": lb_p60,
        "sig_acf_lags(<=60)": ",".join(map(str, sig_acf_preview)) + ("..." if len(sig_acf_lags) > 12 else ""),
        "period_power@7d": p7,
        "period_power@30d": p30,
        "auto_peak_1(period,power)": f"{top_peaks[0][0]:.3g},{top_peaks[0][1]:.3g}" if len(top_peaks) >= 1 else "",
        "auto_peak_2(period,power)": f"{top_peaks[1][0]:.3g},{top_peaks[1][1]:.3g}" if len(top_peaks) >= 2 else "",
        "auto_peak_3(period,power)": f"{top_peaks[2][0]:.3g},{top_peaks[2][1]:.3g}" if len(top_peaks) >= 3 else "",
    })

    # ===== A) 画图（每列四张图）=====
    # 1) 残差序列
    plt.figure(figsize=(12, 3.2), dpi=160)
    plt.plot(r.index, y)
    plt.axhline(0.0, linestyle="--")
    plt.title(f"{c} | detrend={mode} | residual series")
    plt.xlabel("Date")
    plt.ylabel("residual")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()

    # 2) ACF
    plt.figure(figsize=(12, 3.2), dpi=160)
    plot_acf(y, lags=max_lag, ax=plt.gca())
    plt.title(f"{c} | ACF (residual)")
    plt.tight_layout()
    # plt.show()

    # 3) PACF
    plt.figure(figsize=(12, 3.2), dpi=160)
    plot_pacf(y, lags=max_lag, method="ywm", ax=plt.gca())
    plt.title(f"{c} | PACF (residual)")
    plt.tight_layout()
    # plt.show()

    # 4) 周期谱：用 period 作横轴（天），标出 7/30 天
    plt.figure(figsize=(12, 3.2), dpi=160)
    plt.plot(per_arr, pow_arr)
    plt.axvline(7, linestyle="--")
    plt.axvline(30, linestyle="--")
    plt.title(f"{c} | Periodogram (residual) | marks: 7d, 30d")
    plt.xlabel("Period (days)")
    plt.ylabel("Power")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()

# ===== B) 输出汇总表 =====
summary = pd.DataFrame(summary_rows)

# 排序：先看 Ljung-Box(30) p 值（越小越“有自相关”）
if "lb_p(30)" in summary.columns:
    summary = summary.sort_values(["lb_p(30)", "lb_p(7)"], ascending=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
print(summary.to_string(index=False))
