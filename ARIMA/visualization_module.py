import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings

# 忽略绘图时的 UserWarning
warnings.filterwarnings("ignore")

# ================= 1. 全局配置与主题 =================
def default_plot_sets(theme='blue_purple'):
    """
    配置全局绘图风格 (Arial, High DPI) 并返回颜色方案。
    """
    # 全局参数
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "Arial",     # 论文标准字体
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.2,
        # 兼容性设置
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
    })

    # 主题定义
    themes = {
        'blue_purple': {
            'cmap': 'BuPu',
            'bg_color': '#F8F9FA',
            'grid_color': '#E9ECEF',
            # Primary(History), Secondary(Forecast), Accent(CI), Compare
            'colors': {
                'history': '#2E2E2E',      # 黑色 (历史)
                'forecast': '#4B0082',     # 靛青 (预测主线)
                'ci_fill': '#9370DB',      # 适中的紫色 (置信区间)
                'linear_trend': '#FF4500', # 橙红色 (趋势线对比)
                'seasonality': '#1E90FF',  # 宝蓝 (周期项)
                'palette': ['#E0ECF8', '#9EB9D4', '#889AC6', '#6C7DA7', '#4D5B8C', '#2E3971', '#1A1F45'] # A-X 渐变
            }
        }
    }
    return themes.get(theme, themes['blue_purple'])

# 获取颜色配置
THEME = default_plot_sets('blue_purple')

def _ensure_dir():
    if not os.path.exists('paper_plots'):
        os.makedirs('paper_plots')

# ================= 2. 核心绘图函数 =================

def plot_forecast_grid(full_results_dict, forecast_df, title="Fig1_Forecast_Grid"):
    """
    [Fig 1] 4x2 网格图。
    前7张: A-X 各自的预测。
    第8张: 'Success Rate Summary' (Sum of A-F)。
    """
    _ensure_dir()
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'X']
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    colors = THEME['colors']
    
    # --- 1. 绘制 A-X ---
    for i, col in enumerate(categories):
        ax = axes[i]
        res = full_results_dict[col]
        
        # 数据提取
        history = res['model_obj'].data.endog # 原始训练数据
        hist_idx = np.arange(len(history))
        
        # 预测数据 (对齐索引)
        pred_vals = forecast_df[col].values
        future_idx = np.arange(len(history), len(history) + len(pred_vals))
        
        # CI 提取 (需要从原始结果中重新获取，因为 forecast_df 可能被归一化处理过，这里为了展示原始模型能力，建议用原始 CI)
        # 但为了和 Forecast 线匹配，我们使用 forecast_df 的预测值，CI 则按比例缩放(如果有归一化)
        # 这里简化：直接从 res['forecast_data'] 取原始 CI (未归一化的)，仅供参考模型不确定性
        raw_ci_lower = res['forecast_data']['CI_Lower'].values
        raw_ci_upper = res['forecast_data']['CI_Upper'].values
        # 简单截断 CI < 0
        raw_ci_lower = np.maximum(0, raw_ci_lower)

        # 绘图
        ax.plot(hist_idx, history, label='History', color=colors['history'], alpha=0.6, linewidth=1.5)
        ax.plot(future_idx, pred_vals, label='Forecast (SARIMAX)', color=colors['forecast'], linewidth=2.5)
        ax.fill_between(future_idx, raw_ci_lower, raw_ci_upper, color=colors['ci_fill'], alpha=0.3, label='95% CI')
        
        # 标注
        period = res['diagnostics']['Detected_Period']
        order = res['order']
        ax.set_title(f"Category {col} (Period={period}, Order={order})", fontweight='bold')
        ax.grid(True, color=THEME['grid_color'], linestyle='--')
        
        if i == 0: # 只在第一个图显示图例
            ax.legend(loc='upper left')

    # --- 2. 绘制第8张汇总图 (Success Rate A-F) ---
    ax_sum = axes[7]
    
    # 计算历史 Success Rate (A-F sum)
    # 我们需要原始 df 的全部列。这里稍微 trick 一下：
    # 假设 full_results_dict 里存的模型数据长度都一样，拼回去
    hist_len = len(full_results_dict['A']['model_obj'].data.endog)
    hist_sum_AF = np.zeros(hist_len)
    
    for col in ['A', 'B', 'C', 'D', 'E', 'F']:
        hist_sum_AF += full_results_dict[col]['model_obj'].data.endog
        
    pred_sum_AF = forecast_df[['A','B','C','D','E','F']].sum(axis=1).values
    
    hist_idx = np.arange(hist_len)
    future_idx = np.arange(hist_len, hist_len + len(pred_sum_AF))
    
    ax_sum.plot(hist_idx, hist_sum_AF, color='green', label='Hist Success (A-F)', alpha=0.7)
    ax_sum.plot(future_idx, pred_sum_AF, color='darkgreen', linestyle='--', linewidth=2.5, label='Pred Success')
    
    # 绘制分界线
    ax_sum.axvline(x=hist_len, color='red', linestyle=':', label='Forecast Start')
    
    ax_sum.set_title("Overall Success Rate (A-F Sum)", fontweight='bold', color='#2F4F4F')
    ax_sum.set_ylim(80, 105) # 通常 Success Rate 很高
    ax_sum.grid(True, linestyle='--')
    ax_sum.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(f"paper_plots/{title}.png", bbox_inches='tight')
    print(f"  [Plot] Saved {title}.png")
    plt.close()


def plot_stacked_evolution(forecast_df, title="Fig2_Distribution_Evolution"):
    """
    [Fig 2] 堆叠面积图 (Stacked Area Chart)。
    展示 A-X 的分布演变。
    A 在最底，X 在最顶。
    """
    _ensure_dir()
    
    # 准备数据 (Transpose for stackplot)
    # 顺序: A, B, C, D, E, F, X
    cols = ['A', 'B', 'C', 'D', 'E', 'F', 'X']
    data_stack = forecast_df[cols].T.values # Shape: (7, days)
    days = forecast_df.index
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 使用自定义的蓝紫渐变色板
    # 我们可以生成一个渐变色列表
    colors = sns.color_palette(THEME['cmap'], n_colors=7)
    
    ax.stackplot(days, data_stack, labels=cols, colors=colors, alpha=0.85)
    
    ax.set_title("Evolution of Wordle Result Distribution (Forecast)", fontweight='bold')
    ax.set_xlabel("Future Days")
    ax.set_ylabel("Percentage (%)")
    ax.set_xlim(days[0], days[-1])
    ax.set_ylim(0, 100)
    
    # 图例放在外侧
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Category")
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"paper_plots/{title}.png", bbox_inches='tight')
    print(f"  [Plot] Saved {title}.png")
    plt.close()


def plot_decomposition_logic(full_results_dict, target_col='D', compare_cols=None, title="Fig3_Trend_Seasonality_Decomposition"):
    """ [Fig 3] 模型逻辑可视化 (修复数据类型报错版) """
    _ensure_dir()
    if target_col not in full_results_dict:
        print(f"  [Plot Warning] Target {target_col} not found for decomposition plot.")
        return

    res = full_results_dict[target_col]
    model = res['model_obj']
    params = model.params
    exog = model.data.exog 
    endog = model.data.endog 
    
    # [CRITICAL FIX] 强制指定 dtype=float，防止 int64 += float64 报错
    trend_val = np.zeros_like(endog, dtype=float)
    
    # 重构 Trend
    if exog is not None and 'trend_t' in params.index:
        trend_idx = list(model.data.param_names).index('trend_t')
        trend_val += params['trend_t'] * exog[:, trend_idx]
    if 'const' in params.index:
         trend_val += params['const']
         
    # [CRITICAL FIX] 强制指定 dtype=float
    season_val = np.zeros_like(endog, dtype=float)
    
    # 重构 Seasonality
    if exog is not None:
        for name in params.index:
            if 'sin' in name or 'cos' in name:
                idx = list(model.data.param_names).index(name)
                season_val += params[name] * exog[:, idx]
                
    resid = model.resid
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    axes[0].plot(endog, color='black', label='Original Data')
    axes[0].plot(model.fittedvalues, color=THEME['colors']['forecast'], linestyle='--', label='Fitted')
    axes[0].set_title(f"Decomposition for Category {target_col} (Linear Feature)", fontweight='bold')
    axes[0].legend()
    
    axes[1].plot(trend_val, color=THEME['colors']['linear_trend'], linewidth=2)
    axes[1].set_ylabel("Linear Trend Component")
    axes[1].grid(True, linestyle='--')
    
    axes[2].plot(season_val, color=THEME['colors']['seasonality'])
    axes[2].set_ylabel(f"Seasonal Component (Sum of Fourier)")
    axes[2].grid(True, linestyle='--')
    
    axes[3].scatter(np.arange(len(resid)), resid, s=10, color='gray', alpha=0.5)
    axes[3].axhline(0, color='black', linewidth=1)
    axes[3].set_ylabel("Residuals (Noise)")
    axes[3].set_xlabel("Time Steps (Days)")
    
    plt.tight_layout()
    plt.savefig(f"paper_plots/{title}.png", bbox_inches='tight')
    print(f"  [Plot] Saved {title}.png")
    plt.close()


def plot_residual_diagnostics(full_results_dict, target_col='F', title="Fig4_Residual_Diagnostics"):
    """
    [Fig 4] 残差诊断图 (4-pack)。
    选择一个典型类别 (如 F) 展示其残差是否健康。
    """
    _ensure_dir()
    res = full_results_dict[target_col]
    resid = res['model_obj'].resid
    
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2)
    
    # 1. Residuals over time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(resid, color=THEME['colors']['history'])
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_title("Standardized Residuals")
    
    # 2. Histogram vs KDE
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(resid, kde=True, ax=ax2, color=THEME['colors']['ci_fill'], stat='density')
    ax2.set_title("Histogram plus Estimated Density")
    
    # 3. Normal Q-Q
    ax3 = fig.add_subplot(gs[1, 0])
    stats.probplot(resid, dist="norm", plot=ax3)
    ax3.get_lines()[0].set_markerfacecolor(THEME['colors']['forecast'])
    ax3.get_lines()[0].set_markeredgecolor(THEME['colors']['forecast'])
    ax3.get_lines()[1].set_color('red')
    ax3.set_title("Normal Q-Q Plot")
    
    # 4. ACF Plot
    ax4 = fig.add_subplot(gs[1, 1])
    plot_acf(resid, ax=ax4, lags=40, color=THEME['colors']['history'])
    ax4.set_title("Correlogram (ACF)")
    
    plt.suptitle(f"Residual Diagnostics for Category {target_col}", fontweight='bold', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"paper_plots/{title}.png", bbox_inches='tight')
    print(f"  [Plot] Saved {title}.png")
    plt.close()


def plot_confidence_heatmap(forecast_df, full_results_dict, title="Fig5_Confidence_Heatmap"):
    """ [Fig 5] 置信度热力图 (高对比度版: Magma 岩浆色) """
    _ensure_dir()
    
    conf_data = []
    categories = ['A', 'B', 'C', 'D', 'E', 'F', 'X']
    valid_cats = []
    
    for col in categories:
        if col in full_results_dict:
            score_series = full_results_dict[col]['forecast_data']['Confidence_Score']
            conf_data.append(score_series.values)
            valid_cats.append(col)
        
    if not conf_data:
        return

    conf_matrix = np.array(conf_data)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # [MODIFIED] 使用 'magma' (黑/紫 -> 红 -> 亮黄)
    # 适配蓝紫主题的同时提供最强对比度
    # Black/Purple (Low Conf) -> Red -> Bright Yellow (High Conf)
    sns.heatmap(conf_matrix, ax=ax, cmap='magma', vmin=0.5, vmax=1.0, 
                annot=False, cbar_kws={'label': 'Confidence Score (Dark=Low, Bright Yellow=High)'})
    
    ax.set_yticklabels(valid_cats, rotation=0, fontweight='bold')
    ax.set_xlabel("Future Days (Forecast Horizon)")
    ax.set_title("Model Confidence Assessment Heatmap (High Contrast)", fontweight='bold')
    
    xticks = np.arange(0, conf_matrix.shape[1], 10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks + 1)
    
    plt.tight_layout()
    plt.savefig(f"paper_plots/{title}.png", bbox_inches='tight')
    print(f"  [Plot] Saved {title}.png")
    plt.close()

# ================= 3. 统一调用接口 =================

def run_all_plots(full_results_dict, forecast_df):
    """
    主程序调用的唯一入口。
    """
    print("\n======== Starting Visualization Module ========")
    
    # 1. 预测概览 (含汇总)
    plot_forecast_grid(full_results_dict, forecast_df)
    
    # 2. 分布演变 (堆叠图)
    plot_stacked_evolution(forecast_df)
    
    # 3. 逻辑分解 (以线性 E 为例)
    plot_decomposition_logic(full_results_dict, target_col='E')
    
    # 4. 残差诊断 (以 F 为例)
    plot_residual_diagnostics(full_results_dict, target_col='F')
    
    # 5. 置信度热力图
    plot_confidence_heatmap(forecast_df, full_results_dict)
    
    print("======== All Plots Generated in 'paper_plots/' ========\n")