'''
Docstring for Data_processsing.data

Mostly used functions

# 查看前 5 行（最常用，看一眼数据结构）
print(df.head()) 

# 查看后 5 行（看看数据结尾有没有异常）
print(df.tail())

# 查看表格的规模：(行数, 列数)
# 对于 MCM 数据，这能告诉你一共有多少个“分” (points)
print(df.shape)  

# 查看列名（复制粘贴列名时很有用，防止拼写错误）
print(df.columns)

# 打印摘要信息：包括每一列的数据类型、有多少非空数据
# 重点看：Dtype 这一列。
# int64/float64 = 数字（好！）
# object = 文本（如果某一列应该是数字却显示object，说明里面混入了杂质）
print(df.info())

# 对所有数值列进行统计（平均值、标准差、最大最小值）
print(df.describe())

# 查看某一列的唯一值分布（超级好用！）
# 比如：查看赢球方式的分布，看看有多少是 'Winner'，有多少是 'Unforced Error'
# 假设列名是 'winner_shot_type' (具体看你的csv列名)
print(df['winner_shot_type'].value_counts())

# 1. 提取单列（得到 Series）
speeds = df['speed_mph']

# 2. 条件筛选（非常重要！）
# 语法：df[ 条件 ]
# 例如：找出所有发球速度超过 130 mph 的记录
fast_serves = df[df['speed_mph'] > 130]

# 3. 多条件筛选
# 例如：找出 Player1 名字是 "Alcaraz" 且 这一分赢了 的记录
# 注意：两个条件要用圆括号包起来，中间用 & 连接
alcaraz_wins = df[(df['player1'] == 'Alcaraz') & (df['point_victor'] == 1)]

print(f"Alcaraz 赢下了 {len(alcaraz_wins)} 个球")

# 检查每一列有多少个缺失值 (NaN)
print(df.isnull().sum())

# 如果某一列缺失值太多，可能需要删除或者填补
# df = df.dropna()  # 简单粗暴：删掉所有有缺失值的行

# 数值型操作自动忽略空值，可通过修改skipna参数的布尔值控制
'''
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

# 1) 过滤 KPSS 的插值 warning（不影响结论）
warnings.filterwarnings("ignore", category=InterpolationWarning)

# 2) 让 DataFrame 打印不截断列
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.float_format", lambda x: f"{x:.6g}")  # 防止 p 值打印成 0.0 看不清

ALPHA = 0.01  # 显著性水平

def run_tests(x, regression, alpha=ALPHA):
    x = pd.Series(x).dropna()

    adf_stat, adf_p, adf_lags, adf_nobs, *_ = adfuller(x, autolag="AIC", regression=regression)
    kpss_stat, kpss_p, kpss_lags, *_ = kpss(x, regression=regression, nlags="auto")

    # 判定：ADF p<alpha 支持平稳；KPSS p>alpha 支持平稳
    adf_stationary = (adf_p < alpha)
    kpss_stationary = (kpss_p > alpha)

    if adf_stationary and kpss_stationary:
        stationarity = "平稳"
        is_stationary = True
    elif (not adf_stationary) and (not kpss_stationary):
        stationarity = "非平稳"
        is_stationary = False
    else:
        stationarity = "不确定"
        is_stationary = None

    return {
        "regression": regression,       # c / ct
        "n": int(len(x)),
        "adf_stat": float(adf_stat),    # ADF 检验统计量（你说的“检验值”）
        "adf_p": float(adf_p),
        "adf_lags": int(adf_lags),
        "kpss_stat": float(kpss_stat),
        "kpss_p": float(kpss_p),
        "kpss_lags": int(kpss_lags),
        "stationarity": stationarity,   # ✅ 注意：是 stationarity，不是 staitionarity
        "is_stationary": is_stationary
    }

# ===== 读数据 =====
df = pd.read_excel(
    "D:/Files/Study/code/DataProcessing/assessment_models/Data_processsing/2023_MCM_Problem_C_Data.xlsx",
    header=1
)
df.columns = df.columns.map(lambda x: str(x).strip())

time_col = "Date"
df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col]).sort_values(time_col).set_index(time_col)

# 你要的列范围（示例：第5~13列）
cols = df.columns[5:13]

# ===== 批量跑检验 ===== 
results = []
for c in cols:
    s = pd.to_numeric(df[c], errors="coerce").dropna()

    for reg in ["c", "ct"]:
        try:
            res = run_tests(s, regression=reg)
            res["column"] = c
            results.append(res)
        except Exception as e:
            results.append({
                "column": c,
                "regression": reg,
                "error": str(e),
                "stationarity": "失败",
                "is_stationary": None
            })

out = pd.DataFrame(results)

# 列顺序优化（确保你能看到 ADF 和 stationarity）
front = ["column", "regression", "n", "stationarity", "is_stationary",
         "adf_stat", "adf_p", "adf_lags", "kpss_stat", "kpss_p", "kpss_lags", "error"]
out = out[[c for c in front if c in out.columns] + [c for c in out.columns if c not in front]]

# ✅ 用 to_string 强制完整打印（不会吞列）
print(out.to_string(index=False))

# （可选）保存结果，永远不会“看不见列”
out.to_csv("stationarity_results.csv", index=False, encoding="utf-8-sig")
