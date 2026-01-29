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
import numpy as np
import pandas as pd
import os

# 1) 读取
df = pd.read_excel(
    r"D:/Files/Study/code/DataProcessing/assessment_models/Data_processsing/copy_2023Problem_C_Data_Wordle.xlsx",
    header=1
)

# 2) 复制副本（只在副本上操作）
df_copy = df.copy()

# 3) 选取相邻列：'1 try' ~ '7 or more tries (X)'
start_col = "1 try"
end_col = "7 or more tries (X)"

start_idx = df_copy.columns.get_loc(start_col)
end_idx = df_copy.columns.get_loc(end_col)

target_cols = list(df_copy.columns[start_idx:end_idx + 1])

# 4) 清洗成数字（原本是 0-100 的“无单位百分比数”），无法解析 -> NaN
#    然后按你的要求：NaN 变 0
block_raw = df_copy[target_cols]

# 记录“这一行是否这几列全为 NaN”（用于之后把 sum 设为 1000）
all_nan_row_mask = block_raw.isna().all(axis=1)

# 若列里有奇怪空格/不可见字符：先转成字符串并 strip，再 to_numeric
block_num = block_raw.apply(
    lambda col: pd.to_numeric(col.astype(str).str.strip(), errors="coerce")
)

# NaN -> 0
block_num = block_num.fillna(0)

# 写回副本
df_copy[target_cols] = block_num

# 5) 在 df_copy 最后一列生成 sum
sum_col_name = "tries_sum"
df_copy[sum_col_name] = df_copy[target_cols].sum(axis=1)

# 按你的规则：如果该行这几列原始全是 NaN，则 sum 输出 1000
df_copy.loc[all_nan_row_mask, sum_col_name] = 1000

# 6) 用 MAD（Median Absolute Deviation）统计：每列“明显偏离均值(这里用稳健的中位数/MAD)”的行数（口径 A：每列分别计数）
#    常用 modified z-score 阈值 3.5，可按需调整
THRESH = 3.5

def mad_outlier_count(series: pd.Series, thresh: float = THRESH) -> int:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return 0
    mod_z = 0.6745 * (x - med) / mad
    return int(np.nansum(np.abs(mod_z) > thresh))

outlier_counts = pd.Series({c: mad_outlier_count(df_copy[c], THRESH) for c in target_cols})

# 输出：每列异常行数（口径 A）
print("MAD(Modified Z) outlier row counts per column:")
print(outlier_counts.sort_values(ascending=False))

# （可选）如果你想拿到“每列异常行的索引”，取消下面注释即可
# outlier_indices = {}
# for c in target_cols:
#     x = df_copy[c].to_numpy(dtype=float)
#     med = np.nanmedian(x)
#     mad = np.nanmedian(np.abs(x - med))
#     if mad == 0 or np.isnan(mad):
#         outlier_indices[c] = []
#         continue
#     mod_z = 0.6745 * (x - med) / mad
#     outlier_indices[c] = df_copy.index[np.abs(mod_z) > THRESH].tolist()
# print(outlier_indices)

# ---------- 路径准备 ----------
source_path = r"D:/Files/Study/code/DataProcessing/assessment_models/Data_processsing/copy_2023Problem_C_Data_Wordle.xlsx"
out_dir = os.path.dirname(source_path)
out_path = os.path.join(out_dir, "abnormal_value_summary.xlsx")

THRESH = 3.5
records = []

cols_for_outlier = target_cols + ["tries_sum"]

# ---------- 收集所有异常点 ----------
for col in target_cols:
    x = df_copy[col].to_numpy(dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if mad == 0 or np.isnan(mad):
        continue

    mod_z = 0.6745 * (x - med) / mad
    abnormal_mask = np.abs(mod_z) > THRESH

    if not abnormal_mask.any():
        continue  # 这一列没有异常值，跳过

    for idx in df_copy.index[abnormal_mask]:
        records.append({
            "index": idx,
            "abnormal_column": col,
            "value": df_copy.at[idx, col],
            "median": med,
            "MAD": mad,
            "modified_z": mod_z[df_copy.index.get_loc(idx)],
            "tries_sum": df_copy.at[idx, "tries_sum"],
            **df_copy.loc[idx].to_dict()   # 完整行数据
        })

# ---------- 汇总为 DataFrame ----------
abnormal_df = pd.DataFrame(records)

# ---------- 存储 ----------
abnormal_df.to_excel(out_path, index=False)

print(f"Abnormal value summary saved to:\n{out_path}")
print(f"Total abnormal records: {len(abnormal_df)}")
print(f"Abnormal columns: {abnormal_df['abnormal_column'].nunique()}")
