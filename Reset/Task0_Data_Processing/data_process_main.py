import pandas as pd
import numpy as np
import re
import warnings

# ==========================================
# FILE: clean_ops.py (物理清洗层)
# ==========================================
def physical_clean(df_raw):
    """
    负责基础格式清洗、长宽转换与有效性过滤。
    """
    print("[CleanOps] 开始物理清洗...")
    df = df_raw.copy()

    # 1. 基础清理：去空格
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()

    # 2. 列名标准化
    rename_map = {
        'celebrity_age_during_season': 'age',
        'celebrity_industry': 'industry',
        'ballroom_partner': 'partner',
        'celebrity_name': 'celebrity'
    }
    df = df.rename(columns=rename_map)

    # 3. 填充静态缺失值 (防止Melt后丢失信息)
    static_cols = ['celebrity', 'partner', 'industry', 'age', 'season', 'results', 'placement']
    # 简单的填充，确保GroupBy时不会丢行
    for c in static_cols:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown")

    # 4. Melt (宽转长)
    # 识别分数列
    score_cols = [c for c in df.columns if 'week' in c and 'score' in c]
    # 静态列作为ID
    id_vars = [c for c in df.columns if c not in score_cols]
    
    df_melt = pd.melt(df, id_vars=id_vars, value_vars=score_cols, var_name='week_judge', value_name='raw_score')

    # 5. 解析周数
    # 格式: weekX_judgeY_score
    pattern = r'week(\d+)_judge\d+_score'
    df_melt['week'] = df_melt['week_judge'].str.extract(pattern).astype(float)

    # 6. 有效性过滤
    # 去除 NaN 和 0 分 (非参赛周)
    df_clean = df_melt.dropna(subset=['raw_score', 'week'])
    df_clean = df_clean[df_clean['raw_score'] > 0].copy()
    df_clean['week'] = df_clean['week'].astype(int)

    # 7. 聚合为周总分 (Sum over judges)
    # 此时一行代表: Season-Celebrity-Week
    
    # --- FIX START: 去除重复键 ---
    core_keys = ['season', 'celebrity', 'week']
    # 仅添加不在 core_keys 中的静态列
    extra_keys = [c for c in static_cols if c in df_clean.columns and c not in core_keys]
    group_keys = core_keys + extra_keys
    # --- FIX END ---

    df_weekly = df_clean.groupby(group_keys)['raw_score'].sum().reset_index()
    df_weekly.rename(columns={'raw_score': 'weekly_raw_score'}, inplace=True)
    
    print(f"[CleanOps] 物理清洗完成。生成 {len(df_weekly)} 行有效周数据。")
    return df_weekly

# ==========================================
# FILE: rank_ops.py (核心逻辑层)
# ==========================================
def determine_status_and_rank(df_weekly):
    """
    负责身份判定 (Time Gate)、排名计算 (Skip Logic) 和 结果标准化。
    """
    print("[RankOps] 开始核心逻辑计算...")
    
    # --- A. 准备元数据 ---
    # 计算赛季最大周 (Finale Week)
    season_max_weeks = df_weekly.groupby('season')['week'].max().to_dict()
    # 计算个人最大周 (Last Active Week)
    df_weekly['person_max_week'] = df_weekly.groupby(['season', 'celebrity'])['week'].transform('max')
    
    # 提取选手级数据表 (去重)
    contestants = df_weekly[['season', 'celebrity', 'results', 'person_max_week']].drop_duplicates().copy()
    
    # --- B. 身份判定 (Time Gate) ---
    def get_status(row):
        s_max = season_max_weeks[row['season']]
        p_max = row['person_max_week']
        orig_res = str(row['results'])
        
        if "Withdrew" in orig_res:
            return "withdrew", f"Withdrew Week {p_max}"
        elif p_max == s_max:
            # 活到了最后 -> Final
            return "final", None # 稍后填入 "Xth Place"
        else:
            # 未活到最后 -> Eliminated (不管原文写什么)
            return "eliminated", f"Eliminated Week {p_max}"

    status_res = contestants.apply(get_status, axis=1)
    contestants['exit_mode'] = [x[0] for x in status_res]
    contestants['results_temp'] = [x[1] for x in status_res]
    
    # --- C. 排名计算 (Rank Logic: Skip/Min) ---
    # 1. 解析显式排名 (Explicit)
    def parse_explicit(text):
        match = re.search(r'(\d+)(?:st|nd|rd|th)\s+Place', str(text), re.IGNORECASE)
        return int(match.group(1)) if match else np.nan
    
    contestants['explicit_rank'] = contestants['results'].apply(parse_explicit)
    
    # 2. 计算最终排名 (按赛季分组)
    final_ranks = []
    for season, group in contestants.groupby('season'):
        group = group.copy()
        
        # 逻辑：
        # 优先使用 Explicit Rank。
        # 如果没有 Explicit，则使用 Implicit (基于 person_max_week 倒序)。
        # 并列处理：method='min' (1, 1, 3 - 跳号)。
        
        # 构造一个用于排序的临时分值
        # 显式排名越小越好。
        # 存活周数越大越好。
        # 我们创建一个 "Rank Score"：
        # 如果有 Explicit -> Score = Explicit
        # 如果无 -> Score = 1000 - person_max_week (假设没人的 Rank 会超过 1000，且周数越大 Rank 越小)
        # *修正*：混用很危险。不如分两步。
        
        # 策略：全员按 (Exit Week Desc) 排序。
        # 同一周离开的人，视为同组。
        # 该组的 Rank = 1 + (Count of people who exited later).
        # 如果该组有人有 Explicit Rank，则强制覆盖为 Explicit。
        
        # 计算 Implicit Rank (跳号逻辑)
        # Rank = (在该赛季中，exit_week 比我大的人数) + 1
        group['better_count'] = group['person_max_week'].apply(
            lambda x: len(group[group['person_max_week'] > x])
        )
        group['implicit_rank'] = group['better_count'] + 1
        
        # 合并：Explicit 优先
        group['final_rank'] = group['explicit_rank'].fillna(group['implicit_rank'])
        
        # 特殊修正：Implicit Finalist (如 Drew Scott)
        # 如果我是 Final (Implicit=1) 但我没有 Explicit，且同组有人有 Explicit (1, 2, 3)
        # 那么我应该是 Max(Explicit) + 1?
        # 或者简单的：Finalist 内部如果不全，就可能有问题。
        # 简单起见，如果 Explicit 存在，信赖 Explicit。
        # Drew Scott Case: Explicit=NaN, Implicit=1. 
        # S25 显式排名有 1, 2, 3. 
        # 此时 Drew 的 implicit=1 会和冠军冲突。
        # 修正逻辑：检测到冲突（Implicit=1 但 Explicit=NaN，且该赛季有其他 Explicit=1），则顺延。
        # 这种逻辑太复杂，容易过拟合。
        # 简化方案：相信 "Better Count"。
        # 只有在决赛周（大家 exit_week 相同），Implicit 无法区分 1,2,3,4。
        # 此时必须依赖 Explicit。
        # 如果没有 Explicit（如 Drew），赋予决赛圈最低名次。
        
        # 针对决赛圈(Implicit Rank = 1) 的特殊填充：
        if (group['implicit_rank'] == 1).sum() > 1: # 决赛有多人
             max_explicit = group.loc[group['implicit_rank'] == 1, 'explicit_rank'].max()
             if pd.notna(max_explicit):
                 # 填充决赛圈的 NaN 为 max_explicit + 1 (顺延)
                 mask = (group['implicit_rank'] == 1) & (group['explicit_rank'].isna())
                 if mask.any():
                     group.loc[mask, 'final_rank'] = max_explicit + 1
        
        final_ranks.append(group[['season', 'celebrity', 'final_rank']])

    contestants = contestants.drop(columns=['final_rank'], errors='ignore')
    contestants = pd.merge(contestants, pd.concat(final_ranks), on=['season', 'celebrity'])
    
    # --- D. 结果标准化文本 ---
    def standardize_text(row):
        if row['exit_mode'] == 'final':
            r = int(row['final_rank'])
            suffix = 'th'
            if r % 10 == 1 and r != 11: suffix = 'st'
            elif r % 10 == 2 and r != 12: suffix = 'nd'
            elif r % 10 == 3 and r != 13: suffix = 'rd'
            return f"{r}{suffix} Place"
        else:
            return row['results_temp'] # "Eliminated Week X" or "Withdrew Week X"

    contestants['results_display'] = contestants.apply(standardize_text, axis=1)
    
    # 清理中间列
    contestants_clean = contestants[['season', 'celebrity', 'exit_mode', 'final_rank', 'person_max_week', 'results_display']]
    contestants_clean.rename(columns={'person_max_week': 'exit_week_parsed', 'results_display': 'results_parsed'}, inplace=True)
    
    # Merge 回主表
    df_merged = pd.merge(df_weekly, contestants_clean, on=['season', 'celebrity'])
    return df_merged

def calculate_accumulation_and_metric(df):
    """
    负责分数累加 (Strict Reset) 和 指标计算 (Rank/Percent)。
    """
    print("[RankOps] 计算累加与指标...")
    df = df.sort_values(['season', 'celebrity', 'week']).copy()
    
    # --- E. 严格累加 (Strict Reset) ---
    # 规则：若 Week T-1 有人 Eliminated，则 Week T 重置。
    # 1. 标记每周是否有人被淘汰
    # 获取每周的淘汰事件
    elim_events = df[df['exit_mode'] == 'eliminated'].groupby(['season', 'exit_week_parsed']).size()
    # 转为 Set (season, week)
    elim_set = set(elim_events.index)
    
    # 2. 计算 Accumulation Flag & Effective Score
    # 这是一个依赖顺序的过程，按 Season 处理
    
    acc_flags = []
    eff_scores = []
    
    # 这里的逻辑需要精细：
    # Effective Score 的计算需要基于上一周的 Effective Score。
    # 只能用循环（或 Numba），但在 Pandas 这种规模（~4000行）纯 Python 循环也是毫秒级。
    
    # 为了速度，我们按 (season, week) 字典存储 "是否重置"
    # Reset happens AT the start of Week T if Week T-1 had elimination.
    # Map: (season, week) -> bool (Is Reset Week?)
    # Week 1 is always Reset.
    # If (season, w-1) in elim_set -> Week w is Reset.
    
    reset_map = {}
    seasons = df['season'].unique()
    for s in seasons:
        weeks = sorted(df[df['season'] == s]['week'].unique())
        for w in weeks:
            if w == min(weeks):
                reset_map[(s, w)] = True # 第一周总是重置
            else:
                # 检查上一周 (w-1) 是否有人淘汰
                # 注意：如果数据中周数不连续，这里需要逻辑处理。通常 DWTS 是连续的。
                # 用 shift 逻辑：
                prev_w = w - 1
                if (s, prev_w) in elim_set:
                    reset_map[(s, w)] = True # 上周有人死 -> 重置
                else:
                    reset_map[(s, w)] = False # 上周无人死 -> 累加
    
    # 应用
    # 需要按 Celebrity 分组 cumsum，但在 Reset 点断开。
    # 我们可以给每个 Reset 点分配一个新的 "Group ID"。
    
    # 生成 Accumulation Group ID
    # 遍历所有行，太慢。
    # 向量化：为每个 (season, week) 赋予一个 incrementing ID if Reset.
    
    season_week_group = {}
    for s in seasons:
        weeks = sorted(df[df['season'] == s]['week'].unique())
        curr_group = 0
        for w in weeks:
            if reset_map.get((s, w), True):
                curr_group += 1
            season_week_group[(s, w)] = curr_group
            
    df['acc_group'] = df.apply(lambda x: season_week_group.get((x['season'], x['week'])), axis=1)
    
    # 现在可以简单的 GroupBy Sum 了
    df['effective_score'] = df.groupby(['season', 'celebrity', 'acc_group'])['weekly_raw_score'].cumsum()
    df['is_accumulated'] = df['effective_score'] != df['weekly_raw_score']
    
    # --- F. 指标计算 (Judge Metric) ---
    # S1-S2, S28-S34: Rank (Min)
    # S3-S27: Percent
    rank_seasons = [1, 2] + list(range(28, 35))
    
    def get_metric(sub):
        s = sub['season'].iloc[0]
        if s in rank_seasons:
            # Rank: 分数越高，排名越靠前(1)。method='min' (跳号).
            # 注意：Rank 1 是最好的。
            return sub['weekly_raw_score'].rank(ascending=False, method='min')
        else:
            # Percent: 分数 / 总分
            total = sub['weekly_raw_score'].sum()
            return sub['weekly_raw_score'] / total if total > 0 else 0

    df['judge_metric'] = df.groupby(['season', 'week']).apply(get_metric).reset_index(level=[0,1], drop=True)
    
    return df




# ==========================================
# FILE: main.py (主调用层)
# ==========================================

def main():
    # 1. Load
    print(">>> [Main] 读取数据...")
    try:
        df_raw = pd.read_csv('D:/Files/Study/code/DataProcessing/assessment_models/Reset/Raw_data/2026_MCM_Problem_C_Data.csv')
    except FileNotFoundError:
        print("错误：未找到文件 2026_MCM_Problem_C_Data.csv")
        return

    # 2. Call Clean
    df_clean = physical_clean(df_raw)

    # 3. Call Logic
    df_status = determine_status_and_rank(df_clean)
    df_final_calc = calculate_accumulation_and_metric(df_status)

    # 4. Final Formatting
    # 要求的列: season,week,celebrity_name,results,age,industry,partner,weekly_raw_score,effective_score,is_accumulated,exit_mode,final_rank,exit_week_parsed,judge_metric
    
    # 映射列名
    final_cols_map = {
        'season': 'season',
        'week': 'week',
        'celebrity': 'celebrity_name',
        'results': 'ori_results',
        'results_parsed': 'parsed_results',
        'age': 'age',
        'industry': 'industry',
        'partner': 'partner',
        'weekly_raw_score': 'weekly_raw_score',
        'effective_score': 'effective_score',
        'is_accumulated': 'is_accumulated',
        'exit_mode': 'exit_mode',
        'final_rank': 'final_rank',
        'exit_week_parsed': 'exit_week_parsed',
        'judge_metric': 'judge_metric'
    }
    
    df_out = df_final_calc.rename(columns=final_cols_map)[list(final_cols_map.values())]
    
    # 5. Output
    output_filename = 'cleaned_dwts_data_final.csv'
    df_out.to_csv(output_filename, index=False)
    print(f"\n>>> [Main] 成功。文件已保存至: {output_filename}")
    print("\n--- 预览前 5 行 ---")
    print(df_out.head().to_markdown(index=False))
    

# 执行
if __name__ == "__main__":
    main()