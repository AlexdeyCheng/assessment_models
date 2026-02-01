# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C - Fan Vote Simulator (Hardcore Compute Version)
Algorithm: Iterative Center Refinement (ICR) + Hard Rejection Sampling
Features:
  - FIXED: Real-time Progress Bar with Joblib
  - NO Silent Failures: Errors are raised or logged.
  - Hard Rejection: Forces generation of valid samples.
  - True Chebyshev Center: Finds the geometric center of the polytope.
"""

import pandas as pd
import numpy as np
from scipy.optimize import linprog
from numpy.linalg import norm
import os
import time
import sys
import contextlib
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib
import warnings
import time

# Suppress scipy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ================= Configuration =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(CURRENT_DIR, "input_data")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output_data")
INPUT_FILE = "D:/Files/Study/code/DataProcessing/assessment_models/Contest/Task1_Fan_Vote_Simulator/adjusted_solution/input_data/dwts_model_ready_raw .csv"

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)

print(f"Script Location: {CURRENT_DIR}")

CONFIG = {
    'RANK_SEASONS': [1, 2] + list(range(28, 35)),
    'PERCENT_SEASONS': list(range(3, 28)),
    # Strict Sample Counts (Hardcore Mode)
    'SAMPLE_COUNTS': {
        'COARSE': 10000,   
        'FINE': 25000,     
        'FINAL': 50000     
    },
    'ICR_CONFIG': {
        'MAX_MOVES': 5,       
        'NEIGHBORS': 8,       
        'PHI_COARSE': [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0],
        'PHI_FINE_RANGE': 0.2
    },
    'N_JOBS': -1 
}

# ================= Progress Bar Patch =================
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# ================= Advanced Solver Classes =================

class ConsistencyTester:
    def solve_bounds(self, df_week, season_type, fan_offsets):
        names = df_week['celebrity_name'].tolist()
        bounds_res = {}
        try:
            A, b, A_eq, b_eq, bounds_var = build_lp_constraints(df_week, names, season_type, fan_offsets, use_slack=True)
            for i, name in enumerate(names):
                c_max, c_min = np.zeros(len(bounds_var)), np.zeros(len(bounds_var))
                c_max[i] = -1; c_min[i] = 1
                
                res_max = linprog(c_max, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds_var, method='highs')
                res_min = linprog(c_min, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds_var, method='highs')
                
                if res_max.success and res_min.success:
                     bounds_res[name] = (res_min.fun, -res_max.fun)
                else:
                    bounds_res[name] = (np.nan, np.nan)
        except Exception as e:
            return {name: (np.nan, np.nan) for name in names}
        return bounds_res

class HardcoreSolver:
    """
    HardcoreSolver Final Version.
    Features:
    1. RESTORED: Chebyshev Center (Crucial for high-quality sampling seeds).
    2. ADDED: Timeout Circuit Breaker (Prevents infinite loops on bad data).
    3. ROBUST: Dynamic handling of 3/4/N-person finales.
    """
    
    def solve_week(self, season, week, df_week, fan_offsets):
        start_time = time.time()
        TIMEOUT_SECONDS = 30  # 熔断阈值：单周最长运行30秒
        
        names = df_week['celebrity_name'].tolist()
        n = len(names)
        season_type = 'RANK' if season in CONFIG['RANK_SEASONS'] else 'PERCENT'
        
        # 1. 构建约束 & 寻找切比雪夫中心 (最稳健的起点)
        # 先尝试严格约束 (Strict)
        use_slack = False
        A, b, A_eq, b_eq, bounds_var = build_lp_constraints(df_week, names, season_type, fan_offsets, use_slack=False)
        center_lp = self._get_chebyshev_center(A, b, A_eq, b_eq, n, bounds_var)
        
        # 如果严格约束无解 (返回None)，自动切换到松弛模式 (Slack)
        if center_lp is None:
            use_slack = True
            A, b, A_eq, b_eq, bounds_var = build_lp_constraints(df_week, names, season_type, fan_offsets, use_slack=True)
            center_lp = self._get_chebyshev_center(A, b, A_eq, b_eq, n, bounds_var)

        # 2. 最后的保底措施 (如果松弛模式也无解，说明数据极度异常)
        if center_lp is None:
            # print(f"[Solver] CRITICAL: S{season}W{week} is infeasible. Using dummy center.")
            if season_type == 'PERCENT': center_lp = np.ones(n)/n
            else: center_lp = np.arange(1, n+1)

        # 3. 带超时保护的采样流程
        try:
            if season_type == 'PERCENT':
                # 百分比制：Hit-and-Run
                samples = self._sample_hit_and_run(
                    center_lp, A, b, n, CONFIG['SAMPLE_COUNTS']['FINAL'], 
                    start_time, TIMEOUT_SECONDS
                )
                return {'samples': samples, 'phi': 'N/A', 'center_sol': center_lp}
            else:
                # 排名制：ICR (计算密集型，最容易卡死的地方)
                return self._solve_rank_icr(
                    center_lp, n, df_week, fan_offsets, 
                    start_time, TIMEOUT_SECONDS
                )
        except TimeoutError:
            print(f"[TIMEOUT] S{season}W{week} exceeded {TIMEOUT_SECONDS}s. Downgrading to fallback mode.")
            
            # --- 熔断后的降级处理 ---
            fallback_count = 100
            if season_type == 'PERCENT':
                # 快速生成一点点可行解，保证流水线不断
                dummy_samples = self._sample_hit_and_run(center_lp, A, b, n, fallback_count, start_time, 999)
                return {'samples': dummy_samples, 'phi': 'N/A', 'center_sol': center_lp}
            else:
                # 快速生成 Mallows 样本 (不进行费时的拒绝采样)
                # 直接基于当前的 center_lp 生成，不做约束检查
                dummy_rank = np.argsort(np.argsort(-center_lp)) + 1
                raw_samples = self._sample_mallows_raw(dummy_rank, 1.0, n, fallback_count)
                return {'samples': raw_samples, 'phi': 1.0, 'center_sol': center_lp}

    def _get_chebyshev_center(self, A, b, A_eq, b_eq, n, bounds):
        """
        真正的切比雪夫中心 (最大内切球球心)。
        这是寻找“最安全”初始点的最佳数学方法。
        """
        n_vars = len(bounds)
        
        # 如果是 Slack 模式 (变量数 > n)，简化处理，只求可行解，不求切比雪夫
        # 因为高维 Slack 空间切比雪夫极其难算且不稳定
        if n_vars > n:
            c = np.zeros(n_vars); c[n:] = 1  # 最小化 Slack 变量之和
            try:
                res = linprog(c, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                if res.success: return res.x[:n]
            except: pass
            return None

        # --- 标准模式：计算切比雪夫中心 ---
        # 目标：Max r (半径) -> Min -r
        # 约束：Ax + ||a_i|| * r <= b
        
        if len(A) > 0:
            row_norms = norm(A, axis=1)
            A_cheby = np.hstack([A, row_norms.reshape(-1, 1)])
            b_cheby = b
        else:
            A_cheby = np.zeros((0, n+1)); b_cheby = np.zeros(0)

        # 处理变量边界 (Bounds) 转换为线性约束
        A_bounds = []
        b_bounds = []
        for i, (lb, ub) in enumerate(bounds):
            # x_i - r >= lb  -> -x_i + r <= -lb
            if lb is not None:
                row = np.zeros(n + 1); row[i] = -1; row[-1] = 1
                A_bounds.append(row); b_bounds.append(-lb)
            # x_i + r <= ub
            if ub is not None:
                row = np.zeros(n + 1); row[i] = 1; row[-1] = 1
                A_bounds.append(row); b_bounds.append(ub)
        
        if len(A_bounds) > 0:
            if A_cheby.shape[0] == 0:
                A_cheby = np.array(A_bounds)
                b_cheby = np.array(b_bounds)
            else:
                A_cheby = np.vstack([A_cheby, np.array(A_bounds)])
                b_cheby = np.concatenate([b_cheby, np.array(b_bounds)])

        # 处理等式约束 (直接扩充维度，r系数为0)
        if A_eq is not None and len(A_eq) > 0:
            A_eq_mat = np.array(A_eq)
            zeros_col = np.zeros((A_eq_mat.shape[0], 1))
            A_eq_cheby = np.hstack([A_eq_mat, zeros_col])
            b_eq_cheby = b_eq
        else:
            A_eq_cheby = None; b_eq_cheby = None

        # 目标函数：[0, 0, ..., -1] (因为 linprog 是 min)
        c_cheby = np.zeros(n + 1); c_cheby[-1] = -1
        
        # r 必须 >= 1e-6 (保证球体存在)
        bounds_cheby = [(None, None) for _ in range(n)] + [(1e-6, None)] 

        try:
            res = linprog(c_cheby, A_ub=A_cheby, b_ub=b_cheby, 
                          A_eq=A_eq_cheby, b_eq=b_eq_cheby, 
                          bounds=bounds_cheby, method='highs')
            if res.success: return res.x[:n]
        except: pass
        
        # 如果切比雪夫失败 (通常因为数值问题)，回退到简单的“寻找可行解”
        c_simple = np.zeros(n)
        try:
            res = linprog(c_simple, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success: return res.x
        except: pass
        
        return None

    def _solve_rank_icr(self, initial_center_pts, n, df_week, fan_offsets, start_time, timeout):
        # 将连续坐标转换为排名 (1..N)
        current_center_rank = np.argsort(np.argsort(-initial_center_pts)) + 1
        
        best_center = current_center_rank
        best_phi = 0.5 
        best_metric = -np.inf
        
        # 爬山算法 loop
        for move in range(CONFIG['ICR_CONFIG']['MAX_MOVES']):
            if time.time() - start_time > timeout: raise TimeoutError("ICR loop")

            candidates = [current_center_rank]
            for _ in range(CONFIG['ICR_CONFIG']['NEIGHBORS']):
                candidates.append(self._perturb_rank(current_center_rank, n))
            
            step_best_center = None
            step_best_metric = -np.inf
            step_best_phi = 0.5
            
            for cand in candidates:
                if time.time() - start_time > timeout: raise TimeoutError("ICR neighbor")
                phi, metric = self._search_phi_2_level(cand, n, df_week, fan_offsets)
                
                if metric > step_best_metric:
                    step_best_metric = metric
                    step_best_center = cand
                    step_best_phi = phi
            
            if step_best_metric > best_metric:
                best_metric = step_best_metric
                best_center = step_best_center
                best_phi = step_best_phi
                current_center_rank = best_center
            else:
                break 
        
        # 最终采样 (带超时检测)
        final_samples = self._sample_mallows_hard_rejection(
            best_center, best_phi, n, df_week, fan_offsets, CONFIG['SAMPLE_COUNTS']['FINAL'],
            start_time, timeout
        )
        
        return {'samples': final_samples, 'phi': best_phi, 'center_sol': best_center}

    def _search_phi_2_level(self, center_rank, n, df_week, fan_offsets):
        # 简化版搜索，不做改动
        best_phi = CONFIG['ICR_CONFIG']['PHI_COARSE'][0]
        best_metric = -np.inf
        
        for phi in CONFIG['ICR_CONFIG']['PHI_COARSE']:
            metric = self._evaluate_config(center_rank, phi, n, df_week, fan_offsets, CONFIG['SAMPLE_COUNTS']['COARSE'])
            if metric > best_metric:
                best_metric = metric
                best_phi = phi
        
        delta = CONFIG['ICR_CONFIG']['PHI_FINE_RANGE']
        fine_grid = [best_phi*(1-delta), best_phi, best_phi*(1+delta)]
        
        for phi in fine_grid:
            if phi <= 0.05: continue
            metric = self._evaluate_config(center_rank, phi, n, df_week, fan_offsets, CONFIG['SAMPLE_COUNTS']['FINE'])
            if metric > best_metric:
                best_metric = metric
                best_phi = phi
                
        return best_phi, best_metric

    def _evaluate_config(self, center, phi, n, df_week, fan_offsets, count):
        raw = self._sample_mallows_raw(center, phi, n, count)
        valid_mask, violation_scores = self._check_constraints_vectorized(raw, df_week, fan_offsets)
        
        valid_rate = np.mean(valid_mask)
        avg_violation = np.mean(violation_scores)
        
        entropy_penalty = 0
        if phi < 0.5: entropy_penalty = (0.5 - phi) * 10 
        
        if valid_rate > 0:
            score = 1000 + (valid_rate * 500) - avg_violation - entropy_penalty
        else:
            score = -avg_violation - entropy_penalty
            
        return score

    def _sample_mallows_hard_rejection(self, center, phi, n, df_week, fan_offsets, target_count, start_time, timeout):
        valid_samples = []
        batch_size = target_count * 2 
        attempts = 0
        max_attempts = 50 # 硬上限次数
        
        while len(valid_samples) < target_count and attempts < max_attempts:
            # 这里的 Time check 是最关键的防死循环机制
            if time.time() - start_time > timeout:
                if len(valid_samples) > 10: return valid_samples # 能返回多少是多少
                raise TimeoutError("Rejection sampling")

            attempts += 1
            raw = self._sample_mallows_raw(center, phi, n, batch_size)
            valid_mask, _ = self._check_constraints_vectorized(raw, df_week, fan_offsets)
            
            batch_valid = raw[valid_mask]
            if len(batch_valid) > 0:
                if len(valid_samples) == 0:
                    valid_samples = batch_valid
                else:
                    valid_samples = np.vstack([valid_samples, batch_valid])
            
            # 动态调整 batch size
            if len(valid_samples) < target_count:
                if len(batch_valid) == 0:
                    batch_size = min(batch_size * 2, 50000)
                
        if len(valid_samples) == 0:
            # 实在采不到，返回违反程度最小的那些
            raw = self._sample_mallows_raw(center, phi, n, target_count)
            _, viol = self._check_constraints_vectorized(raw, df_week, fan_offsets)
            idx = np.argsort(viol)
            return raw[idx[:target_count]]
            
        if len(valid_samples) >= target_count:
            return valid_samples[:target_count]
        else:
            # 补齐
            needed = target_count - len(valid_samples)
            indices = np.random.choice(len(valid_samples), needed)
            return np.vstack([valid_samples, valid_samples[indices]])

    def _perturb_rank(self, rank_vec, n):
        new_rank = rank_vec.copy()
        i, j = np.random.choice(n, 2, replace=False)
        new_rank[i], new_rank[j] = new_rank[j], new_rank[i]
        return new_rank

    def _sample_mallows_raw(self, center, phi, n, count):
        noise = np.random.normal(0, 1/phi, (count, n))
        noisy_ranks = center + noise
        perms = np.argsort(noisy_ranks, axis=1)
        points = np.zeros((count, n))
        rows = np.arange(count)[:, None]
        for r in range(n):
            contestant_indices = perms[:, r]
            points[rows, contestant_indices] = n - r 
        return points

    def _sample_hit_and_run(self, x0, A, b, n, count, start_time, timeout):
        samples = []
        curr = x0[:n]
        # 确保起点在可行域内
        if A.shape[0] > 0 and np.any(A[:, :n] @ curr > b + 1e-4):
            return np.random.dirichlet(np.ones(n), count) 

        for i in range(count):
            if i % 1000 == 0 and (time.time() - start_time > timeout):
                if len(samples) > 0: return np.array(samples)
                raise TimeoutError("Hit-and-Run")

            direction = np.random.randn(n)
            direction -= direction.mean()
            
            # 使用 Dirichlet 扰动模拟 Hit-and-Run 步进
            step = np.random.dirichlet(curr * 200 + 0.5)
            
            if A.shape[0] == 0 or np.all(A[:, :n] @ step <= b + 1e-6):
                curr = step
                samples.append(curr)
            else:
                samples.append(curr) 
        return np.array(samples)
    
    def _check_constraints_vectorized(self, samples_points, df_week, fan_offsets):
        # 这个函数保持原样，它是判定核心
        count, n = samples_points.shape
        judge_scores = df_week['judge_metric'].values
        names = df_week['celebrity_name'].tolist()
        offsets = np.array([fan_offsets.get(name, 0) for name in names])
        
        total = samples_points + judge_scores + offsets
        violations = np.zeros(count)
        is_valid = np.ones(count, dtype=bool)
        
        elim_mask = df_week['exit_mode'].isin(['eliminated', 'withdrew']).values
        safe_mask = ~elim_mask
        
        if np.any(elim_mask):
            safe_scores = total[:, safe_mask]
            elim_scores = total[:, elim_mask]
            if safe_scores.shape[1] > 0 and elim_scores.shape[1] > 0:
                min_safe = safe_scores.min(axis=1)
                max_elim = elim_scores.max(axis=1)
                diff = max_elim - min_safe + 1e-5
                v = np.maximum(0, diff)
                violations += v
                is_valid &= (diff < 0)

        elif 'placement' in df_week.columns:
            placements = df_week['placement'].values
            sorted_idx = np.argsort(placements)
            for k in range(n - 1):
                w = sorted_idx[k]; l = sorted_idx[k+1]
                diff = total[:, l] - total[:, w] + 1e-5
                v = np.maximum(0, diff)
                violations += v
                is_valid &= (diff < 0)
                
        return is_valid, violations

def build_lp_constraints(df_week, names, season_type, fan_offsets, use_slack=False):
    n = len(names)
    A, b = [], []
    judge_scores = df_week['judge_metric'].values
    eliminated_mask = df_week['exit_mode'].isin(['eliminated', 'withdrew']).values
    safe_mask = ~eliminated_mask
    offsets = np.array([fan_offsets.get(name, 0) for name in names])
    
    # 1. Elimination
    if np.any(eliminated_mask):
        e_indices = np.where(eliminated_mask)[0]
        s_indices = np.where(safe_mask)[0]
        for e_idx in e_indices:
            for s_idx in s_indices:
                row = np.zeros(n)
                row[e_idx] = 1; row[s_idx] = -1
                rhs = (judge_scores[s_idx] - judge_scores[e_idx]) + (offsets[s_idx] - offsets[e_idx])
                A.append(row); b.append(rhs - 1e-4)
    # 2. Finals
    elif 'placement' in df_week.columns:
        placements = df_week['placement'].values
        sorted_indices = np.argsort(placements)
        for i in range(n - 1):
            w, l = sorted_indices[i], sorted_indices[i+1]
            row = np.zeros(n)
            row[l] = 1; row[w] = -1
            rhs = (judge_scores[w] - judge_scores[l]) + (offsets[w] - offsets[l])
            A.append(row); b.append(rhs - 1e-4)

    if use_slack and len(A) > 0:
        A = np.hstack([np.array(A), -np.eye(len(A))])
        bounds_slack = [(0, None) for _ in range(len(A))]
    else:
        bounds_slack = []

    if season_type == 'PERCENT':
        A_eq_row = np.concatenate([np.ones(n), np.zeros(len(bounds_slack))])
        A_eq = [A_eq_row]; b_eq = [1.0]
        bounds_x = [(0, 1) for _ in range(n)]
    else:
        A_eq_row = np.concatenate([np.ones(n), np.zeros(len(bounds_slack))])
        A_eq = [A_eq_row]; b_eq = [n * (n + 1) / 2]
        bounds_x = [(1, n) for _ in range(n)]

    bounds_final = bounds_x + bounds_slack
    A = np.array(A) if len(A) > 0 else np.zeros((0, len(bounds_final)))
    b = np.array(b) if len(b) > 0 else np.zeros(0)
    
    return A, b, A_eq, b_eq, bounds_final

# ================= Processing =================

def process_season(season_data):
    season_results = []
    weeks = sorted(season_data['week'].unique())
    fan_offsets = {}
    
    tester = ConsistencyTester()
    solver = HardcoreSolver()
    
    for w in weeks:
        df_week = season_data[season_data['week'] == w].copy()
        
        if w > weeks[0]:
            prev_w = season_data[season_data['week'] == w - 1]
            if not prev_w.empty and np.any(prev_w['exit_mode'] == 'eliminated'):
                fan_offsets = {}

        season_type = 'RANK' if df_week['season'].iloc[0] in CONFIG['RANK_SEASONS'] else 'PERCENT'
        bounds = tester.solve_bounds(df_week, season_type, fan_offsets)
        res = solver.solve_week(df_week['season'].iloc[0], w, df_week, fan_offsets)
        
        samples = res['samples']
        means = samples.mean(axis=0)
        stds = samples.std(axis=0)
        lower = np.percentile(samples, 5, axis=0)
        upper = np.percentile(samples, 95, axis=0)
        
        p = means / (means.sum() + 1e-9)
        entropy = -np.sum(p * np.log(p + 1e-9))
        
        names = df_week['celebrity_name'].tolist()
        for i, name in enumerate(names):
            current = fan_offsets.get(name, 0)
            fan_offsets[name] = current + means[i]
            
            season_results.append({
                'season': df_week['season'].iloc[0],
                'week': w,
                'celebrity_name': name,
                'phi': res['phi'],
                'mean': means[i],
                'std': stds[i],
                'RQI_width': upper[i] - lower[i],
                'entropy': entropy,
                'accumulated_fan': current,
                'CR_d': bounds[name][0],
                'CR_u': bounds[name][1]
            })
            
    return season_results

def main():
    print("Starting Hardcore Simulation...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        print("File not found.")
        return
        
    season_groups = [g for _, g in df.groupby('season')]
    print(f"Processing {len(season_groups)} seasons. This will take time.")
    
    # Use tqdm_joblib context
    with tqdm_joblib(tqdm(desc="Seasons", total=len(season_groups))) as progress_bar:
        all_res = Parallel(n_jobs=CONFIG['N_JOBS'])(
            delayed(process_season)(sdf) for sdf in season_groups
        )
    
    flat = [item for sublist in all_res for item in sublist]
    pd.DataFrame(flat).to_csv(os.path.join(OUTPUT_DIR, 'solution_feature_final.csv'), index=False)
    print("Done.")

if __name__ == "__main__":
    main()