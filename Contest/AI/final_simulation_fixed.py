# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C - Adjusted Solution: Fan Vote Simulator (Fixed for Finals)
Algorithm: Chebyshev Center Warm-Start + Mallows Double-Level Search + Strict Constraints
Author: Adjusted Solution Bot
"""

import pandas as pd
import numpy as np
from scipy.optimize import linprog
from numpy.linalg import norm
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed

# ================= Configuration =================
INPUT_DIR = r"D:/Files/Study/code/DataProcessing/assessment_models/Contest/Fan_Vote_Simulator1/adjusted_solution/input_data"
OUTPUT_DIR = r"D:/Files/Study/code/DataProcessing/assessment_models/Contest/Fan_Vote_Simulator1/adjusted_solution/output_data"
INPUT_FILE = os.path.join(INPUT_DIR, "dwts_model_ready.csv")

# Ensure Output Directory Exists
if not os.path.exists(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
    except:
        pass 

CONFIG = {
    'RANK_SEASONS': [1, 2] + list(range(28, 35)),
    'PERCENT_SEASONS': list(range(3, 28)),
    'SAMPLE_COUNTS': {
        'COARSE': 5000,
        'FINE': 10000,
        'FINAL': 20000
    },
    'RQI_PERCENTILE': 90, # 90% CI
    'N_JOBS': -1 # Use all cores
}

# ================= Step 1: Consistency Tester (Bounds) =================
class ConsistencyTester:
    """Calculates CR_u and CR_d using Linear Programming Relaxation."""
    
    def solve_bounds(self, df_week, season_type):
        names = df_week['celebrity_name'].tolist()
        n = len(names)
        bounds_res = {}
        
        try:
            eliminated = df_week[df_week['exit_mode'].isin(['eliminated', 'withdrew'])]
            safe = df_week[df_week['exit_mode'].isin(['safe', 'final'])] # 'final' is effectively safe from elimination
            
            A, b = [], []
            
            # --- Constraint Construction ---
            # Mode A: Elimination (Strict: Eliminated < Safe)
            if not eliminated.empty:
                for _, e_row in eliminated.iterrows():
                    for _, s_row in safe.iterrows():
                        row = np.zeros(n)
                        e_idx = names.index(e_row['celebrity_name'])
                        s_idx = names.index(s_row['celebrity_name'])
                        
                        # V_e - V_s < J_s - J_e
                        row[e_idx] = 1
                        row[s_idx] = -1
                        rhs = s_row['judge_metric'] - e_row['judge_metric']
                        
                        A.append(row)
                        b.append(rhs - 1e-5) # Strict inequality

            # Mode B: Finals / Full Ranking (Strict: Rank i > Rank i+1)
            # Only apply if no explicit elimination OR if it's the finals
            # We detect this by checking if 'placement' exists and is valid
            elif 'placement' in df_week.columns and df_week['placement'].notna().all():
                # Sort by placement (1 is best)
                sorted_df = df_week.sort_values('placement')
                sorted_names = sorted_df['celebrity_name'].tolist()
                sorted_judge = sorted_df['judge_metric'].tolist()
                
                # Pairwise constraints for adjacent ranks
                for i in range(n - 1):
                    # Person i (Better) vs Person i+1 (Worse)
                    # Score(Better) > Score(Worse)
                    # V_better + J_better > V_worse + J_worse
                    # V_worse - V_better < J_better - J_worse
                    
                    winner_name = sorted_names[i]
                    loser_name = sorted_names[i+1]
                    winner_idx = names.index(winner_name)
                    loser_idx = names.index(loser_name)
                    
                    row = np.zeros(n)
                    row[loser_idx] = 1
                    row[winner_idx] = -1
                    rhs = sorted_judge[i] - sorted_judge[i+1]
                    
                    A.append(row)
                    b.append(rhs - 1e-5)
            
            else:
                # No elimination and no placement info? Cannot solve bounds.
                return {name: (np.nan, np.nan) for name in names}

            # Common Constraints (Sum = Constant)
            A_eq = [np.ones(n)]
            if season_type == 'RANK':
                b_eq = [n * (n + 1) / 2] # Sum of 1..N
                bounds_var = [(1, n) for _ in range(n)]
            else:
                b_eq = [1.0] # Sum of probability
                bounds_var = [(0, 1) for _ in range(n)]

            # Solve Min and Max for each candidate
            for i, name in enumerate(names):
                # Maximize x_i -> Minimize -x_i
                c_max = np.zeros(n); c_max[i] = -1
                res_max = linprog(c_max, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds_var, method='highs')
                
                # Minimize x_i
                c_min = np.zeros(n); c_min[i] = 1
                res_min = linprog(c_min, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds_var, method='highs')
                
                if res_max.success and res_min.success:
                     bounds_res[name] = (res_min.fun, -res_max.fun)
                else:
                    bounds_res[name] = (np.nan, np.nan)
                    
        except Exception as e:
            # print(f"LP Error: {e}")
            return {name: (np.nan, np.nan) for name in names}
            
        return bounds_res

# ================= Step 2: Hybrid Solver =================
class HybridSolver:
    def _get_chebyshev_center(self, A, b, n, bounds):
        # Maximize r s.t. A_i*x + r*||A_i|| <= b_i
        # Also need to respect box bounds for x.
        # This is complex with general bounds. Simplified:
        # Use simple LP to find *a* feasible point (Analytic center approx) if Chebyshev fails
        
        row_norms = norm(A, axis=1) if len(A) > 0 else np.zeros(0)
        c = np.zeros(n + 1); c[0] = -1 # Maximize r
        
        if len(A) > 0:
            A_lp = np.c_[row_norms, A]
            b_lp = b
        else:
            # No inequality constraints, just box bounds
            # Dummy constraint
            A_lp = np.zeros((0, n+1))
            b_lp = np.zeros(0)
        
        # Bounds for r and x
        new_bounds = [(0, None)] + bounds
        
        # Note: A_eq/b_eq for Sum constraint
        # Equality constraints must handle the extra 'r' variable (coeff 0)
        if bounds[0] == (0, 1): # Percent
             A_eq = np.zeros((1, n+1)); A_eq[0, 1:] = 1; b_eq = [1.0]
        else: # Rank
             A_eq = np.zeros((1, n+1)); A_eq[0, 1:] = 1; b_eq = [n*(n+1)/2]

        res = linprog(c, A_ub=A_lp, b_ub=b_lp, A_eq=A_eq, b_eq=b_eq, bounds=new_bounds, method='highs')
        if res.success:
            return res.x[1:]
        return None

    def solve_week(self, season, week, df_week):
        names = df_week['celebrity_name'].tolist()
        n = len(names)
        eliminated_mask = df_week['exit_mode'].isin(['eliminated', 'withdrew']).values
        safe_mask = ~eliminated_mask
        
        # --- 1. Build Linear Constraints ---
        A, b = [], []
        judge_scores = df_week['judge_metric'].values 
        
        # Logic A: Elimination (Prioritized)
        if np.any(eliminated_mask):
            for e_idx in np.where(eliminated_mask)[0]:
                for s_idx in np.where(safe_mask)[0]:
                    row = np.zeros(n)
                    row[e_idx] = 1; row[s_idx] = -1
                    # V_e - V_s < J_s - J_e
                    rhs = judge_scores[s_idx] - judge_scores[e_idx]
                    A.append(row); b.append(rhs)
        
        # Logic B: Finals / Full Rank (Fallback)
        elif 'placement' in df_week.columns and df_week['placement'].notna().all():
            placements = df_week['placement'].values
            # Sort indices by placement
            sorted_indices = np.argsort(placements) # 0 is Best (Rank 1)
            
            for i in range(n - 1):
                winner_idx = sorted_indices[i]
                loser_idx = sorted_indices[i+1]
                
                # Winner Total > Loser Total
                # V_win + J_win > V_lose + J_lose
                # V_lose - V_win < J_win - J_lose
                row = np.zeros(n)
                row[loser_idx] = 1; row[winner_idx] = -1
                rhs = judge_scores[winner_idx] - judge_scores[loser_idx]
                A.append(row); b.append(rhs)
                
        else:
            return None # Cannot simulate without constraints

        A = np.array(A) if len(A) > 0 else np.zeros((0, n))
        b = np.array(b) if len(b) > 0 else np.zeros(0)

        # --- 2. Branch by System ---
        if season in CONFIG['PERCENT_SEASONS']:
            # Percent System
            center = self._get_chebyshev_center(A, b, n, [(0, 1)] * n)
            if center is None: center = np.ones(n)/n 
            
            samples = self._sample_hit_and_run(center, A, b, n, CONFIG['SAMPLE_COUNTS']['FINAL'])
            
            return {
                'samples': samples,
                'phi': 'N/A',
                'center_sol': center
            }
            
        else:
            # Rank System
            center_pts = self._get_chebyshev_center(A, b, n, [(1, n)] * n)
            if center_pts is None: 
                # Fallback to Judge Order
                center_pts = np.argsort(np.argsort(judge_scores)) + 1
            
            # Map Points -> Rank Permutation (0=Highest Points/Best)
            center_rank_perm = np.argsort(np.argsort(-center_pts)) 
            center_rank_1based = center_rank_perm + 1
            
            # Search Phi
            best_phi = 1.0
            max_valid_rate = -1
            
            coarse_phis = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
            for phi in coarse_phis:
                valid_rate = self._test_mallows(center_rank_perm, phi, n, judge_scores, eliminated_mask, df_week, count=1000)
                if valid_rate > max_valid_rate:
                    max_valid_rate = valid_rate
                    best_phi = phi
            
            fine_phis = [best_phi * 0.8, best_phi, best_phi * 1.2]
            best_phi_final = best_phi
            for phi in fine_phis:
                valid_rate = self._test_mallows(center_rank_perm, phi, n, judge_scores, eliminated_mask, df_week, count=CONFIG['SAMPLE_COUNTS']['COARSE'])
                if valid_rate > max_valid_rate:
                    max_valid_rate = valid_rate
                    best_phi_final = phi
            
            samples = self._sample_mallows_force(center_rank_perm, best_phi_final, n, judge_scores, eliminated_mask, df_week, CONFIG['SAMPLE_COUNTS']['FINAL'])
            
            return {
                'samples': samples,
                'phi': best_phi_final,
                'center_sol': center_rank_1based
            }

    def _sample_hit_and_run(self, x0, A, b, n, count):
        samples = []
        curr = x0
        if A.shape[0] > 0 and np.any(A @ curr > b + 1e-5):
            curr = np.ones(n)/n
            
        for _ in range(count):
            d = np.random.randn(n)
            d -= d.mean()
            d /= norm(d)
            step = np.random.dirichlet(curr * 100 + 0.01)
            
            if A.shape[0] == 0 or np.all(A @ step <= b + 1e-6):
                curr = step
                samples.append(curr)
            else:
                samples.append(curr)
                
        return np.array(samples)

    def _test_mallows(self, center, phi, n, judges, elim_mask, df_week, count):
        if phi < 0.1: noise = np.random.randn(count, n) * 10
        else: noise = np.random.normal(0, 1/phi, (count, n))
        noisy = center + noise
        
        valid_count = 0
        has_elim = np.any(elim_mask)
        
        # Precompute Placement Checks if needed
        check_placement = False
        if not has_elim and 'placement' in df_week.columns:
            check_placement = True
            placements = df_week['placement'].values
            sorted_indices = np.argsort(placements) # Best to Worst
        
        for i in range(count):
            fan_pts_perm = np.argsort(np.argsort(noisy[i]))
            fan_pts = fan_pts_perm + 1 
            total = judges + fan_pts
            
            is_valid = True
            if has_elim:
                # Elimination Logic
                safe_mask = ~elim_mask
                min_elim = np.min(total[elim_mask])
                max_safe = np.max(total[safe_mask]) if np.any(safe_mask) else -999
                if min_elim >= max_safe: is_valid = False
            
            elif check_placement:
                # Finals Logic: Check Pairwise Order
                # We check if Total Score strictly decreases or stays same as rank worsens
                # Actually, strictly decreases is safer for model consistency
                
                # Check adjacent pairs in sorted order
                for k in range(n - 1):
                    w_idx = sorted_indices[k]
                    l_idx = sorted_indices[k+1]
                    if total[w_idx] <= total[l_idx]: # Violation: Winner has less or equal points
                        is_valid = False
                        break
            
            if is_valid: valid_count += 1
                
        return valid_count / count

    def _sample_mallows_force(self, center, phi, n, judges, elim_mask, df_week, target):
        samples = []
        batch = 1000
        
        has_elim = np.any(elim_mask)
        check_placement = False
        if not has_elim and 'placement' in df_week.columns:
            check_placement = True
            placements = df_week['placement'].values
            sorted_indices = np.argsort(placements)

        while len(samples) < target:
            if phi < 0.1: noise = np.random.randn(batch, n) * 10
            else: noise = np.random.normal(0, 1/phi, (batch, n))
            
            noisy = center + noise
            
            for i in range(batch):
                fan_pts_perm = np.argsort(np.argsort(noisy[i]))
                fan_pts = fan_pts_perm + 1
                total = judges + fan_pts
                
                is_valid = True
                if has_elim:
                    safe_mask = ~elim_mask
                    min_elim = np.min(total[elim_mask])
                    max_safe = np.max(total[safe_mask]) if np.any(safe_mask) else -999
                    if min_elim >= max_safe: is_valid = False
                elif check_placement:
                    for k in range(n - 1):
                        w_idx = sorted_indices[k]
                        l_idx = sorted_indices[k+1]
                        if total[w_idx] <= total[l_idx]:
                            is_valid = False
                            break
                            
                if is_valid:
                    samples.append(fan_pts) # Store Points (1..N)
                    if len(samples) >= target: break
        
        return np.array(samples)

# ================= Step 3: Main Pipeline =================
def process_week(season, week, df_week, tester, solver):
    season_type = 'RANK' if season in CONFIG['RANK_SEASONS'] else 'PERCENT'
    
    # 1. Bounds
    bounds = tester.solve_bounds(df_week, season_type)
    
    # 2. Simulation
    res = solver.solve_week(season, week, df_week)
    
    if res is None: return [] # Skip if failed
    
    samples = res['samples']
    # Calculate Metrics
    results = []
    names = df_week['celebrity_name'].tolist()
    
    # Stats
    means = samples.mean(axis=0)
    stds = samples.std(axis=0)
    
    # RQI (90% width)
    lower = np.percentile(samples, 5, axis=0)
    upper = np.percentile(samples, 95, axis=0)
    widths = upper - lower
    
    # Entropy
    if season_type == 'PERCENT':
        p = means / means.sum()
        entropy = -np.sum(p * np.log(p + 1e-9))
    else:
        # For Rank, convert points to relative strength or just use uniform entropy?
        # Rank points are 1..N. Normalize to share?
        p = means / means.sum()
        entropy = -np.sum(p * np.log(p + 1e-9))
        
    for i, name in enumerate(names):
        row = {
            'season': season,
            'week': week,
            'celebrity_name': name,
            'phi': res['phi'],
            'center': str(res['center_sol']), # Store as string
            'mean': means[i],
            'std': stds[i],
            'RQI_width': widths[i],
            'entropy': entropy,
            'CR_d': bounds[name][0],
            'CR_u': bounds[name][1]
        }
        results.append(row)
        
    return results

def main():
    print("Loading Data...")
    df = pd.read_csv(INPUT_FILE)
    
    # Pre-checks
    if 'judge_metric' not in df.columns:
        # Simple fix: Create judge_metric from raw sum if needed, but assuming input is ready
        pass
        
    tester = ConsistencyTester()
    solver = HybridSolver()
    
    tasks = []
    # Group by Season/Week
    groups = df.groupby(['season', 'week'])
    
    print(f"Processing {len(groups)} weeks...")
    
    # Parallel Execution
    results = Parallel(n_jobs=CONFIG['N_JOBS'])(
        delayed(process_week)(s, w, data, tester, solver) 
        for (s, w), data in tqdm(groups)
    )
    
    # Flatten
    final_rows = [item for sublist in results for item in sublist]
    result_df = pd.DataFrame(final_rows)
    
    output_path = os.path.join(OUTPUT_DIR, 'solution_feature_fixed.csv')
    result_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
