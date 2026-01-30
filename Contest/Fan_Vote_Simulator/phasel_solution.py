# -*- coding: utf-8 -*-
"""
MCM 2026 Problem C - Phase 1: Fan Vote Reconstruction System (Final Optimized & Visualized)
Features:
  1. Solvers: Rank (Adaptive Phi Mallows) & Percent (Chebyshev Hit-and-Run)
  2. Optimization: Aggressive Retry for Low Samples (Hard Timeout Limit)
  3. Logging: Detailed calc process (Phi, Center)
  4. Visualization: Scoring Trends with Elimination Ribbons (No Title, With Legend)
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.optimize import linprog
from numpy.linalg import norm
import time
import warnings
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ================= Configuration =================
CONFIG = {
    'RANK_SEASONS': [1, 2] + list(range(28, 35)),
    'PERCENT_SEASONS': list(range(3, 28)),
    'N_SAMPLES': 2000,          # Target valid samples (Balanced for speed/accuracy)
    'MAX_ATTEMPTS': 100000,     # Very high limit to ensure we get samples
    'TIMEOUT_SEC': 30,          # Base timeout per week
    'HARD_TIMEOUT': 60,         # Absolute max time to spend on a hard week
    'OUTPUT_FILE': 'Phase1_FanVote_Est.csv',
    'LOG_FILE': 'Phase1_Calculation_Log.csv'
}

# ================= 1. Certainty Analyzer =================
class CertaintyAnalyzer:
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level

    def analyze(self, samples, candidate_names):
        if samples is None or len(samples) == 0:
            return None, 0
        
        samples = np.array(samples)
        means = np.mean(samples, axis=0)
        std_devs = np.std(samples, axis=0)
        
        min_vals = np.min(samples, axis=0)
        max_vals = np.max(samples, axis=0)
        range_widths = max_vals - min_vals
        
        lower_p = (1 - self.confidence_level) / 2 * 100
        upper_p = 100 - lower_p
        ci_lowers = np.percentile(samples, lower_p, axis=0)
        ci_uppers = np.percentile(samples, upper_p, axis=0)
        ci_widths = ci_uppers - ci_lowers
        
        # Entropy
        p = means / np.sum(means)
        p = p[p > 0]
        system_entropy = -np.sum(p * np.log(p))
        
        metrics = []
        for i, name in enumerate(candidate_names):
            metrics.append({
                'celebrity_name': name,
                'mean': means[i],
                'std': std_devs[i],
                'feasible_range': range_widths[i],
                'ci_width_95': ci_widths[i],
                'ci_lower': ci_lowers[i],
                'ci_upper': ci_uppers[i]
            })
            
        return metrics, system_entropy

# ================= 2. Constraint Builder =================
class ConstraintBuilder:
    def __init__(self, season_type):
        self.season_type = season_type

    def build_percent_constraints(self, df_week):
        eliminated = df_week[df_week['exit_mode'].isin(['eliminated', 'withdrew'])]
        safe = df_week[df_week['exit_mode'].isin(['safe', 'final'])]
        
        A, b = [], []
        names = df_week['celebrity_name'].tolist()
        
        for _, e_row in eliminated.iterrows():
            for _, s_row in safe.iterrows():
                row = np.zeros(len(names))
                e_idx = names.index(e_row['celebrity_name'])
                s_idx = names.index(s_row['celebrity_name'])
                row[e_idx] = 1
                row[s_idx] = -1
                rhs = s_row['judge_metric'] - e_row['judge_metric']
                A.append(row)
                b.append(rhs)
        
        return np.array(A), np.array(b), names

# ================= 3. Solvers (Aggressive Optimization) =================
class PercentPolytopeSolver:
    def _find_chebyshev_center(self, A, b):
        n_constraints, n_dim = A.shape
        row_norms = norm(A, axis=1)
        c = np.zeros(n_dim + 1)
        c[0] = -1 
        A_lp = np.c_[row_norms, A]
        bounds = [(0, None)] + [(0, 1) for _ in range(n_dim)]
        try:
            res = linprog(c, A_ub=A_lp, b_ub=b, bounds=bounds, method='highs')
            if res.success: return res.x[1:]
        except: pass
        return None

    def solve(self, A, b, n_dim, names):
        x0 = self._find_chebyshev_center(A, b)
        center_used = x0 if x0 is not None else np.ones(n_dim)/n_dim
        
        if x0 is None: 
            x0 = np.ones(n_dim) / n_dim
            if A.shape[0] > 0 and np.any(b - A @ x0 < -1e-5):
                 return None, None 

        samples = []
        timeout = CONFIG.get('CURRENT_TIMEOUT', CONFIG['TIMEOUT_SEC'])
        start_time = time.time()
        alpha_scale = 50 
        
        # Aggressive Loop
        while len(samples) < CONFIG['N_SAMPLES']:
            if time.time() - start_time > timeout: break
            
            # Batch generation for speed
            candidates = np.random.dirichlet(x0 * alpha_scale + 0.01, size=100)
            
            if A.shape[0] == 0:
                samples.extend(candidates)
            else:
                # Vectorized check
                valid_mask = np.all(candidates @ A.T <= b.T + 1e-6, axis=1)
                valid_candidates = candidates[valid_mask]
                samples.extend(valid_candidates)
                
        # Trim to N
        if len(samples) > CONFIG['N_SAMPLES']: samples = samples[:CONFIG['N_SAMPLES']]
        
        meta = {
            'phi': 'N/A (Convex)',
            'center': np.array2string(center_used, precision=3, separator=',', suppress_small=True)
        }
        return np.array(samples) if samples else None, meta

class RankMallowsSolver:
    def solve(self, df_week, strict_mode=True):
        names = df_week['celebrity_name'].tolist()
        n = len(names)
        judge_ranks = df_week['judge_metric'].values 
        eliminated_mask = df_week['exit_mode'].isin(['eliminated', 'withdrew']).values
        safe_mask = ~eliminated_mask
        
        samples = []
        center_rank = np.argsort(np.argsort(judge_ranks)) 
        
        timeout = CONFIG.get('CURRENT_TIMEOUT', CONFIG['TIMEOUT_SEC'])
        start_time = time.time()
        
        # Adaptive Strategy:
        # 1. Try Judge Center with various Phis
        # 2. If that fails, try Inverse Center (Reverse Judge)
        
        centers_to_try = [
            ('Judge', center_rank),
            ('Reverse', np.argsort(np.argsort(-judge_ranks))) # Heuristic for shock eliminations
        ]
        phis_to_try = [2.0, 1.0, 0.5, 0.1]
        
        best_phi_used = None
        best_center_name = None
        
        # Search Phase
        for c_name, c_rank in centers_to_try:
            if len(samples) > 100: break # Already found a working path
            
            for phi in phis_to_try:
                if len(samples) > 100: break
                
                # Try a burst
                burst_end = time.time() + 2 # 2 seconds probe
                temp_samples = []
                
                while time.time() < burst_end:
                    if phi < 0.2:
                        fan_rank_perm = np.random.permutation(n)
                    else:
                        noise = np.random.normal(0, 1/phi, n)
                        noisy_ranks = c_rank + noise
                        fan_rank_perm = np.argsort(np.argsort(noisy_ranks))
                    
                    fan_ranks = fan_rank_perm + 1 
                    total_scores = judge_ranks + fan_ranks
                    
                    is_valid = True
                    if strict_mode:
                        if np.any(eliminated_mask):
                            min_elim = np.min(total_scores[eliminated_mask])
                            max_safe = np.max(total_scores[safe_mask]) if np.any(safe_mask) else 999
                            if min_elim < max_safe: is_valid = False
                    else: 
                        all_sorted = np.sort(total_scores)
                        bottom2 = all_sorted[-2]
                        elim_scores = total_scores[eliminated_mask]
                        if not np.any(elim_scores >= bottom2): is_valid = False
                    
                    if is_valid:
                        vote_proxy = np.exp(-0.5 * (fan_ranks - 1))
                        vote_proxy /= vote_proxy.sum()
                        temp_samples.append(vote_proxy)
                
                # If this combo works, commit to it
                if len(temp_samples) > 0:
                    best_phi_used = phi
                    best_center_name = c_name
                    samples.extend(temp_samples)
                    
                    # Fill Phase
                    while len(samples) < CONFIG['N_SAMPLES']:
                        if time.time() - start_time > timeout: break
                        
                        # Repeat logic... (Code duplication minimized for brevity in logic)
                        if phi < 0.2:
                            fan_rank_perm = np.random.permutation(n)
                        else:
                            noise = np.random.normal(0, 1/phi, n)
                            noisy_ranks = c_rank + noise
                            fan_rank_perm = np.argsort(np.argsort(noisy_ranks))
                        
                        fan_ranks = fan_rank_perm + 1
                        total_scores = judge_ranks + fan_ranks
                        
                        is_valid = True
                        if strict_mode:
                            if np.any(eliminated_mask):
                                min_elim = np.min(total_scores[eliminated_mask])
                                max_safe = np.max(total_scores[safe_mask]) if np.any(safe_mask) else 999
                                if min_elim < max_safe: is_valid = False
                        else:
                            all_sorted = np.sort(total_scores)
                            bottom2 = all_sorted[-2]
                            elim_scores = total_scores[eliminated_mask]
                            if not np.any(elim_scores >= bottom2): is_valid = False
                            
                        if is_valid:
                            vote_proxy = np.exp(-0.5 * (fan_ranks - 1))
                            vote_proxy /= vote_proxy.sum()
                            samples.append(vote_proxy)
                    break # Break phi loop
        
        meta = {
            'phi': best_phi_used if best_phi_used else 'None',
            'center_type': best_center_name,
            'center_val': np.array2string(center_rank, precision=0, separator=',')
        }
        return np.array(samples) if samples else None, meta

# ================= 4. Main Pipeline =================
class Pipeline:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.results = []
        self.calc_logs = [] 
        self.certainty_analyzer = CertaintyAnalyzer()
        
    def run(self):
        grouped = self.df.groupby(['season', 'week'])
        pbar = tqdm(total=len(grouped), desc="Processing Weeks")
        
        for (season, week), group in grouped:
            # Skip safe weeks
            if 'eliminated' not in group['exit_mode'].values and 'withdrew' not in group['exit_mode'].values:
                self._record_dummy(season, week, group['celebrity_name'], "No_Elimination")
                pbar.update(1)
                continue

            # Aggressive Retry Logic
            retry_stages = [
                {'timeout': CONFIG['TIMEOUT_SEC'], 'desc': 'Standard'},
                {'timeout': CONFIG['HARD_TIMEOUT'], 'desc': 'Hard_Retry'}
            ]
            
            samples = None
            final_method = ""
            final_meta = {}
            
            for stage in retry_stages:
                CONFIG['CURRENT_TIMEOUT'] = stage['timeout']
                try:
                    if season in CONFIG['PERCENT_SEASONS']:
                        builder = ConstraintBuilder('Percent')
                        A, b, names = builder.build_percent_constraints(group)
                        solver = PercentPolytopeSolver()
                        samples, meta = solver.solve(A, b, len(names), names)
                        final_method = "Percent_Convex"
                        final_meta = meta
                        
                    elif season in CONFIG['RANK_SEASONS']:
                        solver = RankMallowsSolver()
                        final_method = "Rank_Strict"
                        samples, meta = solver.solve(group, strict_mode=True)
                        final_meta = meta
                        
                        if samples is None and season >= 28:
                            samples, meta = solver.solve(group, strict_mode=False)
                            final_method = "Rank_Relaxed_Bottom2"
                            final_meta = meta
                    
                    if samples is not None and len(samples) >= 100: # Threshold for success
                        break 
                        
                except Exception as e:
                    pass
            
            # Record Results & Log
            if samples is not None and len(samples) > 0:
                self._process_samples(season, week, group['celebrity_name'].tolist(), samples, final_method)
                self.calc_logs.append({
                    'season': season,
                    'week': week,
                    'method': final_method,
                    'phi': final_meta.get('phi', 'N/A'),
                    'center_type': final_meta.get('center_type', 'N/A'),
                    'center_val': final_meta.get('center_val', 'N/A'),
                    'sample_count': len(samples)
                })
            else:
                self._record_dummy(season, week, group['celebrity_name'], "Failed_Convergence")
                self.calc_logs.append({
                    'season': season, 'week': week, 'method': 'Failed',
                    'sample_count': 0
                })
                
            pbar.update(1)
            
        self.save()

    def _process_samples(self, season, week, names, samples, method):
        metrics, system_entropy = self.certainty_analyzer.analyze(samples, names)
        for m in metrics:
            self.results.append({
                'season': season,
                'week': week,
                'celebrity_name': m['celebrity_name'],
                'est_vote_share_mean': m['mean'],
                'est_vote_share_std': m['std'],
                'ci_width_95': m['ci_width_95'],
                'feasible_range': m['feasible_range'],
                'method': method,
                'entropy': system_entropy,
                'sample_count': len(samples)
            })

    def _record_dummy(self, season, week, names, method):
        for name in names:
            self.results.append({
                'season': season, 'week': week, 'celebrity_name': name,
                'est_vote_share_mean': np.nan, 'est_vote_share_std': np.nan,
                'method': method, 'entropy': np.nan, 'sample_count': 0
            })

    def save(self):
        pd.DataFrame(self.results).to_csv(CONFIG['OUTPUT_FILE'], index=False)
        pd.DataFrame(self.calc_logs).to_csv(CONFIG['LOG_FILE'], index=False)
        print(f"Results saved to {CONFIG['OUTPUT_FILE']}")
        print(f"Logs saved to {CONFIG['LOG_FILE']}")

# ================= 5. Visualization Function (No Title, With Legend) =================
def plot_scoring_trends(data_path, output_dir='season_plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(data_path)
    seasons = df['season'].unique()
    
    print("Generating plots...")
    for season in tqdm(seasons):
        season_df = df[df['season'] == season]
        
        plt.figure(figsize=(10, 6))
        
        # 1. Calc Ribbon Boundary
        weeks = sorted(season_df['week'].unique())
        elimination_boundary = []
        plot_weeks = []
        
        for w in weeks:
            week_data = season_df[season_df['week'] == w]
            eliminated = week_data[week_data['exit_mode'].isin(['eliminated', 'withdrew'])]
            
            # Boundary is the MAX score of eliminated contestants (The "Danger Line")
            # If Raw Score < this, you were in danger.
            if not eliminated.empty:
                boundary_val = eliminated['weekly_raw_score'].max()
            else:
                boundary_val = 0 
            
            elimination_boundary.append(boundary_val)
            plot_weeks.append(w)
            
        # 2. Plot Lines
        # Use a distinct palette
        unique_celebs = season_df['celebrity_name'].unique()
        palette = sns.color_palette("husl", len(unique_celebs))
        
        sns.lineplot(data=season_df, x='week', y='weekly_raw_score', hue='celebrity_name', 
                     marker='o', palette=palette, alpha=0.8, linewidth=2)
        
        # 3. Plot Ribbon
        plt.fill_between(plot_weeks, 0, elimination_boundary, color='red', alpha=0.15, label='Elimination Zone')
        
        # 4. Styling (No Title)
        plt.xlabel('Week')
        plt.ylabel('Weekly Raw Judge Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        
        plt.savefig(f'{output_dir}/Season_{season}_Trend.png', dpi=150)
        plt.close()
    
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    # Check if file exists, else use dummy path for demo
    data_path = 'D:/Files/Study/code/DataProcessing/assessment_models/Contest/Fan_Vote_Simulator/dwts_model_ready.csv'
    plot_scoring_trends(data_path, output_dir='season_plots')