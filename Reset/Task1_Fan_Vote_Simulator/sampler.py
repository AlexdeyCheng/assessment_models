import numpy as np
import time
import warnings
from typing import Tuple, Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from structures import PolytopeConstraints, SeasonRules

# --- Worker Function (Must be top-level for pickling) ---
def _worker_hard_rejection_batch(seed: int, batch_size: int, model_poly_data, params):
    """
    Independent worker function to generate a batch of samples.
    Args:
        seed: Random seed for this worker.
        batch_size: Number of attempts or target samples (logic dependent).
        model_poly_data: Tuple containing (poly_constraints, pi0, phi).
                         We pass data, not full objects if possible, to simplify pickling.
                         Actually, passing the MallowsModel object might be heavy.
                         Let's reconstruct logic here or pass the model if it pickles well.
    """
    np.random.seed(seed)
    poly, pi0, phi = model_poly_data
    
    # Re-implement Mallows sampling logic locally to avoid complex object passing
    # or assume MallowsModel methods are static/clean.
    # To be safe and fast, we use the logic derived from mallows.py directly.
    
    n_items = len(pi0)
    collected = []
    
    # Batch generation loop
    # We generate a large chunk, validate, and return valid ones.
    # To avoid infinite loops in worker, we try a fixed number of times relative to batch_size
    # Say we try 5 * batch_size attempts to get valid samples.
    
    attempts = batch_size * 5 
    
    # Vectorized Mallows Sample (Inline for performance/pickling safety)
    # 1. Generate Raw Perms
    # RIM Algorithm
    samples_idx = np.zeros((attempts, n_items), dtype=int)
    current_perms = np.zeros((attempts, 1), dtype=int)
    
    for i in range(1, n_items):
        costs = np.arange(i + 1)[::-1]
        logits = -phi * costs
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        insert_positions = np.random.choice(i + 1, size=attempts, p=probs)
        
        new_perms = np.zeros((attempts, i + 1), dtype=int)
        for k in range(attempts):
            new_perms[k] = np.insert(current_perms[k], insert_positions[k], i)
        current_perms = new_perms
        
    # Map back to pi0 items
    for k in range(attempts):
        samples_idx[k] = pi0[current_perms[k]]
        
    # 2. Convert to Ranks
    # ranks[row, perms[row, col]] = col + 1
    x_ranks = np.zeros_like(samples_idx)
    rows = np.arange(attempts)[:, np.newaxis]
    x_ranks[rows, samples_idx] = np.arange(1, n_items + 1)
    
    # 3. Validate
    # Use poly's vectorized validator
    if poly.validator:
        valid_mask = poly.validator(x_ranks)
    else:
        valid_mask = poly.check_linear_feasibility(x_ranks)
        
    valid_samples = x_ranks[valid_mask]
    
    # Return up to batch_size, or whatever we got
    return valid_samples[:batch_size]

class SamplingTimeoutError(Exception):
    pass

class MCMCSampler:
    def __init__(self, polytope: PolytopeConstraints):
        self.poly = polytope
        self.n_dims = len(polytope.bounds)

    def generate_samples(self, 
                         n_samples: int, 
                         mallows_model=None, 
                         mallows_params: Tuple=None, 
                         timeout: float = 100.0) -> Tuple[np.ndarray, Dict]:
        
        start_time = time.time()
        
        try:
            if self.poly.season_rule == SeasonRules.PERCENTAGE_SUM:
                return self._hit_and_run(n_samples, start_time, timeout)
            else:
                if mallows_model is None or mallows_params is None:
                    raise ValueError("Mallows Model required for Rank sampling.")
                return self._hard_rejection_parallel(n_samples, mallows_params, start_time, timeout)
                
        except SamplingTimeoutError as e:
            warnings.warn(f"Sampling Timeout: {e}")
            raise e 

    def _hit_and_run(self, n_samples: int, start_time: float, timeout: float) -> Tuple[np.ndarray, Dict]:
        # Hit-and-Run is hard to parallelize due to Markov Chain nature (sequential dependence).
        # We keep it single-threaded but optimized.
        # ... (Existing H&R code remains the same, omitted for brevity, assume previous logic) ...
        # For simplicity in this fix, I will paste the previous H&R logic back to ensure file completeness.
        
        current_x = self.poly.seed_guess.copy()
        samples = np.zeros((n_samples, self.n_dims))
        count = 0
        
        # Burn-in
        for _ in range(1000):
            current_x = self._hit_and_run_step(current_x)
            if time.time() - start_time > timeout:
                return np.zeros((0, self.n_dims)), {'method': 'hit_and_run', 'status': 'timeout_burnin'}

        while count < n_samples:
            if count % 1000 == 0 and (time.time() - start_time > timeout):
                warnings.warn(f"H&R Timeout. Collected {count}/{n_samples}.")
                return samples[:count], {'method': 'hit_and_run', 'status': 'timeout', 'count': count}
            
            current_x = self._hit_and_run_step(current_x)
            samples[count] = current_x
            count += 1
            
        return samples, {'method': 'hit_and_run', 'status': 'success', 'count': count}

    def _hit_and_run_step(self, x):
        # ... (Same as previous step) ...
        d = np.random.normal(0, 1, self.n_dims)
        d = d - np.sum(d) / self.n_dims # Project
        d = d / np.linalg.norm(d)
        
        lambda_min, lambda_max = -1e9, 1e9
        
        if self.poly.A_ub is not None:
            numerator = self.poly.b_ub - self.poly.A_ub @ x
            denominator = self.poly.A_ub @ d
            with np.errstate(divide='ignore'):
                ratios = numerator / denominator
            
            pos_mask = denominator > 1e-9
            if np.any(pos_mask): lambda_max = min(lambda_max, np.min(ratios[pos_mask]))
            neg_mask = denominator < -1e-9
            if np.any(neg_mask): lambda_min = max(lambda_min, np.max(ratios[neg_mask]))
            
        for i in range(self.n_dims):
            di = d[i]
            xi = x[i]
            if abs(di) > 1e-9:
                val1, val2 = -xi/di, (1.0-xi)/di
                if di > 0: lambda_min = max(lambda_min, val1); lambda_max = min(lambda_max, val2)
                else: lambda_max = min(lambda_max, val1); lambda_min = max(lambda_min, val2)
                    
        if lambda_max <= lambda_min: return x
        lam = np.random.uniform(lambda_min, lambda_max)
        return x + lam * d

    def _hard_rejection_parallel(self, n_samples: int, params: Tuple, start_time: float, timeout: float) -> Tuple[np.ndarray, Dict]:
        """
        Multiprocessing version of Hard Rejection.
        """
        center, phi = params
        
        # Determine number of workers (leave 1 core free ideally, or use max)
        max_workers = min(12, multiprocessing.cpu_count()) 
        # Chunk size
        chunk_size = int(n_samples / max_workers) + 1
        
        collected_samples = []
        
        # Data payload for workers
        # poly needs to be picklable. Dataclasses are picklable.
        poly_data = (self.poly, center, phi)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            # Submit tasks
            for i in range(max_workers):
                seed = int(time.time()) + i * 999
                futures.append(executor.submit(_worker_hard_rejection_batch, seed, chunk_size, poly_data, params))
            
            # Collect results with timeout monitoring
            # Note: as_completed doesn't support global timeout easily across all, 
            # but we can check time inside the loop
            
            for future in as_completed(futures):
                # Check main timeout
                if time.time() - start_time > timeout:
                    warnings.warn(f"Parallel Sampling Timeout. Stopping workers.")
                    # We can't easily kill threads in Executor, but we stop collecting.
                    # The gathered 'collected_samples' is what we return.
                    break
                
                try:
                    # We give each chunk a small timeout relative to total, or just wait
                    # If the worker is stuck in infinite loop, this might hang.
                    # But our worker has finite loop 'attempts'.
                    batch = future.result(timeout=timeout) 
                    if len(batch) > 0:
                        collected_samples.append(batch)
                except Exception as e:
                    # If a worker fails or times out
                    continue
        
        # Combine
        if collected_samples:
            final_samples = np.vstack(collected_samples)
            if len(final_samples) > n_samples:
                final_samples = final_samples[:n_samples]
        else:
            final_samples = np.zeros((0, self.n_dims))
            
        count = len(final_samples)
        status = 'success' if count >= n_samples else 'partial_timeout'
        
        return final_samples, {'method': 'rejection_parallel', 'status': status, 'count': count}