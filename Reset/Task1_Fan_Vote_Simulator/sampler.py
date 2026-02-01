
import numpy as np
import time
from typing import Tuple, Dict, Optional, List
from structures import PolytopeConstraints, SeasonRules
import warnings

class SamplingTimeoutError(Exception):
    """Raised when sampling exceeds the specified time limit."""
    pass

class MCMCSampler:
    """
    Implements Hit-and-Run (for Percentage) and Hard Rejection (for Rank) sampling strategies.
    """
    def __init__(self, polytope: PolytopeConstraints):
        self.poly = polytope
        self.n_dims = len(polytope.bounds)

    def generate_samples(self, 
                         n_samples: int, 
                         mallows_model=None, 
                         mallows_params: Tuple=None, 
                         timeout: float = 100.0) -> Tuple[np.ndarray, Dict]:
        """
        Main entry point for sampling.
        
        Args:
            n_samples: Target number of valid samples.
            mallows_model: Instance of MallowsModel (required for Rank/Bottom2).
            mallows_params: Tuple (center, phi) from ICR search (required for Rank/Bottom2).
            timeout: Max seconds allowed before partial return.
            
        Returns:
            samples: (N, n_dims) array.
            info: Metadata dict (method, hit_rate, etc.).
        """
        start_time = time.time()
        
        try:
            if self.poly.season_rule == SeasonRules.PERCENTAGE_SUM:
                # Continuous Hit-and-Run
                return self._hit_and_run(n_samples, start_time, timeout)
            else:
                # Rank-based Hard Rejection
                if mallows_model is None or mallows_params is None:
                    raise ValueError("Mallows Model and Params required for Rank/Bottom2 sampling.")
                return self._hard_rejection(n_samples, mallows_model, mallows_params, start_time, timeout)
                
        except SamplingTimeoutError as e:
            warnings.warn(f"Sampling Timeout: {e}")
            # Logic handled inside methods, but double check here if needed
            # Usually methods return partial data before raising, or we handle partials here.
            # Ideally, the sub-methods return what they have.
            raise e # Re-raise to let caller handle partial data retrieval if implemented that way
                    # But better: return what we have from sub-functions directly.

    def _hit_and_run(self, n_samples: int, start_time: float, timeout: float, burn_in: int = 1000) -> Tuple[np.ndarray, Dict]:
        """
        Hit-and-Run algorithm for Percentage Constraints (Sum=1).
        """
        # 1. Start at Geometric Center (Seed)
        current_x = self.poly.seed_guess.copy()
        
        # Ensure seed is strictly feasible (Solver might give edge)
        # Add tiny epsilon towards center of bounds if needed, but usually solver is fine.
        
        samples = np.zeros((n_samples, self.n_dims))
        count = 0
        
        # Pre-allocate random directions? No, generate on fly to save memory.
        
        # Burn-in phase
        for _ in range(burn_in):
            current_x = self._hit_and_run_step(current_x)
            if time.time() - start_time > timeout:
                raise SamplingTimeoutError(f"Timeout during burn-in ({time.time()-start_time:.1f}s)")

        # Sampling phase
        while count < n_samples:
            # Check timeout every 100 samples
            if count % 1000 == 0 and (time.time() - start_time > timeout):
                warnings.warn(f"Timeout triggered at {count}/{n_samples} samples.")
                return samples[:count], {'method': 'hit_and_run', 'status': 'timeout', 'count': count}
            
            # Step
            current_x = self._hit_and_run_step(current_x)
            samples[count] = current_x
            count += 1
            
        return samples, {'method': 'hit_and_run', 'status': 'success', 'count': count}

    def _hit_and_run_step(self, x: np.ndarray) -> np.ndarray:
        """Single step of H&R on the simplex."""
        # 1. Generate Direction d
        d = np.random.normal(0, 1, self.n_dims)
        
        # 2. Project d onto Sum=0 plane (to maintain Sum(x)=1)
        # d_proj = d - mean(d) * [1,1,...1] -> since mean([1...1]) is 1? No.
        # Project vector v onto normal n: v_proj = v - (v.n)*n / |n|^2
        # Normal n = [1, 1, ..., 1]. |n|^2 = n_dims.
        d = d - np.sum(d) / self.n_dims
        
        # Normalize direction (optional, helps numerical stability)
        d = d / np.linalg.norm(d)
        
        # 3. Find Line Segment [lambda_min, lambda_max]
        # x_new = x + lambda * d
        # Constraints: A(x + ld) <= b  ->  Ax + l*Ad <= b  -> l*Ad <= b - Ax
        # Bounds: l_min <= x_i + l*d_i <= u_max
        
        lambda_min = -1e9
        lambda_max = 1e9
        
        # A_ub check
        if self.poly.A_ub is not None:
            # Ax_val = self.poly.A_ub @ x # Precomputed usually better, but fast enough
            # Ad_val = self.poly.A_ub @ d
            # dist = self.poly.b_ub - Ax_val
            
            # Optimization: Vectorized
            # (Ad) * l <= (b - Ax)
            
            numerator = self.poly.b_ub - self.poly.A_ub @ x
            denominator = self.poly.A_ub @ d
            
            # If denom > 0: l <= num / denom  (Upper bound)
            # If denom < 0: l >= num / denom  (Lower bound)
            # If denom = 0: 0 <= num (Always true if x feasible)
            
            with np.errstate(divide='ignore'):
                ratios = numerator / denominator
            
            # Upper bounds (denom > epsilon)
            pos_mask = denominator > 1e-9
            if np.any(pos_mask):
                lambda_max = min(lambda_max, np.min(ratios[pos_mask]))
                
            # Lower bounds (denom < -epsilon)
            neg_mask = denominator < -1e-9
            if np.any(neg_mask):
                lambda_min = max(lambda_min, np.max(ratios[neg_mask]))
                
        # Bounds check (0 <= x <= 1)
        # 0 <= x_i + l*d_i <= 1
        # l*d_i >= -x_i   AND   l*d_i <= 1 - x_i
        for i in range(self.n_dims):
            di = d[i]
            xi = x[i]
            if abs(di) > 1e-9:
                val1 = -xi / di
                val2 = (1.0 - xi) / di
                
                if di > 0:
                    # l >= val1, l <= val2
                    lambda_min = max(lambda_min, val1)
                    lambda_max = min(lambda_max, val2)
                else:
                    # l <= val1, l >= val2
                    lambda_max = min(lambda_max, val1)
                    lambda_min = max(lambda_min, val2)
                    
        # 4. Sample lambda
        if lambda_max <= lambda_min:
            return x # Stuck at corner or numerical error
            
        lam = np.random.uniform(lambda_min, lambda_max)
        return x + lam * d

    def _hard_rejection(self, n_samples: int, model, params: Tuple, start_time: float, timeout: float) -> Tuple[np.ndarray, Dict]:
        """
        Rejection Sampling for Rank/Bottom2 logic.
        """
        center, phi = params
        collected_samples = []
        total_generated = 0
        batch_size = 10000 # Matches Mallows optimized size
        
        while len(collected_samples) < n_samples:
            # Check Timeout
            if time.time() - start_time > timeout:
                warnings.warn(f"Timeout in Hard Rejection. Collected {len(collected_samples)}/{n_samples}.")
                if len(collected_samples) == 0:
                     # Emergency Dummy Data: Return center repeated
                     dummy = model._convert_perms_to_ranks(np.array([center]))
                     return np.tile(dummy, (n_samples, 1)), {'method': 'rejection_dummy', 'status': 'timeout'}
                
                # Stack what we have
                res = np.vstack(collected_samples)
                return res, {'method': 'rejection', 'status': 'timeout', 'count': len(res)}
            
            # 1. Generate Batch
            raw_perms = model.sample_mallows(center, phi, n_samples=batch_size)
            x_ranks = model._convert_perms_to_ranks(raw_perms)
            
            # 2. Validate
            if self.poly.validator:
                valid_mask = self.poly.validator(x_ranks)
            else:
                valid_mask = self.poly.check_linear_feasibility(x_ranks)
                
            valid_batch = x_ranks[valid_mask]
            
            if len(valid_batch) > 0:
                collected_samples.append(valid_batch)
                
            total_generated += batch_size
            
            # Efficiency Check (optional breaker if hit rate is abysmal?)
            # For now, rely on timeout.
            
        # Concatenate and Trim
        final_samples = np.vstack(collected_samples)
        if len(final_samples) > n_samples:
            final_samples = final_samples[:n_samples]
            
        hit_rate = len(final_samples) / total_generated if total_generated > 0 else 0
        return final_samples, {'method': 'rejection', 'status': 'success', 'hit_rate': hit_rate}
