
import numpy as np
from typing import List, Tuple, Optional
from structures import PolytopeConstraints, SeasonRules
import warnings

class MallowsModel:
    """
    Implements optimized Mallows Model logic with Batch Validation.
    """
    
    def __init__(self, polytope: PolytopeConstraints):
        self.poly = polytope
        self.n = len(polytope.bounds)

    def sample_mallows(self, pi0: np.ndarray, phi: float, n_samples: int = 5000) -> np.ndarray:
        """
        Vectorized RIM Sampler.
        Returns samples as RANK VECTORS (1..N) if season is RANK-based,
        or Indices (0..N-1) if used internally.
        
        CRITICAL: The constraint checker expects 'x' values.
        For Rank Season: x_i is the Rank (1st, 2nd...).
        This sampler generates Permutations (indices).
        We must convert Permutation -> Rank Vector.
        """
        if phi == 0:
            raw_perms = np.array([np.random.permutation(self.n) for _ in range(n_samples)])
        else:
            # Vectorized RIM implementation
            # Since insertion logic is sequential per item, we vectorise over samples
            samples = np.zeros((n_samples, self.n), dtype=int)
            
            # Init with item 0
            # Iterate items 1 to n-1
            # Current perm length is i+1 after insertion
            
            # To vectorize: We effectively need to track insertion indices for each sample
            # Python loop over items (N is small ~13), vectorized over n_samples (5000)
            
            current_perms = np.zeros((n_samples, 1), dtype=int) # Item 0 is at pos 0
            
            for i in range(1, self.n):
                # Calculate insertion probs for item 'i'
                costs = np.arange(i + 1)[::-1] # Costs: i, i-1, ..., 0
                logits = -phi * costs
                logits -= np.max(logits)
                probs = np.exp(logits)
                probs /= np.sum(probs)
                
                # Sample insertion positions for all samples at once
                # shape (n_samples,)
                insert_positions = np.random.choice(i + 1, size=n_samples, p=probs)
                
                # Insert item 'i' into current_perms at insert_positions
                # This is tricky to vectorise fully without a loop or fancy indexing
                # Since N is small, we can reconstruct
                new_perms = np.zeros((n_samples, i + 1), dtype=int)
                for k in range(n_samples):
                    pos = insert_positions[k]
                    # Insert 'i' at pos
                    # previous items are in current_perms[k]
                    new_perms[k] = np.insert(current_perms[k], pos, i)
                current_perms = new_perms
            
            # Map canonical indices back to pi0 items
            # current_perms contains indices into pi0
            raw_perms = np.zeros((n_samples, self.n), dtype=int)
            for k in range(n_samples):
                raw_perms[k] = pi0[current_perms[k]]

        return raw_perms

    def _convert_perms_to_ranks(self, perms: np.ndarray) -> np.ndarray:
        """
        Converts array of Permutations (who is in pos 0, 1, 2...)
        to Rank Vectors (Contestant 0 is rank X, Contestant 1 is rank Y...).
        
        Args:
            perms: (n_samples, n_items). e.g. [2, 0, 1] means Contestant 2 is 1st.
        Returns:
            ranks: (n_samples, n_items). e.g. [2, 3, 1] means C0 is 2nd, C1 is 3rd, C2 is 1st.
            (Using 1-based ranks for Solver compatibility)
        """
        n_samples, n = perms.shape
        ranks = np.zeros_like(perms)
        # Advanced Indexing
        # rows: [[0,0...], [1,1...]]
        rows = np.arange(n_samples)[:, np.newaxis]
        # For each row, perms[row, col] is the contestant index.
        # We want ranks[row, contestant_index] = col + 1
        ranks[rows, perms] = np.arange(1, n + 1)
        return ranks

    def icr_search(self, initial_phi=1.0, n_coarse=5000, n_fine=10000) -> Tuple[np.ndarray, float]:
        """
        Optimized Iterative Center Refinement.
        """
        # 1. Initial Center from Solver Seed
        seed_continuous = self.poly.seed_guess
        if self.poly.season_rule == SeasonRules.PERCENTAGE_SUM:
             # For %: Higher val = Better Rank. Sort Descending gives Order.
            current_center = np.argsort(seed_continuous)[::-1]
        else:
             # For Rank: Lower val = Better Rank. Sort Ascending.
            current_center = np.argsort(seed_continuous)

        best_center = current_center
        best_phi = initial_phi
        best_rate = -1.0
        
        # 2. Coarse Search
        for _ in range(3): # Max 3 improvements
            neighbors = [best_center]
            # Generate Swap-1 Neighbors
            for i in range(self.n - 1):
                nb = best_center.copy()
                nb[i], nb[i+1] = nb[i+1], nb[i]
                neighbors.append(nb)
            
            # Grid
            phi_grid = np.arange(0.0, 4.1, 0.5)
            
            iter_best_rate = -1
            iter_best_param = (None, None)
            
            for center_cand in neighbors:
                for phi in phi_grid:
                    # BATCH SAMPLING
                    raw_perms = self.sample_mallows(center_cand, phi, n_samples=n_coarse)
                    
                    # CONVERT TO RANKS (Vectorized)
                    x_ranks = self._convert_perms_to_ranks(raw_perms)
                    
                    # BATCH VALIDATION
                    if self.poly.validator:
                        # Logic for complex bottom-2 specific checks if needed
                        valid_mask = self.poly.validator(x_ranks)
                    else:
                        valid_mask = self.poly.check_linear_feasibility(x_ranks)
                    
                    rate = np.mean(valid_mask)
                    
                    if rate > iter_best_rate:
                        iter_best_rate = rate
                        iter_best_param = (center_cand, phi)
            
            if iter_best_rate > best_rate:
                best_rate = iter_best_rate
                best_center, best_phi = iter_best_param
            else:
                break
        
        # 3. Fine Search
        # Range: [Best - 0.3, Best + 0.3]
        phi_start = max(0.0, best_phi - 0.3)
        phi_end = best_phi + 0.35
        phi_fine_grid = np.arange(phi_start, phi_end, 0.05)
        
        for phi in phi_fine_grid:
            raw_perms = self.sample_mallows(best_center, phi, n_samples=n_fine)
            x_ranks = self._convert_perms_to_ranks(raw_perms)
            
            if self.poly.validator:
                valid_mask = self.poly.validator(x_ranks)
            else:
                valid_mask = self.poly.check_linear_feasibility(x_ranks)
                
            rate = np.mean(valid_mask)
            
            if rate > best_rate: # Only update if improvement found
                best_rate = rate
                best_phi = phi
                
        return best_center, best_phi
