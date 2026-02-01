
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Callable, Dict, Any

class SeasonRules(Enum):
    """Defines the scoring and elimination logic for different eras of DWTS."""
    RANK_SUM = "rank_sum"           # S1-S2: Rank(Fan) + Rank(Judge) -> Lowest Eliminated
    PERCENTAGE_SUM = "percentage_sum" # S3-S27: %(Fan) + %(Judge) -> Highest Eliminated
    RANK_BOTTOM2 = "rank_bottom2"   # S28+: Rank(Fan) + Rank(Judge) -> Bottom 2 -> Judges Pick

@dataclass
class Contestant:
    """Basic data holder for a single contestant's weekly state."""
    name: str
    judge_metric: float  
    fan_offset: float = 0.0
    is_accumulated: bool = False
    is_eliminated: bool = False
    is_bottom2_survivor: bool = False 

@dataclass
class PolytopeConstraints:
    """
    Container for the Linear Programming constraints.
    Supports BATCH validation for high-performance sampling.
    """
    A_ub: np.ndarray
    b_ub: np.ndarray
    A_eq: Optional[np.ndarray] = None
    b_eq: Optional[np.ndarray] = None
    bounds: List[tuple] = field(default_factory=list)
    index_to_name: Dict[int, str] = field(default_factory=dict)
    seed_guess: Optional[np.ndarray] = None
    season_rule: SeasonRules = SeasonRules.PERCENTAGE_SUM
    
    # Vectorized Validator Function: func(x_matrix: (N_samples, N_dims)) -> bool_array (N_samples,)
    validator: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def check_linear_feasibility(self, x: np.ndarray, tolerance=1e-6) -> np.ndarray:
        """
        Vectorized check if points x satisfy linear constraints.
        Args:
            x: Shape (n_dims,) for single point OR (n_samples, n_dims) for batch.
        Returns:
            Boolean array of shape (n_samples,) or single bool.
        """
        # Ensure x is at least 2D for consistent broadcasting, or handle 1D gracefully
        x_is_1d = x.ndim == 1
        if x_is_1d:
            x = x[np.newaxis, :] # (1, n_dims)
            
        n_samples = x.shape[0]
        valid_mask = np.ones(n_samples, dtype=bool)

        # 1. Inequality Constraints: A @ x.T <= b
        # A: (m, n_dims), x.T: (n_dims, n_samples) -> (m, n_samples)
        if self.A_ub is not None and len(self.A_ub) > 0:
            # Result shape: (n_constraints, n_samples)
            ineq_diff = self.A_ub @ x.T - self.b_ub[:, np.newaxis]
            # Check if any constraint is violated (> tolerance) per sample
            # any(axis=0) returns (n_samples,) True if any violation exists
            violated = np.any(ineq_diff > tolerance, axis=0)
            valid_mask &= (~violated)

        # 2. Equality Constraints: A @ x.T == b
        if self.A_eq is not None and len(self.A_eq) > 0:
            eq_diff = np.abs(self.A_eq @ x.T - self.b_eq[:, np.newaxis])
            violated = np.any(eq_diff > tolerance, axis=0)
            valid_mask &= (~violated)

        # 3. Bounds Check
        # x shape: (n_samples, n_dims)
        for i, (lower, upper) in enumerate(self.bounds):
            col = x[:, i]
            if lower is not None:
                valid_mask &= (col >= lower - tolerance)
            if upper is not None:
                valid_mask &= (col <= upper + tolerance)
                
        if x_is_1d:
            return valid_mask[0]
        return valid_mask
