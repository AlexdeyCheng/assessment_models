
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
    """
    Basic data holder for a single contestant's weekly state.
    
    Attributes:
        name: Celebrity name.
        judge_metric: effective_score from input (Rank int or Percentage float).
        fan_offset: Accumulated fan score from previous non-elimination weeks (0.0 if reset).
        is_accumulated: Boolean flag indicating if this week carries history (for output metadata).
        is_eliminated: True if this person was eliminated this week.
    """
    name: str
    judge_metric: float  
    fan_offset: float = 0.0
    is_accumulated: bool = False
    is_eliminated: bool = False
    
    # Internal usage for solver logic check
    is_bottom2_survivor: bool = False 

@dataclass
class PolytopeConstraints:
    """
    Container for the Linear Programming constraints defining the 'Legal Fan Vote Space'.
    Represents the polytope P = {x | A_ub @ x <= b_ub, A_eq @ x == b_eq, bounds}.
    """
    # Linear Inequalities: A_ub @ x <= b_ub
    A_ub: np.ndarray
    b_ub: np.ndarray
    
    # Linear Equalities: A_eq @ x == b_eq (Mainly for Percentage sum=1)
    A_eq: Optional[np.ndarray] = None
    b_eq: Optional[np.ndarray] = None
    
    # Variable Bounds: List of (min, max) for each variable x_i
    bounds: List[tuple] = field(default_factory=list)
    
    # Semantic Map: Maps variable index i to Contestant Name
    index_to_name: Dict[int, str] = field(default_factory=dict)
    
    # The geometric center of the polytope (or feasible starting point)
    seed_guess: Optional[np.ndarray] = None
    
    # Validation function for complex logic (e.g., specific Permutation checks for Ranks)
    # Signature: func(fan_values: np.array) -> bool
    validator: Optional[Callable[[np.ndarray], bool]] = None
    
    # Metadata
    season_rule: SeasonRules = SeasonRules.PERCENTAGE_SUM

    def check_linear_feasibility(self, x: np.ndarray, tolerance=1e-6) -> bool:
        """Quick check if a point x satisfies the linear constraints."""
        if self.A_ub is not None and len(self.A_ub) > 0:
            if np.any(self.A_ub @ x - self.b_ub > tolerance):
                return False
        if self.A_eq is not None and len(self.A_eq) > 0:
            if np.any(np.abs(self.A_eq @ x - self.b_eq) > tolerance):
                return False
        # Bounds check
        for i, val in enumerate(x):
            lower, upper = self.bounds[i]
            if val < lower - tolerance or (upper is not None and val > upper + tolerance):
                return False
        return True
