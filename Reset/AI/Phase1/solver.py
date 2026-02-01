
import numpy as np
from scipy.optimize import linprog
import warnings
from typing import List, Tuple, Dict
from structures import SeasonRules, PolytopeConstraints, Contestant

class ConstraintBuilder:
    def __init__(self, season_rule: SeasonRules):
        self.season_rule = season_rule

    def build(self, contestants: List[Contestant], eliminated_name: str) -> PolytopeConstraints:
        """
        Main entry point. Constructs the Polytope defining valid fan votes.
        
        Args:
            contestants: List of Contestant objects for the current week.
            eliminated_name: Name of the person who went home (or "None" if non-elimination).
        """
        # 1. Map names to indices
        names = [c.name for c in contestants]
        n = len(contestants)
        name_map = {i: name for i, name in enumerate(names)}
        
        # 2. Identify Elimination Index
        elim_idx = -1
        if eliminated_name and eliminated_name in names:
            elim_idx = names.index(eliminated_name)
        
        # 3. Route to specific logic
        if self.season_rule == SeasonRules.PERCENTAGE_SUM:
            return self._build_percentage_constraints(contestants, elim_idx, name_map)
        elif self.season_rule == SeasonRules.RANK_SUM:
            return self._build_rank_constraints(contestants, elim_idx, name_map)
        elif self.season_rule == SeasonRules.RANK_BOTTOM2:
            return self._build_bottom2_constraints(contestants, elim_idx, name_map)
        else:
            raise ValueError(f"Unsupported Season Rule: {self.season_rule}")

    def _build_percentage_constraints(self, contestants: List[Contestant], elim_idx: int, name_map: Dict) -> PolytopeConstraints:
        """
        Logic: (Fan%_S + Offset_S) + Judge%_S > (Fan%_E + Offset_E) + Judge%_E
        Standard Linear Programming form.
        """
        n = len(contestants)
        A_ub = []
        b_ub = []
        
        # Constraints: Everyone Safe (S) > Eliminated (E)
        if elim_idx != -1:
            E = contestants[elim_idx]
            for i, S in enumerate(contestants):
                if i == elim_idx: continue
                
                # S_total > E_total
                # (x_S + O_S) + J_S > (x_E + O_E) + J_E
                # x_E - x_S < (O_S - O_E) + (J_S - J_E)
                
                row = np.zeros(n)
                row[elim_idx] = 1  # Coeff for x_E
                row[i] = -1        # Coeff for x_S
                
                # Right Hand Side
                rhs = (S.fan_offset - E.fan_offset) + (S.judge_metric - E.judge_metric)
                
                A_ub.append(row)
                b_ub.append(rhs)

        # Equality: Sum(x) = 1.0 (100%)
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])
        
        # Bounds: 0.0 <= x <= 1.0
        bounds = [(0.0, 1.0) for _ in range(n)]
        
        # Construct Object
        poly = PolytopeConstraints(
            A_ub=np.array(A_ub) if A_ub else np.zeros((0, n)),
            b_ub=np.array(b_ub) if b_ub else np.zeros(0),
            A_eq=A_eq, b_eq=b_eq, bounds=bounds, index_to_name=name_map,
            season_rule=SeasonRules.PERCENTAGE_SUM
        )
        
        # Calculate Center
        poly.seed_guess = self._solve_chebyshev_center(poly)
        return poly

    def _build_rank_constraints(self, contestants: List[Contestant], elim_idx: int, name_map: Dict) -> PolytopeConstraints:
        """
        Logic: Rank(S) + Offset(S) + Judge(S) < Rank(E) + Offset(E) + Judge(E)  (Lower is Better)
        """
        n = len(contestants)
        A_ub = []
        b_ub = []
        
        if elim_idx != -1:
            E = contestants[elim_idx]
            for i, S in enumerate(contestants):
                if i == elim_idx: continue
                
                # Total_S < Total_E (Lower score is better)
                # (Rank_S + O_S + J_S) < (Rank_E + O_E + J_E)
                # Rank_S - Rank_E < (J_E - J_S) + (O_E - O_S)
                
                row = np.zeros(n)
                row[i] = 1         # Coeff for x_S
                row[elim_idx] = -1 # Coeff for x_E
                
                # RHS logic: (Judge_Diff) + (Offset_Diff)
                judge_diff = E.judge_metric - S.judge_metric
                offset_diff = E.fan_offset - S.fan_offset
                rhs = judge_diff + offset_diff
                
                A_ub.append(row)
                b_ub.append(rhs - 0.5) # -0.5 is a small margin to enforce strict inequality in continuous space
        
        # Bounds: 1.0 <= x <= N (Relaxed)
        bounds = [(1.0, float(n)) for _ in range(n)]
        
        poly = PolytopeConstraints(
            A_ub=np.array(A_ub) if A_ub else np.zeros((0, n)),
            b_ub=np.array(b_ub) if b_ub else np.zeros(0),
            bounds=bounds, index_to_name=name_map,
            season_rule=SeasonRules.RANK_SUM
        )
        
        # Special Validator for Ranks
        def rank_validator(x_ranks):
            # For Hard Rejection: Check if permutation works
            return poly.check_linear_feasibility(x_ranks)
            
        poly.validator = rank_validator
        poly.seed_guess = self._solve_chebyshev_center(poly)
        return poly

    def _build_bottom2_constraints(self, contestants: List[Contestant], elim_idx: int, name_map: Dict) -> PolytopeConstraints:
        """
        Logic: Rank(Fan) + Offset + Rank(Judge) -> Bottom 2. Judges save one.
        Constraint: Anyone with WORSE (Higher) Judge Metric than E MUST be Safe.
        Safe means Total_W < Total_E (Better/Lower Total Score).
        """
        n = len(contestants)
        A_ub = []
        b_ub = []
        
        if elim_idx != -1:
            E = contestants[elim_idx]
            
            # Identify "Worse than E" candidates (Higher Judge Rank = Worse)
            worse_candidates_indices = [
                i for i, c in enumerate(contestants) 
                if c.judge_metric > E.judge_metric and i != elim_idx
            ]
            
            # Constraint: Worse Candidate (W) must be SAFE.
            # Total_W < Total_E
            # (Rank_W + O_W + J_W) < (Rank_E + O_E + J_E)
            # Rank_W - Rank_E < (J_E - J_W) + (O_E - O_W)
            
            for w_idx in worse_candidates_indices:
                W = contestants[w_idx]
                
                row = np.zeros(n)
                row[w_idx] = 1
                row[elim_idx] = -1
                
                judge_diff = E.judge_metric - W.judge_metric
                offset_diff = E.fan_offset - W.fan_offset
                rhs = judge_diff + offset_diff
                
                A_ub.append(row)
                b_ub.append(rhs - 0.5)

        # Bounds: 1.0 <= x <= N
        bounds = [(1.0, float(n)) for _ in range(n)]
        
        poly = PolytopeConstraints(
            A_ub=np.array(A_ub) if A_ub else np.zeros((0, n)),
            b_ub=np.array(b_ub) if b_ub else np.zeros(0),
            bounds=bounds, index_to_name=name_map,
            season_rule=SeasonRules.RANK_BOTTOM2
        )
        
        poly.seed_guess = self._solve_chebyshev_center(poly)
        return poly

    def _solve_chebyshev_center(self, poly: PolytopeConstraints) -> np.ndarray:
        """
        Finds the Chebyshev center (deepest point) of the polytope.
        Maximize r, s.t. A_i @ x + r ||A_i|| <= b_i
        """
        try:
            # Dimensions
            n = len(poly.bounds)
            
            # Problem: Maximize r -> Minimize -r
            # Variables: [x_1, ... x_n, r] (n+1 variables)
            c = np.zeros(n + 1)
            c[-1] = -1.0 # Minimize -r
            
            A_ub_aug = []
            b_ub_aug = []
            
            # Add augmented inequality constraints
            if poly.A_ub is not None and len(poly.A_ub) > 0:
                for i in range(len(poly.A_ub)):
                    row = poly.A_ub[i]
                    norm = np.linalg.norm(row)
                    # A_i @ x + r * norm <= b_i
                    # -> [row, norm] @ [x, r] <= b_i
                    new_row = np.append(row, norm)
                    A_ub_aug.append(new_row)
                    b_ub_aug.append(poly.b_ub[i])
            
            # Add Equality Constraints (Extended with 0 for r)
            A_eq_aug = None
            b_eq_aug = None
            if poly.A_eq is not None:
                A_eq_aug = np.hstack([poly.A_eq, np.zeros((len(poly.A_eq), 1))])
                b_eq_aug = poly.b_eq
            
            # Bounds for x (original) + Bound for r (0, None)
            bounds_aug = poly.bounds + [(0.0, None)]
            
            # Use 'highs' method for reliability
            res = linprog(
                c, 
                A_ub=np.array(A_ub_aug) if A_ub_aug else None, 
                b_ub=np.array(b_ub_aug) if b_ub_aug else None,
                A_eq=A_eq_aug,
                b_eq=b_eq_aug,
                bounds=bounds_aug,
                method='highs'
            )
            
            if res.success:
                return res.x[:-1] # Return only x part
            else:
                # Fallback: Geometric Mean of bounds
                return np.array([np.mean(b) for b in poly.bounds])
                
        except Exception as e:
            # Fallback for errors
            return np.zeros(len(poly.bounds))

if __name__ == "__main__":
    # Test Case: Rank Mode with Offset
    print("Running Unit Test...")
    c_s = Contestant("Safe", judge_metric=1, fan_offset=0)   # J=1 (Better)
    c_e = Contestant("Elim", judge_metric=2, fan_offset=5)   # J=2 (Worse), but big Offset penalty(5)
    # Total Safe: Rank_S + 0 + 1
    # Total Elim: Rank_E + 5 + 2
    # Constraint: Total_S < Total_E
    # Rank_S + 1 < Rank_E + 7
    # Rank_S - Rank_E < 6
    
    b = ConstraintBuilder(SeasonRules.RANK_SUM)
    p = b.build([c_s, c_e], "Elim")
    print(f"A_ub: {p.A_ub}")
    print(f"b_ub: {p.b_ub}")
