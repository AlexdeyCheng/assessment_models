import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import Dict, List, Tuple

class StatisticsCalculator:
    """
    Computes statistical metrics from MCMC samples.
    """
    
    @staticmethod
    def compute_stats(samples: np.ndarray, 
                      contestant_names: List[str], 
                      season_mode: str) -> Dict[str, Dict]:
        """
        Calculates Mean, Std, RQI (90% CI), Entropy for each contestant.
        
        Args:
            samples: (N_samples, N_contestants) array.
            contestant_names: List of names corresponding to columns.
            season_mode: 'rank_sum', 'percentage_sum', etc.
            
        Returns:
            Dict mapping Name -> {stat_name: value}
        """
        stats = {}
        n_samples = samples.shape[0]
        
        # Calculate column-wise statistics
        means = np.mean(samples, axis=0)
        stds = np.std(samples, axis=0)
        
        # 90% Confidence Interval (5th and 95th percentiles)
        rqi_d = np.percentile(samples, 5, axis=0)
        rqi_u = np.percentile(samples, 95, axis=0)
        rqi_width = rqi_u - rqi_d
        
        # Entropy Calculation
        entropies = []
        for i in range(samples.shape[1]):
            col_data = samples[:, i]
            
            if "rank" in season_mode:
                # Discrete Entropy for Ranks
                # Count occurrences of each rank (1..N)
                # Note: samples might be floats from mean? No, samples are integers/floats from sampler.
                # Rejection sampler returns ints.
                value_counts = pd.Series(col_data).value_counts(normalize=True)
                ent = entropy(value_counts.values)
            else:
                # Continuous Entropy for Percentages (Histogram Binning)
                # Bin into 50 bins between 0 and 1
                hist, _ = np.histogram(col_data, bins=50, range=(0, 1), density=True)
                # Normalize hist to sum to 1 for Shannon entropy calculation (discrete approx)
                # entropy(pk) requires sum(pk)=1. hist * bin_width approx prob.
                # Let's just use raw counts for robust Shannon entropy of the distribution shape
                hist_counts, _ = np.histogram(col_data, bins=50, range=(0, 1))
                ent = entropy(hist_counts)
                
            entropies.append(ent)
            
        # Pack into dictionary
        for i, name in enumerate(contestant_names):
            stats[name] = {
                'mean': means[i],
                'std': stds[i],
                'RQI_d': rqi_d[i],
                'RQI_u': rqi_u[i],
                'RQI_width': rqi_width[i],
                'entropy': entropies[i],
                'sample_count': n_samples
            }
            
        return stats