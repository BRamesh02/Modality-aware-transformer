# src/evaluation/robustness.py
import numpy as np
import pandas as pd
from scipy import stats
from .performance import compute_metrics

def bootstrap_analysis(
    strategy_returns: pd.Series, 
    n_samples: int = 1000, 
    block_size: int = 22 # ~1 Month
) -> pd.DataFrame:
    """
    Performs Stationary Block Bootstrap to estimate confidence intervals for Sharpe & Return.
    """
    n_obs = len(strategy_returns)
    data = strategy_returns.values
    
    boot_metrics = []
    
    for _ in range(n_samples):
        indices = np.random.randint(0, n_obs - block_size, int(n_obs / block_size) + 2)
        synthetic_idx = []
        for idx in indices:
            synthetic_idx.extend(range(idx, idx + block_size))

        synthetic_idx = synthetic_idx[:n_obs]
        synthetic_ret = pd.Series(data[synthetic_idx])
        
        m = compute_metrics(synthetic_ret)
        boot_metrics.append(m)
        
    df_boot = pd.DataFrame(boot_metrics)
    
    results = df_boot.describe(percentiles=[0.05, 0.5, 0.95]).T
    return results[["5%", "50%", "95%", "std"]]

def compute_t_stat(strategy_returns: pd.Series) -> float:
    """
    Computes the t-statistic for the mean return being > 0.
    t = Mean / SE = Mean / (Std / sqrt(N))
    """
    t_stat, p_val = stats.ttest_1samp(strategy_returns, 0)
    return t_stat