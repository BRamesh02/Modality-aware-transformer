import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from .plots import setup_style

def compute_robustness_metrics(strategy_returns: pd.Series) -> pd.Series:
    """Computes extended statistical metrics."""
    ann_ret = strategy_returns.mean() * 252
    ann_vol = strategy_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    
    t_stat, p_val = stats.ttest_1samp(strategy_returns, 0)
    skew = stats.skew(strategy_returns)
    kurt = stats.kurtosis(strategy_returns)
    win_rate = (strategy_returns > 0).mean()
    
    cum = (1 + strategy_returns).cumprod()
    drawdown = (cum - cum.cummax()) / cum.cummax()
    max_dd = drawdown.min()

    return pd.Series({
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe Ratio": sharpe,
        "t-Stat": t_stat,
        "p-Value": p_val,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Win Rate": win_rate,
        "Max Drawdown": max_dd
    })

def bootstrap_analysis(strategy_returns: pd.Series, n_samples: int = 1000, block_size: int = 22):
    """Performs Stationary Block Bootstrap."""
    original_metrics = compute_robustness_metrics(strategy_returns)
    n_obs = len(strategy_returns)
    data = strategy_returns.values
    
    boot_stats = []
    for _ in range(n_samples):
        num_blocks = int(np.ceil(n_obs / block_size))
        start_indices = np.random.randint(0, n_obs - block_size, num_blocks)
        synthetic_idx = np.concatenate([np.arange(s, s + block_size) for s in start_indices])[:n_obs]
        sample = data[synthetic_idx]
        
        if np.std(sample) > 0:
            s_mean = np.mean(sample) * 252
            s_vol = np.std(sample) * np.sqrt(252)
            boot_stats.append(s_mean / s_vol)
            
    sharpes = np.array(boot_stats)
    df_boot = pd.DataFrame(sharpes, columns=["Sharpe"])
    summary = df_boot.describe(percentiles=[0.05, 0.5, 0.95]).T
    
    return sharpes, summary, original_metrics

def plot_bootstrap_scientific(sharpes: np.ndarray, metrics: pd.Series, title: str, save_path):
    """Histogram of Sharpes with a statistics info box."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.histplot(sharpes, kde=True, ax=ax, color="#1f77b4", edgecolor="white", alpha=0.6, stat="density")
    
    orig_sr = metrics["Sharpe Ratio"]
    lower_5 = np.percentile(sharpes, 5)
    
    ax.axvline(orig_sr, color="#d62728", linestyle="-", linewidth=2, label=f"Actual SR: {orig_sr:.2f}")
    ax.axvline(lower_5, color="grey", linestyle="--", linewidth=1.5, label=f"5% CI: {lower_5:.2f}")
    ax.axvline(0, color="black", linewidth=1)
    
    stats_text = (
        f"Mean SR: {np.mean(sharpes):.2f}\n"
        f"Win Rate: {metrics['Win Rate']:.1%}\n"
        f"Skew: {metrics['Skewness']:.2f}"
    )
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgrey')
    ax.text(0.03, 0.95, stats_text, transform=ax.transAxes, va='top', bbox=props, fontsize=10)
    
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Probability Density")
    
    sns.despine()
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()