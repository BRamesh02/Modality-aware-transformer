import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from .plots import setup_style


def compute_quantile_returns(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    universe_mask: pd.DataFrame = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Computes daily returns for N buckets (Q1=Top, Qn=Bottom)."""
    common = signal.index.intersection(returns.index)
    if universe_mask is not None:
        common = common.intersection(universe_mask.index)
        universe_mask = universe_mask.loc[common]

    signal = signal.loc[common]
    returns = returns.loc[common]
    if universe_mask is not None:
        signal = signal.where(universe_mask > 0)

    ranks = signal.rank(axis=1, pct=True)
    buckets = (
        (ranks * n_bins).apply(np.floor).fillna(-1).astype(int).clip(upper=n_bins - 1)
    )

    stats_dict = {}
    for b in range(n_bins):
        mask = buckets == b
        counts = mask.sum(axis=1).replace(0, np.nan)
        weights = mask.div(counts, axis=0)
        b_ret = (weights.shift(1).fillna(0.0) * returns).sum(axis=1)
        label = f"Q{n_bins - b}"
        stats_dict[label] = b_ret

    df_quantiles = pd.DataFrame(stats_dict)
    cols = [f"Q{i}" for i in range(1, n_bins + 1)]
    return df_quantiles[cols]


def plot_quintiles_scientific(df_quantiles: pd.DataFrame, title: str, save_path):
    """Plots Quintiles with diverging colors and clean log scale."""
    setup_style()
    cum_ret = (1 + df_quantiles).cumprod()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [
        "#1a9641", # Q1: Deep Green
        "#a6d96a", # Q2: Light Green
        "#757575", # Q3: Neutral Grey
        "#fdae61", # Q4: Orange
        "#d7191c"  # Q5: Deep Red
    ]
    
    for i, col in enumerate(cum_ret.columns):
        is_edge = (i == 0) or (i == len(cum_ret.columns) - 1)
        lw = 2.5 if is_edge else 1.0
        alpha = 1.0 if is_edge else 0.6

        label_suffix = (
            " (Top)"
            if i == 0
            else (" (Bottom)" if i == len(cum_ret.columns) - 1 else "")
        )
        ax.plot(
            cum_ret[col],
            color=colors[i],
            linewidth=lw,
            alpha=alpha,
            label=f"{col}{label_suffix}",
        )

    ax.set_title(title, fontsize=14, weight="bold", pad=15)
    ax.set_ylabel("Wealth Index ($1 Initial)", fontsize=12)
    ax.set_yscale("log")

    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())

    sns.despine()
    ax.legend(title="Quintile", loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
