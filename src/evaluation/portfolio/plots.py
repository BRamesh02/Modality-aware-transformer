import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd

def setup_style():
    """Sets a professional style for plots."""
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['font.family'] = 'sans-serif'

def plot_cumulative_log(cum_series_dict: dict, title: str, save_path):
    """
    Plots cumulative equity curves on a Log Scale.
    Args:
        cum_series_dict: Dict { "Label": pd.Series } of cumulative returns.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for label, series in cum_series_dict.items():
        ax.plot(series, label=label, linewidth=2)

    ax.axhline(1.0, color='black', linewidth=1, linestyle='-', alpha=0.5)

    ax.set_title(title, fontsize=14, weight='bold', pad=15)
    ax.set_ylabel("Wealth Index ($1 Initial)", fontsize=12)
    
    ax.set_yscale("log")
    
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    
    sns.despine(ax=ax)
    ax.legend(frameon=True, loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()