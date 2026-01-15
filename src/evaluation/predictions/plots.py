import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
colors = ["#4c72b0", "#55a868", "#c44e52"]

def _save(fig, name, save_dir):
    if save_dir:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / f"{name}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_ic_distribution(daily_ic, model_name, save_dir=None, suffix=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(daily_ic, kde=True, bins=40, ax=ax, color=colors[0], alpha=0.6)
    mean_ic = daily_ic.mean()
    ax.axvline(mean_ic, color=colors[2], linestyle='--', lw=2, label=f"Mean: {mean_ic:.4f}")
    ax.set_title(f"Daily IC Distribution - {model_name} {suffix}")
    ax.legend()
    _save(fig, f"ic_dist_{model_name}{suffix}", save_dir)

def plot_cumulative_ic(daily_ic, model_name, save_dir=None, suffix=""):
    fig, ax = plt.subplots(figsize=(12, 6))
    daily_ic.cumsum().plot(ax=ax, lw=2, color=colors[1])
    ax.set_title(f"Cumulative IC - {model_name} {suffix}")
    ax.set_ylabel("Cumulative Sum")
    _save(fig, f"ic_cumulative_{model_name}{suffix}", save_dir)

def plot_quintile_returns(df_pred, model_name, save_dir=None, suffix=""):
    def get_quintiles(g):
        try: return pd.qcut(g['pred'], 5, labels=False, duplicates='drop')
        except: return np.nan

    df = df_pred.copy()
    df['bucket'] = df.groupby('date_forecast', group_keys=False).apply(get_quintiles)
    ret_per_bucket = df.groupby('bucket')['target'].mean() * 10000 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ret_per_bucket.plot(kind='bar', ax=ax, color=sns.color_palette("RdYlGn", 5), width=0.8)
    ax.set_title(f"Avg Return by Quintile (bps) - {model_name} {suffix}")
    ax.set_ylabel("Basis Points")
    ax.axhline(0, color='black', lw=0.8)
    _save(fig, f"quintiles_{model_name}{suffix}", save_dir)

def plot_comparison_bar(metric_dict, metric_name, save_dir=None, suffix=""):
    names = metric_dict['names']
    values = metric_dict[metric_name]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=names, y=values, palette=[colors[0], colors[1]], ax=ax)
    ax.set_title(f"Model Comparison: {metric_name} {suffix}")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.5f}", color='black', ha="center", va="bottom", fontweight='bold')
    _save(fig, f"compare_{metric_name}{suffix}", save_dir)

def plot_horizon_metrics(df_decay, metric="MAE", save_dir=None):
    """
    Plots the decay of a metric over horizons for multiple models.
    df_decay: DataFrame from compare_horizons_decay
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    cols = [c for c in df_decay.columns if c.startswith(metric)]
    
    for c in cols:
        label = c.replace(f"{metric}_", "")
        ax.plot(df_decay.index, df_decay[c], marker='o', lw=2, label=label)
    
    ax.set_title(f"{metric} Decay by Forecast Horizon")
    ax.set_xlabel("Horizon (Days Ahead)")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(df_decay.index)
    
    _save(fig, f"horizon_decay_{metric}", save_dir)