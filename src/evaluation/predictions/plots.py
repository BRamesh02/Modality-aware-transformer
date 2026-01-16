import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.4)

COLORS = ["#4c72b0", "#55a868", "#c44e52"]


# Utilities
def _save(fig, name, save_dir):
    """Save figure to disk if a directory is provided."""
    if save_dir:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        fig.savefig(path / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# IC plots

def plot_ic_distribution(daily_ic, model_name, save_dir=None, suffix=""):
    """
    Histogram of DAILY IC values.

    Interpretation:
      Shows how stable the model’s cross-sectional predictive power is
      across days. A distribution shifted to the right is better.
    """
    daily_ic = pd.Series(daily_ic).dropna()
    if daily_ic.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(daily_ic, kde=True, bins=40, ax=ax, color=COLORS[0], alpha=0.6)

    mean_ic = float(daily_ic.mean())
    ax.axvline(mean_ic, color=COLORS[2], linestyle="--", lw=2, label=f"Mean: {mean_ic:.4f}")

    ax.set_title(f"Daily IC Distribution – {model_name} {suffix}")
    ax.legend()

    _save(fig, f"ic_dist_{model_name}{suffix}", save_dir)


def plot_cumulative_ic(daily_ic, model_name, save_dir=None, suffix=""):
    """
    Cumulative sum of DAILY IC over time.

    Interpretation:
      Highlights persistence: a steadily increasing curve indicates
      consistent predictive skill over time.
    """
    daily_ic = pd.Series(daily_ic).dropna()
    if daily_ic.empty:
        return

    daily_ic = daily_ic.sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    daily_ic.cumsum().plot(ax=ax, lw=2, color=COLORS[1])

    ax.set_title(f"Cumulative IC – {model_name} {suffix}")
    ax.set_ylabel("Cumulative IC")

    _save(fig, f"ic_cumulative_{model_name}{suffix}", save_dir)



# Quintile plot

def plot_quintile_returns(df_pred, model_name, save_dir=None, suffix=""):
    """
    Average return by prediction quintile.

    Construction:
      - Quintiles are assigned per date based on predictions.
      - Returns are averaged per date, then across dates (equal weight).
      - Values are shown in basis points.

    Interpretation:
      A strong monotonic increase from Q1 to Q5 indicates good ranking power.
      The Q5–Q1 spread summarizes long–short performance.
    """
    if df_pred is None or df_pred.empty:
        return

    df = df_pred.copy()

    def assign_quintiles(g):
        if g["pred"].nunique() < 5:
            return pd.Series(np.nan, index=g.index)
        return pd.qcut(g["pred"], 5, labels=False, duplicates="drop")

    df["bucket"] = df.groupby("date_forecast", group_keys=False).apply(assign_quintiles)
    df = df.dropna(subset=["bucket"])
    if df.empty:
        return

    df["bucket"] = df["bucket"].astype(int)

    # Daily mean return per quintile
    daily_bucket_ret = (
        df.groupby(["date_forecast", "bucket"])["target"]
          .mean()
          .unstack("bucket")
          .sort_index()
    )

    # Average across dates and convert to bps
    avg_ret_bps = daily_bucket_ret.mean(axis=0) * 10000.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        avg_ret_bps.index.astype(str),
        avg_ret_bps.values,
        color=sns.color_palette("RdYlGn", 5),
        width=0.8,
    )

    ax.set_title(f"Average Return by Quintile (bps) – {model_name} {suffix}")
    ax.set_ylabel("Basis Points")
    ax.axhline(0, color="black", lw=0.8)

    # Annotate long–short spread
    if 0 in daily_bucket_ret.columns and 4 in daily_bucket_ret.columns:
        spread = (daily_bucket_ret[4] - daily_bucket_ret[0]).dropna()
        if len(spread):
            spread_bps = spread.mean() * 10000.0
            ax.text(
                0.98, 0.98,
                f"Spread Q5–Q1: {spread_bps:.2f} bps",
                transform=ax.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

    _save(fig, f"quintiles_{model_name}{suffix}", save_dir)



# Model comparison plots

def plot_comparison_bar(metric_dict, metric_name, save_dir=None, suffix=""):
    """
    Bar chart comparing a single metric across models.

    Interpretation:
      Direct visual comparison of model performance on one summary metric
      (e.g. MAE or IC).
    """
    names = metric_dict["names"]
    values = metric_dict[metric_name]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=names,
        y=values,
        hue=names,
        palette=[COLORS[0], COLORS[1]],
        legend=False,
        ax=ax,
    )

    ax.set_title(f"Model Comparison – {metric_name} {suffix}")

    for i, v in enumerate(values):
        label = "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.5f}"
        ax.text(i, 0 if np.isnan(v) else v, label, ha="center", va="bottom", fontweight="bold")

    _save(fig, f"compare_{metric_name}{suffix}", save_dir)



# Horizon decay plot

def plot_horizon_metrics(df_decay, metric="MAE", save_dir=None):
    """
    Plot metric decay as a function of forecast horizon.

    Interpretation:
      Shows how performance deteriorates as predictions are made further
      into the future.
    """
    if df_decay is None or df_decay.empty:
        return

    cols = [c for c in df_decay.columns if c.startswith(metric)]
    if not cols:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for c in cols:
        label = c.replace(f"{metric}_", "")
        ax.plot(df_decay.index, df_decay[c], marker="o", lw=2, label=label)

    ax.set_title(f"{metric} Decay by Forecast Horizon")
    ax.set_xlabel("Horizon")
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xticks(df_decay.index)

    _save(fig, f"horizon_decay_{metric}", save_dir)