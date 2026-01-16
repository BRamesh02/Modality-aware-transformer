import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_style() -> None:
    """Configure a clean, consistent matplotlib/seaborn style for all figures."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.2)

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def save_figure(fig: plt.Figure, name: str, save_dir: str | Path | None) -> None:
    """Save a matplotlib figure as a PNG into save_dir if provided, then close it."""
    if save_dir is None:
        return

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def box_props() -> dict:
    """Return a standard bbox style for text annotations on plots."""
    return {
        "boxstyle": "round",
        "facecolor": "white",
        "alpha": 0.9,
        "edgecolor": "lightgrey",
    }


def plot_ic_distribution(
    daily_ic, model_name: str, save_dir: str | Path | None = None, suffix: str = ""
) -> None:
    """Plot the distribution of daily IC values as a histogram with KDE and basic stats."""
    setup_style()

    s = pd.Series(daily_ic).dropna()
    if s.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        s.values,
        kde=True,
        ax=ax,
        color="#1f77b4",
        edgecolor="white",
        alpha=0.6,
        stat="density",
    )

    mu = float(s.mean())
    sigma = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    n = int(s.shape[0])

    ax.axvline(mu, color="#d62728", linestyle="-", linewidth=2, label=f"Mean: {mu:.4f}")
    ax.axvline(0.0, color="black", linewidth=1)

    stats_text = f"{model_name}\n" f"n = {n}\n" f"μ = {mu:.4f}\n" f"σ = {sigma:.4f}"
    ax.text(
        0.03,
        0.95,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=box_props(),
        fontsize=10,
    )

    ax.set_title(f"Daily IC Distribution {suffix}".strip())
    ax.set_xlabel("Daily IC")
    ax.set_ylabel("Probability Density")

    sns.despine()
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()

    save_figure(fig, f"ic_dist_{model_name}{suffix}", save_dir)


def plot_cumulative_ic(
    daily_ic, model_name: str, save_dir: str | Path | None = None, suffix: str = ""
) -> None:
    """Plot the cumulative sum of daily IC over time."""
    setup_style()

    s = pd.Series(daily_ic).dropna()
    if s.empty:
        return

    s = s.sort_index()
    cum = s.cumsum()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cum.index, cum.values, color="#1f77b4", linewidth=2)

    mu = float(s.mean())
    sigma = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    n = int(s.shape[0])

    stats_text = (
        f"{model_name}\n" f"n = {n}\n" f"μ(IC) = {mu:.4f}\n" f"σ(IC) = {sigma:.4f}"
    )
    ax.text(
        0.03,
        0.95,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=box_props(),
        fontsize=10,
    )

    ax.set_title(f"Cumulative Daily IC {suffix}".strip())
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative IC")

    sns.despine()
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout()

    save_figure(fig, f"ic_cumulative_{model_name}{suffix}", save_dir)


def plot_cumulative_ic_compare(
    daily_ic_dict: dict, save_dir: str | Path | None = None, suffix: str = ""
) -> None:
    """Plot cumulative IC curves for multiple models on a single figure."""
    setup_style()

    series = {k: pd.Series(v).dropna() for k, v in daily_ic_dict.items()}
    series = {k: s for k, s in series.items() if not s.empty}
    if not series:
        return

    df = pd.concat(series, axis=1).dropna().sort_index()
    if df.empty:
        return

    cum = df.cumsum()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    palette = sns.color_palette("tab10", n_colors=len(cum.columns))

    for i, col in enumerate(cum.columns):
        ax.plot(cum.index, cum[col].values, linewidth=2, label=col, color=palette[i])

    lines = []
    for col in df.columns:
        mu = float(df[col].mean())
        sigma = float(df[col].std(ddof=1)) if df.shape[0] > 1 else 0.0
        n = int(df[col].shape[0])
        lines.append(f"{col}: n={n}, mean={mu:.4f}, std={sigma:.4f}")

    ax.text(
        0.03,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=box_props(),
        fontsize=10,
    )

    ax.set_title(f"Cumulative Daily IC (Comparison) {suffix}".strip())
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative IC")

    sns.despine()
    ax.legend(loc="best", frameon=False)
    fig.autofmt_xdate(rotation=30, ha="right")
    plt.tight_layout()

    save_figure(fig, f"ic_cumulative_compare{suffix}", save_dir)


def plot_comparison_boxplot(
    series_dict: dict,
    metric_name: str,
    save_dir: str | Path | None = None,
    suffix: str = "",
    ylabel: str | None = None,
) -> None:
    """Compare daily metric series across models using a boxplot and a stats side panel."""
    setup_style()

    series = {k: pd.Series(v).dropna() for k, v in series_dict.items()}
    series = {k: s for k, s in series.items() if not s.empty}
    if not series:
        return

    labels = list(series.keys())
    data = [series[k].values for k in labels]

    fig = plt.figure(figsize=(9.2, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.6], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")

    palette = sns.color_palette("tab10", n_colors=len(labels))

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.6,
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"color": "black", "linewidth": 1.2},
        capprops={"color": "black", "linewidth": 1.2},
        boxprops={"edgecolor": "black", "linewidth": 1.2},
    )

    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(palette[i])
        box.set_alpha(0.6)

    if metric_name.lower() in {"ic", "rankic"}:
        ax.axhline(0.0, color="black", linewidth=1)

    ax.set_title(f"{metric_name} (Daily Distribution) {suffix}".strip())
    ax.set_xlabel("")
    ax.set_ylabel(ylabel or metric_name)

    sns.despine(ax=ax)

    lines = []
    for name in labels:
        s = series[name]
        mu = float(s.mean())
        sigma = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        n = int(s.shape[0])
        lines.append(f"{name}\n  n = {n}\n  mean = {mu:.4f}\n  std = {sigma:.4f}")

    ax_info.text(
        0.02,
        0.98,
        "\n\n".join(lines),
        ha="left",
        va="top",
        fontsize=10,
        bbox=box_props(),
        transform=ax_info.transAxes,
    )

    plt.tight_layout()
    save_figure(fig, f"compare_{metric_name}{suffix}", save_dir)


def plot_horizon_metrics(
    df_decay: pd.DataFrame, metric: str = "MAE", save_dir: str | Path | None = None
) -> None:
    """Plot a metric as a function of the forecast horizon for one or multiple models."""
    setup_style()

    if df_decay is None or df_decay.empty:
        return

    cols = [c for c in df_decay.columns if c.startswith(metric)]
    if not cols:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5))
    palette = sns.color_palette("tab10", n_colors=len(cols))

    for i, col in enumerate(cols):
        label = col.replace(f"{metric}_", "")
        ax.plot(
            df_decay.index,
            df_decay[col].values,
            marker="o",
            linewidth=2,
            label=label,
            color=palette[i],
        )

    ax.set_title(f"{metric} vs Forecast Horizon")
    ax.set_xlabel("Horizon")
    ax.set_ylabel(metric)
    ax.set_xticks(df_decay.index)

    sns.despine()
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()

    save_figure(fig, f"horizon_decay_{metric}", save_dir)


def plot_prediction_vs_target_distribution_zscore_by_date(
    df_pred: pd.DataFrame,
    model_name: str,
    horizons=(1, 2, 3, 4, 5),
    save_dir: str | Path | None = None,
    min_assets_per_day: int = 10,
    clip_z: float | None = 6.0,
) -> None:
    """Compare prediction and target distributions after per-date cross-sectional z-scoring."""
    setup_style()

    if df_pred is None or df_pred.empty:
        return

    has_horizon = "horizon" in df_pred.columns

    def zscore(x: pd.Series) -> pd.Series:
        """Compute z-scores; return NaNs if the group has zero/invalid variance."""
        sd = x.std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            return pd.Series(np.nan, index=x.index)
        return (x - x.mean()) / sd

    for h in horizons:
        df_h = (
            df_pred[df_pred["horizon"] == h].copy() if has_horizon else df_pred.copy()
        )
        df_h = df_h[["date_forecast", "pred", "target"]].dropna()
        if df_h.empty:
            continue

        counts = df_h.groupby("date_forecast").size()
        keep_dates = counts[counts >= min_assets_per_day].index
        df_h = df_h[df_h["date_forecast"].isin(keep_dates)]
        if df_h.empty:
            continue

        df_h["pred_z"] = df_h.groupby("date_forecast")["pred"].transform(zscore)
        df_h["target_z"] = df_h.groupby("date_forecast")["target"].transform(zscore)
        df_h = df_h.dropna(subset=["pred_z", "target_z"])
        if df_h.empty:
            continue

        if clip_z is not None:
            df_h["pred_z"] = df_h["pred_z"].clip(-clip_z, clip_z)
            df_h["target_z"] = df_h["target_z"].clip(-clip_z, clip_z)

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.kdeplot(
            df_h["target_z"].values,
            ax=ax,
            label="Target (z-scored per date)",
            color="black",
            linestyle="--",
            linewidth=2,
        )
        sns.kdeplot(
            df_h["pred_z"].values,
            ax=ax,
            label=f"Prediction (z-scored) – {model_name}",
            color="#1f77b4",
            linewidth=2,
        )

        mu_p, sd_p = df_h["pred_z"].mean(), df_h["pred_z"].std(ddof=1)
        mu_t, sd_t = df_h["target_z"].mean(), df_h["target_z"].std(ddof=1)
        n = int(df_h.shape[0])
        n_days = int(df_h["date_forecast"].nunique())

        stats_text = (
            f"H={h} (z-score per date)\n"
            f"days={n_days}, n={n}\n\n"
            f"Pred: μ={mu_p:.3f}, σ={sd_p:.3f}\n"
            f"Tgt:  μ={mu_t:.3f}, σ={sd_t:.3f}"
        )

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=box_props(),
            fontsize=10,
        )

        ax.axvline(0.0, color="grey", linewidth=1)
        ax.set_title(f"Pred vs Target Distribution – H={h} ({model_name})")
        ax.set_xlabel("Z-score (within date)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False)

        sns.despine()
        plt.tight_layout()

        save_figure(fig, f"pred_vs_target_by_horizons_{model_name}_H{h}", save_dir)
