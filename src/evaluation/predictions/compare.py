import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.predictions.metrics import diebold_mariano_on_daily_loss


def compare_models(
    df_base: pd.DataFrame,
    df_challenger: pd.DataFrame,
    name_a: str = "Baseline",
    name_b: str = "Challenger",
    h: int = 1,
):
    """
    Compare two models on the same data using daily-first metrics.

    What is computed:
      - Daily MAE per model (cross-sectional MAE per date, then averaged).
      - Daily IC per model (cross-sectional Pearson correlation per date).
      - Paired t-test on the daily IC series.
      - Diebold–Mariano test on the DAILY MAE loss series
        (model = challenger, benchmark = baseline).

    Returns
    -------
    dict or None
        None if models have no overlapping observations.
    """
    merge_cols = ["date_forecast", "permno"]
    if "horizon" in df_base.columns:
        merge_cols.append("horizon")

    df = pd.merge(
        df_base[merge_cols + ["target", "pred"]],
        df_challenger[merge_cols + ["pred"]],
        on=merge_cols,
        how="inner",
        suffixes=(f"_{name_a}", f"_{name_b}"),
    )

    if df.empty:
        return None


    # 1) DAILY MAE (one value per date, then averaged)

    daily_mae_a = (
        df.groupby("date_forecast")
        .apply(lambda g: np.mean(np.abs(g[f"pred_{name_a}"] - g["target"])))
        .sort_index()
    )

    daily_mae_b = (
        df.groupby("date_forecast")
        .apply(lambda g: np.mean(np.abs(g[f"pred_{name_b}"] - g["target"])))
        .sort_index()
    )

    mae_tmp = pd.concat(
        [daily_mae_a.rename("a"), daily_mae_b.rename("b")], axis=1
    ).dropna()

    mae_a = float(mae_tmp["a"].mean()) if len(mae_tmp) else np.nan
    mae_b = float(mae_tmp["b"].mean()) if len(mae_tmp) else np.nan


    # 2) DAILY IC + paired t-test

    daily_ic_a = (
        df.groupby("date_forecast")
        .apply(lambda g: g[f"pred_{name_a}"].corr(g["target"]))
        .sort_index()
    )

    daily_ic_b = (
        df.groupby("date_forecast")
        .apply(lambda g: g[f"pred_{name_b}"].corr(g["target"]))
        .sort_index()
    )

    ic_tmp = pd.concat(
        [daily_ic_a.rename("a"), daily_ic_b.rename("b")], axis=1
    ).dropna()

    if ic_tmp.shape[0] >= 2:
        t_stat_ic, p_val_ic = stats.ttest_rel(
            ic_tmp["a"].values,
            ic_tmp["b"].values,
            nan_policy="omit",
        )
        ic_a = float(ic_tmp["a"].mean())
        ic_b = float(ic_tmp["b"].mean())
    else:
        t_stat_ic, p_val_ic = 0.0, 1.0
        ic_a, ic_b = np.nan, np.nan


    # 3) Diebold–Mariano test on DAILY loss (robust)
    #     model = challenger, benchmark = baseline

    df_a = df_base[merge_cols + ["target", "pred"]].copy()

    if "target" in df_challenger.columns:
        df_b = df_challenger[merge_cols + ["target", "pred"]].copy()
    else:
        df_b = df[merge_cols + ["target", f"pred_{name_b}"]].copy()
        df_b = df_b.rename(columns={f"pred_{name_b}": "pred"})

    dm_stat, p_val_dm = diebold_mariano_on_daily_loss(
        df_pred_model=df_b,
        df_pred_benchmark=df_a,
        criterion="MAE",
        h=h,
        merge_keys=tuple(merge_cols),
    )

    return {
        "names": [name_a, name_b],
        "MAE": [mae_a, mae_b],          # daily-averaged MAE
        "IC": [ic_a, ic_b],             # daily-averaged IC
        "dm_stat": float(dm_stat),
        "p_val_dm": float(p_val_dm),
        "p_val_ic": float(p_val_ic),
        "n_days_mae": int(mae_tmp.shape[0]),
        "n_days_ic": int(ic_tmp.shape[0]),
    }


def compare_horizons_decay(
    df_base: pd.DataFrame,
    df_challenger: pd.DataFrame,
    names=("Baseline", "Challenger"),
):
    """
    Compute MAE and IC for each horizon using daily-first metrics.

    For each horizon:
      - Daily MAE per model -> averaged over dates
      - Daily IC per model  -> averaged over dates

    Returns
    -------
    pd.DataFrame
        Indexed by horizon, with MAE, IC, and number of usable days.
    """
    name_a, name_b = names

    df = pd.merge(
        df_base[["date_forecast", "permno", "horizon", "target", "pred"]],
        df_challenger[["date_forecast", "permno", "horizon", "pred"]],
        on=["date_forecast", "permno", "horizon"],
        how="inner",
        suffixes=(f"_{name_a}", f"_{name_b}"),
    )

    if df.empty:
        return pd.DataFrame()

    def _per_horizon(g):
        daily_mae_a = (
            g.groupby("date_forecast")
            .apply(lambda d: np.mean(np.abs(d[f"pred_{name_a}"] - d["target"])))
        )
        daily_mae_b = (
            g.groupby("date_forecast")
            .apply(lambda d: np.mean(np.abs(d[f"pred_{name_b}"] - d["target"])))
        )
        mae_tmp = pd.concat(
            [daily_mae_a.rename("a"), daily_mae_b.rename("b")], axis=1
        ).dropna()

        daily_ic_a = (
            g.groupby("date_forecast")
            .apply(lambda d: d[f"pred_{name_a}"].corr(d["target"]))
        )
        daily_ic_b = (
            g.groupby("date_forecast")
            .apply(lambda d: d[f"pred_{name_b}"].corr(d["target"]))
        )
        ic_tmp = pd.concat(
            [daily_ic_a.rename("a"), daily_ic_b.rename("b")], axis=1
        ).dropna()

        return pd.Series(
            {
                f"MAE_{name_a}": float(mae_tmp["a"].mean()) if len(mae_tmp) else np.nan,
                f"MAE_{name_b}": float(mae_tmp["b"].mean()) if len(mae_tmp) else np.nan,
                f"IC_{name_a}": float(ic_tmp["a"].mean()) if len(ic_tmp) else np.nan,
                f"IC_{name_b}": float(ic_tmp["b"].mean()) if len(ic_tmp) else np.nan,
                "n_days_mae": int(mae_tmp.shape[0]),
                "n_days_ic": int(ic_tmp.shape[0]),
            }
        )

    return df.groupby("horizon").apply(_per_horizon)