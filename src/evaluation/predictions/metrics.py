"""
metrics.py

Evaluation utilities for cross-sectional forecasting models.

Core principle:
- Metrics are computed per date (cross-section), then averaged over time
  so that each day has equal weight.
- Classic regression metrics are also provided in pooled (panel) form.
- Model comparison uses a Diebold–Mariano test applied to DAILY loss series.
"""


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Helpers

def _safe_series(x) -> pd.Series:
    """Ensure a 1D pandas Series."""
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)


def _sort_dropna(s: pd.Series) -> pd.Series:
    """
    Sort by index (typically dates) and drop NaNs.
    Used for daily metric series before aggregation or testing.
    """
    if s is None:
        return pd.Series(dtype=float)
    s = _safe_series(s)
    try:
        s = s.sort_index()
    except Exception:
        pass
    return s.dropna()


# 1) IC (Pearson) - daily cross-sectional correlation

def calculate_ic_metrics(df_pred: pd.DataFrame):
    """
    Compute daily Pearson IC (pred vs target) and aggregate over time.

    Returns
    -------
    metrics : dict
        IC_Mean, IC_Std, IC_IR, IC_t_stat, IC_p_value, IC_n_days
    daily_ic : pd.Series
        Daily IC series (indexed by date).
    """
    if df_pred.empty:
        return {}, pd.Series(dtype=float)

    daily_ic = df_pred.groupby("date_forecast").apply(
        lambda g: g["pred"].corr(g["target"])
    )
    daily_ic = _sort_dropna(daily_ic)

    if len(daily_ic) < 2:
        return (
            {
                "IC_Mean": 0.0,
                "IC_Std": 0.0,
                "IC_IR": 0.0,
                "IC_t_stat": 0.0,
                "IC_p_value": 1.0,
                "IC_n_days": int(daily_ic.shape[0]),
            },
            daily_ic,
        )

    t_stat, p_val = stats.ttest_1samp(daily_ic.values, 0.0, nan_policy="omit")
    m = float(daily_ic.mean())
    s = float(daily_ic.std(ddof=1))

    return (
        {
            "IC_Mean": m,
            "IC_Std": s,
            "IC_IR": float(m / (s + 1e-9)),
            "IC_t_stat": float(t_stat),
            "IC_p_value": float(p_val),
            "IC_n_days": int(daily_ic.shape[0]),
        },
        daily_ic,
    )


# 2) Directional metrics - computed per day then averaged

def calculate_directional_metrics(df_pred: pd.DataFrame, top_k_percent: float = 0.2):
    """
    Compute daily directional metrics and average over time.

    Metrics:
      - Hit Rate: fraction of correct sign predictions per day
      - Precision@Top-K%: fraction of positive targets among top predictions per day

    Returns
    -------
    metrics : dict
    """
    if df_pred.empty:
        return {}

    daily_hit = df_pred.groupby("date_forecast").apply(
        lambda g: np.mean(np.sign(g["pred"].values) == np.sign(g["target"].values))
    )
    daily_hit = _sort_dropna(daily_hit)

    def get_top_k_prec(g):
        k = int(len(g) * top_k_percent)
        if k <= 0:
            return np.nan
        top = g.nlargest(k, "pred")
        return (top["target"] > 0).sum() / k

    precision_series = df_pred.groupby("date_forecast").apply(get_top_k_prec)
    precision_series = _sort_dropna(precision_series)

    return {
        "Hit_Rate": float(daily_hit.mean()) if len(daily_hit) else np.nan,
        f"Precision_Top_{int(top_k_percent * 100)}%":
            float(precision_series.mean()) if len(precision_series) else np.nan,
        "HitRate_n_days": int(daily_hit.shape[0]),
        "Precision_n_days": int(precision_series.shape[0]),
    }



# 3) Regression metrics - pooled panel metrics

def calculate_regression_metrics(df_pred: pd.DataFrame):
    """
    Compute pooled regression-style error metrics on the full panel.

    Metrics:
      - MSE, MAE, RMSE, R2
    """
    if df_pred.empty:
        return {}

    tmp = df_pred[["target", "pred"]].dropna()
    if tmp.empty:
        return {}

    y_true, y_pred = tmp["target"], tmp["pred"]
    mse = mean_squared_error(y_true, y_pred)

    return {
        "MSE": float(mse),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }



# 4) Rank IC (Spearman) - daily cross-sectional rank correlation

def calculate_rank_ic_metrics(df_pred: pd.DataFrame):
    """
    Compute daily Rank IC (Spearman) and aggregate over time.

    Returns
    -------
    metrics : dict
        RankIC_Mean, RankIC_Std, RankIC_IR,
        RankIC_t_stat, RankIC_p_value, RankIC_n_days
    daily_rank_ic : pd.Series
    """
    if df_pred.empty:
        return {}, pd.Series(dtype=float)

    daily_rank_ic = df_pred.groupby("date_forecast").apply(
        lambda g: g["pred"].corr(g["target"], method="spearman")
    )
    daily_rank_ic = _sort_dropna(daily_rank_ic)

    if len(daily_rank_ic) < 2:
        return (
            {
                "RankIC_Mean": 0.0,
                "RankIC_Std": 0.0,
                "RankIC_IR": 0.0,
                "RankIC_t_stat": 0.0,
                "RankIC_p_value": 1.0,
                "RankIC_n_days": int(daily_rank_ic.shape[0]),
            },
            daily_rank_ic,
        )

    t_stat, p_val = stats.ttest_1samp(daily_rank_ic.values, 0.0, nan_policy="omit")
    m = float(daily_rank_ic.mean())
    s = float(daily_rank_ic.std(ddof=1))

    return (
        {
            "RankIC_Mean": m,
            "RankIC_Std": s,
            "RankIC_IR": float(m / (s + 1e-9)),
            "RankIC_t_stat": float(t_stat),
            "RankIC_p_value": float(p_val),
            "RankIC_n_days": int(daily_rank_ic.shape[0]),
        },
        daily_rank_ic,
    )



# 5) Daily loss metrics (MAE / MSE per date)

def calculate_daily_loss_metrics(df_pred: pd.DataFrame):
    """
    Compute MAE and MSE per date (cross-sectional mean), then summarize over time.

    Returns
    -------
    summary : dict
    daily_mae : pd.Series
    daily_mse : pd.Series
    """
    if df_pred.empty:
        return {}, pd.Series(dtype=float), pd.Series(dtype=float)

    def _daily_mae(g):
        return float(np.mean(np.abs(g["target"].values - g["pred"].values)))

    def _daily_mse(g):
        e = g["target"].values - g["pred"].values
        return float(np.mean(e * e))

    daily_mae = _sort_dropna(df_pred.groupby("date_forecast").apply(_daily_mae))
    daily_mse = _sort_dropna(df_pred.groupby("date_forecast").apply(_daily_mse))

    out = {
        "Daily_MAE_Mean": float(daily_mae.mean()) if len(daily_mae) else np.nan,
        "Daily_MAE_Std": float(daily_mae.std(ddof=1)) if len(daily_mae) else np.nan,
        "Daily_MSE_Mean": float(daily_mse.mean()) if len(daily_mse) else np.nan,
        "Daily_MSE_Std": float(daily_mse.std(ddof=1)) if len(daily_mse) else np.nan,
        "DailyLoss_n_days": int(min(daily_mae.shape[0], daily_mse.shape[0])),
    }
    return out, daily_mae, daily_mse



# 6) Diebold–Mariano test

def diebold_mariano_test(
    y_true,
    y_pred_model,
    y_pred_benchmark=None,
    h: int = 1,
    criterion: str = "MAE",
):
    """
    Diebold–Mariano test with HLN adjustment.

    Intended for time-series inputs (one observation per period).

    Returns
    -------
    dm_stat : float
    p_value : float
    """
    criterion = (criterion or "MAE").upper()

    y_true = np.array(y_true).flatten()
    y_pred_model = np.array(y_pred_model).flatten()
    y_pred_benchmark = (
        np.zeros_like(y_true)
        if y_pred_benchmark is None
        else np.array(y_pred_benchmark).flatten()
    )

    T = len(y_true)
    if T < 2:
        return 0.0, 1.0

    e1 = y_true - y_pred_model
    e2 = y_true - y_pred_benchmark

    if criterion == "MSE":
        d = e1 ** 2 - e2 ** 2
    else:
        d = np.abs(e1) - np.abs(e2)

    d_mean = np.mean(d)
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0

    if h > 1:
        for k in range(1, h):
            cov_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
            gamma_sum += cov_k

    lr_var = gamma_0 + 2 * gamma_sum
    if lr_var <= 0:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(lr_var / T)

    correction = ((T + 1 - 2 * h + (h * (h - 1)) / T) / T) ** 0.5
    adjusted_dm = dm_stat * correction
    p_value = 2 * (1 - stats.t.cdf(np.abs(adjusted_dm), df=T - 1))

    return float(adjusted_dm), float(p_value)


def diebold_mariano_on_daily_loss(
    df_pred_model: pd.DataFrame,
    df_pred_benchmark: pd.DataFrame,
    criterion: str = "MAE",
    h: int = 1,
    merge_keys=("date_forecast", "permno", "horizon"),
):
    """
    Apply Diebold–Mariano test on a daily loss series.

    Steps:
      - align model and benchmark predictions
      - compute one loss value per date
      - run DM test on the resulting time series

    Returns
    -------
    dm_stat : float
    p_value : float
    """
    if df_pred_model.empty or df_pred_benchmark.empty:
        return 0.0, 1.0

    # Keep only merge keys that exist in both DataFrames
    keys = [
        k for k in merge_keys
        if k in df_pred_model.columns and k in df_pred_benchmark.columns
    ]

    m = df_pred_model[keys + ["target", "pred"]].copy()
    b = df_pred_benchmark[keys + ["pred"]].copy()

    merged = m.merge(
        b,
        on=keys,
        how="inner",
        suffixes=("_model", "_bench"),
    )

    if merged.empty:
        return 0.0, 1.0

    def _daily_loss(g, pred_col):
        e = g["target"].values - g[pred_col].values
        if criterion.upper() == "MSE":
            return float(np.mean(e * e))
        return float(np.mean(np.abs(e)))

    daily_model = _sort_dropna(
        merged.groupby("date_forecast").apply(lambda g: _daily_loss(g, "pred_model"))
    )
    daily_bench = _sort_dropna(
        merged.groupby("date_forecast").apply(lambda g: _daily_loss(g, "pred_bench"))
    )

    tmp = pd.concat(
        [daily_model.rename("model"), daily_bench.rename("bench")],
        axis=1,
    ).dropna()

    if tmp.shape[0] < 2:
        return 0.0, 1.0

    # Trick: set y_true=0 and y_pred=-loss so that (y_true - y_pred) = loss
    y_true = np.zeros(tmp.shape[0], dtype=float)
    y_pred_model = -tmp["model"].values
    y_pred_bench = -tmp["bench"].values

    return diebold_mariano_test(
        y_true=y_true,
        y_pred_model=y_pred_model,
        y_pred_benchmark=y_pred_bench,
        h=h,
        criterion=criterion,
    )