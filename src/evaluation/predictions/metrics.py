import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_ic_metrics(df_pred: pd.DataFrame):
    """Calculates Daily IC statistics (Mean, Std, IR, p-value)."""
    if df_pred.empty: return {}
    
    daily_ic = df_pred.groupby('date_forecast').apply(
        lambda x: x['pred'].corr(x['target'])
    )
    
    daily_ic = daily_ic.fillna(0)
    
    if len(daily_ic) < 2:
        return {"IC_Mean": 0, "IC_Std": 0, "IC_IR": 0, "p_value": 1.0}, daily_ic

    t_stat, p_val = stats.ttest_1samp(daily_ic, 0)
    
    return {
        "IC_Mean": daily_ic.mean(),
        "IC_Std": daily_ic.std(),
        "IC_IR": daily_ic.mean() / (daily_ic.std() + 1e-9),
        "t_stat": t_stat,
        "p_value": p_val
    }, daily_ic

def calculate_directional_metrics(df_pred: pd.DataFrame, top_k_percent=0.2):
    """Calculates Hit Rate and Precision for the Top K% predictions."""
    if df_pred.empty: return {}
    
    hits = np.sign(df_pred['pred']) == np.sign(df_pred['target'])
    
    def get_top_k_prec(g):
        k = int(len(g) * top_k_percent)
        if k == 0: return np.nan
        top = g.nlargest(k, 'pred')
        return (top['target'] > 0).sum() / k

    precision_series = df_pred.groupby('date_forecast').apply(get_top_k_prec)
    
    return {
        "Hit_Rate": hits.mean(),
        f"Precision_Top_{int(top_k_percent*100)}%": precision_series.mean()
    }

def calculate_regression_metrics(df_pred: pd.DataFrame):
    """Standard error metrics."""
    if df_pred.empty: return {}
    y_true, y_pred = df_pred['target'], df_pred['pred']
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

def diebold_mariano_test(y_true, y_pred_model, y_pred_benchmark=None, h=1, criterion='MAE'):
    """Robust Diebold-Mariano test with HLN adjustment for h >= 1."""
    y_true = np.array(y_true).flatten()
    y_pred_model = np.array(y_pred_model).flatten()
    y_pred_benchmark = np.zeros_like(y_true) if y_pred_benchmark is None else np.array(y_pred_benchmark).flatten()
    
    T = len(y_true)
    if T == 0: return 0.0, 1.0

    e1 = y_true - y_pred_model
    e2 = y_true - y_pred_benchmark
    d = (e1**2 - e2**2) if criterion == 'MSE' else (np.abs(e1) - np.abs(e2))
    
    d_mean = np.mean(d)
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0
    
    if h > 1:
        for k in range(1, h):
            cov_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
            gamma_sum += cov_k
            
    lr_var = gamma_0 + 2 * gamma_sum
    if lr_var <= 0: return 0.0, 1.0
        
    dm_stat = d_mean / np.sqrt(lr_var / T)
    
    correction = ((T + 1 - 2*h + (h*(h-1))/T) / T)**0.5
    adjusted_dm = dm_stat * correction
    p_value = 2 * (1 - stats.t.cdf(np.abs(adjusted_dm), df=T-1))
    
    return adjusted_dm, p_value