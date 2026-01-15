import pandas as pd
import numpy as np
from scipy import stats
from src.evaluation.predictions.metrics import diebold_mariano_test

def compare_models(df_base, df_challenger, name_a="Baseline", name_b="Challenger", h=1):
    """
    Compares two models for a specific horizon step.
    """
    merge_cols = ['date_forecast', 'permno']
    if 'horizon' in df_base.columns: merge_cols.append('horizon')

    df = pd.merge(
        df_base[merge_cols + ['target', 'pred']],
        df_challenger[merge_cols + ['pred']],
        on=merge_cols,
        suffixes=(f'_{name_a}', f'_{name_b}'),
        how='inner'
    )
    
    if df.empty: return None

    mae_a = np.abs(df[f'pred_{name_a}'] - df['target']).mean()
    mae_b = np.abs(df[f'pred_{name_b}'] - df['target']).mean()
    ic_a = df[f'pred_{name_a}'].corr(df['target'])
    ic_b = df[f'pred_{name_b}'].corr(df['target'])
    
    daily_ic_a = df.groupby('date_forecast').apply(lambda x: x[f'pred_{name_a}'].corr(x['target']))
    daily_ic_b = df.groupby('date_forecast').apply(lambda x: x[f'pred_{name_b}'].corr(x['target']))
    t_stat_ic, p_val_ic = stats.ttest_rel(daily_ic_a, daily_ic_b)
    
    dm_stat, p_val_dm = diebold_mariano_test(
        df['target'].values,
        df[f'pred_{name_b}'].values, 
        df[f'pred_{name_a}'].values, 
        h=h, criterion='MAE'
    )
    
    return {
        "names": [name_a, name_b],
        "MAE": [mae_a, mae_b],
        "IC": [ic_a, ic_b],
        "dm_stat": dm_stat,
        "p_val_dm": p_val_dm,
        "p_val_ic": p_val_ic
    }

def compare_horizons_decay(df_base, df_challenger, names=("Baseline", "Challenger")):
    """
    Aggregates MAE and IC for each horizon available in the data.
    Returns a DataFrame suitable for plotting decay curves.
    """
    df = pd.merge(
        df_base[['date_forecast', 'permno', 'horizon', 'target', 'pred']],
        df_challenger[['date_forecast', 'permno', 'horizon', 'pred']],
        on=['date_forecast', 'permno', 'horizon'],
        suffixes=(f'_{names[0]}', f'_{names[1]}'),
        how='inner'
    )
    
    def calc_horizon_metrics(g):
        return pd.Series({
            f"MAE_{names[0]}": np.mean(np.abs(g[f'pred_{names[0]}'] - g['target'])),
            f"MAE_{names[1]}": np.mean(np.abs(g[f'pred_{names[1]}'] - g['target'])),
            f"IC_{names[0]}": g[f'pred_{names[0]}'].corr(g['target']),
            f"IC_{names[1]}": g[f'pred_{names[1]}'].corr(g['target'])
        })

    return df.groupby('horizon').apply(calc_horizon_metrics)