import numpy as np
import pandas as pd

def compute_drawdown(strategy_returns: pd.Series) -> pd.Series:
    """Computes the drawdown series (0 to -1)."""
    wealth = (1 + strategy_returns).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return drawdown

def compute_metrics(
    strategy_returns: pd.Series, 
    turnover: pd.Series = None,
    period: int = 252
) -> pd.Series:
    """
    Computes financial metrics for a Long-Short strategy.
    
    Args:
        strategy_returns: Daily simple strategy_returns.
        turnover: Daily turnover series (optional).
        period: Annualization factor (252 for daily).
    """
    
    ann_ret = strategy_returns.mean() * period
    ann_vol = strategy_returns.std() * np.sqrt(period)
    
    # Sharpe = Mean / Vol
    if ann_vol == 0:
        sharpe = 0.0
    else:
        sharpe = ann_ret / ann_vol
        
    # Sortino (Downside Volatility)
    downside_strategy_returns = strategy_returns[strategy_returns < 0]
    downside_vol = downside_strategy_returns.std() * np.sqrt(period)
    if downside_vol == 0:
        sortino = 0.0
    else:
        sortino = ann_ret / downside_vol
        
    # Drawdowns
    dd_series = compute_drawdown(strategy_returns)
    max_dd = dd_series.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    

    # Skewness
    skew = strategy_returns.skew()
    
    # Kurtosis
    kurt = strategy_returns.kurtosis()
    
    # Tail Ratio (Upside vs Downside at 5% tails)
    tail_95 = strategy_returns.quantile(0.95)
    tail_05 = abs(strategy_returns.quantile(0.05))
    tail_ratio = tail_95 / tail_05 if tail_05 != 0 else 0.0
    
    # Win Rate
    win_rate = (strategy_returns > 0).mean()
    
    # Turnover 
    ann_turnover = turnover.mean() * period if turnover is not None else 0.0
    
    metrics = {
        "Annualized Return": ann_ret,
        "Annualized Vol": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Tail Ratio": tail_ratio,
        "Win Rate": win_rate,
        "Ann. Turnover": ann_turnover
    }
    
    return pd.Series(metrics)