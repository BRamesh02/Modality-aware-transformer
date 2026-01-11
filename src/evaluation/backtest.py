# src/evaluation/backtest.py
import pandas as pd
import numpy as np
from typing import Dict, Union

def run_backtest(
    signal: pd.DataFrame, 
    returns: Union[pd.DataFrame, pd.Series], 
    direction: int = 1,
    factor_name: str = "Strategy",
    cost_bps: float = 0.0
) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
    """
    Simulates a Dollar-Neutral Long/Short portfolio with Drift-Adjusted Turnover.
    
    Args:
        signal: DataFrame (index=Date, columns=Permno) of raw signals.
        returns: DataFrame (index=Date, columns=Permno) of SIMPLE returns.
                 OR Long DataFrame (columns=[date, permno, return]).
        direction: 1 (Long High) or -1 (Long Low).
        factor_name: Name for the output series.
        cost_bps: Transaction costs in basis points (e.g., 5.0 for 5bps).
    
    Returns:
        Dict with 'returns', 'turnover', 'weights', 'net_returns'.
    """

    if "permno" in returns.columns and "date" in returns.columns:
        returns = returns.pivot(index="date", columns="permno", values="return")
    returns = returns.fillna(0.0)

    common_idx = signal.index.intersection(returns.index).sort_values()
    signal = signal.loc[common_idx]
    returns = returns.loc[common_idx]
    
    if len(common_idx) == 0:
        raise ValueError("Signal and Returns have no overlapping dates!")

    ranks = signal.rank(axis=1, pct=True) * 2 - 1
    ranks = ranks * direction
    
    weights = ranks.sub(ranks.mean(axis=1), axis=0)
    abs_sum = weights.abs().sum(axis=1)
    weights = weights.div(abs_sum.replace(0, np.nan), axis=0).fillna(0.0)
    
    weights_prev = weights.shift(1).fillna(0.0)
    drifted_pos = weights_prev * (1 + returns)
    drifted_weights = drifted_pos.div(drifted_pos.abs().sum(axis=1), axis=0).fillna(0.0)
    daily_turnover = (weights - drifted_weights).abs().sum(axis=1) / 2.0
    
    strat_ret = (weights_prev * returns).sum(axis=1)
    strat_ret.name = factor_name
    
    cost_drag = daily_turnover * (cost_bps / 10000.0)
    net_ret = strat_ret - cost_drag
    net_ret.name = f"{factor_name}_Net"
    
    return {
        "returns": strat_ret,
        "net_returns": net_ret,
        "turnover": daily_turnover,
        "weights": weights
    }