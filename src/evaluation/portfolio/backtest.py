import pandas as pd
import numpy as np
from typing import Dict, Union, Optional


def _ensure_datetime_index(
    df: Union[pd.DataFrame, pd.Series],
) -> Union[pd.DataFrame, pd.Series]:
    """Helper to force the index to be timezone-naive datetime."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def compute_decile_weights(
    signal: pd.DataFrame, universe_mask: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Computes dollar-neutral weights (Gross Exposure = 1.0).
    Logic: Long Top Decile, Short Bottom Decile, Demean, Normalize by Abs Sum.
    """
    signal = _ensure_datetime_index(signal)
    if universe_mask is not None:
        universe_mask = _ensure_datetime_index(universe_mask)
        common_dates = signal.index.intersection(universe_mask.index)
        signal = signal.loc[common_dates]
        universe_mask = universe_mask.loc[common_dates]
        signal = signal.where(universe_mask > 0)

    global_ranks = signal.rank(axis=1, pct=True)

    top_mask = global_ranks >= 0.9  # Top 10%
    bot_mask = global_ranks <= 0.1  # Bottom 10%

    top_local = signal.where(top_mask).rank(axis=1, pct=True)

    bot_local = signal.where(bot_mask).rank(axis=1, pct=True)

    raw_long = top_local.fillna(0.0)
    raw_short = -(1.0 - bot_local).fillna(0.0)

    combined_signal = raw_long + raw_short

    # Demean (Enforce Dollar Neutrality: Net = 0)
    weights_demeaned = combined_signal.sub(combined_signal.mean(axis=1), axis=0)

    # Normalize (Enforce Gross Exposure = 1.0)
    abs_sum = weights_demeaned.abs().sum(axis=1)
    final_weights = weights_demeaned.div(abs_sum.replace(0, np.nan), axis=0).fillna(0.0)

    return final_weights


def run_backtest(
    signal: pd.DataFrame,
    returns: Union[pd.DataFrame, pd.Series],
    direction: int = 1,
    factor_name: str = "Strategy",
    cost_bps: float = 0.0,
    universe_mask: Optional[pd.DataFrame] = None,
) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
    """
    Simulates a standard Dollar-Neutral Portfolio.
    """
    signal = _ensure_datetime_index(signal.copy()) * direction
    returns = _ensure_datetime_index(returns.copy()).fillna(0.0)

    if "permno" in returns.columns:
        returns = returns.pivot(index="date", columns="permno", values="return")

    if universe_mask is not None:
        universe_mask = _ensure_datetime_index(universe_mask.copy())

    common_idx = signal.index.intersection(returns.index)
    if universe_mask is not None:
        common_idx = common_idx.intersection(universe_mask.index)
        universe_mask = universe_mask.loc[common_idx]

    if len(common_idx) == 0:
        raise ValueError("No overlapping dates found between Signal and Returns!")

    signal = signal.loc[common_idx]
    returns = returns.loc[common_idx]

    # Compute Weights
    weights = compute_decile_weights(signal, universe_mask)

    # PnL Simulation
    weights_prev = weights.shift(1).fillna(0.0)

    strat_ret = (weights_prev * returns).sum(axis=1)
    strat_ret.name = factor_name

    # Turnover & Cost Calculation
    drifted_pos = weights_prev * (1 + returns)

    drifted_weights = drifted_pos

    drifted_weights = drifted_weights.reindex_like(weights).fillna(0.0)

    # Turnover = sum of absolute changes in position size / 2 (one-way)
    daily_turnover = (weights - drifted_weights).abs().sum(axis=1) / 2.0

    cost_drag = daily_turnover * (cost_bps / 10000.0)
    net_ret = strat_ret - cost_drag
    net_ret.name = f"{factor_name}_Net"

    return {
        "returns": strat_ret,
        "net_returns": net_ret,
        "turnover": daily_turnover,
        "weights": weights,
    }


def analyze_long_short_autopsy(weights: pd.DataFrame, returns: pd.DataFrame):
    """
    Decomposes the strategy into Long Leg and Short Leg returns.

    Args:
        weights: DataFrame of portfolio weights at time T.
        returns: DataFrame of asset returns at time T+1.

    Returns:
        long_ret (pd.Series): Returns of the Long positions only.
        short_ret (pd.Series): Returns of the Short positions only.
    """
    w_shifted = weights.shift(1).fillna(0.0)

    long_w = w_shifted.where(w_shifted > 0, 0.0)
    long_ret = (long_w * returns).sum(axis=1)

    short_w = w_shifted.where(w_shifted < 0, 0.0)
    short_ret = (short_w * returns).sum(axis=1)

    return long_ret, short_ret
