import pandas as pd
import numpy as np


def compute_vol_scaled_returns(
    df_market: pd.DataFrame, horizon: int = 1
) -> pd.DataFrame:
    """
    Computes the Volatility-Scaled Excess Forward Return.

    Target = (Return_{t+h} - MarketMean_{t+h}) / Volatility_t

    This effectively creates a 'Sharpe Ratio' like target for the specific stock,
    removed of general market movements (Rising Tide).

    Args:
        df_market: Must contain 'permno', 'date', 'mkt_log_ret', 'mkt_volatility'
        horizon: Prediction horizon in days (Default=1)
    """

    df = df_market.copy().sort_values(["permno", "date"])

    volatility_t = np.exp(df["mkt_volatility"])

    if horizon == 1:
        fwd_ret = df.groupby("permno")["mkt_log_ret"].shift(-1)
    else:
        # Sum log returns over horizon, then shift back to t
        fwd_ret = (
            df.groupby("permno")["mkt_log_ret"].rolling(horizon).sum().shift(-horizon)
        )
        fwd_ret = fwd_ret.reset_index(level=0, drop=True)

    df["temp_fwd_ret"] = fwd_ret
    daily_means = df.groupby("date")["temp_fwd_ret"].transform("mean")
    excess_ret = df["temp_fwd_ret"] - daily_means

    horizon_vol = volatility_t * np.sqrt(horizon)

    scaled_ret = excess_ret / (horizon_vol + 1e-9)
    scaled_ret = scaled_ret.clip(-5, 5)

    out = df[["date", "permno"]].copy()
    out["target"] = scaled_ret

    # Drop rows where target couldn't be calculated (last row of each stock)
    out = out.dropna(subset=["target"])

    return out
