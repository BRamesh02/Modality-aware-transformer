import pandas as pd
import numpy as np

def compute_vol_scaled_returns(df_market: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Computes the Volatility-Scaled Forward Return.
    Formula: Target = Return_{t+h} / Volatility_t
    
    Args:
        df_market: Must contain 'permno', 'date', 'mkt_log_ret', 'mkt_volatility'
        horizon: Prediction horizon in days (Default=1 for Daily)
    """
    
    df = df_market.copy().sort_values(["permno", "date"])
    
    # We stored 'mkt_volatility' as log(vol), so we exponentiate back
    # Recall feature logic: log_vol = np.log(vol + 1e-6)
    volatility_t = np.exp(df["mkt_volatility"]) 

    if horizon == 1:
        fwd_ret = df.groupby("permno")["mkt_log_ret"].shift(-1)
    else:
        fwd_ret = df.groupby("permno")["mkt_log_ret"].rolling(horizon).sum().shift(-horizon)
        fwd_ret = fwd_ret.reset_index(level=0, drop=True)
    
    horizon_vol = volatility_t * np.sqrt(horizon)
    
    scaled_ret = fwd_ret / (horizon_vol + 1e-9)
    scaled_ret = scaled_ret.clip(-5, 5)
    
    out = df[["date", "permno"]].copy()
    out["target"] = scaled_ret
    
    out = out.dropna(subset=["target"])
    
    return out