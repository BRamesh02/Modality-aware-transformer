import pandas as pd
import numpy as np

def clean_market_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Cleaning specific to Daily Price/Volume data.
    - Handles Sector filling
    - Handles Price/Volume imputation rules
    """
    df = df.copy()
    df = df.sort_values(["date", "permno"]).drop_duplicates(subset=["date", "permno"], keep="first")

    if 'icbindustry' in df.columns:
        df["icbindustry"] = df["icbindustry"].replace("NOAVAIL", np.nan)
        df["icbindustry"] = df.groupby("permno")["icbindustry"].ffill().bfill()

    df["dlyret"] = df["dlyret"].fillna(0.0)
    df["dlyclose"] = df.groupby("permno")["dlyclose"].ffill()
    df["dlyvol"] = df["dlyvol"].fillna(0.0).clip(lower=0)
    
    df["dlycap"] = 1000 * df.groupby("permno")["dlycap"].ffill().bfill()
    df["shrout"] = 1000 * df.groupby("permno")["shrout"].ffill()

    df["dlyhigh"] = df["dlyhigh"].fillna(df["dlyclose"])
    df["dlylow"] = df["dlylow"].fillna(df["dlyclose"])
    
    return df

def compute_market_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Generate technical indicators derived from Market Data.
    (Liquidity, Volatility, Momentum, etc.)
    """
    df = df.copy()

    init_len = len(df)
    df = df.sort_values("dlyvol", ascending=False).drop_duplicates(
        subset=["date", "permno"], 
        keep="first"
    )
    if len(df) < init_len:
        print(f"  -> Removed {init_len - len(df)} duplicate rows from Market Data.")

    df = df.sort_values(["permno", "date"])
        
    eps = 1e-9 

    # --- Liquidity (Spread & Risk) ---
    mid_price = (df["dlyask"] + df["dlybid"]) / 2
    true_spread = (df["dlyask"] - df["dlybid"]) / (mid_price + eps)
    proxy_spread = (df["dlyhigh"] - df["dlylow"]) / (df["dlyclose"] + eps)
    
    df["liquidity_risk"] = true_spread.combine_first(proxy_spread).fillna(0.0).clip(lower=0.0)
    df["liquidity_risk"] = (
        df.groupby("permno")["liquidity_risk"]
        .transform(lambda x: x.fillna(x.rolling(20, min_periods=1).median()))
        .fillna(0.0)
    )
    df["log_liquidity_risk"] = np.log1p(df["liquidity_risk"])

    # --- Log Returns ---
    df["log_ret"] = np.log1p(df["dlyret"])
    clip_low, clip_high = df["log_ret"].quantile(0.001), df["log_ret"].quantile(0.999)
    df["log_ret"] = df["log_ret"].clip(clip_low, clip_high)

    # --- Volatility (Parkinson) ---
    high_low_ratio = (df["dlyhigh"] / (df["dlylow"] + eps)).clip(lower=1.0)
    const = 1.0 / (4.0 * np.log(2.0))
    df["parkinson_daily"] = const * (np.log(high_low_ratio) ** 2)
    
    df["volatility_parkinson"] = df.groupby("permno")["parkinson_daily"].transform(
        lambda x: np.sqrt(x.rolling(window=21, min_periods=10).mean())
    )
    daily_vol_median = df.groupby("date")["volatility_parkinson"].transform("median")
    df["volatility_parkinson"] = df["volatility_parkinson"].fillna(daily_vol_median).fillna(0.01)
    df["log_volatility_parkinson"] = np.log(df["volatility_parkinson"] + 1e-6)

    # --- Momentum ---
    df["mom_1m"] = df.groupby("permno")["log_ret"].transform(
        lambda x: x.rolling(window=21, min_periods=10).sum()
    ).fillna(0.0)
    df["mom_3m"] = df.groupby("permno")["log_ret"].transform(
        lambda x: x.rolling(window=63, min_periods=30).sum()
    ).fillna(0.0)

    # --- Volume Dynamics ---
    df["vol_ma_20"] = df.groupby("permno")["dlyvol"].transform(
        lambda x: x.rolling(window=20, min_periods=5).mean()
    )
    df["relative_volume"] = df["dlyvol"] / (df["vol_ma_20"] + 1.0)
    df["log_relative_volume"] = np.log1p(df["relative_volume"]).fillna(0.69)

    df["turnover"] = df["dlyvol"] / (df["shrout"] + eps)
    df["turnover"] = df["turnover"].clip(upper=1.0)
    df["log_turnover"] = np.log1p(df["turnover"])

    # --- Size (Rank) ---
    df["log_mkt_cap"] = np.log(df["dlycap"] + 1)
    daily_stats = df.groupby("date")["log_mkt_cap"].agg(["mean", "std"])
    df["cap_mean"] = df["date"].map(daily_stats["mean"])
    df["cap_std"] = df["date"].map(daily_stats["std"])
    df["mkt_cap_rank"] = (df["log_mkt_cap"] - df["cap_mean"]) / (df["cap_std"] + eps)
    
    # --- Drawdown ---
    df["rolling_max_1y"] = df.groupby("permno")["dlyclose"].transform(
        lambda x: x.rolling(window=252, min_periods=21).max()
    )
    df["drawdown"] = (df["dlyclose"] / df["rolling_max_1y"]) - 1.0
    df["drawdown"] = df["drawdown"].fillna(0.0)
    df["log_drawdown"] = np.log1p(df["drawdown"])

    return df

def format_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Selects and renames columns with the strict 'mkt_' prefix.
    """
    col_mapping = {
        "date": "date",
        "permno": "permno",
        "log_ret": "mkt_log_ret",
        "mkt_cap_rank": "mkt_cap_rank",
        "mom_1m": "mkt_mom_1m",
        "mom_3m": "mkt_mom_3m",
        "log_volatility_parkinson": "mkt_volatility",
        "log_drawdown": "mkt_drawdown",
        "log_turnover": "mkt_turnover",
        "log_relative_volume": "mkt_rel_vol",
        "log_liquidity_risk": "mkt_liq_risk"
    }
    
    available_cols = [c for c in col_mapping.keys() if c in df.columns]
    out = df[available_cols].rename(columns=col_mapping)
    out["permno"] = out["permno"].astype("int64")
    return out