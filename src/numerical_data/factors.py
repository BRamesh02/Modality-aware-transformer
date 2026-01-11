import pandas as pd
import numpy as np

def compute_risk_metrics(df_returns: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Computes rolling risk metrics: Beta and Annualized Volatility.
    
    Returns a DataFrame with columns: ['date', 'permno', 'beta_est', 'vol_12m']
    """
    
    df = df_returns.copy().sort_values(["date", "permno"])
    wide_ret = df.pivot(index="date", columns="permno", values="return")
    
    mkt_ret = wide_ret.mean(axis=1)
    
    indexer = wide_ret.rolling(window, min_periods=126)
    
    rolling_cov = indexer.cov(mkt_ret)
    rolling_mkt_var = mkt_ret.rolling(window, min_periods=126).var()
    betas = rolling_cov.div(rolling_mkt_var, axis=0)
    
    vols = indexer.std() * np.sqrt(252)
    
    df_beta = betas.stack().rename("beta_est")
    df_vol = vols.stack().rename("vol_12m")
    
    df_risk = pd.concat([df_beta, df_vol], axis=1).reset_index()
    
    return df_risk

def compute_momentum(
    df_returns: pd.DataFrame, 
    lookback_long: int = 252, 
    lookback_short: int = 21
) -> pd.DataFrame:
    """
    Computes standard Momentum (12M - 1M) using cumulative log returns.
    """
    df = df_returns.copy().sort_values(["permno", "date"])
    
    df["log_price"] = df.groupby("permno")["mkt_log_ret"].cumsum()
    grouped = df.groupby("permno")["log_price"]
    
    df["mom_12_1"] = grouped.shift(lookback_short) - grouped.shift(lookback_long)
    
    return df[["date", "permno", "mom_12_1"]].dropna()

def compute_quality_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a 3-Pillar Quality Score using Rank-Average Logic.
    
    1. Profitability: High ROE
    2. Safety:        Low Debt/Equity
    3. Stability:     Low Volatility (12M)
    """
    df = df.copy()


    r_prof = df.groupby("date")["ratio_roe"].rank(pct=True)
    r_safe = 1 - df.groupby("date")["ratio_de"].rank(pct=True)
    r_stab = 1 - df.groupby("date")["vol_12m"].rank(pct=True)
    
    df["quality_score"] = (r_prof + r_safe + r_stab) / 3.0
    
    def zscore(x):
        std = x.std()
        if pd.isna(std) or std == 0:
            return x * 0.0
        return (x - x.mean()) / std
    
    df["quality_score"] = df.groupby("date")["quality_score"].transform(zscore)
    
    return df

def construct_rank_factor(
    df_data: pd.DataFrame,
    signal_col: str,
    factor_name: str,
    direction: int = 1
) -> pd.Series:
    """
    Standard Rank-Based Factor Construction (Dollar Neutral, Gross Leverage=1).
    """
    
    wide_signal = df_data.pivot_table(index="date", columns="permno", values=signal_col)
    wide_ret = df_data.pivot_table(index="date", columns="permno", values="return")
    
    ranks = 2 * wide_signal.rank(axis=1, pct=True) - 1
    ranks = ranks * direction
    
    weights = ranks.sub(ranks.mean(axis=1), axis=0)
    weights = weights.div(weights.abs().sum(axis=1), axis=0)
    
    strat_ret = (weights.shift(1) * wide_ret).sum(axis=1)
    
    strat_ret.name = factor_name
    return strat_ret