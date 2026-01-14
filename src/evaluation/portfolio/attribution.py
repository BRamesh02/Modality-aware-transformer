# src/evaluation/attribution.py
import pandas as pd
import statsmodels.api as sm

def perform_factor_regression(
    strategy_returns: pd.Series, 
    factor_returns: pd.DataFrame, 
    add_alpha: bool = True
):
    """
    Regresses strategy returns against factor returns to decompose performance.
    Formula: R_strat = alpha + beta_1 * F_1 + ... + beta_n * F_n + epsilon
    
    Args:
        strategy_returns: Series of daily strategy returns.
        factor_returns: DataFrame of factor returns (MKT, HML, etc).
        add_alpha: If True, adds a constant (intercept) to capture Alpha.
    
    Returns:
        model: The fitted statsmodels object.
    """

    common_idx = strategy_returns.index.intersection(factor_returns.index)
    if len(common_idx) < 60:
        raise ValueError("Insufficient overlapping data for regression (<60 days).")
        
    y = strategy_returns.loc[common_idx]
    X = factor_returns.loc[common_idx]
    
    if add_alpha:
        X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    return model

def get_attribution_summary(model) -> pd.DataFrame:
    """
    Formats the regression results into a clean DataFrame.
    """
    df = pd.DataFrame({
        "Beta": model.params,
        "t-Stat": model.tvalues,
        "p-Value": model.pvalues
    })
    
    df["Significant"] = df["p-Value"].apply(lambda x: "✅" if x < 0.05 else "")
    
    if "const" in df.index:
        daily_alpha = df.loc["const", "Beta"]
        ann_alpha = daily_alpha * 252
        df.loc["const", "Ann. Alpha"] = f"{ann_alpha:.2%}"
        df = df.rename(index={"const": "Alpha"})
        
    return df