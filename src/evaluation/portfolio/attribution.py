import pandas as pd
import numpy as np
import statsmodels.api as sm


def perform_factor_regression(
    strategy_returns: pd.Series, factor_returns: pd.DataFrame, add_alpha: bool = True
):
    """
    Regresses strategy returns against factor returns.
    """
    strategy_returns = strategy_returns.astype(float)

    factor_returns = factor_returns.select_dtypes(include=[np.number]).astype(float)

    common_idx = strategy_returns.index.intersection(factor_returns.index)
    if len(common_idx) < 30:
        raise ValueError(
            f"Insufficient overlapping data ({len(common_idx)} days). Need >30."
        )

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
    df = pd.DataFrame(
        {"Beta": model.params, "t-Stat": model.tvalues, "p-Value": model.pvalues}
    )

    df["Significant"] = df["p-Value"].apply(lambda x: "✅" if x < 0.05 else "")

    if "const" in df.index:
        daily_alpha = df.loc["const", "Beta"]
        ann_alpha = daily_alpha * 252
        df.loc["const", "Ann. Alpha"] = ann_alpha
        df = df.rename(index={"const": "Alpha"})

    return df
