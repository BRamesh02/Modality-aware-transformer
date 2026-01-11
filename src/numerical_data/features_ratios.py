import pandas as pd
import numpy as np

def clean_and_merge_ratios(
    df_daily_skeleton: pd.DataFrame, 
    df_ccm: pd.DataFrame, 
    df_ibes: pd.DataFrame
) -> pd.DataFrame:
    """
    Aligns sparse Ratio data to the Daily Market Skeleton.
    Prioritizes CCM, falls back to IBES.
    """

    lookback = pd.Timedelta(days=365) 
    
    df_ccm = df_ccm.sort_values("public_date").drop_duplicates(subset=["permno", "public_date"], keep="last")
    df_ibes = df_ibes.sort_values("public_date").drop_duplicates(subset=["permno", "public_date"], keep="last")
    df_ccm["public_date"] = pd.to_datetime(df_ccm["public_date"]).dt.normalize()
    df_ibes["public_date"] = pd.to_datetime(df_ibes["public_date"]).dt.normalize()
    
    ratio_cols = ["ptb", "pe_exi", "roe", "de_ratio", "divyield"]
    df_ibes = df_ibes.rename(columns={c: f"{c}_ibes" for c in ratio_cols})

    print("[Ratios] Merging CCM data...")
    df_merged = pd.merge_asof(
        df_daily_skeleton.sort_values("date"),
        df_ccm,
        left_on="date",
        right_on="public_date",
        by="permno",
        direction="backward",
        tolerance=lookback
    )

    print("[Ratios] Merging IBES data...")
    df_merged = pd.merge_asof(
        df_merged,
        df_ibes,
        left_on="date",
        right_on="public_date",
        by="permno",
        direction="backward",
        tolerance=lookback,
        suffixes=("", "_drop")
    )

    for col in ratio_cols:
        ibes_col = f"{col}_ibes"
        if ibes_col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(df_merged[ibes_col])

    drop_cols = [c for c in df_merged.columns if "_ibes" in c or "public_date" in c or "_drop" in c]
    df_merged = df_merged.drop(columns=drop_cols)
    
    return df_merged.drop_duplicates(subset=["permno", "date"], keep="last")

def impute_ratios(df_ratios: pd.DataFrame, df_industry: pd.DataFrame) -> pd.DataFrame:
    """
    3-Layer Imputation: Sector Median -> Market Median -> Constant.
    Guarantees 0 missing values.
    """
    print("[Ratios] Starting Imputation...")
    
    df = pd.merge(df_ratios, df_industry, on=["date", "permno"], how="left")
    
    if 'icbindustry' in df.columns:
        df["icbindustry"] = df["icbindustry"].replace("NOAVAIL", np.nan)
        df["icbindustry"] = df.groupby("permno")["icbindustry"].ffill().bfill()
        df["icbindustry"] = df["icbindustry"].fillna("Unknown")

    ratio_cols = ["ptb", "pe_exi", "roe", "de_ratio", "divyield"]
    
    print("  -> Layer 1: Sector Median...")
    ind_medians = df.groupby(["date", "icbindustry"])[ratio_cols].transform("median")
    df[ratio_cols] = df[ratio_cols].fillna(ind_medians)
    
    print("  -> Layer 2: Global Market Median...")
    # Calculate median per date (ignoring sector)
    global_medians = df.groupby("date")[ratio_cols].transform("median")
    df[ratio_cols] = df[ratio_cols].fillna(global_medians)
    
    print("  -> Layer 3: Safety Constant (0)...")
    df[ratio_cols] = df[ratio_cols].fillna(0.0)

    return df.drop(columns=["icbindustry"])

def compute_ratio_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies financial logic and transformations.
    """
    df = df.copy()
    
    df["divyield"] = df["divyield"].fillna(0.0).clip(upper=0.5)

    df["log_dte"] = np.log1p(df["de_ratio"].clip(lower=0, upper=100))
    df["log_dte"] = df["log_dte"].clip(lower=-10, upper=10.0)
    df["log_ptb"] = np.log1p(df["ptb"].clip(lower=-0.99))
    
    df["earnings_yield"] = 1.0 / (df["pe_exi"] + 1e-6)
    df["earnings_yield"] = df["earnings_yield"].clip(-0.5, 0.5)

    df["roe"] = df["roe"].clip(lower=-2, upper=2)
    
    return df

def format_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    col_mapping = {
        "date": "date",
        "permno": "permno",
        "log_ptb": "ratio_pb",
        "earnings_yield": "ratio_ey",
        "roe": "ratio_roe",
        "log_dte": "ratio_de",
        "divyield": "ratio_div_yield"
    }
    available = [c for c in col_mapping.keys() if c in df.columns]
    out = df[available].rename(columns=col_mapping)
    out["permno"] = out["permno"].astype("int64")
    return out