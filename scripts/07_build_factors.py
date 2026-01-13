# scripts/06_build_factors.py
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
load_dotenv(project_root / ".env")

from src.numerical_data.wrds_client import WRDSClient
from src.numerical_data.factors import (
    construct_rank_factor, 
    compute_risk_metrics, 
    compute_quality_score,
    compute_momentum
)

PROCESSED_DIR = project_root / "data" / "processed" / "numerical_data"
UNIVERSE_FILE = PROCESSED_DIR / "sp500_universe.parquet" 
MARKET_FILE = PROCESSED_DIR / "features_market.parquet"
RATIO_FILE = PROCESSED_DIR / "features_ratios.parquet"
RETURNS_FILE = PROCESSED_DIR / "returns.parquet"
OUTPUT_FILE = PROCESSED_DIR / "factors_returns.parquet" 

START_DATE = "2007-01-01"

def main():
    print("--- Step 7: Constructing Rank-Based Equity Factors ---")
    
    print("Loading Data...")
    df_univ_wide = pd.read_parquet(UNIVERSE_FILE)
    df_mkt = pd.read_parquet(MARKET_FILE)
    df_ratio = pd.read_parquet(RATIO_FILE)
    df_ret = pd.read_parquet(RETURNS_FILE)
    
    df_univ_wide.index = pd.to_datetime(df_univ_wide.index).normalize()
    df_mkt["date"] = pd.to_datetime(df_mkt["date"]).dt.normalize()
    df_ratio["date"] = pd.to_datetime(df_ratio["date"]).dt.normalize()
    df_ret["date"] = pd.to_datetime(df_ret["date"]).dt.normalize()
    
    print("Computing Risk Metrics (Beta & Vol)...")
    df_risk = compute_risk_metrics(df_ret, window=252)
    
    print("Computing Momentum (12-1)...")
    df_mom = compute_momentum(df_mkt)
    
    print("Merging Global Feature Set...")
    df_global = pd.merge(df_mkt, df_ret, on=["date", "permno"], how="inner")
    df_global = pd.merge(df_global, df_ratio, on=["date", "permno"], how="left")
    df_global = pd.merge(df_global, df_risk, on=["date", "permno"], how="left")
    df_global = pd.merge(df_global, df_mom, on=["date", "permno"], how="left")
    
    print("Computing Quality...")
    df_global = compute_quality_score(df_global)
    
    print("Applying Universe Mask...")
    s_univ = df_univ_wide.stack()
    valid_pairs = s_univ[s_univ == 1].index
    
    df_univ_long = pd.DataFrame(index=valid_pairs).reset_index()
    df_univ_long.columns = ["date", "permno"]
    df_univ_long["permno"] = df_univ_long["permno"].astype(int)
    
    df_indexed = pd.merge(df_global, df_univ_long, on=["date", "permno"], how="inner")

    df_factors = pd.DataFrame(index=df_univ_wide.index.sort_values()).sort_index()
    df_factors.index.name = "date"
    
    client = WRDSClient()
    try:
        df_rf = client.get_risk_free_rate(START_DATE)
        df_rf["date"] = pd.to_datetime(df_rf["date"]).dt.normalize()
    finally:
        client.close()
    
    df_factors = pd.merge(df_factors, df_rf, on="date", how="left").set_index("date").fillna(0.0)

    print("Constructing Factors...")
    
    univ_ret = df_indexed.groupby("date")["return"].mean()
    rf_aligned = df_factors["rf"].reindex(univ_ret.index).fillna(0.0)
    df_factors["MKT"] = univ_ret - rf_aligned
    
    def build(col, name, d):
        return construct_rank_factor(df_indexed, col, name, direction=d)

    df_factors["SMB"] = build("mkt_cap_rank", "SMB", -1)
    df_factors["HML"] = build("ratio_pb", "HML", -1)
    df_factors["UMD"] = build("mom_12_1", "UMD", 1)
    df_factors["QMJ"] = build("quality_score", "QMJ", 1)
    df_factors["BAB"] = build("beta_est", "BAB", -1)

    df_factors = df_factors.drop(columns=["rf"], errors="ignore").fillna(0.0)
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_factors.to_parquet(OUTPUT_FILE, compression="brotli")
    print(f"SUCCESS: Saved Factors' Returns to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()