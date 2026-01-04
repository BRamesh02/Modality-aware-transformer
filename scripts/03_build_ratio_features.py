import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
load_dotenv(project_root / ".env")

from src.data.wrds_client import WRDSClient
from src.data.features_ratios import (
    clean_and_merge_ratios,
    impute_ratios,
    compute_ratio_indicators,
    format_ratio_features
)

MARKET_FILE = project_root / "data" / "processed" / "features_market.parquet"
OUTPUT_FILE = project_root / "data" / "processed" / "features_ratios.parquet"
START_DATE = "2007-01-01"

def main():
    print("--- Step 3: Building Fundamental Features (Ratios) ---")
    
    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Run Step 2 first! Missing: {MARKET_FILE}")
    
    print("Loading Market Skeleton...")
    df_mkt = pd.read_parquet(MARKET_FILE, columns=["date", "permno"])
    unique_permnos = df_mkt["permno"].unique().tolist()
    
    client = WRDSClient()
    try:
        df_ccm, df_ibes = client.get_ratios_data(unique_permnos, start_date=START_DATE)
        print("[WRDS] Fetching Industry Codes for imputation...")
        df_ind = client.get_industry_classifications(unique_permnos, start_date=START_DATE)
        
    finally:
        client.close()

    df_merged = clean_and_merge_ratios(df_mkt, df_ccm, df_ibes)
    df_imputed = impute_ratios(df_merged, df_ind)
    
    print("Computing Indicators...")
    df_feats = compute_ratio_indicators(df_imputed)
    
    df_final = format_ratio_features(df_feats)
    df_final = df_final.drop_duplicates(subset=["date", "permno"], keep="last")
    
    float_cols = df_final.select_dtypes(include=['float64']).columns
    df_final[float_cols] = df_final[float_cols].astype('float32')

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_FILE, compression='brotli')
    
    print(f"SUCCESS: Saved {OUTPUT_FILE}")
    print("Columns:", df_final.columns.tolist())

if __name__ == "__main__":
    main()