import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
load_dotenv(project_root / ".env")

from src.numerical_data.wrds_client import WRDSClient
from src.numerical_data.features_market import (
    clean_market_prices, 
    compute_market_indicators, 
    format_market_features
)

UNIVERSE_FILE = project_root / "data" / "processed" / "sp500_universe.parquet"
OUTPUT_FILE = project_root / "data" / "processed" / "features_market.parquet"
START_DATE = "2007-01-01" 

def main():
    print("--- Step 2: Building Market Features (Price/Vol) ---")
    
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"Universe file missing: {UNIVERSE_FILE}")
    
    df_universe = pd.read_parquet(UNIVERSE_FILE)
    unique_permnos = df_universe.columns.astype(int).tolist()

    client = WRDSClient()
    try:
        df_raw = client.get_daily_metrics_v2(unique_permnos, start_date=START_DATE)
        print(f"Fetched {len(df_raw)} rows.")
    finally:
        client.close()

    print("Cleaning Market Prices...")
    df_clean = clean_market_prices(df_raw)
    
    print("Computing Market Indicators...")
    df_feats = compute_market_indicators(df_clean)
    
    print("Formatting...")
    df_final = format_market_features(df_feats)
    
    float_cols = df_final.select_dtypes(include=['float64']).columns
    df_final[float_cols] = df_final[float_cols].astype('float32')

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_FILE, compression='brotli')

    
    print(f"SUCCESS: Saved {OUTPUT_FILE}")
    print("Columns:", df_final.columns.tolist())

if __name__ == "__main__":
    main()