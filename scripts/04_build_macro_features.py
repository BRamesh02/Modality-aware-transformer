import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from src.config import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.numerical_data.fred_client import FredClient
from src.numerical_data.features_macro import (
    process_monthly_macro, 
    process_daily_macro, 
    merge_and_format_macro
)

MARKET_FILE = PROJECT_ROOT / "data" / "processed" / "numerical_data" / "features_market.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "numerical_data" / "features_macro.parquet"
START_DATE = "2007-01-01"

def main():
    print("--- Step 4: Building Macroeconomic Features ---")
    
    try:
        fred = FredClient()
    except ValueError as e:
        print(f"SKIPPING MACRO: {e}")
        return

    print("Fetching Monthly Series (Jobs, Inflation)...")
    s_unrate = fred.get_first_release("UNRATE")
    s_cpi = fred.get_first_release("CPIAUCSL")
    s_ppi = fred.get_first_release("PPIACO")
    
    print("Fetching Daily Series (Rates, VIX)...")
    s_curve = fred.get_daily_series("T10Y2Y", START_DATE)
    s_risk = fred.get_daily_series("DGS10", START_DATE)
    s_vix = fred.get_daily_series("VIXCLS", START_DATE)

    print("Processing & aligning lag...")
    df_monthly_proc = process_monthly_macro(s_unrate, s_cpi, s_ppi, START_DATE)
    df_daily_proc = process_daily_macro(s_curve, s_risk, s_vix)
    
    df_macro_full = merge_and_format_macro(df_monthly_proc, df_daily_proc)
    
    if MARKET_FILE.exists():
        print("Aligning to Market Data dates...")
        df_mkt_dates = pd.read_parquet(MARKET_FILE, columns=["date"])["date"].unique()
        df_mkt_dates = pd.DataFrame(index=pd.to_datetime(df_mkt_dates)).sort_index()
        
        df_final = pd.merge(
            df_mkt_dates, 
            df_macro_full, 
            left_index=True, 
            right_index=True, 
            how="left"
        ).ffill()
        
        df_final = df_final.reset_index().rename(columns={"index": "date"})
    else:
        print("Market file not found, saving full macro history.")
        df_final = df_macro_full.reset_index()

    float_cols = df_final.select_dtypes(include=['float64']).columns
    df_final[float_cols] = df_final[float_cols].astype('float32')

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_FILE, compression='brotli')
    
    print(f"SUCCESS: Saved {OUTPUT_FILE}")
    print("Columns:", df_final.columns.tolist())

if __name__ == "__main__":
    main()