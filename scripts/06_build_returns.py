import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "numerical_data"
MARKET_FILE = PROCESSED_DIR / "features_market.parquet"
OUTPUT_FILE = PROCESSED_DIR / "returns.parquet"

def main():
    print("--- Step 6: Building Returns Table ---")
    
    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Market features not found: {MARKET_FILE}")
    
    print(f"Loading: {MARKET_FILE}")
    df_mkt = pd.read_parquet(MARKET_FILE)
    df_ret = df_mkt[["date", "permno"]].copy()
    
    if df_mkt["mkt_log_ret"].max() > 2.0:
        print("Warning: Extreme log returns detected (> 2.0). Check raw data.")
    
    df_ret["return"] = np.exp(df_mkt["mkt_log_ret"]) - 1
    
    print("\n[Statistics]")
    print(f"Mean Daily Return: {df_ret['return'].mean():.6f}")
    print(f"Min Return:        {df_ret['return'].min():.6f}")
    print(f"Max Return:        {df_ret['return'].max():.6f}")
    
    df_ret = df_ret.sort_values(["date", "permno"])
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_ret.to_parquet(OUTPUT_FILE, compression="brotli")
    print(f"\nSUCCESS: Saved Returns table to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
