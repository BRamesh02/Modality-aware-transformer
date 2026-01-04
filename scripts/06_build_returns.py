import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

PROCESSED_DIR = project_root / "data" / "processed"
MKT_FILE = PROCESSED_DIR / "features_market.parquet"
OUTPUT_FILE = PROCESSED_DIR / "returns.parquet"

def main():
    print("--- Step 5: Building Returns Table ---")
    
    if not MKT_FILE.exists():
        raise FileNotFoundError(f"Market features not found: {MKT_FILE}")
    
    print(f"Loading: {MKT_FILE}")
    df_mkt = pd.read_parquet(MKT_FILE)
    
    print("Converting Log Returns -> Returns...")
    
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
