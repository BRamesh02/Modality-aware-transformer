import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.numerical_data.target import compute_vol_scaled_returns

MARKET_FILE = project_root / "data" / "processed" / "features_market.parquet"
OUTPUT_FILE = project_root / "data" / "processed" / "target.parquet"

PREDICTION_HORIZON = 1 

def main():
    print(f"--- Step 5: Building Daily Target (Horizon={PREDICTION_HORIZON}) ---")
    
    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Missing market file: {MARKET_FILE}. Run Step 2 first.")
    
    print("Loading Market Features...")
    df_mkt = pd.read_parquet(
        MARKET_FILE, 
        columns=["date", "permno", "mkt_log_ret", "mkt_volatility"]
    )
    
    df_target = compute_vol_scaled_returns(df_mkt, horizon=PREDICTION_HORIZON)
    
    print("\nTarget Statistics:")
    print(df_target["target"].describe().round(4))

    if df_target["target"].std() > 5.0:
        print("WARNING: Target standard deviation is very high. Check volatility scaling.")
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_target.to_parquet(OUTPUT_FILE, compression="brotli")
    
    print(f"\nSUCCESS: Target saved to {OUTPUT_FILE}")
    print(f"Dimensions: {df_target.shape}")

if __name__ == "__main__":
    main()