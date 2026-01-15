import sys
from pathlib import Path
from dotenv import load_dotenv

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent

sys.path.append(str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.numerical_data.wrds_client import WRDSClient
from src.numerical_data.universe import build_full_presence_matrix

RAW_DIR = PROJECT_ROOT / "data" / "raw"
LEGACY_DIR = RAW_DIR / "wrds_sp5scripts/02_build_market_features.py00_constituents"
GITHUB_FILE = RAW_DIR / "wikipedia_sp500_constituents" / "wikipedia_sp500_constituents.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"/ "numerical_data"
OUTPUT_FILE = OUTPUT_DIR / "sp500_universe.parquet"

def main():
    print("--- Step 1: Starting Universe Construction ---")
    
    client = WRDSClient()
    
    try:
        df_universe = build_full_presence_matrix(
            legacy_dir=LEGACY_DIR,
            github_file=GITHUB_FILE,
            wrds_client=client
        )
        
        if df_universe.empty:
            raise ValueError("Generated universe is empty!")
        
        if df_universe.sum(axis=1).iloc[-1] < 400:
            print("WARNING: Recent universe count is low (<400). Check data sources.")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_universe.to_parquet(OUTPUT_FILE)
        
        print(f"SUCCESS: Universe saved to {OUTPUT_FILE}")
        print(f"Dimensions: {df_universe.shape}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        raise
        
    finally:
        client.close()

if __name__ == "__main__":
    main()