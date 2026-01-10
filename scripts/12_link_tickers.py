import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# --- 1. SETUP PATHS & IMPORTS ---
# Add the project root to sys.path so we can import from 'src'
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

# Import the centralized client from your clean architecture
try:
    from src.data.wrds_client import WRDSClient
except ImportError:
    print("  Could not import WRDSClient from src.data.")
    print("   Ensure you are running this from the 'scripts/' folder and 'src/' exists at the root.")
    sys.exit(1)

# --- 2. ROBUST ENV LOADING ---
def load_env_robust():
    """Tries to find .env in current dir or parent dir."""
    possible_paths = [
        current_dir / ".env",          # scripts/.env
        project_root / ".env"          # root/.env
    ]
    
    env_found = False
    for path in possible_paths:
        if path.exists():
            load_dotenv(path)
            env_found = True
            break
            
    if not env_found:
        print("⚠️  Warning: No .env file found in script dir or parent dir.")

load_env_robust()

# --- CONFIG ---
# BASE_DIR is the project root
BASE_DIR = project_root

INPUT_DIR = BASE_DIR / "data" / "fnspid_features_text_stock_date"
OUTPUT_FILE = BASE_DIR / "data" / "fnspid_features_text_stock_date" / "linked_text_features_stock_date.parquet"
MAPPING_FILE = BASE_DIR / "data" / "raw" / "crsp_ticker_map.parquet"
UNIVERSE_FILE = BASE_DIR / "data" / "processed" / "sp500_universe.parquet"

# --- CORE LOGIC ---

def fetch_and_save_crsp_map():
    print("[WRDS] Attempting to connect to WRDS...")
    try:
        client = WRDSClient()
    except Exception as e:
        print(f"⚠️ Could not connect to WRDS: {e}")
        return None

    print("[WRDS] Fetching CRSP Stocknames history...")
    query = """
        SELECT permno, ticker, namedt, nameenddt, comnam
        FROM crsp.stocknames
        WHERE ticker IS NOT NULL
        AND nameenddt >= '2008-01-01'
    """
    df_names = client.query(query)
    client.close()
    
    df_names["namedt"] = pd.to_datetime(df_names["namedt"])
    df_names["nameenddt"] = pd.to_datetime(df_names["nameenddt"])
    df_names["nameenddt"] = df_names["nameenddt"].fillna(pd.Timestamp.today())
    
    MAPPING_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_names.to_parquet(MAPPING_FILE)
    print(f"✅ Saved CRSP Mapping Table to: {MAPPING_FILE}")
    
    return df_names

def load_mapping_table():
    has_creds = os.getenv("WRDS_USERNAME") is not None
    df_map = None

    if has_creds:
        df_map = fetch_and_save_crsp_map()
    
    if df_map is None:
        if MAPPING_FILE.exists():
            print(f"[Offline Mode] Loading local mapping file: {MAPPING_FILE}")
            df_map = pd.read_parquet(MAPPING_FILE)
        else:
            msg = (
                "\n CRITICAL ERROR: Linking is not possible.\n"
                f"   Searched for map at: {MAPPING_FILE}\n"
                "   Reason: No WRDS credentials loaded AND no local map found.\n"
            )
            sys.exit(msg)
    return df_map

def load_and_concat_parts(input_dir: Path) -> pd.DataFrame:
    if not input_dir.exists():
        print(f" Input directory not found: {input_dir}")
        sys.exit(1)

    files = sorted(input_dir.glob("text_features_stock_date_*.parquet"))
    if not files:
        print(f" No files found in {input_dir} matching 'final_*.parquet'")
        sys.exit(1)

    print(f"Found {len(files)} parts. Concatenating...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"⚠️ Skipping corrupted file {f.name}: {e}")

    if not dfs:
        sys.exit(" No valid dataframes loaded.")

    return pd.concat(dfs, ignore_index=True)

def map_point_in_time(df_target, df_map):
    print(f"Mapping {len(df_target)} rows against {len(df_map)} ticker ranges...")
    
    # 1. Normalize Date
    if "effective_date" in df_target.columns:
        df_target["date"] = pd.to_datetime(df_target["effective_date"]).dt.normalize()
    elif "date" in df_target.columns:
        df_target["date"] = pd.to_datetime(df_target["date"]).dt.normalize()
    else:
        raise KeyError("Input dataframe missing 'effective_date' or 'date' column")

    # 2. Identify Ticker Column
    ticker_col = "stock_symbol" if "stock_symbol" in df_target.columns else "ticker"
    if ticker_col not in df_target.columns:
        raise KeyError(f"Input dataframe missing '{ticker_col}' or 'ticker' column")

    # 3. Clean Strings
    df_target["ticker_clean"] = df_target[ticker_col].astype(str).str.upper().str.strip()
    df_map["ticker_clean"] = df_map["ticker"].astype(str).str.upper().str.strip()
    
    # 4. Merge Logic
    df_target = df_target.reset_index(drop=True).reset_index(names="row_id")
    
    merged = pd.merge(
        df_target, 
        df_map[["permno", "ticker_clean", "namedt", "nameenddt", "comnam"]], 
        on="ticker_clean", 
        how="left"
    )
    
    # 5. Filter Validity Window
    valid_mask = (
        (merged["date"] >= merged["namedt"]) & 
        (merged["date"] <= merged["nameenddt"])
    )
    
    df_matched = merged[valid_mask].copy()
    df_matched = df_matched.drop_duplicates(subset=["row_id"])
    
    # 6. Final Join
    df_final = pd.merge(
        df_target,
        df_matched[["row_id", "permno", "comnam"]],
        on="row_id",
        how="left"
    )
    
    return df_final.drop(columns=["row_id", "ticker_clean"])

def filter_by_universe(df_data, universe_path):
    """
    Filters df_data to only include (Date, Permno) pairs present in the universe file.
    Handles Wide Format: Index=Date, Columns=Permno.
    """
    print(f"\n[Filter] Loading universe file from: {universe_path}")
    
    if not universe_path.exists():
        print(f" Universe file not found. Skipping filter step.")
        return df_data
    
    # 1. Load Wide Universe
    df_wide = pd.read_parquet(universe_path)
    
    # 2. Convert to Long Format (Date, Permno)
    print("  -> Converting universe from Wide to Long format...")
    
    if not isinstance(df_wide.index, pd.DatetimeIndex):
        df_wide.index = pd.to_datetime(df_wide.index)

    series_long = df_wide.stack()
    
    if series_long.dtype == bool:
        series_long = series_long[series_long]
        
    df_univ_long = series_long.reset_index()
    df_univ_long.columns = ["date", "permno", "val"]
    
    # 3. Data Type Standardization
    df_univ_long["date"] = pd.to_datetime(df_univ_long["date"]).dt.normalize()
    df_data["date"] = pd.to_datetime(df_data["date"]).dt.normalize()
    
    df_univ_long["permno"] = pd.to_numeric(df_univ_long["permno"], errors='coerce')
    df_data["permno"] = pd.to_numeric(df_data["permno"], errors='coerce')

    # 4. Filter via Inner Join
    initial_len = len(df_data)
    
    df_filtered = df_data.merge(
        df_univ_long[["date", "permno"]], 
        on=["date", "permno"],
        how="inner"
    )
    
    final_len = len(df_filtered)
    pct_kept = final_len / initial_len if initial_len > 0 else 0
    print(f" Filtered by Universe: {initial_len:,} -> {final_len:,} rows ({pct_kept:.1%} kept)")
    
    return df_filtered

def optimize_and_save(df: pd.DataFrame, output_path: Path):
    """
    Optimizes the dataframe for faster saving.
    """
    print(f"\n[Save] Optimizing data types before writing...")
    
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        print(f"  -> Downcasting {len(float_cols)} float64 columns to float32...")
        df[float_cols] = df[float_cols].astype('float32')
        
    print(f"[Save] Writing to disk: {output_path}...")
    try:
        df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f" SUCCESS: Saved {file_size_mb:.1f} MB to {output_path}")
    except Exception as e:
        print(f" Error saving file: {e}")

def main():
    print("--- Ticker Linking Tool ---")
    
    # 1. Get Mapping Data
    df_map = load_mapping_table()

    # 2. Load Input
    df_friend = load_and_concat_parts(INPUT_DIR)

    # 3. Execute Mapping
    df_linked = map_point_in_time(df_friend, df_map)
    
    # 4. Filter by Universe
    df_linked = filter_by_universe(df_linked, UNIVERSE_FILE)
    
    # 5. Show Stats
    match_rate = df_linked["permno"].notna().mean()
    print(f"\nFinal Match Rate: {match_rate:.2%}")
    
    ticker_col = "stock_symbol" if "stock_symbol" in df_linked.columns else "ticker"
    cols_to_print = ["date", ticker_col, "permno", "comnam"]
    existing_cols = [c for c in cols_to_print if c in df_linked.columns]
    
    print(df_linked[existing_cols].head())

    # 6. Save (Optimized)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    optimize_and_save(df_linked, OUTPUT_FILE)

if __name__ == "__main__":
    main()