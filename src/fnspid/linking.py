import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from src.numerical_data.wrds_client import WRDSClient

def fetch_and_save_crsp_map(mapping_file_path: Path):
    print("[WRDS] Attempting to connect to WRDS...")
    try:
        client = WRDSClient()
    except Exception as e:
        print(f"Could not connect to WRDS: {e}")
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
    
    mapping_file_path.parent.mkdir(parents=True, exist_ok=True)
    df_names.to_parquet(mapping_file_path)
    print(f"Saved CRSP Mapping Table to: {mapping_file_path}")
    
    return df_names

def load_mapping_table(mapping_file_path: Path):
    """
    Loads the CRSP mapping table from WRDS (if env vars exist) or local disk.
    """
    has_creds = os.getenv("WRDS_USERNAME") is not None
    df_map = None

    if has_creds:
        df_map = fetch_and_save_crsp_map(mapping_file_path)
    
    if df_map is None:
        if mapping_file_path.exists():
            print(f"[Offline Mode] Loading local mapping file: {mapping_file_path}")
            df_map = pd.read_parquet(mapping_file_path)
        else:
            msg = (
                "\n Linking is not possible.\n"
                f"   Searched for map at: {mapping_file_path}\n"
                "   Reason: No WRDS credentials loaded AND no local map found.\n"
            )
            sys.exit(msg)
    return df_map

def load_and_concat_parts(input_dir: Path, pattern: str = "features_fnspid_*.parquet") -> pd.DataFrame:
    if not input_dir.exists():
        print(f" Input directory not found: {input_dir}")
        sys.exit(1)

    files = sorted(input_dir.glob(pattern))
    if not files:
        print(f" No files found in {input_dir} matching '{pattern}'")
        sys.exit(1)

    print(f"Found {len(files)} parts. Concatenating...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Skipping corrupted file {f.name}: {e}")

    if not dfs:
        sys.exit(" No valid dataframes loaded.")

    return pd.concat(dfs, ignore_index=True)

def map_point_in_time(df_target, df_map):
    print(f"Mapping {len(df_target)} rows against {len(df_map)} ticker ranges...")
    
    if "effective_date" in df_target.columns:
        df_target["date"] = pd.to_datetime(df_target["effective_date"]).dt.normalize()
    elif "date" in df_target.columns:
        df_target["date"] = pd.to_datetime(df_target["date"]).dt.normalize()
    else:
        raise KeyError("Input dataframe missing 'effective_date' or 'date' column")

    ticker_col = "stock_symbol" if "stock_symbol" in df_target.columns else "ticker"
    if ticker_col not in df_target.columns:
        raise KeyError(f"Input dataframe missing '{ticker_col}' or 'ticker' column")

    df_target["ticker_clean"] = df_target[ticker_col].astype(str).str.upper().str.strip()
    df_map["ticker_clean"] = df_map["ticker"].astype(str).str.upper().str.strip()
    
    df_target = df_target.reset_index(drop=True).reset_index(names="row_id")
    
    merged = pd.merge(
        df_target, 
        df_map[["permno", "ticker_clean", "namedt", "nameenddt", "comnam"]], 
        on="ticker_clean", 
        how="left"
    )
    
    valid_mask = (
        (merged["date"] >= merged["namedt"]) & 
        (merged["date"] <= merged["nameenddt"])
    )
    
    df_matched = merged[valid_mask].copy()
    df_matched = df_matched.drop_duplicates(subset=["row_id"])
    
    df_final = pd.merge(
        df_target,
        df_matched[["row_id", "permno", "comnam"]],
        on="row_id",
        how="left"
    )
    
    return df_final.drop(columns=["row_id", "ticker_clean"])

def filter_by_universe(df_data, universe_path: Path):
    """
    Filters df_data to only include (Date, Permno) pairs present in the universe file.
    """
    print(f"\n[Filter] Loading universe file from: {universe_path}")
    
    if not universe_path.exists():
        print(f" Universe file not found. Skipping filter step.")
        return df_data
    
    df_wide = pd.read_parquet(universe_path)
    
    if not isinstance(df_wide.index, pd.DatetimeIndex):
        df_wide.index = pd.to_datetime(df_wide.index)

    series_long = df_wide.stack()

    df_univ_long = series_long.reset_index()
    df_univ_long.columns = ["date", "permno", "val"]
    
    df_univ_long["date"] = pd.to_datetime(df_univ_long["date"]).dt.normalize()
    df_data["date"] = pd.to_datetime(df_data["date"]).dt.normalize()
    
    df_univ_long["permno"] = pd.to_numeric(df_univ_long["permno"], errors='coerce')
    df_data["permno"] = pd.to_numeric(df_data["permno"], errors='coerce')

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
    Optimizes floats to float32 and saves as Snappy Parquet.
    """
    print(f"\n[Save] Optimizing data types before writing...")
    
    float_cols = df.select_dtypes(include=['float64']).columns
    if len(float_cols) > 0:
        print(f"  -> Downcasting {len(float_cols)} float64 columns to float32...")
        df[float_cols] = df[float_cols].astype('float32')
        
    print(f"[Save] Writing to disk: {output_path}...")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f" SUCCESS: Saved {file_size_mb:.1f} MB to {output_path}")
    except Exception as e:
        print(f" Error saving file: {e}")