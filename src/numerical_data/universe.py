import pandas as pd
import numpy as np
from pathlib import Path
import glob
from typing import List, Dict, Optional
from src.numerical_data.wrds_client import WRDSClient

# --- Constants ---
TICKER_CLEANING_MAP = {
    "LEHMQ": "LEH",   "WAMUQ": "WM",    "EKDKQ": "EK",    "MTLQQ": "GM",
    "SUNEQ": "SUNE",  "BTUUQ": "BTU",   "ANRZQ": "ANR",   "RSHCQ": "RSH",
    "CITGQ": "CIT",   "CCTYQ": "CC",    "ABKFQ": "ABK",   "FNMA": "FNM",
    "FMCC": "FRE",    "BRK.B": "BRK",   "BF.B": "BF",     "PSKY": "PSB",
    "SOLS": "SOL",    "XYZ": None
}

def parse_wrds_legacy_csvs(folder_path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    """
    Step 1: Process the WRDS CSVs (Interval Format).
    Structure: "PERMNO", "Company Name", "Ticker", "SP500 Start", "SP500 End"
    """
    files = sorted(glob.glob(str(folder_path / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    
    print(f"[Legacy] Loading {len(files)} CSV files...")
    df_list = [pd.read_csv(f) for f in files]
    df_raw = pd.concat(df_list)

    def clean_date(d):
        if pd.isna(d): return pd.NaT
        d = str(d).replace(".", "").strip()
        return pd.to_datetime(d, errors='coerce')

    df_raw['start_dt'] = df_raw['SP500 Start'].apply(clean_date)
    df_raw['end_dt'] = df_raw['SP500 End'].apply(clean_date)
    
    s_date = pd.Timestamp(f"{start_year}-01-01")
    e_date = pd.Timestamp(f"{end_year}-12-31")
    timeline = pd.date_range(start=s_date, end=e_date, freq='B')
    
    permnos = sorted(df_raw['PERMNO'].dropna().unique().astype(int))
    matrix = pd.DataFrame(0, index=timeline, columns=permnos, dtype=np.int8)
    
    print("[Legacy] Constructing universe matrix...")
    for _, row in df_raw.iterrows():
        try:
            p = int(row['PERMNO'])
            s = row['start_dt']
            e = row['end_dt']
            if pd.isna(e): e = timeline[-1]
            
            valid_s = max(s, timeline[0])
            valid_e = min(e, timeline[-1])
            
            if valid_s <= valid_e:
                matrix.loc[valid_s:valid_e, p] = 1
        except ValueError:
            continue 
            
    return matrix

def process_github_presence(
    file_path: Path, 
    wrds_client: WRDSClient, 
    start_date: str
) -> pd.DataFrame:
    """
    Step 2: Process the GitHub/Wikipedia Presence Matrix.
    """
    print(f"[Modern] Processing GitHub data from {start_date}...")
    
    df_histconst = pd.read_csv(file_path, index_col=0)
    if "tickers" in df_histconst.columns:
        df_histconst = df_histconst.rename(columns={"tickers": "ticker"})
    
    df_histconst.index = pd.to_datetime(df_histconst.index)
    df_histconst.index.name = "date"  # <--- FORCE NAME TO 'date'
    df_histconst = df_histconst.loc[start_date:]
    df_histconst["ticker"] = df_histconst["ticker"].str.split(",")
    df_exploded = df_histconst.explode("ticker").reset_index()
    df_exploded["clean_ticker"] = df_exploded["ticker"].map(TICKER_CLEANING_MAP).fillna(df_exploded["ticker"])
    df_exploded = df_exploded.dropna(subset=["clean_ticker"])
    
    unique_tickers = df_exploded["clean_ticker"].unique().tolist()
    df_ids = wrds_client.get_ticker_to_permno_mapping(unique_tickers, active_after='2015-01-01')
    
    print("Merging Tickers to PERMNOs...")
    merged = pd.merge(
        left=df_exploded,
        right=df_ids,
        left_on="clean_ticker",
        right_on="ticker",
        how="inner"
    )
    
    valid_links = merged[
        (merged["date"] >= merged["namedt"]) & 
        (merged["date"] <= merged["nameenddt"])
    ].copy()
    
    valid_links = valid_links.drop_duplicates(subset=["date", "permno"])
    
    df_wiki_presence = (
        valid_links.assign(flag=1)
        .pivot(index="date", columns="permno", values="flag")
        .fillna(0)
        .astype(np.int8)
    )
    
    return df_wiki_presence

def fuse_presence_matrices(df_wrds: pd.DataFrame, df_wiki: pd.DataFrame, split_date: str) -> pd.DataFrame:
    """
    Step 3: Suture the WRDS history with the Wiki recent data.
    """
    mask_wrds = df_wrds.index < split_date
    df_past = df_wrds.loc[mask_wrds]

    mask_wiki = df_wiki.index >= split_date
    df_future = df_wiki.loc[mask_wiki]

    print(f"[Fusion] WRDS: {df_past.shape[0]} days | {df_past.shape[1]} stocks")
    print(f"[Fusion] Wiki:  {df_future.shape[0]} days | {df_future.shape[1]} stocks")

    df_combined = pd.concat([df_past, df_future], axis=0, sort=True)
    df_combined = df_combined.fillna(0).astype(np.int8)
    df_combined = df_combined.sort_index()

    n_wrds = df_past.shape[1]
    n_final = df_combined.shape[1]
    n_added = n_final - n_wrds

    print("--- Fusion Complete ---")
    print(f"Total Unique Assets: {n_final}")
    print(f"New Entrants (Post-{split_date}): {n_added}")
    print(f"Timeline: {df_combined.index.min().date()} to {df_combined.index.max().date()}")

    return df_combined

def build_full_presence_matrix(
    legacy_dir: Path, 
    github_file: Path, 
    wrds_client: WRDSClient
) -> pd.DataFrame:
    """
    Orchestrator Function.
    """
    SPLIT_DATE = '2023-01-01'

    df_legacy = parse_wrds_legacy_csvs(legacy_dir, start_year=2009, end_year=2023)
    df_recent = process_github_presence(github_file, wrds_client, start_date=SPLIT_DATE)
    df_fused = fuse_presence_matrices(df_legacy, df_recent, split_date=SPLIT_DATE)
    trading_dates = wrds_client.get_trading_dates(df_fused.index[0])
    current_dates = df_fused.index
    combined_dates = trading_dates.union(current_dates).sort_values()
    df_dense = df_fused.reindex(combined_dates).ffill()
    df_final = df_dense.reindex(trading_dates)
    df_final.index.name, df_final.columns.name = "date", "permno"
    
    return df_final