from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import shutil

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.fnspid.text_functions import clean_str, norm_url, same_or_next_trading_day_nyse

IN_DIR = PROJECT_ROOT / "data" / "raw" / "fnspid"
OUT_DIR = PROJECT_ROOT / "data" / "preprocessed" / "fnspid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELETE_RAW = True  # Set to True only when everything is validated
RAW_GLOB = "raw_*.parquet"

BINS = [
    ("2009-01-01", "2012-12-31"),
    ("2013-01-01", "2014-12-31"),
    ("2015-01-01", "2016-12-31"),
    ("2017-01-01", "2018-12-31"),
    ("2019-01-01", "2020-12-31"),
    ("2021-01-01", "2024-12-31"),
]

KEEP_COLS = ["dt_utc", "stock_symbol", "url", "publisher", "title"]

FINAL_COLS = [
    "effective_date",
    "publication_date",
    "stock_symbol",
    "text",
    "text_source",
    "text_len",
    "publisher",
    "url",
]


def _safe_clean_series(s: pd.Series) -> pd.Series:
    return s.map(clean_str)


def _compute_text_len_words(s: pd.Series) -> pd.Series:
    return s.fillna("").str.split().str.len().astype("int32")


def _out_name(a: str, b: str) -> str:
    return f"preprocessed_{a[:4]}_{b[:4]}.parquet"


def main() -> None:
    print("--- Step 9: Cleaning Raw FNSPID Text Data ---")
    print(f"Reading from: {IN_DIR}")
    print(f"Writing to:   {OUT_DIR}")

    # Check if input directory exists
    if not IN_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {IN_DIR}")

    raw_files = sorted(IN_DIR.glob(RAW_GLOB))
    if not raw_files:
        raise FileNotFoundError(f"No files matching {RAW_GLOB} in {IN_DIR}")

    dfs = []
    total_in = 0

    for fp in tqdm(raw_files, desc="Reading raw files", unit="file"):
        df = pd.read_parquet(fp)

        cols = [c for c in KEEP_COLS if c in df.columns]
        df = df[cols].copy()
        for c in KEEP_COLS:
            if c not in df.columns:
                df[c] = None

        total_in += len(df)

        df["stock_symbol"] = _safe_clean_series(df["stock_symbol"])
        df["publisher"] = _safe_clean_series(df["publisher"])
        df["title"] = _safe_clean_series(df["title"])
        df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True, errors="coerce")
        df["url"] = df["url"].map(norm_url)

        dfs.append(df)

    print(f"Concatenating {len(dfs)} chunks...")
    all_df = pd.concat(dfs, ignore_index=True)

    pub_day = all_df["dt_utc"].dt.tz_convert("UTC").dt.floor("D")
    all_df["publication_date"] = pub_day.dt.date

    all_df["effective_date"] = all_df["publication_date"].map(
        lambda d: (
            same_or_next_trading_day_nyse(d).isoformat() if d is not None else None
        )
    )

    before = len(all_df)

    all_df = all_df.drop_duplicates(
        subset=["url", "stock_symbol", "dt_utc"], keep="first"
    )

    dropped_dups = before - len(all_df)

    all_df["text"] = all_df["title"]
    all_df["text_source"] = "title"
    all_df["text_len"] = _compute_text_len_words(all_df["text"])

    all_df = all_df[
        all_df["effective_date"].notna()
        & all_df["stock_symbol"].notna()
        & all_df["text"].notna()
        & (all_df["text_len"] >= 3)
    ].copy()

    all_df["publication_date"] = all_df["publication_date"].map(
        lambda d: d.isoformat() if d is not None else None
    )

    all_df = all_df[FINAL_COLS].copy()

    eff_dt = pd.to_datetime(all_df["effective_date"], errors="coerce").dt.floor("D")

    total_out = 0
    for a, b in tqdm(BINS, desc="Writing bins", unit="bin"):
        start = pd.Timestamp(a)
        end = pd.Timestamp(b)

        mask = (eff_dt >= start) & (eff_dt <= end)
        g = all_df.loc[mask].copy()
        if g.empty:
            continue

        g = g.sort_values(by=["effective_date", "stock_symbol"], kind="mergesort")

        out_path = OUT_DIR / _out_name(a, b)
        g.to_parquet(out_path, index=False, compression="snappy")
        total_out += len(g)

        print(f"Wrote {len(g):,} rows -> {out_path.name}")

    print("Done.")
    print(f"Input rows (raw total): {total_in:,}")
    print(f"Dropped duplicates:     {dropped_dups:,}")
    print(f"Output rows written:    {total_out:,}")
    print(f"Output dir: {OUT_DIR}")

    if DELETE_RAW and IN_DIR.exists():
        shutil.rmtree(IN_DIR)
        print(f"Deleted raw directory: {IN_DIR}")


if __name__ == "__main__":
    main()
