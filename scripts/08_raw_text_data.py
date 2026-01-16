from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fnspid import text_functions as tf
from src.fnspid.store_data import store_raw_articles_parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest FNSPID (streaming) and store raw rows to Parquet chunks."
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/raw/fnspid",
        help="Output directory for raw_*.parquet",
    )
    p.add_argument(
        "--chunk-size", type=int, default=500_000, help="Rows per output parquet chunk"
    )
    p.add_argument(
        "--dataset", type=str, default="Zihan1004/FNSPID", help="HF dataset repo id"
    )
    p.add_argument("--split", type=str, default="train", help="Dataset split to stream")
    return p.parse_args()


def main() -> None:
    print("--- Step 8: Loading & Saving Raw FNSPID Text Data ---")

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, streaming=True)
    split = ds[args.split]

    split_raw = (
        split.map(tf.make_projector())  # keep useful columns
        .map(tf.build_text_record)  # parse dt_utc + clean strings
        .filter(tf.is_valid_record)  # keep minimally valid rows
    )

    total = store_raw_articles_parquet(
        iterable=split_raw,
        out_dir=out_dir,
        chunk_size=args.chunk_size,
        desc="Storing FNSPID raw",
    )

    print(f"\n Total rows written: {total}")
    print(f"SUCCESS: Saved Raw FNSPID text data to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
