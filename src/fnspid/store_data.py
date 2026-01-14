from pathlib import Path
from typing import Iterable, Dict, Any
import pandas as pd
from tqdm import tqdm

def store_raw_articles_parquet(
    iterable: Iterable[Dict[str, Any]],
    out_dir: str | Path,
    chunk_size: int = 500_000,
    desc: str = "Storing raw articles"
) -> int:
    """
    Store raw article-level records into parquet files by chunks.
    Files are named raw_{chunk_id}.parquet (e.g., raw_0.parquet, raw_1.parquet).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buffer = []
    chunk_id = 0
    total_rows = 0

    for ex in tqdm(iterable, desc=desc, unit="rows"):
        buffer.append({
            "dt_utc": ex.get("dt_utc"),
            "stock_symbol": ex.get("stock_symbol"),
            "url": ex.get("url"),
            "publisher": ex.get("publisher"),
            "author": ex.get("author"),
            "title": ex.get("title"),

            # raw text fields
            "article": ex.get("article"),
            "textrank_summary": ex.get("textrank_summary"),
            "lexrank_summary": ex.get("lexrank_summary"),
            "lsa_summary": ex.get("lsa_summary"),
            "luhn_summary": ex.get("luhn_summary"),
        })

        if len(buffer) >= chunk_size:
            df = pd.DataFrame(buffer)
            # CHANGED: simplified filename to raw_{chunk_id}.parquet
            df.to_parquet(
                out_dir / f"raw_{chunk_id}.parquet",
                index=False
            )
            total_rows += len(df)
            buffer.clear()
            chunk_id += 1

    if buffer:
        df = pd.DataFrame(buffer)
        df.to_parquet(
            out_dir / f"raw_{chunk_id}.parquet",
            index=False
        )
        total_rows += len(df)

    return total_rows