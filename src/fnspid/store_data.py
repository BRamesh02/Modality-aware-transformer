from pathlib import Path
from typing import Iterable, Dict, Any
import gdown

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

    Parameters
    ----------
    iterable : iterable of dict
        Stream yielding cleaned raw article dictionaries.
    out_dir : str or Path
        Output directory for parquet files.
    chunk_size : int
        Number of rows per parquet file.
    desc : str
        Progress bar description.

    Returns
    -------
    int
        Total number of rows written.
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
            df.to_parquet(
                out_dir / f"raw_{chunk_id:02d}.parquet",
                index=False
            )
            total_rows += len(df)
            buffer.clear()
            chunk_id += 1

    # last chunk
    if buffer:
        df = pd.DataFrame(buffer)
        df.to_parquet(
            out_dir / f"raw_{chunk_id:05d}.parquet",
            index=False
        )
        total_rows += len(df)

    return total_rows


def download_drive_files(file_map, out_dir, prefix="", sleep_sec=1.5, ext=".parquet"):
    """
    file_map:
      - list[str]
      - dict[str, str] (period -> id)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    if isinstance(file_map, list):
        for i, fid in enumerate(file_map):
            out = out_dir / f"{prefix}_{i:02d}{ext}"
            paths.append(_download_one(fid, out, sleep_sec))

    elif isinstance(file_map, dict):
        for key, fid in file_map.items():
            safe_key = key.replace("/", "-")
            out = out_dir / f"{prefix}_{safe_key}{ext}"
            paths.append(_download_one(fid, out, sleep_sec))

    return [p for p in paths if p is not None]


def _download_one(fid, out_path, sleep_sec):
    if out_path.exists():
        print(f"Skipping existing: {out_path.name}")
        return out_path

    print(f"Downloading {out_path.name}")
    url = f"https://drive.google.com/uc?id={fid}"

    try:
        gdown.download(url, str(out_path), quiet=False)
        time.sleep(sleep_sec)
        return out_path
    except Exception as e:
        print(f"Error downloading {fid}: {e}")
        return None
