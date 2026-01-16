import time
import gdown
from pathlib import Path


def download_drive_files(file_map, out_dir, prefix="", sleep_sec=1.5, ext=".parquet"):
    """
    Download Google Drive files.

    Parameters
    ----------
    file_map : list[str] or dict[str, str]
        List of file IDs or mapping period -> file ID.
    out_dir : Path
        Output directory.
    prefix : str
        Filename prefix.
    sleep_sec : float
        Pause between downloads.
    ext : str
        File extension.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    if isinstance(file_map, list):
        if len(file_map) == 1:
            out = out_dir / f"{prefix}{ext}"
            p = _download_one(file_map[0], out, sleep_sec)
            return [p] if p is not None else []

        for i, fid in enumerate(file_map):
            out = out_dir / f"{prefix}_{i}{ext}"
            paths.append(_download_one(fid, out, sleep_sec))

    elif isinstance(file_map, dict):
        for key, fid in file_map.items():
            safe_key = key.replace("/", "_")
            if prefix:
                out_name = f"{prefix}_{safe_key}{ext}"
            else:
                out_name = f"{safe_key}{ext}"  # Just the key (e.g., "target.parquet")

            out = out_dir / out_name
            paths.append(_download_one(fid, out, sleep_sec))
    else:
        raise TypeError("file_map must be list or dict")

    return [p for p in paths if p is not None]


def _download_one(fid, out_path: Path, sleep_sec: float):
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
