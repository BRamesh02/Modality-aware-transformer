import sys
import json
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.drive_downloads import download_drive_files

IDS_PATH = project_root / "config" / "drive_ids.json"
DATA_DIR = project_root / "data"

SLEEP_SEC = 1.5
EXT = ".parquet"
VALID_OPTIONS = {
    "raw",
    "preprocessed",
    "fnspid_features_text_stock_date",
    "linked_text_data",
    "all",
}

def main():
    with open(IDS_PATH, "r") as f:
        drive_ids = json.load(f)

    print("\nWhat do you want to download?")
    print("Options:", ", ".join(sorted(VALID_OPTIONS)))
    choice = input(">>> ").strip()

    if choice not in VALID_OPTIONS:
        raise ValueError(f"Option invalide: {choice}")

    if choice == "all":
        for key, value in drive_ids.items():
            print(f"\nDownloading {key}")
            out_dir = DATA_DIR / key
            download_drive_files(
                value,
                out_dir,
                prefix=key,
                sleep_sec=SLEEP_SEC,
                ext=EXT,
            )
    else:
        if choice not in drive_ids:
            raise KeyError(f"Clé '{choice}' absente du JSON")

        print(f"\nDownloading {choice}")
        out_dir = DATA_DIR / choice
        download_drive_files(
            drive_ids[choice],
            out_dir,
            prefix=choice,
            sleep_sec=SLEEP_SEC,
            ext=EXT,
        )

    print("\nTéléchargement terminé")


if __name__ == "__main__":
    main()