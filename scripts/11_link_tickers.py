import sys
from pathlib import Path
from dotenv import load_dotenv

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent
sys.path.append(str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.fnspid.linking import (
    load_mapping_table,
    load_and_concat_parts,
    map_point_in_time,
    filter_by_universe,
    optimize_and_save,
)

BASE_DIR = PROJECT_ROOT

INPUT_DIR = BASE_DIR / "data" / "processed" / "fnspid"

OUTPUT_FILE = INPUT_DIR / "processed_and_linked_text_features.parquet"

MAPPING_FILE = BASE_DIR / "data" / "raw" / "crsp_ticker_map.parquet"
UNIVERSE_FILE = (
    BASE_DIR / "data" / "processed" / "numerical_data" / "sp500_universe.parquet"
)


def main():
    print(
        "--- Step 11: Linking Tickers from FNSPID Databse to Permnos from CRSP Database ---"
    )

    df_map = load_mapping_table(MAPPING_FILE)

    df_fnspid = load_and_concat_parts(INPUT_DIR, pattern="processed_*.parquet")

    df_linked = map_point_in_time(df_fnspid, df_map)

    df_linked = filter_by_universe(df_linked, UNIVERSE_FILE)

    match_rate = df_linked["permno"].notna().mean()
    print(f"\nFinal Match Rate: {match_rate:.2%}")

    ticker_col = "stock_symbol" if "stock_symbol" in df_linked.columns else "ticker"
    cols_to_print = ["date", ticker_col, "permno", "comnam"]
    existing_cols = [c for c in cols_to_print if c in df_linked.columns]

    print(df_linked[existing_cols].head())

    optimize_and_save(df_linked, OUTPUT_FILE)
    print(f"Saved linked file to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
