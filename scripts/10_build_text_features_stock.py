from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.fnspid.bert_features import (  
    extract_bin_suffix,
    process_bin_to_stock_date_features,
)

IN_DIR = project_root / "data" / "fnspid_preprocessed"
OUT_DIR = project_root / "data" / "fnspid_features_text_stock_date"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 512
MAX_LEN = 64
READ_COLS = ["effective_date", "stock_symbol", "text"]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
print("Device:", DEVICE)

def main() -> None:
    files = sorted(IN_DIR.glob("preprocessed_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No preprocessed_*.parquet in {IN_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    clf_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
    enc_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    for f in tqdm(files, desc="Bins → stock×date features", unit="file"):
        suffix = extract_bin_suffix(f.name)
        out_file = OUT_DIR / f"text_features_stock_date_{suffix}.parquet"

        process_bin_to_stock_date_features(
            infile=f,
            outfile=out_file,
            tokenizer=tokenizer,
            clf_model=clf_model,
            enc_model=enc_model,
            read_cols=READ_COLS,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            max_len=MAX_LEN,
        )

    print("Done. Output in:", OUT_DIR)


if __name__ == "__main__":
    main()