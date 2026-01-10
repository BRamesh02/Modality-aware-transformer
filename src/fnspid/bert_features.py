# src/fnspid/text_features.py
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch


def clean_text_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x / norms


def stack_emb_column(emb_series: pd.Series) -> np.ndarray:
    return np.stack(emb_series.to_numpy(), axis=0).astype(np.float32)


def extract_bin_suffix(filename: str) -> str:
    m = re.search(r"preprocessed_(\d{4}_\d{4})\.parquet$", filename)
    if not m:
        raise ValueError(f"Unexpected filename format: {filename}")
    return m.group(1)


def process_bin_to_stock_date_features(
    infile: Path,
    outfile: Path,
    tokenizer,
    clf_model,
    enc_model,
    *,
    read_cols: list[str],
    device: str,
    batch_size: int,
    max_len: int,
) -> None:
    if outfile.exists():
        return

    df = pd.read_parquet(infile, columns=read_cols)
    df["text"] = clean_text_series(df["text"])
    df = df[df["text"].str.len() > 0].copy()

    if len(df) == 0:
        empty = pd.DataFrame(columns=[
            "stock_symbol", "effective_date",
            "n_news", "log_n_news",
            "sent_score_mean", "sent_score_std",
            "sent_pos_mean", "sent_neg_mean", "sent_neu_mean",
            "sent_score_sum", "sent_pos_sum", "sent_neg_sum", "sent_neu_sum",
            "emb_mean",
        ])
        empty.to_parquet(outfile, index=False)
        return

    texts = df["text"].tolist()
    n = len(texts)

    sent_neg = np.empty(n, dtype=np.float32)
    sent_neu = np.empty(n, dtype=np.float32)
    sent_pos = np.empty(n, dtype=np.float32)
    emb_all  = np.empty((n, 768), dtype=np.float32)

    clf_model.eval()
    enc_model.eval()
    use_autocast = (device == "cuda")

    for start in tqdm(range(0, n, batch_size), desc=f"{infile.name}", unit="batch", leave=False):
        batch_texts = texts[start:start + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            out_clf = clf_model(**enc)
            probs = torch.softmax(out_clf.logits, dim=1).cpu().numpy()
            bsz = probs.shape[0]
            sent_neg[start:start + bsz] = probs[:, 0]
            sent_neu[start:start + bsz] = probs[:, 1]
            sent_pos[start:start + bsz] = probs[:, 2]

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out_enc = enc_model(**enc)
                    pooled = mean_pooling(out_enc.last_hidden_state, enc["attention_mask"])
            else:
                out_enc = enc_model(**enc)
                pooled = mean_pooling(out_enc.last_hidden_state, enc["attention_mask"])

            pooled = pooled.cpu().float().numpy()
            pooled = l2_normalize(pooled)
            emb_all[start:start + bsz] = pooled

    df_sig = df[["stock_symbol", "effective_date"]].copy()
    df_sig["sent_neg"] = sent_neg
    df_sig["sent_neu"] = sent_neu
    df_sig["sent_pos"] = sent_pos
    df_sig["sent_score"] = df_sig["sent_pos"] - df_sig["sent_neg"]
    df_sig["emb"] = [row.astype(np.float16) for row in emb_all]

    g = df_sig.groupby(["stock_symbol", "effective_date"], sort=True)

    out = g.agg(
        n_news=("sent_score", "size"),
        sent_score_mean=("sent_score", "mean"),
        sent_score_std=("sent_score", "std"),
        sent_score_sum=("sent_score", "sum"),
        sent_pos_mean=("sent_pos", "mean"),
        sent_neg_mean=("sent_neg", "mean"),
        sent_neu_mean=("sent_neu", "mean"),
        sent_pos_sum=("sent_pos", "sum"),
        sent_neg_sum=("sent_neg", "sum"),
        sent_neu_sum=("sent_neu", "sum"),
    ).reset_index()

    out["log_n_news"] = np.log1p(out["n_news"])
    out["sent_score_std"] = out["sent_score_std"].fillna(0.0)

    rows = []
    keys = []
    for (sym, d), sub in tqdm(g, desc=f"emb_mean {infile.name}", unit="group", leave=False):
        E = stack_emb_column(sub["emb"])
        mu = E.mean(axis=0).astype(np.float16)
        rows.append(mu)
        keys.append((sym, d))

    emb_df = pd.DataFrame({
        "stock_symbol": [k[0] for k in keys],
        "effective_date": [k[1] for k in keys],
        "emb_mean": rows,
    })

    out = out.merge(emb_df, on=["stock_symbol", "effective_date"], how="left")

    col_order = [
        "stock_symbol", "effective_date",
        "n_news", "log_n_news",
        "sent_score_mean", "sent_score_std",
        "sent_pos_mean", "sent_neg_mean", "sent_neu_mean",
        "sent_score_sum", "sent_pos_sum", "sent_neg_sum", "sent_neu_sum",
        "emb_mean",
    ]
    out = out[col_order]

    out.to_parquet(outfile, index=False)