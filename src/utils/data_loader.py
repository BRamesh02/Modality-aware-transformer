import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def load_parquet(file_path):
    """Helper to load parquet and normalize dates."""
    if file_path.exists():
        df = pd.read_parquet(file_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df
    print(f"Warning: {file_path} not found.")
    return None


def load_and_merge_data(data_dir: Path, start_date=str, end_date=str):
    """
    Loads Target, Market, Ratios, Macro, and Text data.
    Merges them into a Point-in-Time dataframe and handles missing embeddings.

    Assumes data_dir points to the root 'data' folder.
    """
    print(f"Reading data from: {data_dir}")

    print("Loading datasets...")

    df_target = load_parquet(
        data_dir / "processed/numerical_data/target.parquet"
    ).astype({"target": "float32"})
    df_market = load_parquet(
        data_dir / "processed/numerical_data/features_market.parquet"
    )
    df_ratios = load_parquet(
        data_dir / "processed/numerical_data/features_ratios.parquet"
    )
    df_macro = load_parquet(
        data_dir / "processed/numerical_data/features_macro.parquet"
    )
    df_text = load_parquet(
        data_dir / "processed/fnspid/process_and_linked_text_features.parquet"
    )

    print("\n--- Merging Data ---")
    df_num = df_target.copy()

    for df_feat, name in [(df_market, "Market"), (df_ratios, "Ratios")]:
        if df_feat is not None:
            df_num = df_num.merge(df_feat, on=["date", "permno"], how="left")
            print(f"Merged {name}: {df_num.shape}")

    if df_macro is not None:
        df_num = df_num.merge(df_macro, on="date", how="left")
        print(f"Merged Macro: {df_num.shape}")
    else:
        print("WARNING : Macro data is missing!")

    if df_text.duplicated(subset=["date", "permno"]).any():
        print("Warning: Duplicate text rows found. Keeping first.")
        df_text = df_text.drop_duplicates(subset=["date", "permno"])

    keep_cols = [
        "date",
        "permno",
        "emb_mean",
        "sent_score_mean",
        "sent_pos_mean",
        "sent_neg_mean",
        "log_n_news",
        "sent_score_std",
    ]

    df_text_keep = df_text[keep_cols].copy()
    df_text_keep["has_news"] = df_text_keep["sent_score_mean"].notna().astype(int)

    df_main = df_num.merge(df_text_keep, on=["date", "permno"], how="left")
    print(f"Merged Text: {df_main.shape}")

    print("Filling NaN values...")
    fill_values = {
        "sent_score_mean": 0.0,
        "sent_pos_mean": 0.0,
        "sent_neg_mean": 0.0,
        "sent_score_std": 0.0,
        "log_n_news": 0.0,
        "has_news": 0.0,
    }
    df_main = df_main.fillna(fill_values)
    df_main = df_main.astype({"has_news": "float32"})

    null_emb_mask = df_main["emb_mean"].isna()
    if null_emb_mask.any():
        zero_vec = np.zeros(768, dtype=np.float32)
        emb_fill_values = pd.Series(
            [zero_vec] * null_emb_mask.sum(),
            index=df_main.index[null_emb_mask],
            dtype=object,
        )
        df_main.loc[null_emb_mask, "emb_mean"] = emb_fill_values

    df_main = df_main.astype({"permno": "float32"})

    print(f"Keeping records between {start_date} and {end_date}...")
    df_main = df_main[df_main["date"].between(start_date, end_date)].copy()

    print("Done! Final Data Shape:", df_main.shape)
    return df_main


def prepare_scaled_fold(
    df_main: pd.DataFrame, num_cols: list[str], split, buffer_days=int
):
    """
    Handles the buffering and scaling for a single Walk-Forward fold.

    Logic:
    1. Extract 'Warm' DataFrames (Official Period + 90 days prior context).
    2. Fit Scaler on the FULL 'Train Warm' set (Buffer + Official Train).
    3. Transform Val/Test using that same scaler.
    """
    train_start, train_end = split["train"]
    val_start, val_end = split["val"]
    test_start, test_end = split["test"]

    buffer_delta = pd.Timedelta(days=buffer_days)
    min_data_date = pd.to_datetime(df_main["date"].min())

    def get_raw_warm_df(start_date, end_date):
        s_date = pd.to_datetime(start_date)
        warm_start = max(min_data_date, s_date - buffer_delta)

        mask = df_main["date"].between(warm_start, end_date)
        return df_main[mask].copy()

    df_train_warm = get_raw_warm_df(train_start, train_end)
    df_val_warm = get_raw_warm_df(val_start, val_end)
    df_test_warm = get_raw_warm_df(test_start, test_end)

    scaler = StandardScaler()

    df_train_warm[num_cols] = scaler.fit_transform(df_train_warm[num_cols])
    if len(df_val_warm) > 0:
        df_val_warm[num_cols] = scaler.transform(df_val_warm[num_cols])

    if len(df_test_warm) > 0:
        df_test_warm[num_cols] = scaler.transform(df_test_warm[num_cols])

    return df_train_warm, df_val_warm, df_test_warm
