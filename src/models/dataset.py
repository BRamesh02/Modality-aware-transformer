import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class FinancialDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        min_date=None,
        max_date=None,
        forecast_horizon=int,
        use_emb: bool = True,
    ):
        self.window_size = window_size
        self.H = forecast_horizon
        self.use_emb = use_emb

        # Sort is critical for the "t-1" indexing to work
        self.df = df.sort_values(["permno", "date"]).reset_index(drop=True)

        # Columns Setup
        exclude_cols = [
            "date",
            "permno",
            "target",
            "emb_mean",
            "sent_score_mean",
            "sent_pos_mean",
            "sent_neg_mean",
            "sent_score_std",
            "log_n_news",
        ]
        self.num_cols = [c for c in self.df.columns if c not in exclude_cols]
        self.text_scalar_cols = [
            "sent_score_mean",
            "sent_pos_mean",
            "sent_neg_mean",
            "sent_score_std",
            "log_n_news",
        ]

        # Convert to Tensors
        self.data_num = torch.tensor(self.df[self.num_cols].values.astype(np.float32))
        self.data_sent = torch.tensor(
            self.df[self.text_scalar_cols].values.astype(np.float32)
        )
        self.data_target = torch.tensor(self.df["target"].values.astype(np.float32))

        # Embeddings
        self.data_emb = None
        self.zero_emb = torch.zeros((window_size, 768), dtype=torch.float32)
        if self.use_emb and "emb_mean" in self.df.columns:
            emb_values = np.stack(self.df["emb_mean"].values).astype(np.float32)
            self.data_emb = torch.tensor(emb_values)

        # --- Valid Index Discovery ---
        # We need to find valid windows that don't cross stock boundaries
        dates = pd.to_datetime(self.df["date"])
        min_ts = pd.Timestamp(min_date) if min_date else dates.min()
        max_ts = pd.Timestamp(max_date) if max_date else dates.max()

        self.indices = []
        permnos = self.df["permno"].values

        # Identify where stocks change to avoid mixing data
        change_points = np.where(permnos[:-1] != permnos[1:])[0] + 1
        start_points = np.concatenate(([0], change_points))
        end_points = np.concatenate((change_points, [len(permnos)]))

        for start, end in zip(start_points, end_points):
            # To predict H steps ahead, we need safe indices up to i + T + H
            # Valid start indices `i`:
            valid_starts = range(start, end - (self.window_size + self.H))

            for i in valid_starts:
                # `t` is the index of the last input day (The "Forecast Date")
                t = i + self.window_size - 1

                # Check if the TARGET date (t+1, which is at index `t` in your shifted logic)
                # falls within the requested range.
                # Note: dates[t] is the date of row `t`. Since row `t` holds target t+1,
                # checking dates[t] is effectively checking the date of the target?
                # Actually, usually 'date' col is the input date.
                # If date col is t, and target is t+1, we want to filter by the date of the target.
                # Let's trust dates[t] is the "Forecast Date" and we filter by that.

                current_date = dates[t]
                # Logic: We only want predictions where the FORECAST DATE is within range
                if min_ts <= current_date <= max_ts:
                    self.indices.append(i)

        print(
            f"Dataset Ready. Samples: {len(self.indices)} (Filtered by {min_date} to {max_date})"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        T = self.window_size
        H = self.H

        # 1. Inputs (Indices i to i+T)
        x_num = self.data_num[i : i + T]
        x_sent = self.data_sent[i : i + T]

        x_emb = self.zero_emb
        if self.data_emb is not None:
            x_emb = self.data_emb[i : i + T]

        # 2. Targets & History
        # t is the index of the last input day
        t = i + T - 1

        # y_future: The ground truth targets.
        # Since df['target'] at row `t` is Return_{t+1}, we start at `t`.
        y_future = self.data_target[t : t + H]

        # y_hist: The past returns for the decoder.
        # We need Return_{t}. Since df['target'] at `t-1` is Return_{t}, we start at `t-1`.
        y_hist = self.data_target[t - 1 : t - 1 + H]

        # 3. Metadata
        # We explicitly grab the "Forecast Date" (Date at row t)
        date_forecast = str(self.df.iloc[t]["date"])
        permno_val = int(self.df.iloc[i]["permno"])

        return {
            "x_num": x_num,
            "x_sent": x_sent,
            "x_emb": x_emb,
            "y_hist": y_hist,
            "y_future": y_future,
            "date_forecast": date_forecast,
            "permno": permno_val,
        }


def get_annual_splits(
    df,
    start_year: int,
    train_years: int,
    val_years: int,
    test_years: int,
    end_year: int = None,
):
    """
    Generates the Walk-Forward years.

    Args:
        end_year: The final year allowed for the 'test' set.
                  If None, it defaults to the last year present in the DataFrame.
    """
    years = sorted(df["date"].dt.year.unique())
    max_data_year = max(years)

    limit_year = end_year if end_year is not None else max_data_year

    splits = []

    current_year = start_year
    while True:
        train_end = current_year + train_years - 1
        val_end = train_end + val_years
        test_end = val_end + test_years

        if test_end > limit_year or test_end > max_data_year:
            break

        splits.append(
            {
                "year": test_end,
                "train": (f"{current_year}-01-01", f"{train_end}-12-31"),
                "val": (f"{train_end+1}-01-01", f"{val_end}-12-31"),
                "test": (f"{val_end+1}-01-01", f"{test_end}-12-31"),
            }
        )
        current_year += 1

    return splits
