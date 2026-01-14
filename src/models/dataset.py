import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class FinancialDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int = 60, min_date=None, max_date=None, forecast_horizon=1):
        """
        Args:
            df: DataFrame containing Features + Context Buffer (past data).
            min_date: (str) The earliest date allowed for a TARGET. 
                      Windows ending before this are discarded (used only for context).
            max_date: (str) The latest date allowed for a TARGET.
        """
        self.window_size = window_size
        self.H = forecast_horizon
        
        self.df = df.sort_values(["permno", "date"]).reset_index(drop=True)

        exclude_cols = [
            "date", "permno", "target", "emb_mean", 
            "sent_score_mean", "sent_pos_mean", "sent_neg_mean", 
            "sent_score_std", "log_n_news",
        ]
        self.num_cols = [c for c in self.df.columns if c not in exclude_cols]
        self.text_scalar_cols = ["sent_score_mean", "sent_pos_mean", "sent_neg_mean", "sent_score_std", "log_n_news"]
        
        print(f"Numerical Features ({len(self.num_cols)}): {self.num_cols}")
        print(f"Text Features bar Embedding Vector ({len(self.text_scalar_cols)}): {self.text_scalar_cols}")

        print("Converting to PyTorch Tensors...")

        self.data_num = torch.tensor(self.df[self.num_cols].values.astype(np.float32))
        
        if "emb_mean" in self.df.columns:
            emb_values = np.stack(self.df["emb_mean"].values).astype(np.float32)
        else:
            emb_values = np.zeros((len(self.df), 768), dtype=np.float32)
            
        # scalar_values = self.df[self.text_scalar_cols].values.astype(np.float32)
        # self.data_text = torch.tensor(np.concatenate([emb_values, scalar_values], axis=1))
        # self.data_target = torch.tensor(self.df["target"].values.astype(np.float32))

        # Embeddings: [N, 768]
        self.data_emb = torch.tensor(emb_values)  # already float32

        # Sentiment scalars: [N, n_sent]
        sent_values = self.df[self.text_scalar_cols].values.astype(np.float32)
        self.data_sent = torch.tensor(sent_values)

        self.data_target = torch.tensor(self.df["target"].values.astype(np.float32))

        dates = pd.to_datetime(self.df['date'])
        
        min_ts = pd.Timestamp(min_date) if min_date else dates.min()
        max_ts = pd.Timestamp(max_date) if max_date else dates.max()

        self.indices = []
        permnos = self.df['permno'].values
        change_points = np.where(permnos[:-1] != permnos[1:])[0] + 1
        start_points = np.concatenate(([0], change_points))
        end_points = np.concatenate((change_points, [len(permnos)]))
        
        for start, end in zip(start_points, end_points):
            n_rows = end - start
            if n_rows > window_size+self.H-1:
                # Potential start indices
                # Input Window: [i : i+60]
                # Target Index: i+60-1 (The last day of the window)
                # valid_starts = range(start, end - window_size + 1)
                valid_starts = range(start+1, end - (window_size+self.H -1)+1)
                
                for i in valid_starts:
                    #target_date = dates[i + window_size - 1]
                    target_date = dates[i + window_size + self.H - 2]
                    if min_ts <= target_date <= max_ts:
                        self.indices.append(i)

        print(f"Dataset Ready. Samples: {len(self.indices)} (Filtered by {min_date} to {max_date})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        T = self.window_size
        H = self.H
        
        x_num = self.data_num[i : i + T]    # [T, F]
        # x_text = self.data_text[i : i + self.window_size]
        x_sent = self.data_sent[i : i + T]  # [T, 5]
        x_emb  = self.data_emb[i : i + T]   # [T, 768]
        # y = self.data_target[i + self.window_size - 1]

        # Time index
        t = i + T - 1                              # dernier jour connu dans la fenêtre
        # y_future = ce que tu veux prédire (H pas dans le futur)
        y_future = self.data_target[t : t + H]   # [H] = y_{t+1}..y_{t+H}

        # y_hist = ce que tu donnes au decoder (teacher forcing)
        # [y_t, y_{t+1}, ..., y_{t+H-1}] (shift d'un pas)
        y_hist = self.data_target[t - 1 : t - 1 + H]    # [H] = r_t..r_{t+H-1}
        
        # Metadata for evaluation
        date_val = str(self.df.iloc[i + T + H - 2]['date'])
        permno_val = int(self.df.iloc[i]['permno'])
        
        # return {"x_num": x_num, "x_text": x_text, "y": y}
        # return {"x_num": x_num, "x_sent": x_sent, "x_emb": x_emb, "y": y}
        return {
            "x_num": x_num,
            "x_sent": x_sent,
            "x_emb": x_emb,
            "y_hist": y_hist,      
            "y_future": y_future,
            "date": date_val,
            "permno": permno_val,
        }


def get_annual_splits(df, start_year=2010, end_year=None, train_years=5, val_years=1, test_years=1):
    """
    Generates the Walk-Forward years.
    
    Args:
        end_year: The final year allowed for the 'test' set. 
                  If None, it defaults to the last year present in the DataFrame.
    """
    years = sorted(df['date'].dt.year.unique())
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
            
        splits.append({
            'year': test_end,
            'train': (f"{current_year}-01-01", f"{train_end}-12-31"),
            'val':   (f"{train_end+1}-01-01", f"{val_end}-12-31"),
            'test':  (f"{val_end+1}-01-01", f"{test_end}-12-31")
        })
        current_year += 1
        
    return splits


