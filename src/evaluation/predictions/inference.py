import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

class WalkForwardEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    @torch.no_grad()
    def predict_fold(self, dataloader, fold_name="Test"):
        self.model.eval()
        all_dfs = [] 
        
        use_non_blocking = getattr(dataloader, 'pin_memory', False)
        
        for batch in tqdm(dataloader, desc=f"Inference {fold_name}"):
            x_num  = batch['x_num'].to(self.device, non_blocking=use_non_blocking)
            x_sent = batch['x_sent'].to(self.device, non_blocking=use_non_blocking)
            
            x_emb = batch['x_emb']
            if x_emb is not None:
                x_emb = x_emb.to(self.device, non_blocking=use_non_blocking)
            
            y_teacher_forcing = batch['y_hist'].to(self.device, non_blocking=use_non_blocking)
            y0_val = y_teacher_forcing[:, 0]
            
            dates = batch['date_forecast'] 
            permnos = batch['permno']
            
            y_hats = self.model.predict(x_num, x_sent, x_emb, y0=y0_val)
            
            y_hats_np = y_hats.cpu().numpy()
            y_true_np = batch['y_future'].cpu().numpy()
            
            batch_size, horizon = y_hats_np.shape
            
            if isinstance(permnos, torch.Tensor):
                permnos_np = permnos.cpu().numpy()
            else:
                permnos_np = np.array(permnos)

            if horizon == 1:
                df_batch = pd.DataFrame({
                    "date_forecast": dates,
                    "permno": permnos_np,
                    "horizon": 1,
                    "pred": y_hats_np.flatten(),
                    "target": y_true_np.flatten(),
                    "fold": fold_name
                })
            else:
                df_batch = pd.DataFrame({
                    "date_forecast": np.repeat(dates, horizon),
                    "permno": np.repeat(permnos_np, horizon),
                    "horizon": np.tile(np.arange(1, horizon + 1), batch_size),
                    "pred": y_hats_np.flatten(),
                    "target": y_true_np.flatten(),
                    "fold": fold_name
                })
            
            all_dfs.append(df_batch)
            
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()