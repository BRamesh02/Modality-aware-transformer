import torch
import pandas as pd
from tqdm import tqdm

class WalkForwardEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    @torch.no_grad()
    def predict_fold(self, dataloader, fold_name="Test"):
        self.model.eval()
        all_preds = []
        
        for batch in tqdm(dataloader, desc=f"Inference {fold_name}"):
            # 1. Inputs
            x_num  = batch['x_num'].to(self.device)
            x_sent = batch['x_sent'].to(self.device)
            x_emb  = batch['x_emb'].to(self.device)
            y0     = batch['y_hist'][:, 0].to(self.device)
            
            # 2. Metadata
            dates = batch['date_forecast'] # List of strings
            permnos = batch['permno']      # Tensor of ints
            
            # 3. Predict
            # Shape: [Batch, Horizon]
            y_hats = self.model.predict(x_num, x_sent, x_emb, y0)
            
            # 4. CPU Extraction
            y_hats_np = y_hats.cpu().numpy()
            y_true_np = batch['y_future'].cpu().numpy()
            
            # 5. Build Records
            batch_size = x_num.size(0)
            horizon = y_hats.shape[1]
            
            for i in range(batch_size):
                p_val = permnos[i].item()
                d_val = dates[i]
                
                for h in range(horizon):
                    all_preds.append({
                        "date_forecast": d_val, # Date we made the decision
                        "permno": p_val,
                        "horizon": h + 1,
                        "pred": float(y_hats_np[i, h]),
                        "target": float(y_true_np[i, h]),
                        "fold": fold_name
                    })
                    
        return pd.DataFrame(all_preds)