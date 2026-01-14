import torch
import pandas as pd
from tqdm import tqdm

class WalkForwardEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    @torch.no_grad()
    def predict_fold(self, dataloader, fold_name="Test"):
        """
        Runs inference on a fold using model.predict()
        """
        self.model.eval()
        all_preds = []
        
        for batch in tqdm(dataloader, desc=f"Inference ({fold_name})"):
            # 1. Inputs
            x_num  = batch['x_num'].to(self.device)
            x_sent = batch['x_sent'].to(self.device)
            x_emb  = batch['x_emb'].to(self.device)
            
            # 2. Autoregressive Start Value (y0)
            # Extracted from y_hist (index 0 is t, index 1 is t+1...)
            # For inference starting at t+1, we need y at t.
            y0 = batch['y_hist'][:, 0].to(self.device)
            
            # 3. Metadata (for DataFrame)
            dates = batch.get('date', [None]*x_num.size(0))
            permnos = batch.get('permno', [None]*x_num.size(0))
            
            # 4. Run Model-Specific Prediction Loop
            # This calls .predict() on whichever model instance was passed
            y_hats = self.model.predict(x_num, x_sent, x_emb, y0)
            
            # 5. Store Results
            y_hats_np = y_hats.cpu().numpy()
            y_true_np = batch['y_future'].cpu().numpy()
            
            for i in range(x_num.size(0)):
                for h in range(y_hats.shape[1]):
                    all_preds.append({
                        "date": dates[i],
                        "permno": permnos[i],
                        "horizon": h + 1,
                        "pred": y_hats_np[i, h],
                        "target": y_true_np[i, h],
                        "fold": fold_name
                    })
                    
        return pd.DataFrame(all_preds)