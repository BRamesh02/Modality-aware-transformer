# run_inference.py
import sys
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/content/drive/MyDrive/MAT_Weighted_Loss/Modality-aware-transformer")
sys.path.append(str(PROJECT_ROOT))

from src.utils.data_loader import prepare_scaled_fold, load_and_merge_data
from src.models.dataset import FinancialDataset
from src.models.datasets import get_annual_splits
from src.evaluation.predictions.inference import WalkForwardEvaluator
from src.training.runner import get_model_instance  # Reuse the Factory!

CONFIG = {
    'window_size': 60,
    'forecast_horizon': 1,
    'batch_size': 4096,
    'use_emb': True,
    'start_year': 2017,
    'end_year': 2023
}

def run_pure_inference(df_main, model_type):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔎 Running Inference Only for {model_type}...")
    
    models_dir = PROJECT_ROOT / "models" / model_type
    all_preds = []
    
    # 1. Get Splits (Same logic as training)
    all_splits = get_annual_splits(df_main, start_year=2010, end_year=CONFIG['end_year'], train_years=5)
    target_splits = [s for s in all_splits if CONFIG['start_year'] <= s['year'] <= CONFIG['end_year']]
    
    # Feature Config (Replicated from runner)
    exclude_cols = ["date", "permno", "target", "emb_mean", "sent_score_mean", "sent_pos_mean", "sent_neg_mean", "sent_score_std", "log_n_news"]
    all_num_cols = [c for c in df_main.columns if c not in exclude_cols]
    scale_cols = [c for c in all_num_cols if c != "has_news"]

    for split in target_splits:
        year = split['year']
        weights_path = models_dir / f"{model_type.lower()}_best_{year}.pt"
        
        if not weights_path.exists():
            print(f"⚠️ Weights not found for {year}: {weights_path}. Skipping.")
            continue
            
        print(f"   Loading {year} weights...")
        
        # Prepare Data
        _, _, df_test = prepare_scaled_fold(df_main, scale_cols, split, buffer_days=90)
        
        test_ds = FinancialDataset(
            df_test, 
            window_size=CONFIG['window_size'], 
            forecast_horizon=CONFIG['forecast_horizon'],
            min_date=split['test'][0], 
            max_date=split['test'][1]
        )
        
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
        
        # Load Model
        model_params = {
            'num_input_dim': len(all_num_cols),
            'n_sent': 5,
            'd_model': 128,
            'forecast_horizon': CONFIG['forecast_horizon'],
            'use_emb': CONFIG['use_emb']
        }
        
        model = get_model_instance(model_type, model_params, device)
        model.load_state_dict(torch.load(weights_path))
        
        # Predict
        evaluator = WalkForwardEvaluator(model, device)
        df_pred = evaluator.predict_fold(test_loader, fold_name=f"{model_type}_{year}")
        all_preds.append(df_pred)
        
    return pd.concat(all_preds, axis=0) if all_preds else pd.DataFrame()

def main():
    print("🔄 Loading Data...")
    df_main = load_and_merge_data(PROJECT_ROOT / "data", start_date="2010-01-01", end_date="2023-12-31")
    
    # Run Inference Only
    df_mat = run_pure_inference(df_main, "MAT")
    df_mat.to_parquet(PROJECT_ROOT / "data/processed/predictions/mat_inference_repro.parquet")
    
    df_can = run_pure_inference(df_main, "Canonical")
    df_can.to_parquet(PROJECT_ROOT / "data/processed/predictions/canonical_inference_repro.parquet")
    
    print("✅ Reproduction Complete.")

if __name__ == "__main__":
    main()