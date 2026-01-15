# run_training.py
import sys
from pathlib import Path
import pandas as pd

from src.config import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT))

from src.utils.data_loader import load_and_merge_data
from src.training.runner import run_walk_forward

# --- CONFIGURATION ---
CONFIG = {
    'window_size': 60,
    'forecast_horizon': 1,
    'batch_size': 4096,
    'use_emb': True,
    'start_year': 2017,
    'end_year': 2023
}

def main():
    print("🔄 Loading Main Data (This may take a minute)...")
    data_dir = PROJECT_ROOT / "data"
    df_main = load_and_merge_data(data_dir, start_date="2010-01-01", end_date="2023-12-31")
    
    # 1. Train & Predict MAT
    print("\n🔹 Starting MAT Training Pipeline...")
    df_mat = run_walk_forward(
        df_main=df_main,
        start_year=CONFIG['start_year'],
        end_year=CONFIG['end_year'],
        model_type="MAT",
        config=CONFIG,
        project_root=PROJECT_ROOT
    )
    
    # Save Predictions
    save_path_mat = data_dir / "processed" / "predictions" / "mat_walkforward.parquet"
    save_path_mat.parent.mkdir(parents=True, exist_ok=True)
    df_mat.to_parquet(save_path_mat)
    print(f"✅ MAT Results saved to {save_path_mat}")

    # 2. Train & Predict Canonical
    print("\n🔹 Starting Canonical Transformer Training Pipeline...")
    df_can = run_walk_forward(
        df_main=df_main,
        start_year=CONFIG['start_year'],
        end_year=CONFIG['end_year'],
        model_type="Canonical",
        config=CONFIG,
        project_root=PROJECT_ROOT
    )
    
    save_path_can = data_dir / "processed" / "predictions" / "canonical_walkforward.parquet"
    df_can.to_parquet(save_path_can)
    print(f"✅ Canonical Results saved to {save_path_can}")

if __name__ == "__main__":
    main()