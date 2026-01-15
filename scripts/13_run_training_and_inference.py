import sys
import os
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.config import CONFIG
from src.utils.data_loader import load_and_merge_data
from src.training.runner import run_walk_forward

def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    print("Loading Main Data (This may take a minute)...")
    data_dir = PROJECT_ROOT / "data"
    df_main = load_and_merge_data(data_dir, start_date=CONFIG["start_date"], end_date=CONFIG["end_date"])
    
    print("\n" + "="*50)
    print("Seeding & Starting MAT Training...")
    print("="*50)
    seed_everything(42)
    
    df_mat = run_walk_forward(
        df_main=df_main,
        model_type="MAT",
        config=CONFIG,
        project_root=PROJECT_ROOT
    )
    
    save_path_mat = data_dir / "processed" / "predictions" / "mat_walkforward.parquet"
    save_path_mat.parent.mkdir(parents=True, exist_ok=True)
    df_mat.to_parquet(save_path_mat)
    print(f"MAT Results saved to {save_path_mat}")

    print("\n" + "="*50)
    print("Re-Seeding & Starting Canonical Training...")
    print("="*50)
    seed_everything(42)
    
    df_can = run_walk_forward(
        df_main=df_main,
        model_type="Canonical",
        config=CONFIG,
        project_root=PROJECT_ROOT
    )
    
    save_path_can = data_dir / "processed" / "predictions" / "canonical_walkforward.parquet"
    df_can.to_parquet(save_path_can)
    print(f"Canonical Results saved to {save_path_can}")

    print("\nFull Inference Run Complete.")

if __name__ == "__main__":
    main()