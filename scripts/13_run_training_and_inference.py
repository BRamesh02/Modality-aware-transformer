import sys
from pathlib import Path
import pandas as pd

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.config import CONFIG
from src.utils.data_loader import load_and_merge_data
from src.training.runner import run_walk_forward


def main():
    print("Loading Main Data (This may take a minute)...")
    data_dir = PROJECT_ROOT / "data"
    df_main = load_and_merge_data(data_dir, start_date=CONFIG["start_date"], end_date=CONFIG["end_date"])
    
    print("\nStarting MAT Training Pipeline...")
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

    print("\nStarting Canonical Transformer Training Pipeline...")
    df_can = run_walk_forward(
        df_main=df_main,
        model_type="Canonical",
        config=CONFIG,
        project_root=PROJECT_ROOT
    )
    
    save_path_can = data_dir / "processed" / "predictions" / "canonical_walkforward.parquet"
    df_can.to_parquet(save_path_can)
    print(f"Canonical Results saved to {save_path_can}")

if __name__ == "__main__":
    main()