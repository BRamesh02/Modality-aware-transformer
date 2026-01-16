import sys
from pathlib import Path
import pandas as pd

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.config import CONFIG
from src.evaluation.predictions.evaluator import ModelEvaluator

def main():
    print("--- Step 15: Evaluating Models' Predictions ---")
    
    data_dir = PROJECT_ROOT / "data" / "processed" / "predictions"
    mat_path = data_dir / "mat_walkforward.parquet"
    can_path = data_dir / "canonical_walkforward.parquet"
    
    if not mat_path.exists() or not can_path.exists():
        print(f"\nCRITICAL ERROR: Prediction files not found in {data_dir}")
        print("   Please run 'scripts/13_run_training.py' or '14_run_inference_only.py' first.")
        return

    print(f"\nLoading predictions...")
    try:
        df_mat = pd.read_parquet(mat_path)
        df_can = pd.read_parquet(can_path)
        print(f"   Loaded MAT predictions: {len(df_mat)} rows")
        print(f"   Loaded Canonical predictions: {len(df_can)} rows")
    except Exception as e:
        print(f"   Error loading parquet files: {e}")
        return
    
    evaluator = ModelEvaluator(PROJECT_ROOT)
    
    primary_h = 1 
    if "primary_eval_horizon" in CONFIG:
        primary_h = CONFIG["primary_eval_horizon"]
    
    print("\n" + "-"*40)
    print(f"STEP 1: Individual Model Reports (Focus: H={primary_h})")
    print("-" * 40)
    
    evaluator.evaluate_single_model(
        df_pred=df_mat, 
        model_name="MAT", 
        primary_horizon=primary_h
    )
    
    evaluator.evaluate_single_model(
        df_pred=df_can, 
        model_name="Canonical", 
        primary_horizon=primary_h
    )
    
    print("\n" + "-"*40)
    print(f"STEP 2: Comparison (Canonical vs MAT)")
    print("-" * 40)
    
    evaluator.compare_models(
        df_base=df_can,       
        df_challenger=df_mat,
        names=("Canonical", "MAT"),
        primary_horizon=primary_h
    )
    
    print("\nEvaluation Complete.")
    print(f"   Tables saved to:  {evaluator.tables_dir}")
    print(f"   Figures saved to: {evaluator.figures_dir}")

if __name__ == "__main__":
    main()