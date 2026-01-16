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
        print(
            "   Please run 'scripts/13_run_training.py' or '14_run_inference_only.py' first."
        )
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

    required = {"date_forecast", "permno", "pred", "target"}
    missing_mat = required - set(df_mat.columns)
    missing_can = required - set(df_can.columns)
    if missing_mat:
        print(f"CRITICAL ERROR: MAT missing columns: {missing_mat}")
        return
    if missing_can:
        print(f"CRITICAL ERROR: Canonical missing columns: {missing_can}")
        return

    df_mat["date_forecast"] = pd.to_datetime(df_mat["date_forecast"], errors="coerce")
    df_can["date_forecast"] = pd.to_datetime(df_can["date_forecast"], errors="coerce")

    # Determine primary horizon
    primary_h = CONFIG.get("primary_eval_horizon", 1)

    # Optional: sort for stable outputs
    sort_cols = ["date_forecast", "permno"]
    if "horizon" in df_mat.columns and "horizon" in df_can.columns:
        sort_cols.append("horizon")
    df_mat = df_mat.sort_values(sort_cols).reset_index(drop=True)
    df_can = df_can.sort_values(sort_cols).reset_index(drop=True)

    # Quick logs
    print(f"   MAT unique dates: {df_mat['date_forecast'].nunique()}")
    print(f"   Canonical unique dates: {df_can['date_forecast'].nunique()}")
    if "horizon" in df_mat.columns:
        print(f"   MAT horizons: {sorted(df_mat['horizon'].unique())}")
    if "horizon" in df_can.columns:
        print(f"   Canonical horizons: {sorted(df_can['horizon'].unique())}")

    evaluator = ModelEvaluator(PROJECT_ROOT)

    print("\n" + "-" * 40)
    print(f"STEP 1: Individual Model Reports (Focus: H={primary_h})")
    print("-" * 40)

    evaluator.evaluate_single_model(
        df_pred=df_mat, model_name="MAT", primary_horizon=primary_h
    )
    evaluator.evaluate_single_model(
        df_pred=df_can, model_name="Canonical", primary_horizon=primary_h
    )

    print("\n" + "-" * 40)
    print("STEP 2: Comparison (Canonical vs MAT)")
    print("-" * 40)

    evaluator.compare_models(
        df_base=df_can,
        df_challenger=df_mat,
        names=("Canonical", "MAT"),
        primary_horizon=primary_h,
    )

    print("\nEvaluation Complete.")
    print(f"   Tables saved to:  {evaluator.tables_dir}")
    print(f"   Figures saved to: {evaluator.figures_dir}")


if __name__ == "__main__":
    main()
