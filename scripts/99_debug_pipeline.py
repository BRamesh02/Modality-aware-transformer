import sys
import shutil
import pandas as pd
import torch
import random
import os
import numpy as np
from pathlib import Path

# --- 1. Setup Project Path ---
current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent
sys.path.append(str(PROJECT_ROOT))

# Import your modules
from src.models.config import CONFIG
from src.utils.data_loader import load_and_merge_data
from src.training.runner import run_walk_forward
from src.evaluation.predictions.evaluator import ModelEvaluator

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

def run_debug():
    print("🐞 STARTING FULL PIPELINE DEBUG (T4 Mode)...")
    seed_everything(42)
    
    # --- 2. Override Config for Debugging ---
    DEBUG_CONFIG = CONFIG.copy()
    DEBUG_CONFIG.update({
        "debug": True,
        
        # 1. Timeline: Exactly 3 years for 1 Split (Train 2012, Val 2013, Test 2014)
        "start_date": "2012-01-01",
        "end_date": "2014-12-31",  
        "train_years": 1,          
        "val_years": 1,            
        "test_years": 1,
        
        # 2. Tiny Compute (Safe for T4)
        "batch_size": 32,          # Small batch
        "epochs": 1,               # Single pass
        "d_model": 16,             # Tiny model
        "nhead": 2,
        "n_layers": 1,
        "forecast_horizon": 2,     # H=2 to test Decay logic
        "primary_eval_horizon": 1, # Evaluate on H=1 accuracy
        
        # 3. Critical Dataloader Fixes for Colab
        "num_workers": 0,          # MUST be 0 to avoid Colab multiprocessing crashes
        "pin_memory": False,       
        "persistent_workers": False, 
        "prefetch_factor": None,     
        
        "models_dir": PROJECT_ROOT / "models" / "debug",
    })
    
    # Ensure debug directories exist
    debug_pred_dir = PROJECT_ROOT / "data" / "processed" / "predictions" / "debug"
    debug_report_dir = PROJECT_ROOT / "reports" / "debug"
    debug_pred_dir.mkdir(parents=True, exist_ok=True)
    debug_report_dir.mkdir(parents=True, exist_ok=True)

    # --- 3. Load & Sample Data ---
    print("\n[1/4] Loading & Sampling Data...")
    
    # A. Load full time range
    df_main = load_and_merge_data(
        PROJECT_ROOT / "data", 
        start_date=DEBUG_CONFIG["start_date"], 
        end_date=DEBUG_CONFIG["end_date"]
    )
    
    # B. Vertical Sampling (Preserve Time Series)
    # Pick first 5 unique stocks and keep their FULL history
    unique_permnos = df_main['permno'].unique()
    if len(unique_permnos) > 5:
        debug_permnos = unique_permnos[:5] 
        df_main = df_main[df_main['permno'].isin(debug_permnos)].copy()
        print(f"   ⚠️ Subsampled to 5 stocks: {debug_permnos}")
    
    print(f"   Debug Data Shape: {df_main.shape}")

    # --- 4. Train & Infer (MAT) ---
    print("\n[2/4] Training MAT (Debug)...")
    try:
        df_mat = run_walk_forward(
            df_main=df_main,
            model_type="MAT",
            config=DEBUG_CONFIG,
            project_root=PROJECT_ROOT
        )
        if df_mat.empty:
            raise ValueError("MAT produced no predictions.")
            
        df_mat.to_parquet(debug_pred_dir / "mat_debug.parquet")
        print("   ✅ MAT Debug run successful.")
    except Exception as e:
        print(f"   ❌ MAT Failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

    # --- 5. Train & Infer (Canonical) ---
    print("\n[3/4] Training Canonical (Debug)...")
    try:
        df_can = run_walk_forward(
            df_main=df_main,
            model_type="Canonical",
            config=DEBUG_CONFIG,
            project_root=PROJECT_ROOT
        )
        if df_can.empty:
            raise ValueError("Canonical produced no predictions.")
            
        df_can.to_parquet(debug_pred_dir / "can_debug.parquet")
        print("   ✅ Canonical Debug run successful.")
    except Exception as e:
        print(f"   ❌ Canonical Failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

    # --- 6. Run Evaluation ---
    print("\n[4/4] Running Evaluation Suite...")
    try:
        evaluator = ModelEvaluator(PROJECT_ROOT)
        # Override internal paths for debug output
        evaluator.figures_dir = debug_report_dir / "figures"
        evaluator.tables_dir = debug_report_dir / "tables"
        evaluator.figures_dir.mkdir(parents=True, exist_ok=True)
        evaluator.tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Individual Eval (Sanity Check)
        print("   > Evaluating MAT...")
        evaluator.evaluate_single_model(df_mat, "MAT", primary_horizon=1)
        
        print("   > Evaluating Canonical...")
        evaluator.evaluate_single_model(df_can, "Canonical", primary_horizon=1)
        
        # Comparison Eval (The Real Test)
        print("   > Running Comparison...")
        evaluator.compare_models(
            df_base=df_can,
            df_challenger=df_mat,
            names=("Canonical", "MAT"),
            primary_horizon=1
        )
        
        print(f"   ✅ Evaluation successful. Check {debug_report_dir}")
        
    except Exception as e:
        print(f"   ❌ Evaluation Failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

    print("\n🐞 DEBUG RUN COMPLETE. PIPELINE IS VALID.")

if __name__ == "__main__":
    run_debug()