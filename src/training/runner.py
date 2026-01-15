import torch
import torch.optim as optim
import pandas as pd
import gc
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Project Imports
from src.utils.data_loader import prepare_scaled_fold
from src.models.dataset import FinancialDataset
from src.models.dataset import get_annual_splits
from src.models.architectures.mat import MAT
from src.models.architectures.canonical_transformer import CanonicalTransformer 
from src.training.engine import train_epoch, validate_epoch
from src.training.callbacks import EarlyStopping
from src.training.losses import WeightedMSE
from src.evaluation.predictions.inference import WalkForwardEvaluator

# --- 1. THE FACTORY ---
def get_model_instance(model_type: str, params: dict, device: str):
    """
    Instantiates the correct model architecture based on the string tag.
    """
    # Common arguments for both models based on your definitions
    common_args = {
        'num_input_dim': params['num_input_dim'],
        'n_sent': params['n_sent'],
        'd_model': params['d_model'],
        'forecast_horizon': params['forecast_horizon'],
        'use_emb': params.get('use_emb', True) # Default to True if not specified
    }

    if model_type == "MAT":
        return MAT(**common_args).to(device)
        
    elif model_type == "Canonical":
        # Your CanonicalTransformer explicitly asks for these args too
        return CanonicalTransformer(
            **common_args,
            # You can add specific defaults here if needed, e.g.:
            # nhead=4, 
            # enc_layers=2
        ).to(device)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# --- 2. TRAINER ---
def train_model_for_year(
    year: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type: str,       # <--- NEW PARAM
    model_params: dict,
    device: str,
    save_dir: Path
):
    print(f"   >>> Training {model_type} for Test Year: {year}")
    
    # Use Factory to get model
    model = get_model_instance(model_type, model_params, device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = WeightedMSE(alpha=100.0)
    
    # Unique save path per architecture
    save_path = save_dir / f"{model_type.lower()}_best_{year}.pt"
    
    early_stopping = EarlyStopping(patience=5, path=str(save_path), verbose=False)
    
    EPOCHS = 20
    for epoch in range(EPOCHS):
        # The training engine doesn't care which model it is, 
        # as long as forward() arguments match!
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss = validate_epoch(model, val_loader, criterion, device)
        
        early_stopping(v_loss, model)
        if early_stopping.early_stop:
            print(f"      [Early Stopping] Epoch {epoch+1} - Val Loss: {v_loss:.6f}")
            break
            
    return save_path

# --- 3. MASTER RUNNER ---
def run_walk_forward(
    df_main: pd.DataFrame,
    start_year: int, # e.g. 2017
    end_year: int,   # e.g. 2023
    model_type: str,
    config: dict,
    project_root: Path
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Walk-Forward for {model_type} on {device}...")
    
    models_dir = project_root / "models" / model_type
    models_dir.mkdir(exist_ok=True, parents=True)
    
    all_years_predictions = []
    
    # Feature identification
    exclude_cols = ["date", "permno", "target", "emb_mean", "sent_score_mean", "sent_pos_mean", "sent_neg_mean", "sent_score_std", "log_n_news"]
    all_num_cols = [c for c in df_main.columns if c not in exclude_cols]
    scale_cols = [c for c in all_num_cols if c != "has_news"]
    
    # --- 1. GENERATE ALL SPLITS ---
    # We generate the schedule starting from your data's actual beginning (2010)
    # Adjust train_years if you want longer history (e.g. 5, 7, or 10)
    print("Generating split schedule...")
    all_splits = get_annual_splits(
        df_main, 
        start_year=2010,      # Data start
        end_year=end_year,    # Limit
        train_years=5,        # Fixed Sliding Window size
        val_years=1, 
        test_years=1
    )
    
    # --- 2. FILTER FOR TARGET YEARS ---
    # We only run the years requested in arguments (2017-2023)
    target_splits = [s for s in all_splits if start_year <= s['year'] <= end_year]
    
    if not target_splits:
        print(f"⚠️ No splits found for range {start_year}-{end_year}. Check your data dates.")
        return pd.DataFrame()

    # --- 3. MASTER LOOP ---
    for split in target_splits:
        year = split['year'] # The Test Year
        
        print(f"\n" + "="*60)
        print(f"🚀 PROCESSING YEAR: {year} ({model_type})")
        print(f"   Train: {split['train']}")
        print(f"   Val:   {split['val']}")
        print(f"   Test:  {split['test']}")
        print("="*60)
        
        # A. Prepare Data (Using the dict directly!)
        # prepare_scaled_fold is compatible with the dict returned by get_annual_splits
        df_train, df_val, df_test = prepare_scaled_fold(df_main, scale_cols, split, buffer_days=90)
        
        # B. Create Datasets (Using exact dates from the split dict)
        ws = config['window_size']
        fh = config['forecast_horizon']
        
        # Note: split['train'] is a tuple ("start", "end")
        train_ds = FinancialDataset(df_train, window_size=ws, forecast_horizon=fh, 
                                    min_date=split['train'][0], max_date=split['train'][1])
        
        val_ds   = FinancialDataset(df_val,   window_size=ws, forecast_horizon=fh,
                                    min_date=split['val'][0],   max_date=split['val'][1])
        
        test_ds  = FinancialDataset(df_test,  window_size=ws, forecast_horizon=fh,
                                    min_date=split['test'][0],  max_date=split['test'][1])
        
        # C. Loaders
        bs = config['batch_size']
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=4, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=4, pin_memory=False)
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=4, pin_memory=False)
        
        # D. Train
        model_params = {
            'num_input_dim': len(all_num_cols),
            'n_sent': 5,
            'd_model': 128,
            'forecast_horizon': fh,
            'use_emb': config.get('use_emb', True)
        }
        
        best_weights_path = train_model_for_year(
            year, train_loader, val_loader, 
            model_type, model_params, 
            device, models_dir
        )
        
        # E. Inference
        print(f"   >>> Running Inference for {year}...")
        model = get_model_instance(model_type, model_params, device)
        model.load_state_dict(torch.load(best_weights_path))
        
        evaluator = WalkForwardEvaluator(model, device)
        df_pred = evaluator.predict_fold(test_loader, fold_name=f"{model_type}_{year}")
        all_years_predictions.append(df_pred)
        
        # F. Cleanup
        del df_train, df_val, df_test, train_ds, val_ds, test_ds, model, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    # --- AGGREGATE ---
    print("\n✅ Walk-Forward Complete.")
    if not all_years_predictions:
        return pd.DataFrame()
        
    master_df = pd.concat(all_years_predictions, axis=0)
    master_df = master_df.sort_values(['date_forecast', 'permno'])
    
    return master_df