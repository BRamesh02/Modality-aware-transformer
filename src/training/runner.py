import torch
import torch.optim as optim
import pandas as pd
import math
import gc
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.data_loader import prepare_scaled_fold
from src.models.dataset import FinancialDataset
from src.models.dataset import get_annual_splits
from src.models.architectures.mat import MAT
from src.models.architectures.canonical_transformer import CanonicalTransformer 
from src.training.engine import train_epoch, validate_epoch
from src.training.callbacks import EarlyStopping
from src.training.losses import WeightedMSE
from src.evaluation.predictions.inference import WalkForwardEvaluator


def get_model_instance(model_type: str, config: dict, device: str):
    """
    Instantiates the correct model architecture based on the string tag.
    """

    common_args = {
        'num_input_dim': config['num_input_dim'],
        'n_sent': config['sent_input_dim'],
        'd_model': config['d_model'],
        "nhead": config["nhead"],
        "enc_layers": config["n_layers"],
        "dec_layers": config["n_layers"],
        "dropout": config["dropout"],
        'forecast_horizon': config['forecast_horizon'],
        'use_emb': config.get('use_emb', True)
    }

    if model_type == "MAT":
        return MAT(**common_args).to(device)
        
    elif model_type == "Canonical":
        return CanonicalTransformer(
            **common_args,
        ).to(device)
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train_model_for_year(
    year: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type: str,
    model_config: dict,
    device: str,
    save_dir: Path
):
    print(f"Training {model_type} for Test Year: {year}...")
    
    model = get_model_instance(model_type, model_config, device)
    
    optimizer = optim.AdamW(model.parameters(), lr=model_config["learning_rate"], weight_decay=model_config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=model_config["learning_rate"],             
    steps_per_epoch=len(train_loader),
    epochs=model_config["epochs"],                
    pct_start=0.3,            # Warmup for 30% of time
    anneal_strategy='cos'
    )
    
    if model_config["criterion"] == "weighted_MSE":
        criterion = WeightedMSE(alpha=model_config["MSE_weight"])
    else:
        criterion = torch.nn.L1Loss()
    
    save_path = save_dir / f"{model_type.lower()}_best_{year}.pt"
    
    early_stopping = EarlyStopping(patience=5, path=str(save_path), verbose=False)
    
    EPOCHS = model_config["epochs"]
    for epoch in range(EPOCHS):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        v_loss = validate_epoch(model, val_loader, criterion, device)

        early_stopping(v_loss, model)
        if early_stopping.early_stop:
            print(f"      [Early Stopping] Epoch {epoch+1} - Val Loss: {v_loss:.6f}")
            break
            
    return save_path

def run_walk_forward(
    df_main: pd.DataFrame,
    model_type: str,
    config: dict,
    project_root: Path
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Walk-Forward for {model_type} on {device}...")
    
    models_dir = project_root / "models" / model_type
    models_dir.mkdir(exist_ok=True, parents=True)
    
    start_year = pd.to_datetime(config["start_date"]).year
    end_year = pd.to_datetime(config["end_date"]).year
    
    all_years_predictions = []
    
    exclude_cols = ["date", "permno", "target", "emb_mean", "sent_score_mean", "sent_pos_mean", "sent_neg_mean", "sent_score_std", "log_n_news"]
    all_num_cols = [c for c in df_main.columns if c not in exclude_cols]
    scale_cols = [c for c in all_num_cols if c != "has_news"]
    
    print("Generating split schedule...")
    all_splits = get_annual_splits(
        df_main, 
        start_year=start_year,      
        end_year=end_year,    
        train_years=config["train_years"],        
        val_years=config["val_years"], 
        test_years=config["test_years"],
    )
    
    target_splits = [s for s in all_splits if start_year <= s['year'] <= end_year]
    
    if not target_splits:
        print(f"No splits found for range {start_year}-{end_year}. Check your data dates.")
        return pd.DataFrame()

    for split in target_splits:
        year = split['year']
        
        print("\n" + "="*60)
        print(f"PROCESSING YEAR: {year} ({model_type})...")
        print(f"   Train: {split['train']}")
        print(f"   Val:   {split['val']}")
        print(f"   Test:  {split['test']}")
        print("="*60)
        
        df_train, df_val, df_test = prepare_scaled_fold(df_main, scale_cols, split, buffer_days=math.ceil(1.5*config["window_size"]))
        
        ws = config['window_size']
        fh = config['forecast_horizon']
        
        train_ds = FinancialDataset(df_train, window_size=ws, forecast_horizon=fh, 
                                    min_date=split['train'][0], max_date=split['train'][1], use_emb=config["use_emb"])
        
        val_ds   = FinancialDataset(df_val,   window_size=ws, forecast_horizon=fh,
                                    min_date=split['val'][0],   max_date=split['val'][1], use_emb=config["use_emb"])
        
        test_ds  = FinancialDataset(df_test,  window_size=ws, forecast_horizon=fh,
                                    min_date=split['test'][0],  max_date=split['test'][1], use_emb=config["use_emb"])
        
        bs = config['batch_size']
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=config["num_workers"], pin_memory=config["pin_memory"], persistent_workers=config["persistent_workers"])
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=config["num_workers"], pin_memory=config["pin_memory"], persistent_workers=config["persistent_workers"])
        test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=config["num_workers"], pin_memory=config["pin_memory"], persistent_workers=config["persistent_workers"])
        
        best_weights_path = train_model_for_year(
            year, train_loader, val_loader, 
            model_type, config, 
            device, models_dir
        )
        

        print(f"   >>> Running Inference for {year}...")
        model = get_model_instance(model_type, config, device)
        model.load_state_dict(torch.load(best_weights_path))
        
        evaluator = WalkForwardEvaluator(model, device)
        df_pred = evaluator.predict_fold(test_loader, fold_name=f"{model_type}_{year}")
        all_years_predictions.append(df_pred)
        
        del df_train, df_val, df_test, train_ds, val_ds, test_ds, model, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    print("\n Walk-Forward Complete.")
    if not all_years_predictions:
        return pd.DataFrame()
        
    master_df = pd.concat(all_years_predictions, axis=0)
    master_df = master_df.sort_values(['date_forecast', 'permno'])
    
    return master_df