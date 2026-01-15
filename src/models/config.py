CONFIG = {
    # ==========================================
    # 1. DATA & SPLIT (The Timeline)
    # ==========================================
    "start_date": "2010-01-01",
    "end_date": "2023-12-15",
    
    # 5-1-1 Walk-Forward Split. 
    # This simulates retraining the model every year to adapt to regime changes.
    "train_years": 5,
    "val_years": 1,
    "test_years": 1,

    # ==========================================
    # 2. MODEL INPUTS
    # ==========================================
    # Lookback window: 60 trading days (~3 months). Standard for catching quarterly trends.
    "window_size": 60,       
    
    # Prediction: The model outputs a curve of 10 days (T+1 to T+10).
    "forecast_horizon": 10,
    
    # Whivh horizon is focused by predictions evaluation
    "primary_eval_horizon": 1,   
    # Controls Early Stopping.
    "validation_horizon": 5,
    
    # Feature counts (Must match your CSV columns exactly).
    "num_input_dim": 22,     
    "sent_input_dim": 5,     
    
    # Multimodality Switch.
    # True = Uses BERT embeddings (Heavy VRAM usage, Richer Signal). 
    # False = Scalar Sentiment scores only (Faster, lighter).
    "use_emb": True,         

    # ==========================================
    # 3. HARDWARE TUNING
    # ==========================================
    "num_workers": 4,        
    
    # RAM Locking. Essential for fast CPU->GPU transfer. Keep True.
    "pin_memory": True,     
    
    # Keep workers alive. Prevents "spool up" lag at the start of every epoch.
    "persistent_workers": True,
    
    # Buffer size. 4 batches waiting on deck to ensure GPU never idles.
    "prefetch_factor": 4,

    # ==========================================
    # 4. TRAINING DYNAMICS
    # ==========================================
    "epochs": 20,
    
    # Stop if IC (Correlation) doesn't improve for 5 epochs.
    "patience": 5,           
    
    "batch_size": 16384,      
    
    # The Optimizer's Goal (HuberLoss, MAE, weighted_MSE). 
    "criterion": "HuberLoss",     
    "HuberLoss_delta": 1.0, # (Normalized Z-score 1.0 = 1 Standard Deviation)
    # If criterion is weighted_MSE.
    "MSE_weight": 10,

    # Learning Rate Strategy.
    "learning_rate": 7e-4,
    
    # Regularization. 1e-4 prevents weights from growing too large.
    "weight_decay": 1e-4,
    
    # Warmup. First 10% of training ramps up LR (stabilizes gradients).
    "pct_start": 0.1,
    
    # ==========================================
    # 5. ARCHITECTURE (The Size)
    # ==========================================
    # Model Width
    "d_model": 128,
    
    # Parallel Attention heads.
    "nhead": 4,
    
    # Depth.
    "n_layers": 2,         
    
    # Noise injection. 0.2 drops 20% of connections to force robust learning.
    "dropout": 0.2
}