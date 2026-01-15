CONFIG = {
    # --- Data Shapes ---
    "start_date":"2010-01-01",
    "end_date":"2023-12-15",
    "train_years": 5,
    "val_years": 1,
    "test_years":1,
    'window_size': 60,       # Input lookback (T)x
    'forecast_horizon': 1,   # Prediction target (H)
    'num_input_dim': 22,             # Number of sentiment features
    'sent_input_dim': 5,             # Number of sentiment features
    'use_emb': True,         # Use text embeddings (768 dim)
    
    # DataLoader Optimization
    'num_workers': 4,        
    'pin_memory': False,     
    'persistent_workers': True,
    
    # --- Training ---
    'epochs': 20,
    'patience': 5,           # Early stopping patience
    'batch_size': 4096,      # Optimized for A100
    "criterions": "MAE",     # MAE or weighted_MSE
    "MSE_weight": 10,
    'learning_rate': 5e-4,
    "weight_decay": 1e-4,
    
    # --- Transformers Architectures ---
    'd_model': 128,
    'nhead': 4,
    'n_layers': 2,         # Encoder/Decoder layers
    'dropout': 0.1
}