

CONFIG = {
    # ==========================================
    # 1. DATA & SPLIT CONFIGURATION
    # ==========================================
    # The absolute start and end dates for the entire dataset processing.
    "start_date": "2010-01-01",
    "end_date": "2023-12-15",
    
    # Walk-Forward Validation parameters (Sliding Window logic).
    # Train on 5 years -> Validate on 1 year -> Test on 1 year.
    "train_years": 5,
    "val_years": 1,
    "test_years": 1,

    # ==========================================
    # 2. MODEL INPUT SHAPES
    # ==========================================
    # How many past days the model sees (T). 60 days is approx. 1 trading quarter.
    "window_size": 60,       
    
    # How many days into the future to predict (H). 1 = Next Day Return.
    "forecast_horizon": 5,
    "primary_eval_horizon": 1,   
    
    # Number of scalar features per day (Price, Volatility, PE ratio, etc.).
    "num_input_dim": 22,     
    
    # Number of sentiment features (e.g., Positive, Negative, Neutral scores).
    "sent_input_dim": 5,     
    
    # If True, includes the 768-dimensional BERT embeddings in the model input.
    # Setting to False significantly reduces VRAM usage but loses semantic news info.
    "use_emb": True,         

    # ==========================================
    # 3. DATALOADER OPTIMIZATION (HARDWARE)
    # ==========================================
    # Number of CPU sub-processes used to load data. 
    "num_workers": 4,        
    
    # If True, locks data in RAM for faster transfer to GPU. 
    # Set to False if you run out of System RAM (RAM crash), otherwise True is faster.
    "pin_memory": True,     
    
    # If True, keeps worker threads alive between epochs. 
    # Reduces the "pause" at the start of every new epoch.
    "persistent_workers": True,
    
    # Number of batches loaded in advance by each worker. 
    # Helps prevent the GPU from waiting for the CPU (prevents starvation).
    "prefetch_factor": 2,

    # ==========================================
    # 4. TRAINING HYPERPARAMETERS
    # ==========================================
    # Maximum number of passes through the training data.
    "epochs": 20,
    
    # Early Stopping: Stop training if Validation Loss doesn't improve for this many epochs.
    "patience": 5,           
    
    # Number of samples per gradient update. 
    # 4096 is optimized for A100/High-VRAM GPUs. Reduce to 1024 or 512 for smaller GPUs.
    "batch_size": 16384,      
    
    # Loss Function: "MAE" (L1Loss) is robust to outliers. 
    # "weighted_MSE" penalizes errors on high-volatility days more heavily.
    "criterion": "MAE",     
    
    # Only used if criterion == "weighted_MSE". 
    # Controls how much extra penalty is applied to large errors.
    "MSE_weight": 10,
    
    # Peak Learning Rate for the OneCycleLR scheduler.
    "learning_rate": 7e-4,
    
    # L2 Regularization penalty. Helps prevent overfitting by keeping weights small.
    "weight_decay": 1e-4,
    
    # ==========================================
    # 5. TRANSFORMER ARCHITECTURE
    # ==========================================
    # The internal dimension size of the Transformer (Hidden Size).
    # Larger = Smarter but slower and more prone to overfitting.
    "d_model": 128,
    
    # Number of Attention Heads. Allows the model to focus on different parts of the sequence.
    # d_model must be divisible by nhead (128 / 4 = 32).
    "nhead": 4,
    
    # Number of stacked Transformer Encoder/Decoder layers.
    "n_layers": 2,         
    
    # Dropout probability (0.1 = 10% of neurons turned off randomly during training).
    "dropout": 0.1
}