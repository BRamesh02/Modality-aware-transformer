import sys
import os
from pathlib import Path

def get_project_root():
    """
    Detects if we are in Colab or Local and returns the correct root.
    """
    if 'google.colab' in sys.modules:
        return Path("/content/drive/MyDrive/MAT_Weighted_Loss/Modality-aware-transformer")
    else:
        return Path(__file__).resolve().parent.parent

PROJECT_ROOT = get_project_root()

DEFAULT_CONFIG = {
    'window_size': 60,
    'forecast_horizon': 1,
    'batch_size': 4096,
    'use_emb': True,
    'start_year': 2017,
    'end_year': 2023
}