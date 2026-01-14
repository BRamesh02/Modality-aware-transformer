# Modality-Aware Transformer for Financial Time Series and Text

This repository contains an implementation and empirical study of a Modality-Aware Transformer (MAT) applied to financial prediction tasks using multimodal data, combining numerical time series (e.g. stock returns) and textual information (e.g. financial news).

The project is conducted in the context of a research-oriented machine learning study for the **ENSAE course Advanced Machine Learning_** given by M. Stromme.

**Authors**  
- Ramesh Brian  
- Sicard Audric  

---

## References

This project follows the experimental protocol of the original Modality-aware Transformer for Financial Time series Forecasting by Emami and al. (2023), paper, with comparisons against strong Transformer-based baselines.
> Available at: https://arxiv.org/abs/2310.01232

---
## Project Overview

Financial markets are influenced by both historical numerical signals and unstructured textual information. Classical Transformers struggle to efficiently fuse heterogeneous modalities.

The Modality-Aware Transformer (MAT) addresses this limitation by:
	•	Encoding each modality separately
	•	Using learned modality-aware queries
	•	Performing cross-modal attention in a structured manner

This project:
	•	Implements a clean and modular version of MAT in PyTorch
	•	Compares MAT against standard Transformer baselines
	•	Evaluates performance on financial return prediction tasks

---

## Repository Structure

```
Modality-aware-transformer/
├─ README.md                      # Project overview, structure, and usage
├─ requirements.txt               # Python dependencies
├─ notebooks/
│  ├─ encoders_decoders.ipynb      # Encoder/decoder experiments and sanity checks
│  ├─ test_model.ipynb             # End-to-end model testing notebook
│  └─ text_data.ipynb              # Text data exploration and preprocessing
├─ scripts/
│  ├─ 01_build_universe.py         # Build asset universe and identifiers
│  ├─ 02_build_market_features.py  # Generate market-level numerical features
│  ├─ 03_build_ratio_features.py   # Generate accounting ratio features
│  ├─ 04_build_macro_features.py   # Generate macroeconomic features
│  ├─ 05_build_target.py           # Build prediction targets
│  ├─ 06_build_returns.py          # Compute returns series
│  ├─ 07_build_factors.py          # Compute factor features
│  ├─ 08_raw_text_data.py          # Collect raw text data
│  ├─ 09_clean_text_data.py        # Clean and normalize text data
│  ├─ 10_build_text_features_stock.py # Build stock-level text features
│  ├─ 11_link_tickers.py           # Link tickers to identifiers across sources
│  └─ 12_drive_gather.py           # Gather data artifacts from shared drive
└─ src/
   ├─ evaluation/
   │  ├─ predictions/
   │  │  └─ inference.py           # Prediction/inference utilities
   │  └─ portfolio/
   │     ├─ attribution.py         # Performance attribution metrics
   │     ├─ backtest.py            # Portfolio backtesting logic
   │     ├─ performance.py         # Return and risk metrics
   │     └─ robustness.py          # Robustness checks and stress tests
   ├─ fnspid/
   │  ├─ bert_features.py          # BERT-based feature extraction
   │  ├─ linking.py                # Text-to-identifier linking helpers
   │  ├─ store_data.py             # Data storage routines
   │  └─ text_functions.py         # Text preprocessing utilities
   ├─ models/
   │  ├─ architectures/
   │  │  ├─ canonical_transformer.py # Baseline transformer architecture
   │  │  └─ mat.py                  # Modality-Aware Transformer architecture
   │  ├─ decoders/
   │  │  ├─ canonical_decoder.py    # Baseline decoder
   │  │  └─ mat_decoder.py          # MAT decoder
   │  ├─ encoders/
   │  │  ├─ canonical_encoder.py    # Baseline encoder
   │  │  ├─ mat_encoder.py          # MAT encoder
   │  │  └─ mat_encoder_weighted.py # MAT encoder with modality weighting
   │  ├─ layers/
   │  │  ├─ feature_attention.py    # Feature-level attention layers
   │  │  ├─ masks.py                # Attention mask utilities
   │  │  └─ positional_encoding.py  # Positional encoding layers
   │  └─ dataset.py                 # Dataset and dataloader definitions
   ├─ numerical_data/
   │  ├─ factors.py                 # Factor construction logic
   │  ├─ features_macro.py          # Macro features computation
   │  ├─ features_market.py         # Market features computation
   │  ├─ features_ratios.py         # Ratio features computation
   │  ├─ fred_client.py             # FRED data client
   │  ├─ target.py                  # Target construction
   │  ├─ universe.py                # Universe selection logic
   │  └─ wrds_client.py             # WRDS data client
   ├─ training/
   │  ├─ callbacks.py               # Training callbacks and logging
   │  ├─ engine.py                  # Training/evaluation engine
   │  ├─ train_mat.py               # MAT training entry point
   │  └─ train_transformer.py       # Baseline transformer training entry point
   └─ utils/
      ├─ data_loader.py             # Shared data loading utilities
      └─ drive_downloads.py         # Drive download helpers
```

---

## Usage to reproduce results

1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variables (create `.env` at repo root)

```bash
WRDS_USERNAME=your_wrds_username
FRED_API_KEY=your_fred_key
```

Notes:
- WRDS is required for numerical data (Steps 1-3 and 7).
- FRED is optional; Step 4 will skip macro features if `FRED_API_KEY` is missing.

3. Option A: download prebuilt artifacts from Google Drive

```bash
python scripts/12_drive_gather.py
```

4. Option B: build datasets from scratch (run in order)

Numerical pipeline:
```bash
python scripts/01_build_universe.py
python scripts/02_build_market_features.py
python scripts/03_build_ratio_features.py
python scripts/04_build_macro_features.py
python scripts/05_build_target.py
python scripts/06_build_returns.py
python scripts/07_build_factors.py
```

Text pipeline:
```bash
python scripts/08_raw_text_data.py
python scripts/09_clean_text_data.py
python scripts/10_build_text_features_stock.py
python scripts/11_link_tickers.py
```

Notes:
- Step 8 streams FNSPID via Hugging Face; Step 10 downloads FinBERT weights.
- `scripts/11_link_tickers.py` expects `data/raw/crsp_ticker_map.parquet`.

5. Build the merged dataset (numerical + text)

```bash
python - <<'PY'
from pathlib import Path
from src.utils.data_loader import load_and_merge_data

df = load_and_merge_data(Path("data"))
print(df.shape)
PY
```

If the linked text file name differs, align it with the loader expectation in
`src/utils/data_loader.py` (the loader reads
`data/processed/fnspid/process_and_linked_text_features.parquet`).

6. Train models


Then run:

```bash

```
