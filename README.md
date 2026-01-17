# Modality-Aware Transformer for Financial Time Series and Text

This repository contains an implementation and empirical study of a Modality-Aware Transformer (MAT) applied to financial prediction tasks using multimodal data, combining numerical time series (e.g. stock returns) and textual information (e.g. financial news).

The project is conducted in the context of a research-oriented machine learning study for the **ENSAE course Advanced Machine Learning (2025)** given by A. Stromme.

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
-	Encoding each modality separately
-	Using learned modality-aware queries
-	Performing cross-modal attention in a structured manner

This project:
-	Implements a clean and modular version of MAT in PyTorch
-  Implements a standard Transformer in PyTorch
-	Compares MAT against standard Transformer baselines
-	Evaluates performance on financial return prediction tasks
---
## Code Provenance and Originality

This project is primarily original code written for the course. The core model implementations (MAT architecture, encoders/decoders, training loop, walk-forward runner, and evaluation pipeline) were implemented from scratch based on the paper’s high-level description and adapted to our data format. We relied on standard open-source libraries (PyTorch, pandas, scikit-learn) for model layers, tensor ops, and preprocessing. Where external methods were used (e.g., [FinBERT](https://arxiv.org/abs/1908.10063) from Araci, 2019), we used the publicly available pretrained model via the [Hugging Face API](https://huggingface.co/ProsusAI/finbert) and adapted the feature extraction pipeline to our dataset. Any borrowed ideas were re-implemented to fit our walk-forward protocol and multimodal input design. 

---
## Repository Structure

### Global overview

```text
Modality-aware-transformer/
├─ README.md                         # Project overview, structure, and usage
├─ requirements.txt                  # Python dependencies
├─ data/                             # Raw and processed datasets
├─ models/                           # Saved weights (walk-forward checkpoints)
├─ reports/                          # Tables and figures from evaluations
├─ notebooks/                        # Exploratory notebooks
├─ scripts/                          # Data pipeline and execution
└─ src/                              # Core library
```

<details>
<summary><strong>data/</strong></summary>

```text
├─ raw/                              # Raw inputs (WRDS, Wikipedia, FNSPID, etc.)
└─ processed/                        # Processed numerical/text features
   ├─ numerical_data/                # Market, ratios, macro, targets, returns
   ├─ fnspid/                        # Text features (processed + linked)
   └─ predictions/                   # Walk-forward prediction outputs
```
</details>

<details>
<summary><strong>models/</strong></summary>

```text
├─ MAT/                              # MAT checkpoints per test year
├─ Canonical/                        # Canonical checkpoints per test year
└─ mat_best.pt                       # Legacy/single checkpoint
```
</details>

<details>
<summary><strong>reports/</strong></summary>

```text
├─ report.pdf                       # Final project report
├─ predictions/
│  ├─ tables/                        # Metrics tables 
│  └─ figures/                       # Prediction plots 
└─ portfolio_analysis/
   ├─ tables/                        # Portfolio analysis tables
   └─ figures/                       # Portfolio analysis plots
```
</details>

<details>
<summary><strong>notebooks/</strong></summary>

```text
└─ text_data.ipynb                   # Text data exploration 
```
</details>


<details>
<summary><strong>scripts/</strong></summary>

```text
├─ 01_build_universe.py              # Build asset universe and identifiers
├─ 02_build_market_features.py       # Generate market-level numerical features
├─ 03_build_ratio_features.py        # Generate accounting ratio features
├─ 04_build_macro_features.py        # Generate macroeconomic features
├─ 05_build_target.py                # Build prediction targets
├─ 06_build_returns.py               # Compute returns series
├─ 07_build_factors.py               # Compute factor features
├─ 08_raw_text_data.py               # Collect raw text data
├─ 09_clean_text_data.py             # Clean and normalize text data
├─ 10_build_text_features_stock.py   # Build stock-level text features
├─ 11_link_tickers.py                # Link tickers to identifiers across sources
├─ 12_gather_data_from_drive.py      # Gather data artifacts from shared drive
├─ 13_run_training_and_inference.py  # Train and run inference end-to-end
├─ 14_run_inference_only.py          # Run inference with a trained model
├─ 15_run_predictions_evaluation.py  # Evaluate predictions and generate plots
└─ 16_run_portfolios_analysis.py     # Portfolio analysis and reporting
```
</details>

<details>
<summary><strong>src/</strong></summary>

  <details>
  <summary><strong>evaluation/</strong></summary>

  ```text
  ├─ predictions/
  │  ├─ compare.py                  # Compare model predictions
  │  ├─ evaluator.py                # Walk-forward evaluation utilities
  │  ├─ inference.py                # Prediction/inference utilities
  │  ├─ metrics.py                  # Prediction metrics
  │  └─ plots.py                    # Prediction plots
  └─ portfolio/
     ├─ attribution.py              # Performance attribution metrics
     ├─ backtest.py                 # Portfolio backtesting logic
     ├─ performance.py              # Return and risk metrics
     ├─ plots.py                    # Portfolio analysis plots
     ├─ quantile_analysis.py        # Quantile portfolio analysis
     ├─ robustness.py               # Robustness checks and stress tests
     └─ signals.py                  # Signal construction utilities
  ```
  </details>

  <details>
  <summary><strong>fnspid/</strong></summary>

  ```text
  ├─ bert_features.py               # BERT-based feature extraction
  ├─ linking.py                     # Text-to-identifier linking helpers
  ├─ store_data.py                  # Data storage routines
  └─ text_functions.py              # Text preprocessing utilities
  ```
  </details>

  <details>
  <summary><strong>models/</strong></summary>

  ```text
  ├─ architectures/
  │  ├─ canonical_transformer.py    # Baseline transformer architecture
  │  └─ mat.py                      # Modality-Aware Transformer architecture
  ├─ decoders/
  │  ├─ canonical_decoder.py        # Baseline decoder
  │  └─ mat_decoder.py              # MAT decoder
  ├─ encoders/
  │  ├─ canonical_encoder.py        # Baseline encoder
  │  ├─ mat_encoder.py              # MAT encoder
  │  └─ mat_encoder_weighted.py     # MAT encoder with modality weighting
  ├─ layers/
  │  ├─ feature_attention.py        # Feature-level attention layers
  │  ├─ masks.py                    # Attention mask utilities
  │  └─ positional_encoding.py      # Positional encoding layers
  ├─ config.py                      # Model configuration defaults
  └─ dataset.py                     # Dataset and dataloader definitions
  ```
  </details>

  <details>
  <summary><strong>numerical_data/</strong></summary>

  ```text
  ├─ factors.py                     # Factor construction logic
  ├─ features_macro.py              # Macro features computation
  ├─ features_market.py             # Market features computation
  ├─ features_ratios.py             # Ratio features computation
  ├─ fred_client.py                 # FRED data client
  ├─ target.py                      # Target construction
  ├─ universe.py                    # Universe selection logic
  └─ wrds_client.py                 # WRDS data client
  ```
  </details>

  <details>
  <summary><strong>training/</strong></summary>

  ```text
  ├─ callbacks.py                   # Training callbacks and logging
  ├─ engine.py                      # Training/evaluation engine
  ├─ losses.py                      # Training loss functions
  └─ runner.py                      # Training runner/orchestration
  ```
  </details>

  <details>
  <summary><strong>utils/</strong></summary>

  ```text
  ├─ data_loader.py                 # Shared data loading utilities
  └─ drive_downloads.py             # Drive download helpers
  ```
  </details>
</details>



---

## Usage to reproduce results

<details>
<summary><strong>Step 1 — Install dependencies</strong></summary>

```bash
git clone https://github.com/BRamesh02/Modality-aware-transformer.git
cd Modality-aware-transformer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>Step 2 — Set environment variables (.env)</strong></summary>

```bash
WRDS_USERNAME=your_wrds_username
FRED_API_KEY=your_fred_key
```

Notes:
- WRDS is required for numerical data ( in Scripts 1-3 (Option B) and 7).
- FRED is optional : Step 4 will skip macro features if `FRED_API_KEY` is missing.
</details>

<details>
<summary><strong>Step 3 — Choose your data source</strong></summary>

  <details>
  <summary><strong>Option A — Download prebuilt artifacts from Google Drive</strong></summary>

  Note: place `drive_ids.json` in `config/` at the repo root before running this.

  ```bash
  python scripts/12_gather_data_from_drive.py
  ```
  Choose **1. Text Data (type 1)**, you can have access to all the data at each step. But for analysis choose **4. processed_and_linked (type 4)**.
  Then re run the script, choose **2. Numerical Data (type 2)**, and then type **1. processed (type 1)**.
  
  Note : In the menu, you can also select **Predictions (or 3)** to download the predictions of the models directly in 
  `data/processed/predictions`. If you do this, you can skip Steps 4–7 and go directly to Step 8 to regenerate prediction tables/figures.
  </details>

  <details>
  <summary><strong>Option B — Build datasets from scratch (run in order)</strong></summary>

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
  </details>
</details>

<details>
<summary><strong> Step 4 — Build the merged dataset (numerical + text)</strong></summary>

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

Note: You can also see an exemple of creating the merged dataset in `notebooks/data.ipynb`
</details>

<details>
<summary><strong>Step 5 — Configure the experiment</strong></summary>

Edit `src/models/config.py` to set dates, horizons, batch size, and whether to
use text embeddings (`use_emb`).
</details>

<details>
<summary><strong>Step 6 — Train and run inference </strong></summary>

```bash
python scripts/13_run_training_and_inference.py
```

Outputs:
- `data/processed/predictions/mat_walkforward.parquet`
- `data/processed/predictions/canonical_walkforward.parquet`
</details>

<details>
<summary><strong>Step 7 — Inference only (reuse saved weights)</strong></summary>

```bash
python scripts/14_run_inference_only.py
```
</details>

<details>
<summary><strong>Step 8 — Evaluate predictions and generate plots</strong></summary>

```bash
python scripts/15_run_predictions_evaluation.py
```

Outputs:
- `reports/predictions/tables/*.csv`
- `reports/predictions/figures/*.png`
</details>

<details>
<summary><strong>Step 9 — Run portfolio analysis and reports</strong></summary>

```bash
python scripts/16_run_portfolios_analysis.py
```

Outputs:
- `reports/portfolio_analysis/tables/*.csv`
- `reports/portfolio_analysis/figures/*.png`
</details>
