import sys
import torch
import pandas as pd
import math
from pathlib import Path
from torch.utils.data import DataLoader

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.data_loader import prepare_scaled_fold, load_and_merge_data
from src.models.dataset import FinancialDataset
from src.models.dataset import get_annual_splits
from src.evaluation.predictions.inference import WalkForwardEvaluator
from src.training.runner import get_model_instance  # Reuse the Factory!

from src.models.config import CONFIG


def run_pure_inference(df_main: pd.DataFrame, model_type: str, config: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Inference Only for {model_type}...")

    models_dir = PROJECT_ROOT / "models" / model_type
    all_preds = []

    start_year = pd.to_datetime(config["start_date"]).year
    end_year = pd.to_datetime(config["end_date"]).year
    all_splits = get_annual_splits(
        df_main,
        start_year=start_year,
        train_years=config["train_years"],
        val_years=config["val_years"],
        test_years=config["test_years"],
        end_year=end_year,
    )
    target_splits = [s for s in all_splits if start_year <= s["year"] <= end_year]

    exclude_cols = [
        "date",
        "permno",
        "target",
        "emb_mean",
        "sent_score_mean",
        "sent_pos_mean",
        "sent_neg_mean",
        "sent_score_std",
        "log_n_news",
    ]
    all_num_cols = [c for c in df_main.columns if c not in exclude_cols]
    scale_cols = [c for c in all_num_cols if c != "has_news"]

    for split in target_splits:
        year = split["year"]
        weights_path = models_dir / f"{model_type.lower()}_best_{year}.pt"

        if not weights_path.exists():
            print(f"Weights not found for {year}: {weights_path}. Skipping.")
            continue

        print(f"   Loading {year} weights...")

        _, _, df_test = prepare_scaled_fold(
            df_main,
            scale_cols,
            split,
            buffer_days=math.ceil(1.5 * config["window_size"]),
        )

        test_ds = FinancialDataset(
            df_test,
            window_size=config["window_size"],
            forecast_horizon=config["forecast_horizon"],
            min_date=split["test"][0],
            max_date=split["test"][1],
            use_emb=config["use_emb"],
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=config["pin_memory"],
            persistent_workers=config["persistent_workers"],
            prefetch_factor=config["prefetch_factor"],
        )

        model = get_model_instance(model_type, CONFIG, device)
        state_dict = torch.load(weights_path)

        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            from collections import OrderedDict

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("_orig_mod.", "")
                new_state_dict[name] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict)

        evaluator = WalkForwardEvaluator(model, device)
        df_pred = evaluator.predict_fold(test_loader, fold_name=f"{model_type}_{year}")
        all_preds.append(df_pred)

    return pd.concat(all_preds, axis=0) if all_preds else pd.DataFrame()


def main():
    print("Loading Data...")
    df_main = load_and_merge_data(
        PROJECT_ROOT / "data",
        start_date=CONFIG["start_date"],
        end_date=CONFIG["end_date"],
    )

    df_mat = run_pure_inference(df_main, "MAT", CONFIG)
    df_mat.to_parquet(
        PROJECT_ROOT / "data/processed/predictions/mat_inference_repro.parquet"
    )

    df_can = run_pure_inference(df_main, "Canonical", CONFIG)
    df_can.to_parquet(
        PROJECT_ROOT / "data/processed/predictions/canonical_inference_repro.parquet"
    )

    print("Reproduction Complete.")


if __name__ == "__main__":
    main()
