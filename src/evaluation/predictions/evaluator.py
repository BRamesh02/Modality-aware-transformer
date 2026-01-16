import pandas as pd
from pathlib import Path
import sys


current_file = Path(__file__).resolve()
project_root = current_file.parents[3]
sys.path.append(str(project_root))

from src.evaluation.predictions.metrics import (
    calculate_ic_metrics,
    calculate_rank_ic_metrics,
    calculate_directional_metrics,
    calculate_regression_metrics,
    calculate_daily_loss_metrics,
    diebold_mariano_on_daily_loss,
)

from src.evaluation.predictions.compare import (
    compare_models,
    compare_horizons_decay,
)

from src.evaluation.predictions.plots import (
    plot_ic_distribution,
    plot_cumulative_ic,
    plot_quintile_returns,
    plot_comparison_bar,
    plot_horizon_metrics,
)


# ---------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------

class ModelEvaluator:
    """
    High-level evaluation pipeline for prediction models.
    """

    def __init__(self, project_root_path=None):
        self.root = Path(project_root_path) if project_root_path else project_root
        self.figures_dir = self.root / "reports/predictions/figures"
        self.tables_dir = self.root / "reports/predictions/tables"

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Internal: evaluate one slice
    # -----------------------------------------------------------------

    def _evaluate_slice(self, df_slice, model_name, horizon, is_primary=False):
        if df_slice is None or df_slice.empty:
            return {}

        ic_res, daily_ic = calculate_ic_metrics(df_slice)
        rank_ic_res, _ = calculate_rank_ic_metrics(df_slice)
        dir_res = calculate_directional_metrics(df_slice)
        reg_res = calculate_regression_metrics(df_slice)
        daily_loss_res, _, _ = calculate_daily_loss_metrics(df_slice)

        merge_keys = ["date_forecast", "permno"]
        if "horizon" in df_slice.columns:
            merge_keys.append("horizon")

        df_zero = df_slice[merge_keys + ["target"]].copy()
        df_zero["pred"] = 0.0

        dm_stat, dm_p = diebold_mariano_on_daily_loss(
            df_pred_model=df_slice[merge_keys + ["target", "pred"]],
            df_pred_benchmark=df_zero,
            criterion="MAE",
            h=horizon if horizon is not None else 1,
            merge_keys=tuple(merge_keys),
        )

        metrics = {
            "Model": model_name,
            "Horizon": horizon if horizon is not None else "Pooled",
            **ic_res,
            **rank_ic_res,
            **dir_res,
            **reg_res,
            **daily_loss_res,
            "DM_P_Zero": float(dm_p),
        }

        if is_primary:
            suffix = f"_H{horizon}" if horizon is not None else "_Pooled"
            plot_ic_distribution(daily_ic, model_name, self.figures_dir, suffix)
            plot_cumulative_ic(daily_ic, model_name, self.figures_dir, suffix)
            plot_quintile_returns(df_slice, model_name, self.figures_dir, suffix)

        return metrics

    # -----------------------------------------------------------------
    # Single-model evaluation
    # -----------------------------------------------------------------

    def evaluate_single_model(self, df_pred, model_name="Model", primary_horizon=1):
        print(f"\nEvaluating model: {model_name}")

        if df_pred is None or df_pred.empty:
            print("No predictions provided.")
            return pd.DataFrame()

        all_metrics = []

        if "horizon" in df_pred.columns:
            horizons = sorted(df_pred["horizon"].unique())

            for h in horizons:
                df_h = df_pred[df_pred["horizon"] == h]
                m = self._evaluate_slice(
                    df_h,
                    model_name,
                    h,
                    is_primary=(h == primary_horizon),
                )
                if m:
                    all_metrics.append(m)

            if len(horizons) > 1:
                pooled = self._evaluate_slice(df_pred, model_name, None, is_primary=False)
                if pooled:
                    all_metrics.append(pooled)

        else:
            m = self._evaluate_slice(df_pred, model_name, primary_horizon, is_primary=True)
            if m:
                all_metrics.append(m)

        if not all_metrics:
            return pd.DataFrame()

        df_summary = pd.DataFrame(all_metrics)

        # -------- Rename columns for human-readable output --------
        rename_map = {
            "IC_Mean": "IC_Daily_Mean",
            "IC_IR": "IC_Daily_IR",
            "RankIC_Mean": "RankIC_Daily_Mean",
            "RankIC_IR": "RankIC_Daily_IR",
            "Hit_Rate": "HitRate_Daily",
            "Daily_MAE_Mean": "MAE_Daily_Mean",
            "Daily_MSE_Mean": "MSE_Daily_Mean",
            "MAE": "MAE_Pooled",
            "MSE": "MSE_Pooled",
            "R2": "R2_Pooled",
            "DM_P_Zero": "DM_pvalue_vs_Zero",
        }

        df_summary = df_summary.rename(
            columns={k: v for k, v in rename_map.items() if k in df_summary.columns}
        )

        preferred = [
            "Model", "Horizon",
            "IC_Daily_Mean", "IC_Daily_IR",
            "RankIC_Daily_Mean", "RankIC_Daily_IR",
            "HitRate_Daily",
            "MAE_Daily_Mean", "MSE_Daily_Mean",
            "MAE_Pooled", "MSE_Pooled", "R2_Pooled",
            "DM_pvalue_vs_Zero",
        ]
        ordered = [c for c in preferred if c in df_summary.columns]
        df_summary = df_summary[ordered + [c for c in df_summary.columns if c not in ordered]]

        out_path = self.tables_dir / f"metrics_summary_{model_name}.csv"
        df_summary.to_csv(out_path, index=False)
        print(f"Metrics saved to {out_path}")

        return df_summary

    # -----------------------------------------------------------------
    # Model comparison
    # -----------------------------------------------------------------

    def compare_models(self, df_base, df_challenger, names=("Baseline", "Challenger"), primary_horizon=1):
        print(f"\nComparing models: {names[0]} vs {names[1]}")

        if df_base.empty or df_challenger.empty:
            return

        if "horizon" in df_base.columns:
            df_base_h = df_base[df_base["horizon"] == primary_horizon]
            df_chall_h = df_challenger[df_challenger["horizon"] == primary_horizon]
        else:
            df_base_h = df_base
            df_chall_h = df_challenger

        if not df_base_h.empty and not df_chall_h.empty:
            res = compare_models(df_base_h, df_chall_h, names[0], names[1], h=primary_horizon)

            if res:
                plot_comparison_bar(res, "IC", self.figures_dir, suffix=f"_H{primary_horizon}")
                plot_comparison_bar(res, "MAE", self.figures_dir, suffix=f"_H{primary_horizon}")

                summary = pd.DataFrame({
                    "Metric": [
                        "MAE_Daily_Mean",
                        "IC_Daily_Mean",
                        "DM_pvalue_Challenger_vs_Baseline",
                        "IC_pvalue_Difference",
                        "IC_n_days",
                        "MAE_n_days",
                    ],
                    names[0]: [
                        res["MAE"][0],
                        res["IC"][0],
                        "-",
                        "-",
                        res["n_days_ic"],
                        res["n_days_mae"],
                    ],
                    names[1]: [
                        res["MAE"][1],
                        res["IC"][1],
                        res["p_val_dm"],
                        res["p_val_ic"],
                        res["n_days_ic"],
                        res["n_days_mae"],
                    ],
                })

                out_path = self.tables_dir / f"compare_H{primary_horizon}.csv"
                summary.to_csv(out_path, index=False)
                print(f"Primary horizon comparison saved to {out_path}")

        if "horizon" in df_base.columns and "horizon" in df_challenger.columns:
            df_decay = compare_horizons_decay(df_base, df_challenger, names)

            if df_decay is not None and not df_decay.empty:
                df_decay = df_decay.rename(columns={
                    f"MAE_{names[0]}": f"MAE_Daily_{names[0]}",
                    f"MAE_{names[1]}": f"MAE_Daily_{names[1]}",
                    f"IC_{names[0]}": f"IC_Daily_{names[0]}",
                    f"IC_{names[1]}": f"IC_Daily_{names[1]}",
                    "n_days_mae": "MAE_n_days",
                    "n_days_ic": "IC_n_days",
                })

                plot_horizon_metrics(df_decay, "MAE_Daily", self.figures_dir)
                plot_horizon_metrics(df_decay, "IC_Daily", self.figures_dir)

                out_path = self.tables_dir / "comparison_horizon_decay.csv"
                df_decay.to_csv(out_path)
                print("Horizon decay analysis saved.")


if __name__ == "__main__":
    print(f"Evaluator ready. Project root: {project_root}")