import pandas as pd
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
project_root = current_file.parents[3] 
sys.path.append(str(project_root))

from src.evaluation.predictions.metrics import (
    calculate_ic_metrics, 
    calculate_directional_metrics, 
    calculate_regression_metrics, 
    diebold_mariano_test
)
from src.evaluation.predictions.compare import compare_models, compare_horizons_decay
from src.evaluation.predictions.plots import (
    plot_ic_distribution, 
    plot_cumulative_ic, 
    plot_quintile_returns, 
    plot_comparison_bar,
    plot_horizon_metrics
)

class ModelEvaluator:
    def __init__(self, project_root_path=None):
        """
        Orchestrates model evaluation and plotting.
        Args:
            project_root_path (Path, optional): Overrides the detected root.
        """
        self.root = Path(project_root_path) if project_root_path else project_root
        self.figures_dir = self.root / "reports/figures"
        self.tables_dir = self.root / "reports/tables"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def _evaluate_slice(self, df_slice, model_name, h, is_primary=False):
        """
        Internal helper to calculate metrics for a specific data slice (horizon or pooled).
        Only generates heavy plots for the primary horizon.
        """
        if df_slice.empty:
            return {}

        ic_res, daily_ic = calculate_ic_metrics(df_slice)
        dir_res = calculate_directional_metrics(df_slice)
        reg_res = calculate_regression_metrics(df_slice)
        
        test_h = h if h is not None else 1 
        dm_stat, dm_p = diebold_mariano_test(
            df_slice['target'], df_slice['pred'], None, h=test_h
        )
        
        metrics = {
            "Model": model_name,
            "Horizon": h if h is not None else "Pooled",
            **ic_res,
            **dir_res,
            **reg_res,
            "DM_Stat_Zero": dm_stat,
            "DM_P_Zero": dm_p
        }
        
        if is_primary:
            suffix = f"_H{h}" if h is not None else "_Pooled"
            plot_ic_distribution(daily_ic, model_name, self.figures_dir, suffix)
            plot_cumulative_ic(daily_ic, model_name, self.figures_dir, suffix)
            plot_quintile_returns(df_slice, model_name, self.figures_dir, suffix)
            
        return metrics

    def evaluate_single_model(self, df_pred, model_name="MAT", primary_horizon=1):
        """
        Runs evaluation for a single model across all available horizons.
        """
        print(f"\nEVALUATING MODEL: {model_name}")
        all_metrics = []
        
        if 'horizon' in df_pred.columns:
            horizons = sorted(df_pred['horizon'].unique())
            print(f"   Found horizons: {horizons} (Primary focus: H={primary_horizon})")
            
            for h in horizons:
                is_focus = (h == primary_horizon)
                df_h = df_pred[df_pred['horizon'] == h]
                
                if not df_h.empty:
                    m = self._evaluate_slice(df_h, model_name, h, is_primary=is_focus)
                    all_metrics.append(m)
        else:
            print("   ⚠️ No 'horizon' column found. Assuming single horizon.")
            m = self._evaluate_slice(df_pred, model_name, primary_horizon, is_primary=True)
            all_metrics.append(m)

        if 'horizon' in df_pred.columns and len(all_metrics) > 1:
            m_pooled = self._evaluate_slice(df_pred, model_name, None, is_primary=False)
            all_metrics.append(m_pooled)

        if all_metrics:
            df_summary = pd.DataFrame(all_metrics)
            cols = ["Model", "Horizon", "IC_Mean", "IC_IR", "Hit_Rate", "MAE", "DM_P_Zero"]
            existing_cols = [c for c in cols if c in df_summary.columns]
            other_cols = [c for c in df_summary.columns if c not in cols]
            df_summary = df_summary[existing_cols + other_cols]
            
            save_path = self.tables_dir / f"metrics_summary_{model_name}.csv"
            df_summary.to_csv(save_path, index=False)
            print(f"   💾 Metrics summary saved to: {save_path}")
        
        return df_summary

    def compare_models(self, df_base, df_challenger, names=("Canonical", "MAT"), primary_horizon=1):
        """
        Runs head-to-head comparison:
        1. Deep dive on Primary Horizon (Bar charts).
        2. Decay analysis across all horizons (Line charts).
        """
        print(f"\n COMPARING: {names[0]} vs {names[1]}")
        
        if 'horizon' in df_base.columns and 'horizon' in df_challenger.columns:
            df_base_h1 = df_base[df_base['horizon'] == primary_horizon]
            df_chall_h1 = df_challenger[df_challenger['horizon'] == primary_horizon]
        else:
            df_base_h1 = df_base
            df_chall_h1 = df_challenger

        if not df_base_h1.empty and not df_chall_h1.empty:
            res_h1 = compare_models(df_base_h1, df_chall_h1, names[0], names[1], h=primary_horizon)
            
            if res_h1:
                plot_comparison_bar(res_h1, "IC", self.figures_dir, suffix=f"_H{primary_horizon}")
                plot_comparison_bar(res_h1, "MAE", self.figures_dir, suffix=f"_H{primary_horizon}")
                
                summary = {
                    "Metric": ["MAE", "IC", "DM_P_Value", "IC_T_P_Value"],
                    f"{names[0]}": [res_h1["MAE"][0], res_h1["IC"][0], "-", "-"],
                    f"{names[1]}": [res_h1["MAE"][1], res_h1["IC"][1], res_h1["p_val_dm"], res_h1["p_val_ic"]]
                }
                pd.DataFrame(summary).to_csv(self.tables_dir / f"compare_H{primary_horizon}.csv", index=False)
                print(f"   Primary Horizon (H={primary_horizon}) comparison completed.")
        else:
            print(f"   Primary horizon H={primary_horizon} not found in predictions.")

        if 'horizon' in df_base.columns and 'horizon' in df_challenger.columns:
            df_decay = compare_horizons_decay(df_base, df_challenger, names)
            
            if not df_decay.empty:
                plot_horizon_metrics(df_decay, "MAE", self.figures_dir)
                plot_horizon_metrics(df_decay, "IC", self.figures_dir)
                
                df_decay.to_csv(self.tables_dir / "comparison_horizon_decay.csv")
                print(f"   Horizon decay analysis saved.")

if __name__ == "__main__":
    print(f"Evaluator module ready. Root: {project_root}")