import sys
import pandas as pd
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.evaluation.portfolio.signals import SignalFactory
from src.evaluation.portfolio.backtest import run_backtest, analyze_long_short_autopsy
from src.evaluation.portfolio.performance import compute_metrics
from src.evaluation.portfolio.attribution import perform_factor_regression, get_attribution_summary
from src.evaluation.portfolio.quantile_analysis import compute_quantile_returns, plot_quintiles_scientific
from src.evaluation.portfolio.robustness import bootstrap_analysis, plot_bootstrap_scientific
from src.evaluation.portfolio.plots import plot_cumulative_log

MAT_PRED_PATH = PROJECT_ROOT / "data/processed/predictions/mat_walkforward.parquet"
CANONICAL_PRED_PATH = PROJECT_ROOT / "data/processed/predictions/canonical_walkforward.parquet"
RETURNS_PATH = PROJECT_ROOT / "data/processed/numerical_data/returns.parquet"
UNIVERSE_PATH = PROJECT_ROOT / "data/processed/numerical_data/sp500_universe.parquet"
FACTORS_PATH = PROJECT_ROOT / "data/processed/numerical_data/factors_returns.parquet" 

OUTPUT_DIR = PROJECT_ROOT / "reports/portfolio_analysis"
COST_BPS = 0.0

def main():
    (OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    print(f"Starting Scientific Portfolio Analysis (Cost = {COST_BPS} bps)...\n")

    print("   [1/5] Loading Market Data...")
    df_returns = pd.read_parquet(RETURNS_PATH)
    
    if "return" in df_returns.columns:
        ret_col = "return"
    elif "ret" in df_returns.columns:
        ret_col = "ret"
    else:
        raise ValueError(f"Could not find return column. Available: {df_returns.columns}")
        
    if "permno" in df_returns.columns:
        returns_matrix = df_returns.pivot(index='date', columns='permno', values=ret_col).fillna(0.0)
    else:
        returns_matrix = df_returns.fillna(0.0)
        
    df_universe = pd.read_parquet(UNIVERSE_PATH)
    if not pd.api.types.is_datetime64_any_dtype(df_universe.index): 
        df_universe.index = pd.to_datetime(df_universe.index)

    df_factors = None
    if FACTORS_PATH.exists():
        df_factors = pd.read_parquet(FACTORS_PATH)
        if not pd.api.types.is_datetime64_any_dtype(df_factors.index): 
            df_factors.index = pd.to_datetime(df_factors.index)
        print("      Factors loaded.")
    else:
        print("      Factors file not found.")

    print("\n   [2/5] Generating Signals...")
    models = {}
    if MAT_PRED_PATH.exists():
        models["MAT"] = SignalFactory(pd.read_parquet(MAT_PRED_PATH)).get_all_signals()
    if CANONICAL_PRED_PATH.exists():
        models["Canonical"] = SignalFactory(pd.read_parquet(CANONICAL_PRED_PATH)).get_all_signals()

    if not models:
        print("   No models found. Exiting.")
        return

    print("\n   [3/5] Running Horizon Decay Analysis...")
    decay_strategies = ["h1_only", "h1_h5_mean", "h1_h10_mean"]
    results = []
    best_strategies = {m: (None, -999) for m in models.keys()}

    for model_name, signals in models.items():
        cum_series_dict = {} # Collect curves for plotting
        
        for strat_name, signal_df in signals.items():
            bt = run_backtest(signal_df, returns_matrix, cost_bps=COST_BPS, universe_mask=df_universe)
            stats = compute_metrics(bt['net_returns'], bt['turnover'])
            
            results.append({
                "Model": model_name,
                "Strategy": strat_name,
                "Sharpe": stats['Sharpe Ratio'],
                "Ann. Return": stats['Annualized Return'],
                "Turnover": stats['Ann. Turnover']
            })
            
            if stats['Sharpe Ratio'] > best_strategies[model_name][1]:
                best_strategies[model_name] = (strat_name, stats['Sharpe Ratio'])

            if strat_name in decay_strategies:
                label = f"{strat_name} (SR: {stats['Sharpe Ratio']:.2f})"
                cum_series_dict[label] = (1 + bt['net_returns']).cumprod()
        
        if cum_series_dict:
            plot_cumulative_log(
                cum_series_dict,
                title=f"{model_name}: Signal Decay Analysis",
                save_path=OUTPUT_DIR / "figures" / f"decay_curves_{model_name}.png"
            )

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "tables" / "all_strategies_metrics.csv", index=False)

    print("\n   [4/5] Selected Champions:")
    for m, (strat, sr) in best_strategies.items():
        print(f"      {m}: {strat} (SR: {sr:.2f})")

    print("\n   [5/5] Running Deep Dive...")
    attribution_list = []
    autopsy_list = []

    for model_name, (best_strat, _) in best_strategies.items():
        if best_strat is None: continue
        
        print(f"      ...Analyzing {model_name} Champion: {best_strat}")
        signal = models[model_name][best_strat]
        bt = run_backtest(signal, returns_matrix, cost_bps=COST_BPS, universe_mask=df_universe)
        net_returns = bt['net_returns']
        
        q_df = compute_quantile_returns(signal, returns_matrix, df_universe, n_bins=5)
        plot_quintiles_scientific(
            q_df, 
            title=f"{model_name} ({best_strat}) - Quintile Separation",
            save_path=OUTPUT_DIR / "figures" / f"quintiles_{model_name}.png"
        )
        
        sharpes, summary_df, metrics_series = bootstrap_analysis(net_returns)
        plot_bootstrap_scientific(
            sharpes, 
            metrics_series,
            title=f"{model_name} ({best_strat}) - Robustness",
            save_path=OUTPUT_DIR / "figures" / f"bootstrap_{model_name}.png"
        )
        
        if df_factors is not None:
            try:
                model = perform_factor_regression(net_returns, df_factors)
                attr_summary = get_attribution_summary(model)
                attr_summary['Model'] = model_name
                attr_summary['Strategy'] = best_strat
                attr_summary = attr_summary.reset_index().rename(columns={"index": "Factor"})
                attribution_list.append(attr_summary)
            except Exception: pass

        long_ret, short_ret = analyze_long_short_autopsy(bt['weights'], returns_matrix)
        autopsy_list.append({
            "Model": model_name, "Strategy": best_strat, "Leg": "Long",
            "Return": long_ret.mean()*252, "Vol": long_ret.std()*np.sqrt(252)
        })
        autopsy_list.append({
            "Model": model_name, "Strategy": best_strat, "Leg": "Short",
            "Return": short_ret.mean()*252, "Vol": short_ret.std()*np.sqrt(252)
        })

    if attribution_list: 
        pd.concat(attribution_list).to_csv(OUTPUT_DIR / "tables" / "champion_attribution.csv", index=False)
    
    if autopsy_list: 
        pd.DataFrame(autopsy_list).to_csv(OUTPUT_DIR / "tables" / "champion_autopsy.csv", index=False)

    print(f"\nPortfolio Analysis Complete. Reports saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()