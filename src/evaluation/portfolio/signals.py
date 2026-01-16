import pandas as pd
import numpy as np
from typing import Dict


class SignalFactory:
    """
    A factory class to transform raw Long-Format model predictions
    into tradeable Wide-Format signal matrices (Index=Date, Columns=Permno).
    """

    def __init__(self, df_results: pd.DataFrame):
        """
        Args:
            df_results: The raw output from the inference loop.
                        Must contain columns: ['date_forecast', 'permno', 'horizon', 'pred']
        """
        self.raw_df = df_results.copy()

        if "date_forecast" in self.raw_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.raw_df["date_forecast"]):
                self.raw_df["date_forecast"] = pd.to_datetime(
                    self.raw_df["date_forecast"]
                )

        print(
            "   [SignalFactory] Pivoting raw predictions into multi-horizon tensor..."
        )
        self.df_wide = self.raw_df.pivot(
            index=["date_forecast", "permno"], columns="horizon", values="pred"
        )

    def get_signal(self, strategy_name: str) -> pd.DataFrame:
        """
        Dispatcher method to get a specific signal matrix.
        """
        method_name = f"_build_{strategy_name.lower()}"
        if hasattr(self, method_name):
            print(f"   [SignalFactory] Building Strategy: {strategy_name}")
            return getattr(self, method_name)()
        else:
            raise ValueError(
                f"Strategy '{strategy_name}' not implemented in SignalFactory."
            )

    def get_all_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dictionary of all available strategy matrices.
        """
        strategies = [
            "h1_only",
            "h1_h5_mean",
            "h1_h10_mean",
            "smart_decay",
            "conviction",
        ]
        return {strat: self.get_signal(strat) for strat in strategies}

    # STRATEGY IMPLEMENTATIONS

    def _build_h1_only(self) -> pd.DataFrame:
        """
        Strategy 1:
        Uses ONLY the next-day prediction (H=1).
        Pros: Highest raw Alpha.
        Cons: Highest noise and turnover.
        """
        return self.df_wide[1].unstack()

    def _build_h1_h5_mean(self) -> pd.DataFrame:
        """
        Strategy 2: The 'Weekly Trend'
        Arithmetic Mean of H=1 to H=5.
        Pros: Smoother signal, lower turnover.
        """
        return self.df_wide.loc[:, 1:5].mean(axis=1).unstack()

    def _build_h1_h10_mean(self) -> pd.DataFrame:
        """
        Strategy 3: The 'Bi-Weekly Trend'
        Arithmetic Mean of H=1 to H=10.
        Pros: Very stable.
        Cons: May lag significantly (slow reaction to news).
        """
        return self.df_wide.loc[:, 1:10].mean(axis=1).unstack()

    def _build_smart_decay(self) -> pd.DataFrame:
        """
        Strategy 4: The 'Smart Decay' (H=1 to H=10)
        Inverse Horizon Weighting (1/h).
        Gives H=1 full weight (1.0), H=2 (0.5) ... H=10 (0.1).
        Pros: Balances immediate accuracy (H1) with short-term trend stability.
        """
        horizons = np.arange(1, 11)
        weights = 1 / horizons
        weights = weights / weights.sum()
        decay_score = self.df_wide.loc[:, 1:10].dot(weights)

        return decay_score.unstack()

    def _build_conviction(self) -> pd.DataFrame:
        """
        Strategy 5: The 'Conviction' (Sign Concordance)
        Only takes a position if H=1 (Next Day) and H=5 (Next Week) agree on direction.
        If they disagree (e.g., H1 > 0 but H5 < 0), signal is forced to 0.0 (Neutral).
        Pros: high 'Hit Rate', avoids false breakouts.
        """
        h1 = self.df_wide[1]
        h5 = self.df_wide[5]
        concordance_mask = (h1 * h5) > 0

        conviction_score = h1.where(concordance_mask, 0.0)

        return conviction_score.unstack()
