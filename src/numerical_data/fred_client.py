import os
import pandas as pd
from fredapi import Fred
from typing import Optional

class FredClient:
    def __init__(self, api_key: Optional[str] = None):
        """
        Wrapper for the Federal Reserve Economic Data (FRED) API.
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED_API_KEY not found in environment variables.")
        
        self.client = Fred(api_key=self.api_key)

    def get_first_release(self, series_id: str) -> pd.Series:
        """
        Fetches the unrevised 'First Release' of a series.
        Crucial for preventing Look-Ahead Bias in backtests.
        """
        print(f"[FRED] Fetching First Release: {series_id}")
        # Note: get_series_first_release returns the value as it was first reported
        return self.client.get_series_first_release(series_id)

    def get_daily_series(self, series_id: str, start_date: str) -> pd.Series:
        """
        Fetches standard daily series (e.g., VIX, Yields).
        """
        print(f"[FRED] Fetching Daily Series: {series_id}")
        return self.client.get_series(series_id, observation_start=start_date)