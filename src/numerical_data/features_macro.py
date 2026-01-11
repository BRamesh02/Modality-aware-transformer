import pandas as pd
import numpy as np

def process_monthly_macro(
    s_unrate: pd.Series, 
    s_cpi: pd.Series, 
    s_ppi: pd.Series, 
    start_date: str
) -> pd.DataFrame:
    """
    Cleans Monthly Economic Data and handles Reporting Lags.
    """
    buffer_date = pd.to_datetime(start_date) - pd.DateOffset(months=14)
    
    data = {}
    
    # --- Unemployment ---
    s_un = pd.to_numeric(s_unrate, errors='coerce')
    s_un = s_un[s_un.index >= buffer_date]
    
    data["unemp_rate"] = s_un.div(100).clip(0.0, 0.3)
    data["unemp_delta"] = s_un.diff().div(100)

    # --- Inflation (CPI) ---
    s_c = pd.to_numeric(s_cpi, errors='coerce')
    s_c = s_c[s_c.index >= buffer_date]
    data["cpi_yoy"] = s_c.pct_change(12).clip(lower=-0.2, upper=0.3)

    # --- Producer Prices (PPI) ---
    s_p = pd.to_numeric(s_ppi, errors='coerce')
    s_p = s_p[s_p.index >= buffer_date]
    data["ppi_yoy"] = s_p.pct_change(12).clip(lower=-0.2, upper=0.3)

    df = pd.DataFrame(data)
    df.index.name = "date"
    df.index = df.index + pd.DateOffset(months=1, days=15)
    df_daily = df.resample("B").ffill()
    
    return df_daily

def process_daily_macro(
    s_yield_curve: pd.Series, 
    s_risk_free: pd.Series, 
    s_vix: pd.Series
) -> pd.DataFrame:
    """
    Cleans Daily Financial Data (Rates, Volatility).
    """
    df = pd.DataFrame({
        "yield_curve": s_yield_curve / 100.0, # T10Y2Y
        "risk_free": s_risk_free / 100.0,     # DGS10
        "vix": s_vix                  # VIXCLS
    })
    
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    
    df = df.shift(1)
    
    df["log_vix"] = np.log1p(df["vix"])
    df = df.drop(columns=["vix"])
    
    return df

def merge_and_format_macro(
    df_monthly: pd.DataFrame, 
    df_daily: pd.DataFrame
) -> pd.DataFrame:
    """
    Fuses the two streams and applies 'macro_' namespace.
    """
    df_merged = pd.merge(
        left=df_monthly, 
        right=df_daily, 
        left_index=True, 
        right_index=True, 
        how="outer" 
    )
    
    df_merged = df_merged.ffill().dropna()
    
    col_mapping = {
        "unemp_rate": "macro_unemp_rate",
        "unemp_delta": "macro_unemp_delta",
        "cpi_yoy": "macro_cpi_yoy",
        "ppi_yoy": "macro_ppi_yoy",
        "yield_curve": "macro_yield_curve",
        "risk_free": "macro_risk_free",
        "log_vix": "macro_vix"
    }
    
    df_out = df_merged.rename(columns=col_mapping)
    return df_out

