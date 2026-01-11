import wrds
import pandas as pd
from typing import List, Optional
import os

class WRDSClient:
    def __init__(self, username: Optional[str] = None):
        """
        Initializes connection to WRDS.
        """
        self.username = username or os.getenv("WRDS_USERNAME")
        self.db = wrds.Connection(wrds_username=self.username)

    def get_ticker_to_permno_mapping(self, tickers: List[str], active_after: str = '2022-01-01') -> pd.DataFrame:
        """
        Fetches the historical Ticker -> PERMNO mapping from CRSP.
        Used to map the tickers in the GitHub/Wikipedia presence matrix to unique PERMNOs.
        
        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT']).
            active_after: Date string. Optimization to avoid fetching 1960s data 
                          for tickers that are currently active.
        """
    
        if not tickers:
            return pd.DataFrame(columns=['permno', 'ticker', 'namedt', 'nameenddt'])

        # Escape tickers for SQL
        safe_tickers = [t.replace("'", "") for t in tickers]
        ticker_str = "'" + "','".join(safe_tickers) + "'"
        
        query = f"""
        SELECT
            permno,
            ticker,
            comnam AS company_name,
            namedt,  -- Start Date for this ticker
            nameenddt -- End Date for this ticker
        FROM crsp.stocknames 
        WHERE ticker IN ({ticker_str})
        AND nameenddt >= '{active_after}'
        """
        
        print(f"[WRDS] Fetching Ticker mapping for {len(tickers)} symbols...")
        df_map = self.db.raw_sql(query)
        
        df_map['namedt'] = pd.to_datetime(df_map['namedt'])
        df_map['nameenddt'] = pd.to_datetime(df_map['nameenddt'])
        df_map['permno'] = df_map['permno'].astype(int)
        
        return df_map

    def get_trading_dates(self, start_date: str) -> pd.Index:
        """
        Fetches daily trading dates from crsp.wrds_dsfv2_query.
        """
        query = f"""
        SELECT DISTINCT
            dlycaldt AS date
        FROM crsp.wrds_dsfv2_query 
        WHERE dlycaldt >= '{start_date}'
        """
        
        print(f"[WRDS] Fetching trading dates starting {start_date}...")
        df = self.db.raw_sql(query)
        
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        df = df.sort_values('date').drop_duplicates()
        print(f"Found {len(df):,} valid trading days.")
        return df.set_index("date").index
        
    
    def get_daily_metrics_v2(self, permnos: list, start_date: str) -> pd.DataFrame:
        """
        Fetches daily price/volume data from crsp.wrds_dsfv2_query.
        """
        if not permnos:
            return pd.DataFrame()

        permno_str = ",".join([str(p) for p in permnos])
        start_int = int(start_date.replace("-", ""))

        query = f"""
        SELECT 
            permno,
            yyyymmdd as date,
            dlyret,   -- Total Return
            dlyprc,   -- Price
            dlyclose, -- Close
            dlycap,   -- Market Cap (Size)
            shrout,   -- Shares Outstanding
            dlyvol,   -- Volume
            dlyhigh,  -- High (Volatility)
            dlylow,   -- Low (Volatility)
            dlyask,   -- Ask (Spread)
            dlybid,   -- Bid (Spread)
            icbindustry
        FROM 
            crsp.wrds_dsfv2_query
        WHERE
            permno in ({permno_str})
            AND yyyymmdd >= {start_int}
        """
        
        print(f"[WRDS] Fetching Price Data (DSF V2) for {len(permnos)} assets...")
        df = self.db.raw_sql(query)
        df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
        df['permno'] = df['permno'].astype(int)
        
        return df
    
    def get_ratios_data(self, permnos: list, start_date: str):
        """
        Fetches valuation ratios from two sources (CCM and IBES) for robust coverage.
        Returns: (df_ccm, df_ibes)
        """
        if not permnos:
            return pd.DataFrame(), pd.DataFrame()

        permno_str = ",".join([str(p) for p in permnos])
        
        cols = """
            permno,
            public_date,
            ptb,       -- Price to Book
            pe_exi,    -- P/E (Exclude Extra Items)
            roe,       -- Return on Equity
            de_ratio,  -- Debt to Equity
            divyield   -- Dividend Yield
        """

        print(f"[WRDS] Fetching Ratios (CCM) for {len(permnos)} assets...")
        q_ccm = f"""
        SELECT {cols} FROM wrdsapps.firm_ratio_ccm
        WHERE permno IN ({permno_str}) AND public_date >= '{start_date}'
        ORDER BY permno, public_date
        """
        df_ccm = self.db.raw_sql(q_ccm)
        if not df_ccm.empty:
            df_ccm['public_date'] = pd.to_datetime(df_ccm['public_date'])
            df_ccm['permno'] = df_ccm['permno'].astype(int)

        print(f"[WRDS] Fetching Ratios (IBES) for {len(permnos)} assets...")
        q_ibes = f"""
        SELECT {cols} FROM wrdsapps.firm_ratio_ibes
        WHERE permno IN ({permno_str}) AND public_date >= '{start_date}'
        ORDER BY permno, public_date
        """
        df_ibes = self.db.raw_sql(q_ibes)
        if not df_ibes.empty:
            df_ibes['public_date'] = pd.to_datetime(df_ibes['public_date'])
            df_ibes['permno'] = df_ibes['permno'].astype(int)

        return df_ccm, df_ibes

    def get_industry_classifications(self, permnos: list, start_date: str) -> pd.DataFrame:
        """
        Fetches ICB Industry codes for Sector Imputation.
        """
        permno_str = ",".join([str(p) for p in permnos])
        start_int = int(start_date.replace("-", ""))
        
        query = f"""
        SELECT permno, yyyymmdd as date, icbindustry
        FROM crsp.wrds_dsfv2_query
        WHERE permno IN ({permno_str}) AND yyyymmdd >= {start_int}
        """
        df = self.db.raw_sql(query)
        df['date'] = pd.to_datetime(df['date'], format="%Y%m%d")
        df['permno'] = df['permno'].astype(int)
        return df

    def get_risk_free_rate(self, start_date: str) -> pd.DataFrame:
        """
        Fetches the Daily Risk-Free Rate (Rf) from Fama-French Factors.
        """
        query = f"""
        SELECT date, rf
        FROM ff.factors_daily
        WHERE date >= '{start_date}'
        """
        print("[WRDS] Fetching Risk-Free Rate...")
        df = self.db.raw_sql(query)
        df["date"] = pd.to_datetime(df["date"])
        return df
    
    def close(self):
        """Closes the WRDS connection."""
        self.db.close()