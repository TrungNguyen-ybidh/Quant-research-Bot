"""
FRED API Client
Fetches interest rate data from Federal Reserve Economic Data (FRED)
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from src.utils.logger import info, success, error


class FREDClient:
    """
    Client for fetching economic data from FRED API.
    
    Requires fredapi package: pip install fredapi
    API key from: https://fred.stlouisfed.org/docs/api/api_key.html
    """
    
    # Central bank rate series IDs
    RATE_SERIES = {
        'USD': 'DFF',           # Federal Funds Effective Rate (daily)
        'EUR': 'ECBDFR',        # ECB Deposit Facility Rate
        'GBP': 'BOERUKM',       # Bank of England Bank Rate
        'JPY': 'IRSTCI01JPM156N',  # Japan Short-Term Interest Rate
        'CHF': 'IRSTCI01CHM156N',  # Switzerland Short-Term Rate
        'AUD': 'IRSTCI01AUM156N',  # Australia Short-Term Rate
        'NZD': 'IRSTCI01NZM156N',  # New Zealand Short-Term Rate
        'CAD': 'IRSTCI01CAM156N',  # Canada Short-Term Rate
    }
    
    # Treasury yield series
    YIELD_SERIES = {
        'US2Y': 'DGS2',         # 2-Year Treasury
        'US10Y': 'DGS10',       # 10-Year Treasury
        'US30Y': 'DGS30',       # 30-Year Treasury
        'US3M': 'DTB3',         # 3-Month T-Bill
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED client.
        
        Args:
            api_key: FRED API key. If None, reads from:
                     1. FRED_API_KEY environment variable
                     2. Falls back to None (will fail on fetch)
        """
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        self._fred = None
        self._connected = False
        
        if self.api_key:
            self._connect()
        else:
            error("No FRED API key provided. Set FRED_API_KEY environment variable or pass api_key parameter.")
    
    def _connect(self) -> bool:
        """Establish connection to FRED API."""
        try:
            from fredapi import Fred
            self._fred = Fred(api_key=self.api_key)
            self._connected = True
            success("Connected to FRED API")
            return True
        except ImportError:
            error("fredapi not installed. Run: pip install fredapi")
            return False
        except Exception as e:
            error(f"Failed to connect to FRED: {e}")
            return False
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected and self._fred is not None
    
    def fetch_series(self, 
                     series_id: str, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> Optional[pd.Series]:
        """
        Fetch a single FRED series.
        
        Args:
            series_id: FRED series ID (e.g., 'DFF', 'DGS10')
            start_date: Start date string 'YYYY-MM-DD' (default: 5 years ago)
            end_date: End date string 'YYYY-MM-DD' (default: today)
        
        Returns:
            Series with datetime index, or None if failed
        """
        if not self.is_connected:
            error("FRED client not connected")
            return None
        
        # Default date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        try:
            info(f"Fetching FRED series: {series_id}")
            data = self._fred.get_series(series_id, start_date, end_date)
            
            if data is not None and len(data) > 0:
                success(f"Fetched {len(data)} observations for {series_id}")
                return data
            else:
                error(f"No data returned for {series_id}")
                return None
                
        except Exception as e:
            error(f"Failed to fetch {series_id}: {e}")
            return None
    
    def fetch_interest_rate(self, 
                            currency: str,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Optional[pd.Series]:
        """
        Fetch interest rate for a specific currency.
        
        Args:
            currency: Currency code (USD, EUR, GBP, JPY, etc.)
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            Series with interest rate data
        """
        currency = currency.upper()
        
        if currency not in self.RATE_SERIES:
            error(f"Unknown currency: {currency}. Available: {list(self.RATE_SERIES.keys())}")
            return None
        
        series_id = self.RATE_SERIES[currency]
        return self.fetch_series(series_id, start_date, end_date)
    
    def fetch_all_rates(self,
                        currencies: Optional[List[str]] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch interest rates for multiple currencies.
        
        Args:
            currencies: List of currency codes. Default: all available
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            DataFrame with currency rates as columns
        """
        if currencies is None:
            currencies = list(self.RATE_SERIES.keys())
        
        rates = {}
        for currency in currencies:
            rate = self.fetch_interest_rate(currency, start_date, end_date)
            if rate is not None:
                rates[currency] = rate
        
        if not rates:
            error("No rates fetched")
            return pd.DataFrame()
        
        # Combine into DataFrame
        df = pd.DataFrame(rates)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Forward fill missing values (rates don't change daily)
        df = df.ffill()
        
        return df
    
    def fetch_yield_curve(self,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch US Treasury yield curve data.
        
        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            DataFrame with yields as columns
        """
        yields = {}
        for name, series_id in self.YIELD_SERIES.items():
            data = self.fetch_series(series_id, start_date, end_date)
            if data is not None:
                yields[name] = data
        
        if not yields:
            error("No yield data fetched")
            return pd.DataFrame()
        
        df = pd.DataFrame(yields)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().ffill()
        
        return df
    
    def compute_rate_differential(self,
                                   base_currency: str,
                                   quote_currency: str,
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None) -> Optional[pd.Series]:
        """
        Compute interest rate differential between two currencies.
        
        For pair like EURUSD (EUR/USD):
            - base_currency = EUR
            - quote_currency = USD
            - differential = EUR_rate - USD_rate
        
        Positive differential = carry for going long the pair
        
        Args:
            base_currency: Base currency (first in pair)
            quote_currency: Quote currency (second in pair)
            start_date: Start date
            end_date: End date
        
        Returns:
            Series with rate differential
        """
        base_rate = self.fetch_interest_rate(base_currency, start_date, end_date)
        quote_rate = self.fetch_interest_rate(quote_currency, start_date, end_date)
        
        if base_rate is None or quote_rate is None:
            error(f"Could not fetch rates for {base_currency}/{quote_currency}")
            return None
        
        # Align indices
        combined = pd.DataFrame({
            'base': base_rate,
            'quote': quote_rate
        }).ffill()
        
        differential = combined['base'] - combined['quote']
        differential.name = f"{base_currency}_{quote_currency}_diff"
        
        return differential


def get_fred_client(api_key: Optional[str] = None) -> FREDClient:
    """
    Factory function to get FRED client.
    
    Args:
        api_key: Optional API key
    
    Returns:
        FREDClient instance
    """
    return FREDClient(api_key=api_key)


# Convenience functions
def fetch_rate_differential(pair: str, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            api_key: Optional[str] = None) -> Optional[pd.Series]:
    """
    Fetch rate differential for an FX pair.
    
    Args:
        pair: FX pair like 'EURUSD', 'GBPUSD'
        start_date: Start date
        end_date: End date
        api_key: FRED API key
    
    Returns:
        Series with rate differential
    """
    # Parse pair
    if len(pair) != 6:
        error(f"Invalid pair format: {pair}. Expected 6 characters like EURUSD")
        return None
    
    base = pair[:3].upper()
    quote = pair[3:].upper()
    
    client = get_fred_client(api_key)
    return client.compute_rate_differential(base, quote, start_date, end_date)