"""
External Data Module
Provides clients for fetching external market data (FRED, Yahoo Finance)
"""

from src.data.external.fred_client import (
    FREDClient,
    get_fred_client,
    fetch_rate_differential
)

from src.data.external.yahoo_client import (
    YahooClient,
    get_yahoo_client,
    fetch_dxy,
    fetch_spx,
    fetch_vix,
    fetch_gold
)

from src.data.external.external_store import (
    ExternalDataStore,
    get_external_store
)


__all__ = [
    # FRED
    'FREDClient',
    'get_fred_client',
    'fetch_rate_differential',
    
    # Yahoo
    'YahooClient',
    'get_yahoo_client',
    'fetch_dxy',
    'fetch_spx',
    'fetch_vix',
    'fetch_gold',
    
    # Store
    'ExternalDataStore',
    'get_external_store',
]