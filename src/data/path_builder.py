"""
Path Builder - Standardized file path construction
Solves consistent naming, no broken paths, no duplicate formats
"""

import os
from src.data.timeframes import timeframe_to_suffix


def sanitize_symbol(symbol: str) -> str:
    """
    Sanitize symbols into safe filenames.
    
    Example:
       EURUSD -> EURUSD
       XAU/USD -> XAU_USD
       GBPUSD=X -> GBPUSD_X
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD", "XAU/USD")
    
    Returns:
        Sanitized symbol string
    """
    return symbol.replace("/", "_").replace("=", "_").replace(" ", "_")


def parquet_path(base_dir: str, symbol: str, timeframe: str) -> str:
    """
    Construct a standardized Parquet path.
    
    Example output:
       data/raw/clock/EURUSD_1h.parquet
    
    Args:
        base_dir: Base directory (e.g., "data/raw/clock" or "data/raw/intrinsic")
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Timeframe (e.g., "1 hour", "1h") - uses suffix for filename
    
    Returns:
        Full file path
    """
    sym = sanitize_symbol(symbol)
    tf_suffix = timeframe_to_suffix(timeframe)
    return os.path.join(base_dir, f"{sym}_{tf_suffix}.parquet")

