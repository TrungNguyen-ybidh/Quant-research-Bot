"""
Data Converters - Transform IBKR data to pandas DataFrames
Handles bar-to-DataFrame conversion and timestamp transformations
"""

import pandas as pd
from typing import List
from src.utils.logger import info, warning


def convert_ibkr_timestamp(date_str: str) -> pd.Timestamp:
    """
    Convert IBKR datetime string to UTC-aware pandas Timestamp.
    
    IBKR formats:
    - "20240701 17:15:00" (no timezone, assumed to be US/Eastern)
    - "20240701 17:15:00 US/Eastern" (with timezone)
    
    Args:
        date_str: IBKR timestamp string
    
    Returns:
        pandas Timestamp with UTC timezone
    """
    # Handle format with timezone: "20240701 17:15:00 US/Eastern"
    if " US/Eastern" in date_str:
        # Remove timezone suffix
        date_part = date_str.replace(" US/Eastern", "")
        # Parse datetime
        ts = pd.to_datetime(date_part, format="%Y%m%d %H:%M:%S")
        # Localize to US/Eastern (IBKR's default timezone)
        ts = ts.tz_localize("America/New_York")
        # Convert to UTC
        return ts.tz_convert("UTC")
    
    # Handle format without timezone: "20240701 17:15:00"
    try:
        ts = pd.to_datetime(date_str, format="%Y%m%d %H:%M:%S")
        # Assume US/Eastern if no timezone specified (IBKR default)
        ts = ts.tz_localize("America/New_York")
        return ts.tz_convert("UTC")
    except Exception:
        # Last resort: let pandas try to parse it
        ts = pd.to_datetime(date_str)
        if ts.tzinfo is None:
            # If no timezone, assume US/Eastern and convert to UTC
            ts = ts.tz_localize("America/New_York")
            return ts.tz_convert("UTC")
        return ts.tz_convert("UTC")


def format_timestamp_for_ibkr(timestamp: pd.Timestamp) -> str:
    """
    Convert pandas Timestamp to IBKR endDateTime format.
    
    IBKR format: "YYYYMMDD-HH:MM:SS" (with dash, UTC timezone) or
                 "YYYYMMDD HH:MM:SS US/Eastern" (with timezone)
    
    Args:
        timestamp: pandas Timestamp (will be converted to UTC if needed)
    
    Returns:
        IBKR-formatted timestamp string with explicit timezone
    """
    # Convert to UTC if needed
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC")
    else:
        timestamp = timestamp.tz_localize("UTC")
    
    # Format: "YYYYMMDD-HH:MM:SS" (with dash and UTC timezone)
    # IBKR prefers this format with explicit UTC timezone
    return timestamp.strftime("%Y%m%d-%H:%M:%S")


def bars_to_dataframe(bars: List, symbol: str = None, timeframe: str = None) -> pd.DataFrame:
    """
    Convert IBKR historical data bars to pandas DataFrame.
    
    Extracts all OHLCV into DataFrame, adds symbol and timeframe columns,
    sorts by timestamp, and drops duplicate timestamps.
    
    Note: Does NOT preprocess weekends or gaps (that is for preprocessor).
    
    Args:
        bars: List of IBKR bar objects from historicalData callback
        symbol: Trading symbol (e.g., "EURUSD") - optional
        timeframe: Timeframe (e.g., "1 hour") - optional
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume, symbol, timeframe
        Sorted by timestamp (ascending)
    """
    if not bars:
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if symbol:
            columns.append('symbol')
        if timeframe:
            columns.append('timeframe')
        return pd.DataFrame(columns=columns)
    
    data = []
    for bar in bars:
        # Convert IBKR timestamp to pandas datetime
        timestamp = convert_ibkr_timestamp(bar.date)
        
        row = {
            'timestamp': timestamp,
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': int(bar.volume) if hasattr(bar, 'volume') and bar.volume else 0
        }
        
        # Add symbol and timeframe if provided
        if symbol:
            row['symbol'] = symbol
        if timeframe:
            row['timeframe'] = timeframe
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort by timestamp (IBKR may return in reverse order)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Drop duplicate timestamps (keep last)
    df = df.drop_duplicates(subset=['timestamp'], keep='last')
    
    info(f"Converted {len(bars)} bars to DataFrame ({len(df)} rows after deduplication)")
    
    return df

