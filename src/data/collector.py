"""
Data Collector - Cleans and Saves Raw Parquet Files
Fetches data from IBKR and saves to Parquet format
Supports incremental updates with safety windows and max fetch caps
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from src.utils.logger import info, success, error, warning
from src.data.path_builder import parquet_path
from src.data.loader import load_parquet
from src.data.validator import validate_df
from src.data.converters import format_timestamp_for_ibkr, bars_to_dataframe
from src.data.timeframes import timeframe_to_bar_size
from src.data.metadata import get_metadata_tracker
from src.api.ibkr_client import fetch_fx_history


def convert_duration_to_ibkr_format(duration_str: str) -> str:
    """
    Convert duration string to IBKR-compatible format.
    
    IBKR requires durations > 365 days to be in years format.
    Example: "1825 D" -> "5 Y", "3650 D" -> "10 Y"
    
    Args:
        duration_str: Duration string (e.g., "365 D", "1825 D")
    
    Returns:
        IBKR-compatible duration string (e.g., "365 D", "5 Y")
    """
    if not duration_str or " " not in duration_str:
        return duration_str
    
    parts = duration_str.strip().split()
    if len(parts) != 2:
        return duration_str
    
    value_str, unit = parts
    try:
        value = int(value_str)
    except ValueError:
        return duration_str
    
    # If days > 365, convert to years
    if unit.upper() == "D" and value > 365:
        years = value / 365.0
        # Round to nearest integer or 1 decimal place
        if years == int(years):
            return f"{int(years)} Y"
        else:
            return f"{years:.1f} Y"
    
    return duration_str


def save_parquet(df: pd.DataFrame, base_dir: str, symbol: str, timeframe: str):
    """
    Save a DataFrame to Parquet using pyarrow.
    
    Args:
        df: DataFrame to save
        base_dir: Base directory (e.g., "data/raw")
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Timeframe (e.g., "1 hour", "1h")
    
    Returns:
        Path to saved file, or None if failed
    """
    try:
        # Validate before saving
        if not validate_df(df):
            error("DataFrame validation failed, not saving")
            return None
        
        # Build full path
        path = parquet_path(base_dir, symbol, timeframe)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save
        df.to_parquet(path, engine="pyarrow", index=False)
        
        success(f"Saved Parquet: {path} ({len(df)} rows)")
        return path
        
    except Exception as e:
        error(f"Failed to save Parquet: {e}")
        return None


def get_last_timestamp(base_dir: str, symbol: str, timeframe: str) -> Optional[pd.Timestamp]:
    """
    Get the last timestamp from existing Parquet file.
    
    Args:
        base_dir: Base directory (e.g., "data/raw")
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Timeframe (e.g., "1 hour", "1h")
    
    Returns:
        Last timestamp as pandas Timestamp, or None if file doesn't exist
    """
    df = load_parquet(base_dir, symbol, timeframe)
    
    if df is None or df.empty:
        return None
    
    if 'timestamp' not in df.columns:
        warning("DataFrame has no 'timestamp' column")
        return None
    
    # Get last timestamp
    last_ts = df['timestamp'].max()
    info(f"Last timestamp in existing data: {last_ts}")
    return last_ts


def calculate_duration_from_timestamp(last_ts: pd.Timestamp, buffer_days: int = 1) -> str:
    """
    Calculate IBKR duration string from last timestamp.
    Adds buffer to ensure we don't miss any data.
    
    Args:
        last_ts: Last timestamp in existing data
        buffer_days: Buffer days to add (default: 1)
    
    Returns:
        Duration string (e.g., "5 D", "30 D")
    """
    now = pd.Timestamp.now(tz='UTC')
    delta = now - last_ts
    
    # Add buffer
    total_days = delta.days + buffer_days
    
    # IBKR requires at least 1 day
    if total_days < 1:
        total_days = 1
    
    return f"{total_days} D"


def save_incremental(
    new_df: pd.DataFrame,
    base_dir: str,
    symbol: str,
    timeframe: str,
    deduplicate: bool = True
) -> Optional[str]:
    """
    Save data incrementally by appending to existing Parquet file.
    
    Args:
        new_df: New DataFrame to append
        base_dir: Base directory (e.g., "data/raw")
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Timeframe (e.g., "1 hour", "1h")
        deduplicate: Whether to remove duplicate timestamps (default: True)
    
    Returns:
        Path to saved file, or None if failed
    """
    if new_df.empty:
        warning("New DataFrame is empty, nothing to save")
        return None
    
    # Validate new data
    if not validate_df(new_df):
        error("New DataFrame validation failed")
        return None
    
    # Load existing data
    existing_df = load_parquet(base_dir, symbol, timeframe)
    
    if existing_df is None or existing_df.empty:
        # No existing data, just save new data
        info("No existing data found, saving new data")
        return save_parquet(new_df, base_dir, symbol, timeframe)
    
    # Combine existing and new data
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Sort by timestamp
    if 'timestamp' in combined_df.columns:
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates if requested
        if deduplicate:
            before = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
            after = len(combined_df)
            if before != after:
                info(f"Removed {before - after} duplicate timestamps")
    
    # Save combined data
    return save_parquet(combined_df, base_dir, symbol, timeframe)


# Safety windows for incremental updates (in minutes)
SAFETY_WINDOWS = {
    "1 min": 10,    # 10 minutes
    "5 mins": 30,   # 30 minutes
    "15 mins": 60,  # 1 hour
    "1 hour": 120,  # 2 hours
    "4 hours": 480, # 8 hours
    "1 day": 2880,  # 2 days (in minutes)
}

# Max fetch caps for incremental updates (in minutes)
MAX_FETCH_CAPS = {
    "1 min": 180,      # 180 minutes (3 hours)
    "5 mins": 1440,    # 24 hours
    "15 mins": 2880,   # 2 days
    "1 hour": 20160,   # 14 days (in minutes)
    "4 hours": 40320,  # 28 days (in minutes)
    "1 day": 129600,   # 90 days (in minutes)
}


def _calculate_incremental_duration(
    last_ts: pd.Timestamp,
    timeframe: str
) -> tuple[str, str]:
    """
    Calculate duration and end_date_time for incremental update.
    
    Args:
        last_ts: Last timestamp in existing data
        timeframe: Timeframe string
    
    Returns:
        Tuple of (duration_str, end_date_time_str)
    """
    now = pd.Timestamp.now(tz='UTC')
    delta = now - last_ts
    
    # Get safety window in minutes
    safety_minutes = SAFETY_WINDOWS.get(timeframe, 60)
    if safety_minutes is None:
        safety_minutes = 60
    
    # Calculate minutes since last timestamp
    total_seconds = delta.total_seconds()
    if total_seconds is None or total_seconds < 0:
        total_seconds = 0
    total_minutes = int(total_seconds / 60)
    
    # Add safety window
    fetch_minutes = total_minutes + safety_minutes
    
    # Apply max fetch cap
    max_minutes = MAX_FETCH_CAPS.get(timeframe, 1440)
    if max_minutes is None:
        max_minutes = 1440
    # Ensure both values are integers before comparison
    fetch_minutes = int(fetch_minutes) if fetch_minutes is not None else 0
    max_minutes = int(max_minutes) if max_minutes is not None else 1440
    fetch_minutes = min(fetch_minutes, max_minutes)
    
    # Convert to days for IBKR duration string
    fetch_days = max(1, int(fetch_minutes / 1440))  # At least 1 day
    
    duration_str = f"{fetch_days} D"
    # Convert to IBKR format (days > 365 must be in years)
    duration_str = convert_duration_to_ibkr_format(duration_str)
    end_date_time_str = format_timestamp_for_ibkr(last_ts)
    
    return duration_str, end_date_time_str


def update_raw_data(pair: str, timeframe: str, cfg: Dict[str, Any]) -> bool:
    """
    Update raw data for a pair/timeframe combination.
    
    Rules:
    1. If Parquet does not exist: full history mode
    2. If Parquet exists: incremental mode
    3. Full History: duration based on timeframe from config
    4. Incremental: uses safety windows and max fetch caps
    5. Always dedupe by timestamp
    6. Always sort by timestamp after merging
    7. Always update metadata
    
    Args:
        pair: FX pair symbol (e.g., "EURUSD")
        timeframe: Timeframe (e.g., "1 hour")
        cfg: Configuration dictionary
    
    Returns:
        True if successful, False otherwise
    """
    info(f"Updating raw data for {pair} ({timeframe})")
    
    # Get config values
    data_source = cfg.get("data_source", {})
    historical_request = cfg.get("historical_request", {})
    storage = cfg.get("storage", {})
    
    host = data_source.get("host", "127.0.0.1")
    port = data_source.get("port", 7497)
    client_id = data_source.get("client_id", 1)
    what_to_show = historical_request.get("whatToShow", "MIDPOINT")
    
    # Raw clock files always save into data/raw/clock/
    base_dir = storage.get("data_paths", {}).get("raw_clock", "data/raw/clock")
    
    # Convert timeframe to IBKR bar size (use human-readable string for IBKR)
    bar_size = timeframe_to_bar_size(timeframe)
    
    # Check if Parquet exists
    last_ts = get_last_timestamp(base_dir, pair, timeframe)
    
    if last_ts is None:
        # Full history mode
        first_download = historical_request.get("first_download_duration", {})
        duration = first_download.get(timeframe, "365 D")
        # Convert to IBKR format (days > 365 must be in years)
        duration = convert_duration_to_ibkr_format(duration)
        end_date_time = ""
        info(f"Full history mode: fetching {duration} of data")
    else:
        # Incremental mode
        duration, end_date_time = _calculate_incremental_duration(last_ts, timeframe)
        info(f"Incremental mode: fetching {duration} from {last_ts}")
    
    # Fetch data from IBKR
    bars = fetch_fx_history(
        pair=pair,
        bar_size=bar_size,
        duration=duration,
        host=host,
        port=port,
        client_id=client_id,
        what_to_show=what_to_show,
        end_date_time=end_date_time
    )
    
    if not bars:
        error(f"No data received for {pair} ({timeframe})")
        return False
    
    # Convert bars to DataFrame
    df = bars_to_dataframe(bars, symbol=pair, timeframe=timeframe)
    
    if df.empty:
        error(f"Converted DataFrame is empty for {pair} ({timeframe})")
        return False
    
    # Save data (incremental if exists, otherwise full)
    if last_ts is None:
        path = save_parquet(df, base_dir, pair, timeframe)
    else:
        path = save_incremental(df, base_dir, pair, timeframe, deduplicate=True)
    
    if path is None:
        error(f"Failed to save data for {pair} ({timeframe})")
        return False
    
    # Load final data to get accurate row count and last timestamp
    final_df = load_parquet(base_dir, pair, timeframe)
    if final_df is not None and not final_df.empty:
        rows = len(final_df)
        final_last_ts = final_df['timestamp'].max()
        
        # Update metadata
        tracker = get_metadata_tracker()
        tracker.record_update(pair, timeframe, rows, final_last_ts)
        
        success(f"Successfully updated {pair} ({timeframe}): {rows} rows, last_ts={final_last_ts}")
        return True
    else:
        error(f"Failed to load final data for {pair} ({timeframe})")
        return False


def update_all_raw(cfg: Dict[str, Any]) -> None:
    """
    Update raw data for all pairs and timeframes from config.
    
    Args:
        cfg: Configuration dictionary
    """
    from src.utils.symbols_loader import load_fx_pairs
    from src.data.timeframes import TIMEFRAME_MAP
    
    # Get pairs and timeframes from config
    pairs = load_fx_pairs()
    timeframes = cfg.get("timeframes", [])
    
    if not pairs:
        error("No FX pairs found in symbols.yaml")
        return
    
    if not timeframes:
        error("No timeframes found in config.yaml")
        return
    
    # Validate all timeframes exist in TIMEFRAME_MAP
    for tf in timeframes:
        if tf not in TIMEFRAME_MAP:
            error(f"Timeframe '{tf}' in config does not exist in TIMEFRAME_MAP. Available: {list(TIMEFRAME_MAP.keys())}")
            raise ValueError(f"Invalid timeframe in config: {tf}")
    
    info(f"Starting update for {len(pairs)} pairs and {len(timeframes)} timeframes")
    
    total = len(pairs) * len(timeframes)
    current = 0
    
    for pair in pairs:
        for timeframe in timeframes:
            current += 1
            info(f"[{current}/{total}] Processing {pair} ({timeframe})")
            try:
                update_raw_data(pair, timeframe, cfg)
            except Exception as e:
                error(f"Failed to update {pair} ({timeframe}): {e}")
    
    success(f"Completed update for all pairs and timeframes")
