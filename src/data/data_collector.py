"""
Data Collector - High-level data collection with incremental updates
Orchestrates IBKR fetching, conversion, and saving
"""

import pandas as pd
from typing import Optional
from src.utils.logger import info, success, error
from src.api.ibkr_client import fetch_fx_history
from src.data.converters import bars_to_dataframe, format_timestamp_for_ibkr
from src.data.collector import (
    save_parquet,
    save_incremental,
    get_last_timestamp,
    calculate_duration_from_timestamp
)
from src.data.loader import load_parquet
from src.data.metadata import get_metadata_tracker


def collect_fx_data(
    pair: str,
    base_dir: str = "data/raw",
    timeframe: str = "1 hour",
    bar_size: str = "1 hour",
    duration: str = "365 D",
    incremental: bool = True,
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1
) -> Optional[pd.DataFrame]:
    """
    Collect FX data from IBKR with optional incremental updates.
    
    This is the main function for data collection:
    1. Check for existing data (if incremental)
    2. Fetch only new bars from IBKR
    3. Convert bars to DataFrame
    4. Save incrementally or overwrite
    
    Args:
        pair: FX pair symbol (e.g., "EURUSD")
        base_dir: Base directory for Parquet files (default: "data/raw")
        timeframe: Timeframe identifier (default: "1 hour")
        bar_size: Bar size for IBKR request (default: "1 hour")
        duration: Duration string for IBKR (default: "365 D")
        incremental: Whether to use incremental updates (default: True)
        host: IBKR host (default: "127.0.0.1")
        port: IBKR port (default: 7497)
        client_id: IBKR client ID (default: 1)
    
    Returns:
        DataFrame with collected data, or None if failed
    """
    info(f"Collecting data for {pair} ({timeframe})")
    
    # Check for existing data if incremental
    end_date_time = ""
    actual_duration = duration
    
    if incremental:
        last_ts = get_last_timestamp(base_dir, pair, timeframe)
        
        if last_ts is not None:
            # Calculate duration from last timestamp
            actual_duration = calculate_duration_from_timestamp(last_ts, buffer_days=1)
            end_date_time = format_timestamp_for_ibkr(last_ts)
            info(f"Incremental update: fetching from {last_ts} to now ({actual_duration})")
        else:
            info("No existing data found, performing full fetch")
    
    # Fetch data from IBKR
    bars = fetch_fx_history(
        pair=pair,
        bar_size=bar_size,
        duration=actual_duration,
        host=host,
        port=port,
        client_id=client_id,
        end_date_time=end_date_time
    )
    
    if not bars:
        error(f"No data received for {pair}")
        return None
    
    # Convert bars to DataFrame
    df = bars_to_dataframe(bars)
    
    if df.empty:
        error(f"Converted DataFrame is empty for {pair}")
        return None
    
    # Save data
    if incremental:
        path = save_incremental(df, base_dir, pair, timeframe)
    else:
        path = save_parquet(df, base_dir, pair, timeframe)
    
    if path is None:
        error(f"Failed to save data for {pair}")
        return None
    
    # Update metadata
    # Load final data to get accurate row count
    final_df = load_parquet(base_dir, pair, timeframe)
    if final_df is not None:
        tracker = get_metadata_tracker()
        tracker.update(pair, timeframe, len(final_df))
    
    success(f"Successfully collected data for {pair}")
    
    return final_df if final_df is not None else df

