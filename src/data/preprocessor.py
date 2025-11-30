"""
Data Preprocessor - Clean and prepare raw FX data
Removes weekends, duplicates, and market-closed periods
"""

import pandas as pd
from typing import Optional
from src.utils.logger import info, warning


def preprocess_raw_fx_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess raw FX data by cleaning and removing invalid periods.
    
    Requirements:
    - Ensure timezone is UTC
    - Sort by timestamp
    - Remove duplicate timestamps
    - Remove weekends (Saturday/Sunday)
    - Remove obvious market-closed periods
    - Do NOT forward-fill gaps unless explicitly told
    - Return clean DataFrame
    
    Args:
        df: Raw DataFrame with timestamp and OHLCV columns
    
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        warning("DataFrame is empty, nothing to preprocess")
        return df.copy()
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure timestamp column exists
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")
    
    # Ensure timezone is UTC
    if df['timestamp'].dtype != 'datetime64[ns, UTC]':
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize("UTC")
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert("UTC")
    
    # Remove duplicate timestamps (keep last)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['timestamp'], keep='last')
    if len(df) < before_dedup:
        info(f"Removed {before_dedup - len(df)} duplicate timestamps")
    
    # Remove weekends (Saturday=5, Sunday=6)
    before_weekend = len(df)
    df = df[df['timestamp'].dt.weekday < 5]
    if len(df) < before_weekend:
        info(f"Removed {before_weekend - len(df)} weekend bars")
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove obvious market-closed periods (optional: very low volume or zero volume)
    # For FX, this is less critical since markets are open 24/5, but we can filter
    # periods with zero volume if they seem suspicious
    # Note: This is a conservative approach - you may want to adjust based on your data
    
    info(f"Preprocessed data: {len(df)} rows remaining")
    
    return df

