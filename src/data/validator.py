"""
Data Validator - Validate DataFrames
Checks for missing timestamps, empty datasets, out-of-order indices, NaN problems
"""

import pandas as pd
from src.utils.logger import warning, success, error


def validate_df(df: pd.DataFrame, require_timestamp: bool = True) -> bool:
    """
    Validate data before saving or after loading.
    
    Args:
        df: DataFrame to validate
        require_timestamp: Whether to require a 'timestamp' column
    
    Returns:
        True if valid, False otherwise
    """
    if df.empty:
        error("DataFrame is empty.")
        return False
    
    # Check for timestamp column if required
    if require_timestamp and 'timestamp' not in df.columns:
        error("DataFrame missing required 'timestamp' column.")
        return False
    
    # Check if index is sorted (if timestamp is in index)
    if 'timestamp' in df.columns:
        if not df['timestamp'].is_monotonic_increasing:
            warning("Timestamp column is not sorted. Consider sorting before saving.")
    
    # Check for NaN values
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        warning(f"Data contains {nan_count} NaN values.")
        # Show which columns have NaNs
        nan_cols = df.columns[df.isna().any()].tolist()
        warning(f"Columns with NaN: {', '.join(nan_cols)}")
    
    # Check for required OHLCV columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warning(f"Missing OHLC columns: {', '.join(missing_cols)}")
    
    # Check for duplicate timestamps
    if 'timestamp' in df.columns:
        duplicates = df['timestamp'].duplicated().sum()
        if duplicates > 0:
            warning(f"Found {duplicates} duplicate timestamps.")
    
    success("Data validation complete.")
    return True

