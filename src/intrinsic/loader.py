"""
Intrinsic Loader - Load 1-minute clock-time Parquet files
Loads the source data for intrinsic-time conversion
Robust fallback handling for corrupted/compatibility issues
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from src.utils.logger import info, success, error, warning
from src.data.path_builder import parquet_path


def load_clock_data(symbol: str, base_dir: str = "data/raw/clock") -> pd.DataFrame:
    """
    Load 1-minute clock-time Parquet file for a symbol.
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        base_dir: Base directory for clock-time data (default: "data/raw/clock")
    
    Returns:
        DataFrame with 1-minute bars, or None if failed
    """
    try:
        path = parquet_path(base_dir, symbol, "1 min")
        df = load_clock_file(path)
        
        if df is None or df.empty:
            error(f"Loaded empty DataFrame for {symbol}")
            return None
        
        # Ensure timestamp column exists and is sorted
        if 'timestamp' not in df.columns:
            error(f"Missing 'timestamp' column in {symbol} data")
            return None
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        success(f"Loaded clock data: {path} ({len(df)} rows)")
        return df
        
    except FileNotFoundError:
        error(f"Clock-time file not found for {symbol}")
        return None
    except Exception as e:
        error(f"Error loading clock data for {symbol}: {e}")
        return None


def load_clock_file(path):
    """Load Parquet file with robust fallback recovery."""
    info(f"Loading clock data from: {path}")

    # --- Attempt 1: PyArrow standard ---
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as e1:
        warning(f"PyArrow standard failed: {e1}")

    # --- Attempt 2: PyArrow simple retry ---
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception as e2:
        warning(f"PyArrow fallback failed: {e2}")

    # --- Explicitly skip fastparquet high-level engine ---
    warning("Skipping fastparquet engine by design.")

    # --- FINAL RECOVERY MODE: manual parquet extraction (no to_pandas) ---
    warning("[RECOVERY] Attempting manual parquet extraction via pyarrow...")

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        frames = []

        # Iterate row groups manually
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)

            # Build pure-Python dict:
            # Convert Arrow table to pandas DataFrame row by row
            df_rg = table.to_pandas()
            frames.append(df_rg)

        # Concatenate all row groups
        if frames:
            df = pd.concat(frames, ignore_index=True)
            
            # Clean nested objects, lists, ndarrays
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(_clean_nested_value)
            
            success(f"[RECOVERY] Manual pyarrow extraction succeeded: {path}")
            return df
        else:
            error("[RECOVERY FAILED] No row groups found in parquet file")
            return None

    except Exception as e:
        error(f"[RECOVERY FAILED] Manual pyarrow extraction could not recover file: {e}")
        import traceback
        traceback.print_exc()
        return None


def _clean_nested_value(x):
    """Flatten or convert list/array-like cells to scalar."""
    if isinstance(x, (list, np.ndarray, tuple)):
        return x[0] if len(x) else np.nan
    return x


def calculate_mid_price(df: pd.DataFrame) -> pd.Series:
    """
    Calculate mid-price from DataFrame.
    
    Uses (bid + ask)/2 if available, otherwise uses 'close' price.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Series with mid-price values
    """
    if 'bid' in df.columns and 'ask' in df.columns:
        return (df['bid'] + df['ask']) / 2.0
    elif 'close' in df.columns:
        return df['close']
    else:
        raise ValueError("DataFrame must have either (bid, ask) or 'close' column")
