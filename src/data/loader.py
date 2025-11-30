"""
Data Loader - Fast Parquet Loading
Efficiently loads Parquet files with optional batching
Robust fallback handling for corrupted/compatibility issues
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from src.utils.logger import info, success, error, warning
from src.data.path_builder import parquet_path


def load_parquet(base_dir: str, symbol: str, timeframe: str):
    """
    Load a Parquet dataset using pyarrow with robust fallback recovery.
    
    Args:
        base_dir: Base directory (e.g., "data/raw")
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Timeframe (e.g., "1 hour", "1h")
    
    Returns:
        DataFrame, or None if failed
    """
    path = parquet_path(base_dir, symbol, timeframe)
    return load_parquet_file(path)


def load_parquet_file(path):
    """
    Load Parquet file with robust fallback recovery.
    
    Args:
        path: Path to Parquet file
    
    Returns:
        DataFrame or None if failed
    """
    info(f"Loading Parquet: {path}")
    
    # --- Attempt 1: PyArrow standard ---
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        success(f"Loaded Parquet: {path} (rows={len(df)})")
        return df
    except Exception as e1:
        warning(f"PyArrow standard failed: {e1}")
    
    # --- Attempt 2: PyArrow simple retry ---
    try:
        df = pd.read_parquet(path, engine="pyarrow")
        success(f"Loaded Parquet (retry): {path} (rows={len(df)})")
        return df
    except Exception as e2:
        warning(f"PyArrow fallback failed: {e2}")
    
    # --- Explicitly skip fastparquet high-level engine ---
    warning("Skipping fastparquet engine by design.")
    
    # --- FINAL RECOVERY MODE: manual column-by-column extraction ---
    warning("[RECOVERY] Attempting manual column-by-column extraction...")
    
    try:
        pf = pq.ParquetFile(path)
        frames = []
        
        # Iterate row groups manually
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            
            # Manual extraction: read each column as Arrow array
            data_dict = {}
            column_names = []
            
            # Get column names - handle corrupted metadata
            try:
                # Try to get column names from schema
                schema = table.schema
                for i, field in enumerate(schema):
                    col_name = field.name
                    # Handle if column name is stored as array
                    if isinstance(col_name, (list, np.ndarray)):
                        col_name = str(col_name[0]) if len(col_name) > 0 else f"col_{i}"
                    elif not isinstance(col_name, str):
                        col_name = str(col_name)
                    column_names.append(col_name)
            except Exception:
                # Fallback: use generic column names
                column_names = [f"col_{i}" for i in range(table.num_columns)]
            
            # Extract each column as numpy array
            for i, col_name in enumerate(column_names):
                try:
                    col = table.column(i)
                    # Convert Arrow array to numpy
                    arr = col.to_numpy(zero_copy_only=False)
                    
                    # Handle nested arrays
                    if arr.dtype == object:
                        # Flatten nested arrays
                        arr = np.array([_clean_nested_value(x) for x in arr])
                    
                    data_dict[col_name] = arr
                except Exception as e:
                    warning(f"Failed to extract column {col_name}: {e}")
                    # Create empty column
                    data_dict[col_name] = np.full(len(table), np.nan)
            
            # Create DataFrame from dict
            if data_dict:
                df_rg = pd.DataFrame(data_dict)
                frames.append(df_rg)
        
        # Concatenate all row groups
        if frames:
            df = pd.concat(frames, ignore_index=True)
            
            # Clean nested objects, lists, ndarrays in all columns
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(_clean_nested_value)
            
            # Try to identify and rename columns if needed
            df = _fix_column_names(df)
            
            success(f"[RECOVERY] Manual extraction succeeded: {path} (rows={len(df)})")
            return df
        else:
            error("[RECOVERY FAILED] No data extracted from parquet file")
            return None
    
    except Exception as e:
        error(f"[RECOVERY FAILED] Manual extraction could not recover file: {e}")
        import traceback
        traceback.print_exc()
        return None


def _clean_nested_value(x):
    """Flatten or convert list/array-like cells to scalar."""
    if isinstance(x, (list, np.ndarray, tuple)):
        return x[0] if len(x) > 0 else np.nan
    return x


def _fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to fix column names if they're corrupted.
    
    Common column names for OHLCV data.
    """
    # Expected columns for clock-time data
    expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'bid', 'ask', 'symbol', 'timeframe']
    
    # If we have generic column names, try to map them
    if all(col.startswith('col_') for col in df.columns):
        # Try to infer from data types and positions
        # This is a best-guess approach
        if len(df.columns) >= 5:
            # Assume: timestamp, open, high, low, close, volume, ...
            mapping = {}
            for i, expected in enumerate(expected_cols[:len(df.columns)]):
                mapping[f'col_{i}'] = expected
            df = df.rename(columns=mapping)
    
    return df

