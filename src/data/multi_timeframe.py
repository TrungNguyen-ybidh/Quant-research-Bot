"""
Multi-Timeframe Data Manager
Aligns and manages data from multiple timeframes for feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from src.utils.logger import info, warning, error
from src.data.loader import load_parquet
from src.data.path_builder import parquet_path


def align_timeframes(
    dataframes: Dict[str, pd.DataFrame],
    how: str = 'right',
    method: str = 'ffill',
    tolerance: Optional[pd.Timedelta] = None
) -> Dict[str, pd.DataFrame]:
    """
    Align multiple DataFrames from different timeframes to a common index.
    
    Args:
        dataframes: Dictionary mapping timeframe names to DataFrames
                   e.g., {'1m': df1m, '5m': df5m, '1h': df1h}
        how: Alignment method - 'left', 'right', 'inner', 'outer' (default: 'right')
        method: Forward fill method - 'ffill', 'bfill', 'interpolate' (default: 'ffill')
        tolerance: Maximum time distance for alignment (default: None)
    
    Returns:
        Dictionary of aligned DataFrames with common index
    """
    if not dataframes:
        return {}
    
    info(f"Aligning {len(dataframes)} timeframes using '{how}' alignment")
    
    # Ensure all DataFrames have datetime index
    aligned = {}
    indices = []
    
    for tf_name, df in dataframes.items():
        if df is None or df.empty:
            warning(f"DataFrame for '{tf_name}' is empty, skipping")
            continue
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                error(f"DataFrame '{tf_name}' has no datetime index or timestamp column")
                continue
        
        aligned[tf_name] = df
        indices.append(df.index)
    
    if not aligned:
        error("No valid DataFrames to align")
        return {}
    
    # Determine target index based on 'how' parameter
    if how == 'right':
        # Use the finest (most frequent) timeframe as target
        target_idx = min(indices, key=len)
    elif how == 'left':
        # Use the coarsest (least frequent) timeframe as target
        target_idx = max(indices, key=len)
    elif how == 'inner':
        # Intersection of all indices
        target_idx = indices[0]
        for idx in indices[1:]:
            target_idx = target_idx.intersection(idx)
    elif how == 'outer':
        # Union of all indices
        target_idx = indices[0]
        for idx in indices[1:]:
            target_idx = target_idx.union(idx)
    else:
        raise ValueError(f"Invalid 'how' parameter: {how}. Must be 'left', 'right', 'inner', or 'outer'")
    
    # Align all DataFrames to target index
    result = {}
    for tf_name, df in aligned.items():
        if df.index.equals(target_idx):
            # Already aligned
            result[tf_name] = df
        else:
            # Reindex with forward fill
            if method == 'ffill':
                aligned_df = df.reindex(target_idx, method='ffill', tolerance=tolerance)
            elif method == 'bfill':
                aligned_df = df.reindex(target_idx, method='bfill', tolerance=tolerance)
            elif method == 'interpolate':
                aligned_df = df.reindex(target_idx).interpolate(method='time')
            else:
                aligned_df = df.reindex(target_idx, method=method, tolerance=tolerance)
            
            result[tf_name] = aligned_df
    
    info(f"Aligned {len(result)} DataFrames to index with {len(target_idx)} timestamps")
    
    return result


def load_multi_timeframe(
    symbol: str,
    timeframes: List[str],
    base_dir: str = "data/raw/clock"
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple timeframes for a symbol.
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        timeframes: List of timeframes to load (e.g., ["1 min", "5 mins", "1 hour"])
        base_dir: Base directory for data files
    
    Returns:
        Dictionary mapping timeframe names to DataFrames
    """
    info(f"Loading multi-timeframe data for {symbol}: {timeframes}")
    
    dataframes = {}
    
    for tf in timeframes:
        try:
            df = load_parquet(base_dir, symbol, tf)
            if df is not None and not df.empty:
                dataframes[tf] = df
                info(f"Loaded {symbol} {tf}: {len(df)} rows")
            else:
                warning(f"No data found for {symbol} {tf}")
        except Exception as e:
            error(f"Error loading {symbol} {tf}: {e}")
    
    return dataframes


def merge_multi_timeframe_features(
    aligned_dataframes: Dict[str, pd.DataFrame],
    prefix_timeframe: bool = True
) -> pd.DataFrame:
    """
    Merge features from multiple aligned timeframes into a single DataFrame.
    
    Args:
        aligned_dataframes: Dictionary of aligned DataFrames (from align_timeframes)
        prefix_timeframe: Whether to prefix column names with timeframe (default: True)
    
    Returns:
        Merged DataFrame with all features
    """
    if not aligned_dataframes:
        return pd.DataFrame()
    
    info(f"Merging features from {len(aligned_dataframes)} timeframes")
    
    merged = None
    
    for tf_name, df in aligned_dataframes.items():
        if df is None or df.empty:
            continue
        
        # Prefix columns with timeframe if requested
        if prefix_timeframe:
            df_renamed = df.add_prefix(f"{tf_name}_")
        else:
            df_renamed = df.copy()
        
        # Merge
        if merged is None:
            merged = df_renamed
        else:
            merged = merged.join(df_renamed, how='outer')
    
    if merged is None:
        return pd.DataFrame()
    
    info(f"Merged DataFrame shape: {merged.shape}")
    
    return merged


def get_timeframe_hierarchy() -> Dict[str, int]:
    """
    Get timeframe hierarchy (finer to coarser).
    
    Returns:
        Dictionary mapping timeframe to hierarchy level (lower = finer)
    """
    return {
        "1 min": 1,
        "5 mins": 2,
        "15 mins": 3,
        "1 hour": 4,
        "4 hours": 5,
        "1 day": 6
    }


def select_target_timeframe(
    dataframes: Dict[str, pd.DataFrame],
    target: str = 'finest'
) -> Optional[str]:
    """
    Select target timeframe for alignment.
    
    Args:
        dataframes: Dictionary of DataFrames
        target: 'finest', 'coarsest', or specific timeframe name
    
    Returns:
        Selected timeframe name or None
    """
    if not dataframes:
        return None
    
    hierarchy = get_timeframe_hierarchy()
    
    if target == 'finest':
        # Find finest (lowest hierarchy level)
        available = [tf for tf in dataframes.keys() if tf in hierarchy]
        if not available:
            return list(dataframes.keys())[0]
        return min(available, key=lambda x: hierarchy[x])
    
    elif target == 'coarsest':
        # Find coarsest (highest hierarchy level)
        available = [tf for tf in dataframes.keys() if tf in hierarchy]
        if not available:
            return list(dataframes.keys())[0]
        return max(available, key=lambda x: hierarchy[x])
    
    elif target in dataframes:
        return target
    
    else:
        warning(f"Target timeframe '{target}' not found, using finest")
        return select_target_timeframe(dataframes, 'finest')

