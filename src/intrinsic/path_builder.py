"""
Intrinsic Path Builder - Output file naming and directory management
Constructs paths for intrinsic-time Parquet files
"""

import os
from src.data.path_builder import sanitize_symbol
from src.data.timeframes import INTRINSIC_MAP


def intrinsic_parquet_path(base_dir: str, symbol: str, delta: float) -> str:
    """
    Construct path for intrinsic-time Parquet file.
    
    Example output:
       data/raw/intrinsic/EURUSD_dc010.parquet
    
    Args:
        base_dir: Base directory (e.g., "data/raw/intrinsic")
        symbol: Trading symbol (e.g., "EURUSD")
        delta: Directional-change threshold (e.g., 0.0010)
    
    Returns:
        Full file path
    """
    sym = sanitize_symbol(symbol)
    
    # Get suffix from INTRINSIC_MAP
    if delta not in INTRINSIC_MAP:
        raise ValueError(f"Delta {delta} not found in INTRINSIC_MAP. Available: {list(INTRINSIC_MAP.keys())}")
    
    suffix = INTRINSIC_MAP[delta]["suffix"]
    return os.path.join(base_dir, f"{sym}_{suffix}.parquet")


def delta_to_suffix(delta: float) -> str:
    """
    Convert delta threshold to file suffix.
    
    Args:
        delta: Directional-change threshold (e.g., 0.0010)
    
    Returns:
        Suffix string (e.g., "dc010")
    """
    if delta not in INTRINSIC_MAP:
        raise ValueError(f"Delta {delta} not found in INTRINSIC_MAP")
    return INTRINSIC_MAP[delta]["suffix"]

