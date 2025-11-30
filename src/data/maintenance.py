"""
Maintenance Utilities - Rebuild metadata from existing Parquet files
"""

import os
from pathlib import Path
from typing import Dict, Optional
from src.utils.logger import info, success, error, warning
from src.data.loader import load_parquet
from src.data.metadata import MetadataTracker
from src.data.timeframes import TIMEFRAME_MAP


def rebuild_metadata(base_dir: str = "data/raw/clock", metadata_file: str = "data/raw/metadata.json"):
    """
    Rebuild metadata.json by scanning existing Parquet files.
    
    Scans data/raw/clock/ for all *.parquet files, loads each to get
    row count and last timestamp, then rebuilds metadata entries.
    
    Args:
        base_dir: Directory to scan for Parquet files (default: "data/raw/clock")
        metadata_file: Path to metadata JSON file (default: "data/raw/metadata.json")
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        error(f"Directory not found: {base_dir}")
        return
    
    info(f"Scanning {base_dir} for Parquet files...")
    
    # Find all parquet files
    parquet_files = list(base_path.glob("*.parquet"))
    
    if not parquet_files:
        warning(f"No Parquet files found in {base_dir}")
        return
    
    info(f"Found {len(parquet_files)} Parquet files")
    
    # Create new metadata tracker (will create fresh metadata)
    tracker = MetadataTracker(metadata_file)
    tracker.metadata = {}  # Start with empty metadata
    
    # Reverse lookup: suffix -> timeframe string
    suffix_to_timeframe = {}
    for timeframe, mapping in TIMEFRAME_MAP.items():
        suffix_to_timeframe[mapping["suffix"]] = timeframe
    
    processed = 0
    errors = 0
    
    for parquet_file in parquet_files:
        # Parse filename: SYMBOL_SUFFIX.parquet -> (SYMBOL, SUFFIX)
        filename = parquet_file.stem  # Remove .parquet extension
        parts = filename.split("_", 1)
        
        if len(parts) != 2:
            warning(f"Skipping {parquet_file.name}: filename format not SYMBOL_SUFFIX")
            errors += 1
            continue
        
        symbol = parts[0]
        suffix = parts[1]
        
        # Find timeframe from suffix
        timeframe = suffix_to_timeframe.get(suffix)
        if not timeframe:
            warning(f"Skipping {parquet_file.name}: suffix '{suffix}' not in TIMEFRAME_MAP")
            errors += 1
            continue
        
        # Load parquet file directly to get metadata
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_file, engine="pyarrow")
            
            if df is None or df.empty:
                warning(f"Skipping {parquet_file.name}: failed to load or empty")
                errors += 1
                continue
            
            # Get row count and last timestamp
            rows = len(df)
            
            if 'timestamp' not in df.columns:
                warning(f"Skipping {parquet_file.name}: no timestamp column")
                errors += 1
                continue
            
            last_ts = df['timestamp'].max()
            
            # Rebuild metadata entry
            tracker.record_update(symbol, timeframe, rows, last_ts)
            processed += 1
            info(f"Processed {parquet_file.name}: {rows} rows, last_ts={last_ts}")
            
        except Exception as e:
            error(f"Error processing {parquet_file.name}: {e}")
            errors += 1
            continue
    
    success(f"Metadata rebuild complete: {processed} files processed, {errors} errors")
    info(f"Metadata saved to {metadata_file}")


if __name__ == "__main__":
    """
    CLI entry point for maintenance utilities.
    
    Usage:
        python -m src.data.maintenance
    """
    rebuild_metadata()

