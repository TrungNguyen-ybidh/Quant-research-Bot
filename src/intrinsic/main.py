"""
Intrinsic-Time Engine Main Orchestrator
Processes clock-time data into intrinsic-time events
"""

import os
import pandas as pd
from typing import Dict, Any, List
from src.utils.logger import info, success, error, warning
from src.intrinsic.loader import load_clock_data
from src.intrinsic.event_builder import build_intrinsic_events
from src.intrinsic.path_builder import intrinsic_parquet_path
from src.intrinsic.validators import validate_intrinsic_events
from src.intrinsic.metadata import get_intrinsic_metadata_tracker


def process_symbol_delta(
    symbol: str,
    delta: float,
    base_dir: str = "data/raw/intrinsic",
    clock_dir: str = "data/raw/clock"
) -> bool:
    """
    Process a single symbol/delta combination.
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        delta: Directional-change threshold (e.g., 0.0010)
        base_dir: Output directory for intrinsic data
        clock_dir: Input directory for clock-time data
    
    Returns:
        True if successful, False otherwise
    """
    info(f"Processing {symbol} with delta {delta}")
    
    # Load clock-time data
    df = load_clock_data(symbol, clock_dir)
    if df is None or df.empty:
        error(f"Failed to load clock data for {symbol}")
        return False
    
    # Build intrinsic events
    events_df = build_intrinsic_events(df, delta)
    if events_df.empty:
        warning(f"No events generated for {symbol} with delta {delta}")
        return False
    
    # Validate events
    if not validate_intrinsic_events(events_df):
        error(f"Validation failed for {symbol} with delta {delta}")
        return False
    
    # Build output path
    output_path = intrinsic_parquet_path(base_dir, symbol, delta)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to Parquet
    try:
        events_df.to_parquet(output_path, engine="pyarrow", index=False)
        success(f"Saved intrinsic events: {output_path} ({len(events_df)} events)")
    except Exception as e:
        error(f"Failed to save intrinsic events: {e}")
        return False
    
    # Update metadata
    try:
        dc_start = events_df['dc_timestamp'].min() if not events_df.empty else None
        dc_end = events_df['dc_timestamp'].max() if not events_df.empty else None
        source_clock_file = f"{symbol}_1m.parquet"
        tracker = get_intrinsic_metadata_tracker()
        tracker.record_update(symbol, delta, len(events_df), dc_start, dc_end, source_clock_file)
    except Exception as e:
        warning(f"Failed to update metadata: {e}")
    
    return True


def run_intrinsic_engine(cfg: Dict[str, Any] = None) -> None:
    """
    Main entry point for intrinsic-time engine.
    
    Args:
        cfg: Configuration dictionary (if None, loads from config.yaml)
    """
    if cfg is None:
        from src.utils.config_loader import ConfigLoader
        config_loader = ConfigLoader(config_path="config/config.yaml")
        cfg = config_loader.config
    
    process_all_intrinsic(cfg)


def process_all_intrinsic(cfg: Dict[str, Any]) -> None:
    """
    Process all symbols and deltas from configuration.
    
    Discovers all symbols in clock directory and processes each with all thresholds.
    
    Args:
        cfg: Configuration dictionary from config.yaml
    """
    info("Starting intrinsic-time engine")
    
    # Get thresholds from config
    intrinsic_config = cfg.get('intrinsic_time', {})
    thresholds = intrinsic_config.get('thresholds', [])
    
    if not thresholds:
        error("No thresholds found in intrinsic_time configuration")
        return
    
    info(f"Found {len(thresholds)} thresholds: {thresholds}")
    
    # Get directories
    base_dir = cfg.get('storage', {}).get('data_paths', {}).get('raw_intrinsic', 'data/raw/intrinsic')
    clock_dir = cfg.get('storage', {}).get('data_paths', {}).get('raw_clock', 'data/raw/clock')
    
    # Discover all symbols in clock directory
    from pathlib import Path
    clock_path = Path(clock_dir)
    
    if not clock_path.exists():
        error(f"Clock directory not found: {clock_dir}")
        return
    
    # Find all 1-minute parquet files
    parquet_files = list(clock_path.glob("*_1m.parquet"))
    
    if not parquet_files:
        error(f"No 1-minute parquet files found in {clock_dir}")
        return
    
    # Extract symbols from filenames
    symbols = []
    for file in parquet_files:
        # Extract symbol from filename: SYMBOL_1m.parquet
        symbol = file.stem.replace("_1m", "")
        symbols.append(symbol)
    
    symbols = sorted(set(symbols))  # Remove duplicates and sort
    info(f"Discovered {len(symbols)} symbols in clock directory: {symbols}")
    
    # Process each symbol/delta combination
    total = len(symbols) * len(thresholds)
    processed = 0
    failed = 0
    
    for symbol in symbols:
        for delta in thresholds:
            if process_symbol_delta(symbol, delta, base_dir, clock_dir):
                processed += 1
            else:
                failed += 1
    
    success(f"Intrinsic-time engine complete: {processed}/{total} successful, {failed} failed")


if __name__ == "__main__":
    """
    CLI entry point for intrinsic-time engine.
    
    Usage:
        python -m src.intrinsic.main
    """
    try:
        run_intrinsic_engine()
    except Exception as e:
        error(f"Fatal error: {e}")
        raise

