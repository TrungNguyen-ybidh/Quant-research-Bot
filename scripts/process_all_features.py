"""
Batch Feature Engineering - Process all symbols
Runs feature engineering on all symbols and saves to data/processed
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from typing import List, Optional
from src.utils.logger import info, success, error, warning
from src.utils.symbols_loader import load_fx_pairs
from src.utils.config_loader import ConfigLoader
from src.data.loader import load_parquet
from src.data.path_builder import sanitize_symbol
from src.feature_engineering import FeatureGenerator
from src.features.transformations import FeatureTransformer


def process_symbol_features(
    symbol: str,
    timeframe: str = "1 hour",
    input_dir: str = "data/raw/clock",
    output_dir: str = "data/processed",
    apply_transformations: bool = False,
    transform_config: Optional[dict] = None
) -> bool:
    """
    Process features for a single symbol.
    
    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Timeframe to use (default: "1 hour")
        input_dir: Input directory for raw data (default: "data/raw/clock")
        output_dir: Output directory for processed features (default: "data/processed")
        apply_transformations: Whether to apply transformations (default: False)
        transform_config: Optional transformation configuration
    
    Returns:
        True if successful, False otherwise
    """
    info(f"Processing features for {symbol} ({timeframe})")
    
    # Load data
    df = load_parquet(input_dir, symbol, timeframe)
    if df is None or df.empty:
        error(f"Failed to load data for {symbol} {timeframe}")
        return False
    
    # Ensure timestamp index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    if not isinstance(df.index, pd.DatetimeIndex):
        error(f"DataFrame for {symbol} does not have DatetimeIndex")
        return False
    
    # Generate features
    try:
        generator = FeatureGenerator()
        features = generator.compute_all_features(df, track_metadata=True)
        
        if features.empty:
            warning(f"No features generated for {symbol}")
            return False
        
        info(f"Generated {len(features.columns)} features for {symbol}")
        
        # Apply transformations if requested
        if apply_transformations and transform_config:
            transformer = FeatureTransformer()
            features = transformer.transform_dataframe(features, transform_config)
            info(f"Applied transformations to {symbol}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Build output path
        sanitized_symbol = sanitize_symbol(symbol)
        from src.data.timeframes import timeframe_to_suffix
        tf_suffix = timeframe_to_suffix(timeframe)
        output_path = os.path.join(output_dir, f"{sanitized_symbol}_{tf_suffix}_features.parquet")
        
        # Save features
        # Extract metadata before saving (metadata can't be serialized to Parquet)
        metadata = features.attrs.pop('metadata', None)
        
        # Save features DataFrame
        features.to_parquet(output_path, engine="pyarrow", index=True)
        
        # Save metadata separately (as JSON)
        if metadata:
            import json
            metadata_path = output_path.replace('.parquet', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        success(f"Saved features for {symbol}: {output_path} ({len(features)} rows, {len(features.columns)} features)")
        return True
        
    except Exception as e:
        error(f"Error processing {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_all_symbols(
    timeframe: str = "1 hour",
    input_dir: str = "data/raw/clock",
    output_dir: str = "data/processed",
    symbols: Optional[List[str]] = None,
    apply_transformations: bool = False,
    transform_config: Optional[dict] = None
) -> dict:
    """
    Process features for all symbols with a single timeframe.
    
    Args:
        timeframe: Timeframe to use (default: "1 hour")
        input_dir: Input directory for raw data (default: "data/raw/clock")
        output_dir: Output directory for processed features (default: "data/processed")
        symbols: Optional list of symbols (if None, loads from symbols.yaml)
        apply_transformations: Whether to apply transformations (default: False)
        transform_config: Optional transformation configuration
    
    Returns:
        Dictionary with processing statistics
    """
    info("=" * 60)
    info("Starting batch feature engineering for all symbols")
    info("=" * 60)
    
    # Load symbols
    if symbols is None:
        symbols = load_fx_pairs()
    
    if not symbols:
        error("No symbols found to process")
        return {'success': 0, 'failed': 0, 'total': 0}
    
    info(f"Found {len(symbols)} symbols to process: {symbols}")
    info(f"Processing timeframe: {timeframe}")
    
    # Process each symbol
    success_count = 0
    failed_count = 0
    failed_symbols = []
    
    for i, symbol in enumerate(symbols, 1):
        info(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
        
        if process_symbol_features(
            symbol=symbol,
            timeframe=timeframe,
            input_dir=input_dir,
            output_dir=output_dir,
            apply_transformations=apply_transformations,
            transform_config=transform_config
        ):
            success_count += 1
        else:
            failed_count += 1
            failed_symbols.append(symbol)
    
    # Summary
    info("\n" + "=" * 60)
    info("Batch Feature Engineering Complete")
    info("=" * 60)
    info(f"Total symbols: {len(symbols)}")
    info(f"Successful: {success_count}")
    info(f"Failed: {failed_count}")
    
    if failed_symbols:
        warning(f"Failed symbols: {', '.join(failed_symbols)}")
    
    return {
        'success': success_count,
        'failed': failed_count,
        'total': len(symbols),
        'failed_symbols': failed_symbols
    }


def discover_available_symbols(input_dir: str = "data/raw/clock") -> List[str]:
    """
    Discover symbols that have data files in the input directory.
    
    Args:
        input_dir: Input directory to scan for Parquet files
    
    Returns:
        List of available symbols
    """
    from pathlib import Path
    
    input_path = Path(input_dir)
    if not input_path.exists():
        warning(f"Input directory does not exist: {input_dir}")
        return []
    
    # Find all Parquet files
    parquet_files = list(input_path.glob("*.parquet"))
    
    if not parquet_files:
        warning(f"No Parquet files found in {input_dir}")
        return []
    
    # Extract unique symbols from filenames
    # Format: SYMBOL_TIMEFRAME.parquet (e.g., EURUSD_1m.parquet)
    symbols = set()
    for file in parquet_files:
        # Extract symbol (everything before the last underscore)
        parts = file.stem.rsplit('_', 1)
        if len(parts) == 2:
            symbol = parts[0]
            symbols.add(symbol)
    
    return sorted(list(symbols))


def process_all_symbols_all_timeframes(
    timeframes: Optional[List[str]] = None,
    input_dir: str = "data/raw/clock",
    output_dir: str = "data/processed",
    symbols: Optional[List[str]] = None,
    apply_transformations: bool = False,
    transform_config: Optional[dict] = None,
    discover_symbols: bool = True
) -> dict:
    """
    Process features for all symbols and all timeframes.
    
    Args:
        timeframes: List of timeframes to process (if None, loads from config)
        input_dir: Input directory for raw data (default: "data/raw/clock")
        output_dir: Output directory for processed features (default: "data/processed")
        symbols: Optional list of symbols (if None, discovers from data directory)
        apply_transformations: Whether to apply transformations (default: False)
        transform_config: Optional transformation configuration
        discover_symbols: If True, discover symbols from data directory (default: True)
    
    Returns:
        Dictionary with processing statistics
    """
    info("=" * 60)
    info("Starting batch feature engineering for all symbols and all timeframes")
    info("=" * 60)
    
    # Load timeframes from config if not provided
    if timeframes is None:
        config_loader = ConfigLoader(config_path="config/config.yaml")
        cfg = config_loader.config
        timeframes = cfg.get('timeframes', ["1 hour"])
    
    # Discover or load symbols
    if symbols is None:
        if discover_symbols:
            symbols = discover_available_symbols(input_dir)
            info(f"Discovered {len(symbols)} symbols from data directory: {symbols}")
        else:
            symbols = load_fx_pairs()
            info(f"Loaded {len(symbols)} symbols from config: {symbols}")
    
    if not symbols:
        error("No symbols found to process")
        return {'success': 0, 'failed': 0, 'total_combinations': 0}
    
    if not timeframes:
        error("No timeframes found to process")
        return {'success': 0, 'failed': 0, 'total_combinations': 0}
    
    info(f"Found {len(symbols)} symbols to process: {symbols}")
    info(f"Found {len(timeframes)} timeframes to process: {timeframes}")
    
    total_combinations = len(symbols) * len(timeframes)
    info(f"Total combinations to process: {total_combinations}")
    
    # Process each symbol and timeframe combination
    success_count = 0
    failed_count = 0
    failed_combinations = []
    current_combination = 0
    
    for symbol in symbols:
        for timeframe in timeframes:
            current_combination += 1
            info(f"\n[{current_combination}/{total_combinations}] Processing {symbol} ({timeframe})...")
            
            if process_symbol_features(
                symbol=symbol,
                timeframe=timeframe,
                input_dir=input_dir,
                output_dir=output_dir,
                apply_transformations=apply_transformations,
                transform_config=transform_config
            ):
                success_count += 1
            else:
                failed_count += 1
                failed_combinations.append(f"{symbol}_{timeframe}")
    
    # Summary
    info("\n" + "=" * 60)
    info("Batch Feature Engineering Complete")
    info("=" * 60)
    info(f"Total combinations: {total_combinations}")
    info(f"Successful: {success_count}")
    info(f"Failed: {failed_count}")
    
    if failed_combinations:
        warning(f"Failed combinations ({len(failed_combinations)}): {', '.join(failed_combinations[:10])}...")
    
    return {
        'success': success_count,
        'failed': failed_count,
        'total_combinations': total_combinations,
        'failed_combinations': failed_combinations
    }


def main():
    """Main entry point."""
    # Load configuration
    config_loader = ConfigLoader(config_path="config/config.yaml")
    cfg = config_loader.config
    
    # Get paths from config
    data_paths = cfg.get('storage', {}).get('data_paths', {})
    input_dir = data_paths.get('raw_clock', 'data/raw/clock')
    output_dir = data_paths.get('processed', data_paths.get('features', 'data/processed'))
    
    # Get ALL timeframes from config
    timeframes = cfg.get('timeframes', ["1 hour"])
    
    # Optional: Transformation config
    transform_config = {
        'realized_vol_15m': 'zscore',
        'realized_vol_5m': 'zscore',
        'skew': 'percentile',
        'kurtosis': 'winsorize',
        'trend_strength': 'robust'
    }
    
    # Process all symbols and all timeframes
    # discover_symbols=True means only process symbols that have data files
    results = process_all_symbols_all_timeframes(
        timeframes=timeframes,  # Process all timeframes
        input_dir=input_dir,
        output_dir=output_dir,
        discover_symbols=True,  # Only process symbols with data files
        apply_transformations=False,  # Set to True to apply transformations
        transform_config=transform_config if False else None
    )
    
    success(f"\nProcessing complete: {results['success']}/{results['total_combinations']} successful")


if __name__ == "__main__":
    main()

