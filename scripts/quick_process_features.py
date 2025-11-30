"""
Quick Feature Processing Script
Process EURUSD for all timeframes
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.data.loader import load_parquet
from src.feature_engineering import FeatureGenerator
from src.data.path_builder import sanitize_symbol
from src.data.timeframes import timeframe_to_suffix
import pandas as pd
import os

# Configuration
SYMBOL = "EURUSD"  # Process EURUSD only
INPUT_DIR = "data/raw/clock"
OUTPUT_DIR = "data/processed"

# Load timeframes from config
config_loader = ConfigLoader(config_path="config/config.yaml")
cfg = config_loader.config
timeframes = cfg.get('timeframes', ["1 hour"])

print(f"Processing {SYMBOL} for {len(timeframes)} timeframes: {timeframes}")

# Initialize generator
generator = FeatureGenerator()

# Process each timeframe
success_count = 0
failed_count = 0

for i, timeframe in enumerate(timeframes, 1):
    print(f"\n[{i}/{len(timeframes)}] Processing {SYMBOL} ({timeframe})...")
    
    try:
        # Load data
        df = load_parquet(INPUT_DIR, SYMBOL, timeframe)
        if df is None:
            print(f"  ❌ Failed to load {SYMBOL} {timeframe}")
            failed_count += 1
            continue
        
        # Set timestamp index if needed
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        # Generate features
        features = generator.compute_all_features(df, track_metadata=True)
        
        if features.empty:
            print(f"  ❌ No features generated for {SYMBOL} {timeframe}")
            failed_count += 1
            continue
        
        # Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        sanitized_symbol = sanitize_symbol(SYMBOL)
        tf_suffix = timeframe_to_suffix(timeframe)
        output_path = os.path.join(OUTPUT_DIR, f"{sanitized_symbol}_{tf_suffix}_features.parquet")
        
        # Extract metadata before saving (metadata can't be serialized to Parquet)
        metadata = features.attrs.pop('metadata', None)
        
        # Save features DataFrame
        features.to_parquet(output_path, engine="pyarrow", index=True)
        
        # Save metadata separately as JSON
        if metadata:
            import json
            metadata_path = os.path.join(OUTPUT_DIR, f"{sanitized_symbol}_{tf_suffix}_features_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        print(f"  ✅ Saved {len(features.columns)} features to {output_path}")
        success_count += 1
        
    except Exception as e:
        print(f"  ❌ Error processing {SYMBOL} {timeframe}: {e}")
        import traceback
        traceback.print_exc()
        failed_count += 1

print(f"\n✅ Complete: {success_count} successful, {failed_count} failed")
