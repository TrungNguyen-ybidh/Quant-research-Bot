"""
Institution-Grade Feature Engineering Example
Demonstrates all advanced features: multi-timeframe, dependencies, transformations, versioning
"""

import pandas as pd
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import FeatureGenerator, FeatureTransformer
from src.data.multi_timeframe import align_timeframes, load_multi_timeframe, merge_multi_timeframe_features


def example_multi_timeframe():
    """Example: Multi-timeframe feature engineering"""
    print("=" * 60)
    print("Example 1: Multi-Timeframe Feature Engineering")
    print("=" * 60)
    
    # Create sample data for different timeframes
    dates_1m = pd.date_range('2024-01-01', periods=100, freq='1min')
    dates_5m = pd.date_range('2024-01-01', periods=20, freq='5min')
    dates_1h = pd.date_range('2024-01-01', periods=4, freq='1H')
    
    df_1m = pd.DataFrame({
        'open': 1.1000 + pd.Series(range(100)) * 0.0001,
        'high': 1.1005 + pd.Series(range(100)) * 0.0001,
        'low': 0.9995 + pd.Series(range(100)) * 0.0001,
        'close': 1.1002 + pd.Series(range(100)) * 0.0001,
        'volume': 1000 + pd.Series(range(100)) * 10,
    }, index=dates_1m)
    
    df_5m = pd.DataFrame({
        'open': 1.1000 + pd.Series(range(20)) * 0.0005,
        'high': 1.1005 + pd.Series(range(20)) * 0.0005,
        'low': 0.9995 + pd.Series(range(20)) * 0.0005,
        'close': 1.1002 + pd.Series(range(20)) * 0.0005,
        'volume': 5000 + pd.Series(range(20)) * 50,
    }, index=dates_5m)
    
    df_1h = pd.DataFrame({
        'open': 1.1000 + pd.Series(range(4)) * 0.002,
        'high': 1.1005 + pd.Series(range(4)) * 0.002,
        'low': 0.9995 + pd.Series(range(4)) * 0.002,
        'close': 1.1002 + pd.Series(range(4)) * 0.002,
        'volume': 30000 + pd.Series(range(4)) * 300,
    }, index=dates_1h)
    
    # Align timeframes
    dataframes = {
        '1m': df_1m,
        '5m': df_5m,
        '1h': df_1h
    }
    
    aligned = align_timeframes(dataframes, how='right', method='ffill')
    
    print(f"\nOriginal timeframes:")
    print(f"  1m: {len(df_1m)} rows")
    print(f"  5m: {len(df_5m)} rows")
    print(f"  1h: {len(df_1h)} rows")
    
    print(f"\nAligned timeframes:")
    for tf, df in aligned.items():
        print(f"  {tf}: {len(df)} rows")
    
    # Generate features for each timeframe
    generator = FeatureGenerator()
    
    features_by_tf = {}
    for tf, df in aligned.items():
        print(f"\nComputing features for {tf}...")
        features = generator.compute_all_features(df)
        features_by_tf[tf] = features
        print(f"  Generated {len(features.columns)} features")
    
    # Merge features
    merged_features = merge_multi_timeframe_features(features_by_tf, prefix_timeframe=True)
    print(f"\nMerged features shape: {merged_features.shape}")
    print(f"Total features: {len(merged_features.columns)}")
    
    return merged_features


def example_dependency_system():
    """Example: Feature dependency resolution"""
    print("\n" + "=" * 60)
    print("Example 2: Feature Dependency System")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'open': 1.1000 + pd.Series(range(100)) * 0.0001,
        'high': 1.1005 + pd.Series(range(100)) * 0.0001,
        'low': 0.9995 + pd.Series(range(100)) * 0.0001,
        'close': 1.1002 + pd.Series(range(100)) * 0.0001,
        'volume': 1000 + pd.Series(range(100)) * 10,
    }, index=dates)
    
    generator = FeatureGenerator()
    
    # Get dependency graph
    dependency_graph = generator._build_dependency_graph()
    
    print(f"\nFeature dependencies found:")
    for feature, deps in dependency_graph.items():
        print(f"  {feature} depends on: {deps}")
    
    # Compute features (dependencies will be resolved automatically)
    print(f"\nComputing features with dependency resolution...")
    features = generator.compute_all_features(df, track_metadata=True)
    
    print(f"\nComputed {len(features.columns)} features")
    print(f"Features computed in dependency order")
    
    # Show metadata
    metadata = generator.get_metadata()
    print(f"\nComputation metadata:")
    print(f"  Schema version: {metadata.get('schema_version')}")
    print(f"  Feature version: {metadata.get('feature_version')}")
    print(f"  Timestamp: {metadata.get('timestamp')}")
    print(f"  Features computed: {len(metadata.get('computed_features', {}))}")
    
    return features


def example_transformations():
    """Example: Feature transformations"""
    print("\n" + "=" * 60)
    print("Example 3: Feature Transformations")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'open': 1.1000 + pd.Series(range(100)) * 0.0001,
        'high': 1.1005 + pd.Series(range(100)) * 0.0001,
        'low': 0.9995 + pd.Series(range(100)) * 0.0001,
        'close': 1.1002 + pd.Series(range(100)) * 0.0001,
        'volume': 1000 + pd.Series(range(100)) * 10,
    }, index=dates)
    
    # Generate features
    generator = FeatureGenerator()
    features = generator.compute_all_features(df)
    
    print(f"\nOriginal features shape: {features.shape}")
    
    # Apply transformations
    transformer = FeatureTransformer()
    
    # Configure transformations
    transform_config = {
        'realized_vol_15m': 'zscore',
        'realized_vol_5m': {'type': 'minmax', 'window': 20},
        'skew': 'percentile',
        'kurtosis': {'type': 'winsorize', 'limits': (0.05, 0.05)},
        'trend_strength': {'type': 'robust', 'window': 20}
    }
    
    # Transform features
    transformed = transformer.transform_dataframe(features, transform_config)
    
    print(f"\nTransformed features shape: {transformed.shape}")
    print(f"\nTransformation summary:")
    for col in transform_config.keys():
        if col in features.columns:
            orig_mean = features[col].mean()
            trans_mean = transformed[col].mean()
            print(f"  {col}: mean {orig_mean:.4f} -> {trans_mean:.4f}")
    
    return transformed


def example_version_control():
    """Example: Schema version control and reproducibility"""
    print("\n" + "=" * 60)
    print("Example 4: Schema Version Control")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'open': 1.1000 + pd.Series(range(100)) * 0.0001,
        'high': 1.1005 + pd.Series(range(100)) * 0.0001,
        'low': 0.9995 + pd.Series(range(100)) * 0.0001,
        'close': 1.1002 + pd.Series(range(100)) * 0.0001,
        'volume': 1000 + pd.Series(range(100)) * 10,
    }, index=dates)
    
    generator = FeatureGenerator()
    
    # Get version info
    schema_version = generator.get_schema_version()
    feature_version = generator.get_feature_version()
    
    print(f"\nVersion Information:")
    print(f"  Schema version: {schema_version}")
    print(f"  Feature version: {feature_version}")
    
    # Generate features with metadata tracking
    features = generator.compute_all_features(df, track_metadata=True)
    
    # Access metadata
    metadata = features.attrs.get('metadata', {})
    
    print(f"\nFeature Computation Metadata:")
    print(f"  Schema version: {metadata.get('schema_version')}")
    print(f"  Feature version: {metadata.get('feature_version')}")
    print(f"  Timestamp: {metadata.get('timestamp')}")
    print(f"  Features computed: {len(metadata.get('computed_features', {}))}")
    
    # Show sample feature metadata
    computed = metadata.get('computed_features', {})
    if computed:
        sample_feature = list(computed.keys())[0]
        sample_meta = computed[sample_feature]
        print(f"\nSample feature metadata ({sample_feature}):")
        print(f"  Category: {sample_meta.get('category')}")
        print(f"  Computation time: {sample_meta.get('computation_time'):.4f}s")
        print(f"  Shape: {sample_meta.get('shape')}")
        print(f"  Null count: {sample_meta.get('null_count')}")
    
    # Save metadata for reproducibility
    print(f"\nMetadata saved in DataFrame.attrs for reproducibility")
    
    return features


def example_complete_pipeline():
    """Example: Complete institution-grade pipeline"""
    print("\n" + "=" * 60)
    print("Example 5: Complete Institution-Grade Pipeline")
    print("=" * 60)
    
    # Step 1: Load multi-timeframe data
    print("\nStep 1: Multi-timeframe data loading")
    # (In practice, would load from files)
    # dataframes = load_multi_timeframe("EURUSD", ["1 min", "5 mins", "1 hour"])
    
    # Step 2: Align timeframes
    print("Step 2: Align timeframes")
    # aligned = align_timeframes(dataframes, how='right')
    
    # Step 3: Generate features with dependencies
    print("Step 3: Generate features with dependency resolution")
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'open': 1.1000 + pd.Series(range(100)) * 0.0001,
        'high': 1.1005 + pd.Series(range(100)) * 0.0001,
        'low': 0.9995 + pd.Series(range(100)) * 0.0001,
        'close': 1.1002 + pd.Series(range(100)) * 0.0001,
        'volume': 1000 + pd.Series(range(100)) * 10,
    }, index=dates)
    
    generator = FeatureGenerator()
    features = generator.compute_all_features(df, track_metadata=True)
    
    # Step 4: Apply transformations
    print("Step 4: Apply feature transformations")
    transformer = FeatureTransformer()
    transform_config = {
        'realized_vol_15m': 'zscore',
        'skew': 'percentile',
        'kurtosis': 'winsorize'
    }
    transformed = transformer.transform_dataframe(features, transform_config)
    
    # Step 5: Access version and metadata
    print("Step 5: Version control and metadata")
    metadata = features.attrs.get('metadata', {})
    
    print(f"\nComplete Pipeline Summary:")
    print(f"  Schema version: {metadata.get('schema_version')}")
    print(f"  Feature version: {metadata.get('feature_version')}")
    print(f"  Timestamp: {metadata.get('timestamp')}")
    print(f"  Features generated: {len(features.columns)}")
    print(f"  Features transformed: {len(transform_config)}")
    print(f"  Final shape: {transformed.shape}")
    
    return transformed


if __name__ == "__main__":
    print("Institution-Grade Feature Engineering System")
    print("=" * 60)
    
    try:
        # Run all examples
        example_multi_timeframe()
        example_dependency_system()
        example_transformations()
        example_version_control()
        example_complete_pipeline()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

