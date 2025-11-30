"""
Feature Engineering Example
Demonstrates how to use the feature engineering system
"""

import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.feature_engineering import FeatureGenerator
from src.data.loader import load_parquet


def example_basic_usage():
    """Example: Basic feature generation"""
    print("=" * 60)
    print("Example 1: Basic Feature Generation")
    print("=" * 60)
    
    # Load some data (example)
    # df = load_parquet("data/raw/clock", "EURUSD", "1 hour")
    
    # Create sample data for demonstration
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 1.1000 + pd.Series(range(100)) * 0.0001,
        'high': 1.1005 + pd.Series(range(100)) * 0.0001,
        'low': 0.9995 + pd.Series(range(100)) * 0.0001,
        'close': 1.1002 + pd.Series(range(100)) * 0.0001,
        'volume': 1000 + pd.Series(range(100)) * 10,
        'bid': 1.1000 + pd.Series(range(100)) * 0.0001,
        'ask': 1.1004 + pd.Series(range(100)) * 0.0001,
    })
    df = df.set_index('timestamp')
    
    # Initialize feature generator
    generator = FeatureGenerator()
    
    # Compute all features
    features_df = generator.compute_all_features(df)
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Features shape: {features_df.shape}")
    print(f"\nComputed features:")
    print(features_df.columns.tolist()[:10])  # Show first 10
    
    return features_df


def example_category_features():
    """Example: Compute features from specific category"""
    print("\n" + "=" * 60)
    print("Example 2: Category-Specific Features")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 1.1000 + pd.Series(range(100)) * 0.0001,
        'high': 1.1005 + pd.Series(range(100)) * 0.0001,
        'low': 0.9995 + pd.Series(range(100)) * 0.0001,
        'close': 1.1002 + pd.Series(range(100)) * 0.0001,
        'volume': 1000 + pd.Series(range(100)) * 10,
    })
    df = df.set_index('timestamp')
    
    # Initialize feature generator
    generator = FeatureGenerator()
    
    # Compute only volatility features
    vol_features = generator.compute_category_features(df, category='volatility')
    
    print(f"\nVolatility features shape: {vol_features.shape}")
    print(f"Volatility features: {vol_features.columns.tolist()}")
    
    return vol_features


def example_single_feature():
    """Example: Compute a single feature"""
    print("\n" + "=" * 60)
    print("Example 3: Single Feature Computation")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 1.1000 + pd.Series(range(100)) * 0.0001,
        'high': 1.1005 + pd.Series(range(100)) * 0.0001,
        'low': 0.9995 + pd.Series(range(100)) * 0.0001,
        'close': 1.1002 + pd.Series(range(100)) * 0.0001,
        'volume': 1000 + pd.Series(range(100)) * 10,
    })
    df = df.set_index('timestamp')
    
    # Initialize feature generator
    generator = FeatureGenerator()
    
    # Compute single feature
    rsi_feature = generator.compute_feature('realized_vol_20m', df)
    
    if rsi_feature is not None:
        print(f"\nFeature computed successfully")
        print(f"Feature shape: {rsi_feature.shape}")
        print(f"First 5 values:\n{rsi_feature.head()}")
    else:
        print("\nFeature computation failed")
    
    return rsi_feature


def example_list_features():
    """Example: List available features"""
    print("\n" + "=" * 60)
    print("Example 4: List Available Features")
    print("=" * 60)
    
    generator = FeatureGenerator()
    
    # List all features
    all_features = generator.list_features()
    print(f"\nTotal features: {len(all_features)}")
    
    # List features by category
    for category in ['volatility', 'trend', 'microstructure']:
        features = generator.list_features(category=category)
        print(f"\n{category} features ({len(features)}):")
        print(f"  {features[:5]}...")  # Show first 5


def example_feature_info():
    """Example: Get feature information"""
    print("\n" + "=" * 60)
    print("Example 5: Feature Information")
    print("=" * 60)
    
    generator = FeatureGenerator()
    
    # Get info about a feature
    info = generator.get_feature_info('realized_vol_20m')
    
    if info:
        print(f"\nFeature info:")
        print(f"  Module: {info.get('module')}")
        print(f"  Function: {info.get('function')}")
        print(f"  Description: {info.get('description')}")
        print(f"  Enabled: {info.get('enabled')}")
        print(f"  Params: {info.get('params')}")


if __name__ == "__main__":
    print("Feature Engineering System - Usage Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_usage()
        example_category_features()
        example_single_feature()
        example_list_features()
        example_feature_info()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

