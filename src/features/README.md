# Feature Engineering System

**Institution-grade** feature engineering system for quantitative research.

## Overview

This module provides a comprehensive, extensible feature engineering system that:

- Computes 50-200+ features consistently
- Uses YAML registry for automatic feature registration
- Allows adding new features with ONE YAML line + ONE function
- Produces clean ML-ready feature matrices
- Works within retail limitations (IBKR, Oanda, etc.)

### 🎯 Institution-Grade Features

- ✅ **Multi-Timeframe Data Manager** - Align and merge features from multiple timeframes
- ✅ **Feature Dependency System** - Automatic dependency resolution and ordering
- ✅ **Transformation Layer** - Pre-ML feature scaling and normalization
- ✅ **Schema Version Control** - Full reproducibility tracking

See [INSTITUTION_GRADE.md](INSTITUTION_GRADE.md) for detailed documentation on advanced features.

## Directory Structure

```
src/features/
├── __init__.py              # Module exports
├── registry.yaml            # Feature registry (YAML)
├── intrinsic_time.py        # DC/intrinsic-time features
├── volatility.py            # Volatility features
├── microstructure.py        # Microstructure features
├── trend.py                 # Trend features
├── fx_factors.py            # FX-specific factors
├── cross_asset.py           # Cross-asset features
├── session.py               # Trading session features
├── liquidity.py             # Liquidity features
└── utils.py                 # Utility functions

src/feature_engineering/
├── __init__.py
└── feature_generator.py     # Main orchestrator
```

## Quick Start

### Basic Usage

```python
from src.feature_engineering import FeatureGenerator
from src.data.loader import load_parquet

# Load data
df = load_parquet("data/raw/clock", "EURUSD", "1 hour")

# Initialize generator
generator = FeatureGenerator()

# Compute all features (with dependency resolution and metadata tracking)
features_df = generator.compute_all_features(df, track_metadata=True)

print(f"Computed {len(features_df.columns)} features")
print(f"Schema version: {generator.get_schema_version()}")
```

### Multi-Timeframe Usage

```python
from src.data.multi_timeframe import load_multi_timeframe, align_timeframes
from src.feature_engineering import FeatureGenerator

# Load multiple timeframes
dataframes = load_multi_timeframe("EURUSD", ["1 min", "5 mins", "1 hour"])

# Align to common index
aligned = align_timeframes(dataframes, how='right')

# Generate features
generator = FeatureGenerator()
features = generator.compute_all_features(aligned['1m'])
```

### With Transformations

```python
from src.feature_engineering import FeatureGenerator, FeatureTransformer

# Generate features
generator = FeatureGenerator()
features = generator.compute_all_features(df)

# Apply transformations
transformer = FeatureTransformer()
transform_config = {
    'realized_vol_15m': 'zscore',
    'skew': 'percentile',
    'kurtosis': 'winsorize'
}
transformed = transformer.transform_dataframe(features, transform_config)
```

### Compute Specific Categories

```python
# Compute only volatility features
vol_features = generator.compute_category_features(df, category='volatility')

# Compute only trend features
trend_features = generator.compute_category_features(df, category='trend')
```

### Compute Single Feature

```python
# Compute a single feature
rsi = generator.compute_feature('realized_vol_20m', df)
```

## Feature Categories

### 1. Intrinsic Time Features (`intrinsic_time.py`)

Directional-change and intrinsic-time based features:
- `dc_return_x`: Directional-change return
- `overshoot_return_x`: Overshoot return after DC
- `overshoot_ratio_x`: Ratio of overshoot to DC return
- `event_frequency`: Number of DC events per unit time
- `clustering`: Temporal clustering of events
- `multi_delta_agreement`: Agreement across thresholds
- `intrinsic_trend_strength`: Trend strength in intrinsic time

### 2. Volatility Features (`volatility.py`)

OHLC-based volatility structure:
- `realized_vol_*`: Realized volatility (multiple windows)
- `bipower_variation`: Jump-robust volatility
- `jump_component`: Jump component of volatility
- `volatility_of_vol`: Volatility of volatility
- `parkinson_volatility`: Parkinson volatility estimator
- `skew`: Return skewness
- `kurtosis`: Return kurtosis
- `variance_ratio`: Variance ratio test
- `hurst_exponent`: Hurst exponent (long memory)

### 3. Microstructure Features (`microstructure.py`)

Retail-friendly microstructure features:
- `micro_price`: Microstructure price (bid-ask weighted)
- `synthetic_spread`: Synthetic bid-ask spread
- `relative_spread`: Relative spread (spread / mid price)
- `wick_ratio`: Upper wick to lower wick ratio
- `body_ratio`: Body size to total range ratio
- `rejection_wicks`: Rejection wick indicator
- `standardized_range`: Range standardized by volatility
- `volume_adv_ratio`: Volume to average daily volume ratio

### 4. Trend Features (`trend.py`)

Trend detection and strength:
- `sma_ratio_10_20`: SMA(10) / SMA(20) ratio
- `ema_ratio_12_26`: EMA(12) / EMA(26) ratio
- `adx`: Average Directional Index
- `trend_strength`: Trend strength indicator
- `price_position`: Price position in recent range

### 5. FX Factors (`fx_factors.py`)

FX-specific factor models:
- `carry_factor`: Carry trade factor
- `momentum_factor`: Momentum factor
- `value_factor`: Value factor
- `volatility_factor`: Volatility factor

### 6. Cross-Asset Features (`cross_asset.py`)

Cross-asset relationships:
- `correlation_rolling`: Rolling correlation with other assets
- `relative_strength`: Relative strength vs other assets
- `cross_asset_momentum`: Cross-asset momentum signal

### 7. Session Features (`session.py`)

Trading session indicators:
- `london_session`: London session indicator
- `new_york_session`: New York session indicator
- `asia_session`: Asia session indicator
- `session_overlap`: Session overlap indicator
- `session_volatility`: Volatility by trading session

### 8. Liquidity Features (`liquidity.py`)

Market liquidity measures:
- `bid_ask_spread`: Bid-ask spread
- `spread_percentile`: Spread percentile in distribution
- `liquidity_score`: Composite liquidity score

## Adding New Features

### Step 1: Create Feature Function

Add a function to the appropriate module:

```python
# In src/features/volatility.py
def compute_my_new_feature(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute my new feature.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size
        **kwargs: Additional parameters
    
    Returns:
        Series with feature values
    """
    # Your computation here
    result = df['close'].rolling(window=window).mean()
    return result.fillna(0.0)
```

### Step 2: Register in YAML

Add to `src/features/registry.yaml`:

```yaml
volatility:
  my_new_feature:
    module: volatility
    function: compute_my_new_feature
    description: "My new feature description"
    enabled: true
    params:
      window: 20
```

That's it! The feature is now automatically available.

## Feature Registry

The `registry.yaml` file controls:
- Which features are enabled/disabled
- Feature parameters
- Feature descriptions
- Module and function mappings

### Enabling/Disabling Features

```python
generator = FeatureGenerator()

# Disable a feature
generator.disable_feature('realized_vol_5m')

# Enable a feature
generator.enable_feature('realized_vol_5m')
```

### Listing Features

```python
# List all features
all_features = generator.list_features()

# List features by category
vol_features = generator.list_features(category='volatility')
```

### Getting Feature Info

```python
info = generator.get_feature_info('realized_vol_20m')
print(info['description'])
print(info['params'])
```

## Function Signature

All feature functions must follow this signature:

```python
def compute_feature_name(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute feature.
    
    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional parameters from registry
    
    Returns:
        Series with feature values (same index as df)
    """
    # Implementation
    return pd.Series(...)
```

## Requirements

- pandas
- numpy
- pyyaml
- scipy (for transformations)

## Examples

- `examples/feature_engineering_example.py` - Basic usage examples
- `examples/institution_grade_features_example.py` - Advanced features examples

## Notes

- All features return `pd.Series` with the same index as input DataFrame
- Features handle missing data gracefully (fill with 0.0 or NaN as appropriate)
- Features are vectorized for performance
- Registry-based system allows easy feature management without code changes
- Dependencies are automatically resolved (no manual ordering needed)
- Metadata is tracked for full reproducibility
- Transformations are applied before ML (not during feature computation)

## Advanced Features

For detailed documentation on institution-grade features, see:

- **[INSTITUTION_GRADE.md](INSTITUTION_GRADE.md)** - Complete guide to advanced features
  - Multi-Timeframe Manager - Align and merge features from multiple timeframes
  - Dependency System - Automatic feature dependency resolution
  - Transformation Layer - Pre-ML feature scaling
  - Version Control - Schema versioning and reproducibility

