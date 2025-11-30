# Quantitative Research Bot

A production-grade quantitative research system for FX trading, featuring institutional-quality data collection, intrinsic-time transformation, and comprehensive feature engineering.

## 🎯 Overview

This system provides a complete pipeline for quantitative FX research:

1. **Data Collection** - Automated data fetching from Interactive Brokers (IBKR)
2. **Intrinsic-Time Transformation** - Convert clock-time data to event-driven intrinsic time
3. **Feature Engineering** - Generate 50-200+ features with dependency resolution
4. **Multi-Timeframe Support** - Align and merge features across timeframes
5. **ML-Ready Output** - Clean feature matrices with full reproducibility tracking

### Key Features

- ✅ **Automated Data Collection** - Incremental updates from IBKR with robust error handling
- ✅ **Intrinsic-Time Processing** - Directional-change (DC) event detection and overshoot tracking
- ✅ **Institution-Grade Feature Engineering** - 50-200+ features across 8 categories
- ✅ **YAML-Driven Configuration** - All parameters configurable without code changes
- ✅ **Multi-Timeframe Alignment** - Seamlessly combine features from different timeframes
- ✅ **Feature Dependency System** - Automatic dependency resolution and computation ordering
- ✅ **Schema Version Control** - Full reproducibility with metadata tracking
- ✅ **Robust Data Handling** - Recovery mechanisms for corrupted Parquet files
- ✅ **Production-Ready** - Comprehensive logging, validation, and error handling

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Intrinsic Time](#intrinsic-time)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Scripts](#scripts)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Interactive Brokers TWS or IB Gateway (for data collection)
- 10GB+ free disk space (for historical data)

### Setup

1. **Clone the repository**
   
   git clone <repository-url>
   cd quant-research-bot
   2. **Create a virtual environment**sh
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   3. **Install dependencies**
   pip install -r requirements.txt
   4. **Configure IBKR Connection**
   - Install and run Interactive Brokers TWS or IB Gateway
   - Enable API connections in TWS/Gateway settings
   - Update `config/config.yaml` with your connection details:
     
     data_source:
       host: "127.0.0.1"
       port: 7497  # 7497=Paper, 7496=Live
       client_id: 10
     5. **Verify Installation**
   python -c "from src.utils.config_loader import ConfigLoader; print('Installation successful!')"
   ## ⚡ Quick Start

### 1. Collect Data

from src.data.data_collector import collect_fx_data

# Collect data for EURUSD
df = collect_fx_data(
    pair="EURUSD",
    timeframe="1 hour",
    incremental=True  # Only fetch new data
)Or use the main script:
python src/main.py### 2. Generate Features
on
from src.feature_engineering import FeatureGenerator
from src.data.loader import load_parquet

# Load data
df = load_parquet("data/raw/clock", "EURUSD", "1 hour")
df = df.set_index('timestamp')

# Generate features
generator = FeatureGenerator()
features = generator.compute_all_features(df, track_metadata=True)

# Save features
features.to_parquet("data/processed/EURUSD_1h_features.parquet")### 3. Process All Symbols

# Process features for all symbols and timeframes
python scripts/process_all_features.py

# Quick process for EURUSD only
python scripts/quick_process_features.py## ⚙️ Configuration

All configuration is managed through YAML files. **No hard-coded values.**

### Main Configuration (`config/config.yaml`)

Key sections:

- **`data_source`** - IBKR connection settings
- **`timeframes`** - List of timeframes to collect
- **`historical_request`** - Data request parameters
- **`intrinsic_time`** - DC thresholds for intrinsic-time conversion
- **`storage`** - Data paths and file formats
- **`ml`** - Machine learning model settings

### Symbols Configuration (`config/symbols.yaml`)

Define FX pairs organized by groups:
- Majors (EURUSD, GBPUSD, USDJPY, etc.)
- Yen Crosses (EURJPY, GBPJPY, etc.)
- Gold (XAUUSD)

See [CONFIG_README.md](CONFIG_README.md) for detailed configuration documentation.

## 🏗️ Architecture

### System Components

```
┌─────────────────┐
│   Data Source   │  (IBKR API)
│   (IBKR TWS)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Collector │  Collect & save raw OHLCV data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Intrinsic Time  │  Convert to event-driven DC events
│   Transformer   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Feature       │  Generate 50-200+ features
│   Engineering   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Pipeline    │  Model training & prediction
└─────────────────┘
```

### Module Structure

- **`src/api/`** - IBKR API client and contract builders
- **`src/data/`** - Data collection, loading, validation, and preprocessing
- **`src/intrinsic/`** - Intrinsic-time transformation (DC events)
- **`src/features/`** - Feature computation modules (8 categories)
- **`src/feature_engineering/`** - Feature orchestration and dependency resolution
- **`src/models/`** - Machine learning models and regime detection
- **`src/utils/`** - Configuration loading, logging, helpers

## 📊 Data Pipeline

### Step 1: Data Collection

Collects OHLCV data from IBKR with incremental updates:

```python
from src.data.data_collector import collect_fx_data

df = collect_fx_data(
    pair="EURUSD",
    timeframe="1 hour",
    incremental=True  # Only fetch new bars
)
```

**Output**: Parquet files in `data/raw/clock/` (e.g., `EURUSD_1h.parquet`)

### Step 2: Intrinsic-Time Transformation

Converts clock-time data to event-driven intrinsic time using directional-change (DC) events:

```python
from src.intrinsic.main import process_symbol_delta

process_symbol_delta(
    symbol="EURUSD",
    delta=0.0010,  # 10 pips threshold
    timeframe="1 hour"
)
```

**Output**: Parquet files in `data/raw/intrinsic/` (e.g., `EURUSD_1h_dc_0.0010.parquet`)

**Key Concepts**:
- **DC Events**: Price movements exceeding a threshold (delta) from previous extreme
- **Overshoot Phase**: Price movement after DC event before next DC event
- **Event Structure**: Each event contains return, duration, overshoot ratio, etc.

See [Intrinsic Time Documentation](#intrinsic-time) for details.

### Step 3: Feature Engineering

Generates comprehensive features from raw and intrinsic-time data:

```python
from src.feature_engineering import FeatureGenerator

generator = FeatureGenerator()
features = generator.compute_all_features(df, track_metadata=True)
```

**Output**: 
- Feature Parquet files in `data/processed/` (e.g., `EURUSD_1h_features.parquet`)
- Metadata JSON files (e.g., `EURUSD_1h_features_metadata.json`)

**Feature Categories**:
1. **Intrinsic-Time Features** - DC returns, overshoot ratios, event clustering
2. **Volatility Features** - Realized vol, bipower variation, jump components
3. **Microstructure Features** - Micro-price, synthetic spread, wick ratios
4. **Trend Features** - Moving averages, momentum indicators
5. **FX Factors** - Carry, momentum, value factors
6. **Cross-Asset Features** - Correlations with other pairs
7. **Session Features** - Trading session indicators
8. **Liquidity Features** - Volume-based liquidity measures

## 🔬 Feature Engineering

### Feature Registry System

Features are defined in `src/features/registry.yaml`:

```yaml
intrinsic_time:
  dc_return_x:
    enabled: true
    module: intrinsic_time
    function: compute_dc_return_x
    depends_on: []
    parameters:
      delta: 0.0010
```

### Adding New Features

1. **Add function to appropriate module** (`src/features/*.py`):
   ```python
   def compute_my_feature(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
       """Compute my custom feature"""
       return df['close'].rolling(window).mean()
   ```

2. **Register in `registry.yaml`**:
   ```yaml
   trend:
     my_feature:
       enabled: true
       module: trend
       function: compute_my_feature
       depends_on: []
       parameters:
         window: 20
   ```

That's it! The feature will be automatically included in feature computation.

### Feature Dependencies

Features can depend on other features:

```yaml
volatility:
  jump_component:
    depends_on: [realized_volatility, bipower_variation]
```

The system automatically resolves dependencies and computes features in the correct order.

### Multi-Timeframe Features

Combine features from multiple timeframes:

```python
from src.data.multi_timeframe import align_timeframes

# Load multiple timeframes
df_1m = load_parquet("data/raw/clock", "EURUSD", "1 min")
df_5m = load_parquet("data/raw/clock", "EURUSD", "5 mins")
df_1h = load_parquet("data/raw/clock", "EURUSD", "1 hour")

# Align to 1-minute bars
aligned = align_timeframes(df_1m, df_5m, df_1h, how='right')

# Generate features on aligned data
features = generator.compute_all_features(aligned['1m'])
```

### Feature Transformations

Apply transformations before ML:

```python
from src.features.transformations import FeatureTransformer

transformer = FeatureTransformer()
transformed = transformer.fit_transform(features, method='zscore')
```

Available transformations:
- `zscore` - Standardization (mean=0, std=1)
- `minmax` - Min-max scaling (0-1 range)
- `robust` - Robust scaling (median and IQR)
- `percentile` - Percentile-based scaling
- `winsorize` - Outlier capping
- `log` - Logarithmic transformation

## ⏱️ Intrinsic Time

### What is Intrinsic Time?

Intrinsic time transforms clock-time data into event-driven data based on price movements, not time. This is crucial for FX markets where volatility varies significantly.

### Directional-Change (DC) Model

**DC Event**: A price movement that exceeds a threshold (delta) from the previous extreme.

**Example** (delta = 0.0010 = 10 pips):
- Price starts at 1.1000
- Price moves to 1.1010 (10 pips up) → **DC Event** (upward)
- Price continues to 1.1025 → **Overshoot Phase** (15 pips overshoot)
- Price reverses to 1.1015 (10 pips down from 1.1025) → **DC Event** (downward)

### Event Structure

Each intrinsic event contains:
- `dc_return` - Return from previous extreme to DC point
- `overshoot_return` - Return from DC point to next extreme
- `overshoot_ratio` - Ratio of overshoot to DC return
- `duration` - Clock-time duration of the event
- `dc_timestamp` - Timestamp of the DC event

### Usage

```python
from src.intrinsic.main import process_symbol_delta

# Process with 10 pips threshold
process_symbol_delta("EURUSD", delta=0.0010, timeframe="1 hour")

# Load intrinsic-time data
from src.intrinsic.loader import load_intrinsic_data
intrinsic_df = load_intrinsic_data("EURUSD", delta=0.0010, timeframe="1 hour")
```

### Benefits

1. **Volatility Normalization** - Events represent similar price movements regardless of time
2. **Regime Independence** - Works across different market regimes
3. **Better Signal-to-Noise** - Focuses on significant price movements
4. **Cross-Timeframe Consistency** - Same delta works across all timeframes

## 💻 Usage Examples

### Example 1: Basic Feature Generation

```python
from src.feature_engineering import FeatureGenerator
from src.data.loader import load_parquet

# Load data
df = load_parquet("data/raw/clock", "EURUSD", "1 hour")
df = df.set_index('timestamp')

# Generate features
generator = FeatureGenerator()
features = generator.compute_all_features(df)

print(f"Generated {len(features.columns)} features")
print(features.head())
```

### Example 2: Process Single Symbol

```python
from src.data.loader import load_parquet
from src.feature_engineering import FeatureGenerator
import pandas as pd

# Load data
df = load_parquet("data/raw/clock", "EURUSD", "1 hour")
if 'timestamp' in df.columns:
    df = df.set_index('timestamp')

# Generate features
generator = FeatureGenerator()
features = generator.compute_all_features(df, track_metadata=True)

# Save
features.to_parquet("data/processed/EURUSD_1h_features.parquet")
```

### Example 3: Multi-Timeframe Analysis

```python
from src.data.multi_timeframe import align_timeframes, load_multi_timeframe
from src.feature_engineering import FeatureGenerator

# Load multiple timeframes
data = load_multi_timeframe("EURUSD", ["1 min", "5 mins", "1 hour"])

# Align to 1-minute bars
aligned = align_timeframes(
    data['1 min'],
    data['5 mins'],
    data['1 hour'],
    how='right'
)

# Generate features
generator = FeatureGenerator()
features = generator.compute_all_features(aligned['1m'])
```

### Example 4: View Feature Metadata

```python
import json

# Load metadata
with open("data/processed/EURUSD_1h_features_metadata.json") as f:
    metadata = json.load(f)

print(f"Schema Version: {metadata['schema_version']}")
print(f"Feature Version: {metadata['feature_version']}")
print(f"Computed Features: {len(metadata['computed_features'])}")

# View feature stats
for feat_name, feat_info in metadata['computed_features'].items():
    print(f"{feat_name}: {feat_info['null_count']} nulls")
```

## 📁 Project Structure

```
quant-research-bot/
├── config/
│   ├── config.yaml          # Main configuration
│   └── symbols.yaml         # FX pairs definition
├── data/
│   ├── raw/
│   │   ├── clock/           # Raw OHLCV data (Parquet)
│   │   └── intrinsic/       # Intrinsic-time events (Parquet)
│   └── processed/           # Feature matrices (Parquet + JSON metadata)
├── src/
│   ├── api/                 # IBKR API client
│   ├── data/                # Data collection, loading, validation
│   ├── intrinsic/           # Intrinsic-time transformation
│   ├── features/            # Feature computation modules
│   ├── feature_engineering/ # Feature orchestration
│   ├── models/              # ML models
│   └── utils/               # Configuration, logging, helpers
├── scripts/
│   ├── process_all_features.py    # Batch feature processing
│   └── quick_process_features.py  # Quick single-symbol processing
├── examples/                # Usage examples
├── notebooks/               # Jupyter notebooks
├── outputs/                 # Models, plots, reports
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Scripts

### `scripts/process_all_features.py`

Process features for all symbols and all timeframes:

```bash
python scripts/process_all_features.py
```

Options:
- Processes all symbols found in `data/raw/clock/`
- Processes all timeframes from config
- Saves features and metadata to `data/processed/`

### `scripts/quick_process_features.py`

Quick processing for EURUSD only (all timeframes):

```bash
python scripts/quick_process_features.py
```

### `src/main.py`

Main entry point for data collection:

```bash
python src/main.py
```

Collects data for all symbols and timeframes defined in config.

## 🐛 Troubleshooting

### Common Issues

**1. IBKR Connection Errors**
- Ensure TWS/Gateway is running
- Check API settings in TWS (Enable ActiveX and Socket Clients)
- Verify port number (7497 for Paper, 7496 for Live)
- Check firewall settings

**2. Parquet File Errors**
- The system includes robust recovery mechanisms
- If files are corrupted, delete and re-download
- Check disk space availability

**3. Module Import Errors**
- Ensure virtual environment is activated
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python path includes project root

**4. Feature Computation Errors**
- Check data quality (missing columns, NaN values)
- Verify feature registry YAML syntax
- Check feature dependencies are correctly defined

**5. Memory Issues**
- Process symbols/timeframes individually
- Use smaller timeframes for initial testing
- Consider processing in batches

### Getting Help

1. Check logs in `outputs/logs/bot.log`
2. Review configuration files
3. Check example scripts in `examples/`
4. Review feature documentation in `src/features/README.md`

## 📚 Additional Documentation

- [Configuration Guide](CONFIG_README.md) - Detailed configuration documentation
- [Feature Engineering Guide](src/features/README.md) - Feature system documentation
- [Environment Setup](docs/env_setup.md) - Detailed setup instructions

## 🔮 Future Enhancements

- [ ] Real-time data streaming
- [ ] Advanced regime detection models
- [ ] Backtesting framework
- [ ] Portfolio optimization
- [ ] Web dashboard
- [ ] Additional data sources (Oanda, etc.)


## 👤 Author

[Trung Nguyen]

---

**Built for quantitative FX research with institutional-grade quality.**
```

This README covers:
- Overview and features
- Installation steps
- Quick start examples
- Configuration details
- Architecture overview
- Data pipeline explanation
- Feature engineering guide
- Intrinsic time concepts
- Usage examples
- Project structure
- Scripts documentation
- Troubleshooting guide

Should I add or modify any sections?
