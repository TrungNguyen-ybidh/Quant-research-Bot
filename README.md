# Quantitative Research Bot

A production-grade quantitative research system for FX market analysis, featuring institutional-quality data collection, intrinsic-time transformation, and comprehensive feature engineering.

> **Note**: This is a research tool developed for CS 3200 at Northeastern University. It analyzes and describes market conditions but does **NOT** generate trading signals or recommendations. The goal is to understand market behavior, not to trade.

## 🎯 Overview

This system provides a complete pipeline for quantitative FX research:

1. **Data Collection** - Automated data fetching from Interactive Brokers (IBKR)
2. **Intrinsic-Time Transformation** - Convert clock-time data to event-driven intrinsic time
3. **Feature Engineering** - Generate 78 features across 8 categories with dependency resolution
4. **Multi-Timeframe Support** - Align and merge features across timeframes
5. **Market State Detection** - Regime, volatility, and transition models using neural networks
6. **Market State Analysis** - Unified market state description for research
7. **ML-Ready Output** - Clean feature matrices with full reproducibility tracking

### Key Features

- ✅ **Automated Data Collection** - Incremental updates from IBKR with robust error handling
- ✅ **Intrinsic-Time Processing** - Directional-change (DC) event detection and overshoot tracking
- ✅ **Institution-Grade Feature Engineering** - 78 features across 8 categories
- ✅ **YAML-Driven Configuration** - All parameters configurable without code changes
- ✅ **Multi-Timeframe Alignment** - Seamlessly combine features from different timeframes
- ✅ **Feature Dependency System** - Automatic dependency resolution and computation ordering
- ✅ **Schema Version Control** - Full reproducibility with metadata tracking
- ✅ **Robust Data Handling** - Recovery mechanisms for corrupted Parquet files
- ✅ **Market State Detection** - Regime, volatility, and transition models with neural networks
- ✅ **Market State Analysis** - Unified state description combining regime, volatility, and transition models
- ✅ **Production-Ready** - Comprehensive logging, validation, and error handling

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Intrinsic Time](#intrinsic-time)
- [Machine Learning Models](#machine-learning-models)
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
│   Feature       │  Generate 78 features
│   Engineering   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Pipeline    │  Model training & prediction
│  - Regime       │  - RegimeClassifier
│  - Volatility   │  - VolatilityRegimeClassifier
│  - Transition   │  - TransitionDetector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Market State    │  Combine models → State description
│ Analyzer        │  (Research tool, NO trading signals)
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

## 🤖 Machine Learning Models

### Research Objectives

This tool answers the following research questions:
- **"What regime is the market in?"** - Trending (up/down) vs ranging
- **"What is the volatility state?"** - Low/normal/high/crisis conditions
- **"Is a regime transition likely?"** - Early warning of state changes
- **"How do conditions vary across currency pairs?"** - Cross-market analysis

**What this tool does NOT do:**
- Generate buy/sell signals
- Recommend position sizes
- Execute trades
- Promise any returns

### Model Performance

**Baseline Performance (EURUSD 1h):**

| Model | Classes | Accuracy | Notes |
|-------|---------|----------|-------|
| Regime Classifier | RANGING, TREND_UP, TREND_DOWN | ~33% (1h), ~40% (1d) | Best on AUDUSD; rarely predicts RANGING |
| Volatility Classifier | LOW, NORMAL, HIGH, CRISIS | **77%** avg | Strong performance vs 25% baseline |
| Transition Detector | STABLE, TRANSITIONING | 76% | ⚠️ Over-predicts transitions (98.8% of predictions) |

**Known Issues:**
- Transition detector predicts transitions 98.8% of the time due to class imbalance
- Regime classifier rarely predicts RANGING (only 4.4% of predictions)
- USDCAD performs worst on regime classification, best on volatility

**Trained Models:**
- **55 regime models** (11 pairs × 5 timeframes: 1m, 5m, 15m, 1h, 1d)
- **11 volatility models** (11 pairs × 1h only)
- **11 transition models** (11 pairs × 1h only)
- **Total: 77 trained PyTorch models**

### State Detection Models

The system includes three specialized models for market state detection:

#### 1. Regime Classifier
Detects market regimes: **RANGING**, **TREND_UP**, or **TREND_DOWN**

```python
from src.models.state import RegimeClassifier

# Load trained model
model = RegimeClassifier.load("EURUSD", "1h", "v1")

# Predict regime
regime, confidence = model.predict(features)
print(f"Regime: {regime}, Confidence: {confidence:.2%}")
```

**Training:**
```bash
python scripts/train_regime_model.py --symbol EURUSD --timeframe 1h
```

#### 2. Volatility Regime Classifier
Detects volatility regimes: **LOW**, **NORMAL**, **HIGH**, or **CRISIS**

```python
from src.models.state import VolatilityRegimeClassifier

model = VolatilityRegimeClassifier.load("EURUSD", "1h", "v1")
volatility, confidence = model.predict(features)
```

**Training:**
```bash
python scripts/train_volatility_model.py --symbol EURUSD --timeframe 1h
```

#### 3. Transition Detector
Predicts regime transitions: **STABLE** or **CHANGING**

```python
from src.models.state import TransitionDetector

model = TransitionDetector.load("EURUSD", "1h", "v1")
transition, probability = model.predict(features)
```

**Training:**
```bash
python scripts/train_transition_model.py --symbol EURUSD --timeframe 1h
```

### Market State Analyzer

Combines all three models into a unified market state description for research:

```python
from src.models.state import MarketStateAnalyzer

# Initialize analyzer (NOT aggregator)
analyzer = MarketStateAnalyzer(symbol="EURUSD", timeframe="1h")
analyzer.load_models()

# Get market state (NOT trading signals)
state = analyzer.get_market_state(features)

# Access state information
print(state.regime)              # 'TREND_UP'
print(state.regime_confidence)   # 0.67
print(state.volatility_regime)   # 'NORMAL'
print(state.volatility_confidence) # 0.81
print(state.is_transitioning)    # False
print(state.transition_probability) # 0.12
print(state.state_code)          # 'TREND_UP_NORMAL_STABLE'

# Pretty print full summary
print(state.summary())
```

**MarketState Fields:**
- `regime` - Current regime (RANGING, TREND_UP, TREND_DOWN)
- `regime_confidence` - Confidence in regime prediction (0-1)
- `volatility_regime` - Volatility state (LOW, NORMAL, HIGH, CRISIS)
- `volatility_confidence` - Confidence in volatility prediction (0-1)
- `is_transitioning` - Boolean indicating expected regime change
- `transition_probability` - Probability of transition (0-1)
- `state_code` - Compact code (e.g., "TREND_UP_NORMAL_STABLE")
- `summary()` - Human-readable formatted summary

### Multi-Pair Analyzer

Analyze market state across multiple currency pairs for cross-market research:

```python
from src.models.state import MultiPairAnalyzer

# Initialize for multiple pairs
multi_analyzer = MultiPairAnalyzer(
    symbols=["EURUSD", "GBPUSD", "USDJPY"],
    timeframe="1h"
)
multi_analyzer.load_all_models()

# Get states for all pairs
features_dict = {
    "EURUSD": eurusd_features,
    "GBPUSD": gbpusd_features,
    "USDJPY": usdjpy_features
}
states = multi_analyzer.get_all_states(features_dict)

# Research summary
summary = multi_analyzer.get_market_summary(states)
print(f"Dominant regime: {summary['dominant_regime']}")
print(f"Dominant volatility: {summary['dominant_volatility']}")
print(f"Pairs transitioning: {summary['n_transitioning']}/{summary['n_pairs']}")

# Print research dashboard
multi_analyzer.print_dashboard(states)
```

### Model Architecture

All models use **PyTorch neural networks** (MLP or LSTM) with:
- Multi-layer architecture (128 → 64 → 32 units)
- Dropout regularization (0.2-0.3)
- Feature scaling and selection
- Time-series cross-validation
- Comprehensive metrics tracking

**Data Pipeline:**
- Preprocessing: DataFrame → Scaled DataFrame
- Feature Selection: DataFrame → Selected DataFrame
- Model Input: DataFrame → NumPy array → PyTorch tensor
- The `predict()` methods automatically handle DataFrame to NumPy conversion

**Model Outputs:**
- Trained models saved to `models/trained/`
- Config and metrics saved to `outputs/models/`
- Full reproducibility with version tracking

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

### Example 5: Get Market State Analysis

```python
from src.models.state import MarketStateAnalyzer
from src.data.loader import load_parquet
from src.feature_engineering import FeatureGenerator

# Load data and generate features
df = load_parquet("data/raw/clock", "EURUSD", "1 hour")
df = df.set_index('timestamp')

generator = FeatureGenerator()
features = generator.compute_all_features(df)

# Initialize market state analyzer
analyzer = MarketStateAnalyzer(symbol="EURUSD", timeframe="1h")
analyzer.load_models()

# Get current market state
state = analyzer.get_market_state(features.tail(100))  # Use last 100 bars

# Display full summary
print(state.summary())

# Access individual components
print(f"\nRegime: {state.regime}")
print(f"Regime Confidence: {state.regime_confidence:.1%}")
print(f"Volatility: {state.volatility_regime}")
print(f"Is Transitioning: {state.is_transitioning}")
print(f"State Code: {state.state_code}")
```

### Example 6: Multi-Pair Market Analysis

```python
from src.models.state import MultiPairAnalyzer
from src.data.loader import load_parquet
from src.feature_engineering import FeatureGenerator

# Load features for multiple pairs
symbols = ["EURUSD", "GBPUSD", "USDJPY"]
features_dict = {}

generator = FeatureGenerator()

for symbol in symbols:
    df = load_parquet("data/raw/clock", symbol, "1 hour")
    df = df.set_index('timestamp')
    features = generator.compute_all_features(df)
    features_dict[symbol] = features.tail(100)

# Initialize multi-pair analyzer
multi_analyzer = MultiPairAnalyzer(symbols=symbols, timeframe="1h")
multi_analyzer.load_all_models()

# Get states for all pairs
states = multi_analyzer.get_all_states(features_dict)

# Research summary
summary = multi_analyzer.get_market_summary(states)
print(f"Dominant Regime: {summary['dominant_regime']}")
print(f"Dominant Volatility: {summary['dominant_volatility']}")
print(f"Pairs Transitioning: {summary['n_transitioning']}/{summary['n_pairs']}")
print(f"Average Regime Confidence: {summary['avg_regime_confidence']:.1%}")

# Print research dashboard
multi_analyzer.print_dashboard(states)
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
│   │   ├── state/           # State detection models
│   │   │   ├── regime_classifier.py      # Regime detection
│   │   │   ├── volatility_regime.py      # Volatility regime detection
│   │   │   ├── transition_detector.py    # Transition detection
│   │   │   └── state_aggregator.py       # Market state analyzer
│   │   ├── baseline_predictor.py         # Baseline models
│   │   ├── feature_selection.py          # Feature selection
│   │   └── preprocessing.py              # Data preprocessing
│   └── utils/               # Configuration, logging, helpers
├── scripts/
│   ├── process_all_features.py    # Batch feature processing
│   ├── quick_process_features.py  # Quick single-symbol processing
│   ├── train_regime_model.py      # Train regime classifier
│   ├── train_volatility_model.py  # Train volatility classifier
│   ├── train_transition_model.py  # Train transition detector
│   ├── test_state_aggregator.py   # Test state aggregator
│   ├── test_state_analyzer.py     # Analyze state predictions
│   ├── analyze_full_dataset.py    # Batch predictions on full dataset
│   └── generate_market_report.py  # Generate market state reports
├── examples/                # Usage examples
├── notebooks/               # Jupyter notebooks
├── outputs/                 # Models, plots, reports
│   ├── models/              # Model configs and metrics
│   │   ├── regime/          # Regime model outputs
│   │   └── volatility/      # Volatility model outputs
│   ├── logs/                # Application logs
│   ├── plots/               # Generated plots
│   └── reports/             # Analysis reports
├── models/                  # Trained model files
│   ├── trained/             # Saved PyTorch models (.pt)
│   ├── scalers/             # Feature scalers (.pkl)
│   └── selectors/           # Feature selectors (.pkl)
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

### `scripts/train_regime_model.py`

Train regime classification model:

```bash
python scripts/train_regime_model.py --symbol EURUSD --timeframe 1h
```

### `scripts/train_volatility_model.py`

Train volatility regime classification model:

```bash
python scripts/train_volatility_model.py --symbol EURUSD --timeframe 1h
```

### `scripts/train_transition_model.py`

Train regime transition detection model:

```bash
python scripts/train_transition_model.py --symbol EURUSD --timeframe 1h
```

### `scripts/test_state_analyzer.py`

Test the market state analyzer with real data:

```bash
python scripts/test_state_analyzer.py --symbol EURUSD --timeframe 1h
```

Analyzes market state predictions across historical data and provides research insights.

### `scripts/analyze_full_dataset.py`

Run batch predictions on full dataset for comprehensive analysis:

```bash
python scripts/analyze_full_dataset.py --symbol EURUSD --timeframe 1h
```

Performs batch analysis across entire historical datasets to understand model behavior over time.

### `scripts/generate_market_report.py`

Generate comprehensive market state reports for multiple pairs:

```bash
python scripts/generate_market_report.py --pairs EURUSD GBPUSD USDJPY --timeframe 1h
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

**5. Model Prediction Errors**
- Ensure features DataFrame matches training data structure
- Check that all required features are present
- Verify model files are loaded correctly
- The `predict()` methods handle DataFrame inputs automatically

**6. Memory Issues**
- Process symbols/timeframes individually
- Use smaller timeframes for initial testing
- Consider processing in batches
- Use `analyze_full_dataset.py` for batch predictions on large datasets

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

- [ ] Improve regime classifier performance (better class balancing)
- [ ] Fix transition detector class imbalance
- [ ] Real-time data streaming
- [ ] Enhanced evaluation metrics and visualizations
- [ ] Web dashboard for market state monitoring
- [ ] Additional data sources (Oanda, etc.)
- [ ] Multi-asset class support (stocks, crypto)


## 👤 Author

[Trung Nguyen]

---

**Built for quantitative FX research at Northeastern University CS 3200.**
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
