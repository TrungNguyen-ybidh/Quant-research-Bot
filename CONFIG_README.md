# Configuration System

## Overview

The quantitative research bot uses a YAML-based configuration system. **All parameters are defined in `config.yaml` - no hard-coded values.**

## Files

- **`config.yaml`** - Main configuration file with all system parameters
- **`config_loader.py`** - Python module to load and validate configuration

## Quick Start

```python
from config_loader import get_config

# Load configuration
config = get_config()

# Access configuration values
data_source = config.get_data_source()
symbol = config.get_symbol()
timeframe = config.get_timeframe()

# Or use dot notation
host = config.get('data_source.host')
port = config.get('data_source.port')
```

## Configuration Sections

### 1. Data Source (`data_source`)
- **provider**: Data provider (IBKR, ib_insync)
- **host**: IBKR API host (usually 127.0.0.1)
- **port**: Port number
  - `7496` = TWS Paper Trading
  - `7497` = TWS Live Trading
  - `4001` = IB Gateway Paper
  - `4002` = IB Gateway Live
- **client_id**: Unique client identifier
- **timeout**: Connection timeout in seconds

### 2. Symbol (`symbol`)
- **primary**: Main trading symbol
- **sec_type**: Security type (CASH, STK, FUT, OPT)
- **exchange**: Exchange name
- **currency**: Base currency
- **additional**: List of additional symbols for multi-asset strategies

### 3. Timeframe (`timeframe`)
- **primary**: Primary timeframe string
- **lookback_days**: Days of historical data to fetch
- **bar_size**: Bar size for historical data

Valid bar sizes:
- Seconds: `1 sec`, `5 secs`, `10 secs`, `15 secs`, `30 secs`
- Minutes: `1 min`, `5 mins`, `15 mins`, `30 mins`
- Hours: `1 hour`, `2 hours`, `4 hours`, `8 hours`
- Days: `1 day`, `1 week`, `1 month`

### 4. Indicators (`indicators`)
Configure technical indicators:
- **moving_averages**: SMA, EMA, WMA with configurable periods
- **momentum**: RSI, MACD, Stochastic
- **volatility**: Bollinger Bands, ATR
- **volume**: Volume SMA, OBV

### 5. Features (`features`)
Feature engineering configuration:
- **version**: Feature version identifier
- **groups**: Feature groups (price, technical, lag, rolling, time)
- **selection**: Feature selection method and thresholds

### 6. Machine Learning (`ml`)
ML model configuration:
- **model**: Model type and algorithm
- **neural_network**: LSTM/GRU specific settings
- **training**: Train/val/test splits
- **scaling**: Feature scaling method
- **regime**: Regime detection parameters

### 7. Storage (`storage`)
Data storage paths and formats:
- **database**: Database type and path
- **data_paths**: Directory paths for raw data, features, models, plots, reports
- **formats**: File formats (parquet, csv, hdf5, pkl)

### 8. Logging (`logging`)
Logging configuration:
- **level**: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **file**: Log file path
- **console**: Enable console logging
- **max_bytes**: Max log file size
- **backup_count**: Number of backup files

## Usage Examples

### Example 1: Using Config in Code

```python
from config_loader import get_config
from ibapi.contract import Contract

config = get_config()
symbol_config = config.get_symbol()

# Create contract from config
contract = Contract()
contract.symbol = symbol_config.primary
contract.secType = symbol_config.sec_type
contract.exchange = symbol_config.exchange
contract.currency = symbol_config.currency
```

### Example 2: Accessing Nested Values

```python
config = get_config()

# Get ML learning rate
learning_rate = config.get('ml.neural_network.learning_rate')

# Get indicator settings
rsi_period = config.get('indicators.momentum.rsi.period')
```

### Example 3: Validating Configuration

```python
from config_loader import get_config

config = get_config()
try:
    config.validate()
    print("Configuration is valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Best Practices

1. **Never hard-code values** - Always use `config.yaml`
2. **Validate on startup** - Call `config.validate()` before running
3. **Use typed accessors** - Use `get_data_source()`, `get_symbol()`, etc. for type safety
4. **Version control** - Keep `config.yaml` in version control (exclude secrets)
5. **Environment-specific configs** - Consider separate configs for dev/prod

## Modifying Configuration

Simply edit `config.yaml` and reload:

```python
from config_loader import reload_config

# After modifying config.yaml
reload_config()
```

## Configuration Validation

The config loader validates:
- Required fields are present
- Port numbers are in valid range (1-65535)
- Bar sizes are valid IBKR bar sizes
- ML train/val/test splits sum to 1.0
- All directory paths are created automatically

