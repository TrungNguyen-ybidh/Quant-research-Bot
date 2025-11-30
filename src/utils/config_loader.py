"""
Configuration Loader for Quantitative Research Bot
Loads and validates configuration from YAML files
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DataSourceConfig:
    """Data source configuration"""
    provider: str
    host: str
    port: int
    client_id: int
    timeout: int


@dataclass
class SymbolConfig:
    """Symbol configuration"""
    primary: str
    sec_type: str
    exchange: str
    currency: str
    additional: list


@dataclass
class TimeframeConfig:
    """Timeframe configuration"""
    primary: str
    lookback_days: int
    bar_size: str


@dataclass
class MLConfig:
    """Machine learning configuration"""
    model_type: str
    algorithm: str
    hidden_layers: list
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    train_split: float
    val_split: float
    test_split: float
    random_seed: int


class ConfigLoader:
    """Loads and manages configuration from YAML files"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize config loader
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}\n"
                "Please create config.yaml in the project root."
            )
        
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate and create directories
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary directories from config"""
        paths = self.config.get('storage', {}).get('data_paths', {})
        for key, path in paths.items():
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Create logs directory
        log_file = self.config.get('logging', {}).get('file', 'outputs/logs/bot.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path (e.g., 'data_source.host')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_data_source(self) -> DataSourceConfig:
        """Get data source configuration"""
        ds = self.config.get('data_source', {})
        return DataSourceConfig(
            provider=ds.get('provider', 'IBKR'),
            host=ds.get('host', '127.0.0.1'),
            port=ds.get('port', 7496),
            client_id=ds.get('client_id', 1),
            timeout=ds.get('timeout', 15)
        )
    
    def get_symbol(self) -> SymbolConfig:
        """Get symbol configuration"""
        sym = self.config.get('symbol', {})
        return SymbolConfig(
            primary=sym.get('primary', 'EUR'),
            sec_type=sym.get('sec_type', 'CASH'),
            exchange=sym.get('exchange', 'IDEALPRO'),
            currency=sym.get('currency', 'USD'),
            additional=sym.get('additional', [])
        )
    
    def get_timeframe(self) -> TimeframeConfig:
        """Get timeframe configuration"""
        tf = self.config.get('timeframe', {})
        return TimeframeConfig(
            primary=tf.get('primary', '1 hour'),
            lookback_days=tf.get('lookback_days', 365),
            bar_size=tf.get('bar_size', '1 hour')
        )
    
    def get_indicators(self) -> Dict[str, Any]:
        """Get indicator configuration"""
        return self.config.get('indicators', {})
    
    def get_features(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self.config.get('features', {})
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get machine learning configuration"""
        return self.config.get('ml', {})
    
    def get_storage_paths(self) -> Dict[str, str]:
        """Get storage path configuration"""
        return self.config.get('storage', {}).get('data_paths', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get('logging', {})
    
    def reload(self) -> None:
        """Reload configuration from file"""
        self.load()
    
    def validate(self) -> bool:
        """
        Validate configuration
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Validate data source
        if not self.get('data_source.host'):
            raise ValueError("data_source.host is required")
        
        if not (1 <= self.get('data_source.port', 0) <= 65535):
            raise ValueError("data_source.port must be between 1 and 65535")
        
        # Validate symbol
        if not self.get('symbol.primary'):
            raise ValueError("symbol.primary is required")
        
        # Validate timeframe
        valid_bar_sizes = [
            '1 sec', '5 secs', '10 secs', '15 secs', '30 secs',
            '1 min', '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins', '30 mins',
            '1 hour', '2 hours', '3 hours', '4 hours', '8 hours',
            '1 day', '1 week', '1 month'
        ]
        
        bar_size = self.get('timeframe.bar_size', '')
        if bar_size and bar_size not in valid_bar_sizes:
            raise ValueError(f"Invalid bar_size: {bar_size}. Must be one of {valid_bar_sizes}")
        
        # Validate ML splits sum to 1.0
        train = self.get('ml.training.train_split', 0)
        val = self.get('ml.training.val_split', 0)
        test = self.get('ml.training.test_split', 0)
        
        if abs(train + val + test - 1.0) > 0.01:
            raise ValueError(f"ML training splits must sum to 1.0 (got {train + val + test})")
        
        return True


# Global config instance (singleton pattern)
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Get global configuration instance
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance


def reload_config() -> None:
    """Reload global configuration"""
    global _config_instance
    if _config_instance is not None:
        _config_instance.reload()


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = get_config()
        config.validate()
        print("[OK] Configuration loaded and validated successfully!")
        print(f"\nData Source: {config.get_data_source()}")
        print(f"Symbol: {config.get_symbol()}")
        print(f"Timeframe: {config.get_timeframe()}")
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")

