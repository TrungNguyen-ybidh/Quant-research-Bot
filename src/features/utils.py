"""
Feature Engineering Utilities
Helper functions for feature computation
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def calculate_mid_price(df: pd.DataFrame) -> pd.Series:
    """
    Calculate mid-price from DataFrame.
    
    Uses (bid + ask)/2 if available, otherwise uses 'close' price.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Series with mid-price values
    """
    if 'bid' in df.columns and 'ask' in df.columns:
        return (df['bid'] + df['ask']) / 2.0
    elif 'close' in df.columns:
        return df['close']
    else:
        raise ValueError("DataFrame must have either (bid, ask) or 'close' column")


def calculate_returns(df: pd.DataFrame, price_col: str = 'close', method: str = 'simple') -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price (default: 'close')
        method: 'simple' or 'log' (default: 'simple')
    
    Returns:
        Series with returns
    """
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in DataFrame")
    
    price = df[price_col]
    
    if method == 'log':
        return np.log(price / price.shift(1))
    else:  # simple
        return price.pct_change()


def safe_divide(numerator: Union[pd.Series, np.ndarray], 
                denominator: Union[pd.Series, np.ndarray],
                fill_value: float = 0.0) -> Union[pd.Series, np.ndarray]:
    """
    Safe division that handles zero denominators.
    
    Args:
        numerator: Numerator values
        denominator: Denominator values
        fill_value: Value to use when denominator is zero (default: 0.0)
    
    Returns:
        Division result with zeros filled
    """
    if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
        result = numerator / denominator.replace(0, np.nan)
        return result.fillna(fill_value)
    else:
        result = np.divide(numerator, denominator, 
                          out=np.zeros_like(numerator, dtype=float),
                          where=(denominator != 0))
        return result


def rolling_apply_safe(series: pd.Series, 
                       window: int, 
                       func: callable,
                       min_periods: Optional[int] = None) -> pd.Series:
    """
    Apply rolling function with safe handling of insufficient data.
    
    Args:
        series: Input series
        window: Rolling window size
        func: Function to apply
        min_periods: Minimum periods required (default: window)
    
    Returns:
        Series with rolling function applied
    """
    if min_periods is None:
        min_periods = window
    
    return series.rolling(window=window, min_periods=min_periods).apply(func, raw=True)


def normalize_series(series: pd.Series, 
                     method: str = 'zscore',
                     window: Optional[int] = None) -> pd.Series:
    """
    Normalize a series using various methods.
    
    Args:
        series: Input series
        method: 'zscore', 'minmax', 'robust' (default: 'zscore')
        window: Rolling window for normalization (None = global)
    
    Returns:
        Normalized series
    """
    if window is None:
        # Global normalization
        if method == 'zscore':
            return (series - series.mean()) / series.std()
        elif method == 'minmax':
            return (series - series.min()) / (series.max() - series.min())
        elif method == 'robust':
            median = series.median()
            mad = (series - median).abs().median()
            return (series - median) / (mad * 1.4826)  # MAD to std conversion
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    else:
        # Rolling normalization
        if method == 'zscore':
            mean = series.rolling(window).mean()
            std = series.rolling(window).std()
            return (series - mean) / std
        elif method == 'minmax':
            min_val = series.rolling(window).min()
            max_val = series.rolling(window).max()
            return (series - min_val) / (max_val - min_val)
        elif method == 'robust':
            median = series.rolling(window).median()
            mad = series.rolling(window).apply(lambda x: (x - x.median()).abs().median())
            return (series - median) / (mad * 1.4826)
        else:
            raise ValueError(f"Unknown normalization method: {method}")


def get_session_hours(timestamp: pd.Timestamp) -> dict:
    """
    Get trading session hours for a timestamp.
    
    Args:
        timestamp: Timestamp to check
    
    Returns:
        Dictionary with session indicators
    """
    hour = timestamp.hour
    weekday = timestamp.weekday()
    
    # London: 8:00-17:00 GMT (UTC)
    london_start = 8
    london_end = 17
    
    # New York: 13:00-22:00 GMT (UTC)
    ny_start = 13
    ny_end = 22
    
    # Asia: 23:00-8:00 GMT (UTC) (spans midnight)
    asia_start = 23
    asia_end = 8
    
    is_london = london_start <= hour < london_end
    is_ny = ny_start <= hour < ny_end
    is_asia = hour >= asia_start or hour < asia_end
    
    # Overlaps
    london_ny_overlap = ny_start <= hour < london_end
    london_asia_overlap = hour >= asia_start or hour < london_end
    
    return {
        'london': is_london,
        'new_york': is_ny,
        'asia': is_asia,
        'london_ny_overlap': london_ny_overlap,
        'london_asia_overlap': london_asia_overlap
    }


def ensure_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has a datetime index.
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with datetime index
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
        return df
    
    raise ValueError("DataFrame must have 'timestamp' column or datetime index")


def fill_missing_values(series: pd.Series, method: str = 'forward') -> pd.Series:
    """
    Fill missing values in a series.
    
    Args:
        series: Input series
        method: 'forward', 'backward', 'interpolate', 'zero' (default: 'forward')
    
    Returns:
        Series with missing values filled
    """
    if method == 'forward':
        return series.fillna(method='ffill')
    elif method == 'backward':
        return series.fillna(method='bfill')
    elif method == 'interpolate':
        return series.interpolate()
    elif method == 'zero':
        return series.fillna(0)
    else:
        raise ValueError(f"Unknown fill method: {method}")

