"""
FX Factor Features
FX-specific factor models (carry, momentum, value, volatility)
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.features.utils import calculate_returns, normalize_series, safe_divide


def compute_carry_factor(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute carry trade factor (interest rate differential).
    
    Note: This is a placeholder. In practice, would need interest rate data.
    For now, uses forward-looking return as proxy.
    
    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional parameters
    
    Returns:
        Series with carry factor (normalized)
    """
    # Placeholder: In real implementation, would fetch interest rates
    # For now, return zeros or use a proxy
    
    # Proxy: Use forward return as carry signal
    if 'close' in df.columns:
        returns = calculate_returns(df, price_col='close', method='simple')
        # Shift forward to use as proxy
        carry_proxy = returns.shift(-1).fillna(0.0)
        return normalize_series(carry_proxy, method='zscore')
    else:
        return pd.Series(0.0, index=df.index)


def compute_momentum_factor(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute momentum factor (past return).
    
    Args:
        df: DataFrame with OHLCV data
        window: Lookback window (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with momentum factor (normalized)
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    close = df['close']
    
    # Momentum = (close / close[window] - 1)
    momentum = (close / close.shift(window) - 1.0)
    
    # Normalize
    momentum_normalized = normalize_series(momentum, method='zscore', window=window*2)
    
    return momentum_normalized.fillna(0.0)


def compute_value_factor(df: pd.DataFrame, window: int = 60, **kwargs) -> pd.Series:
    """
    Compute value factor (deviation from mean).
    
    Measures how far price is from its historical mean.
    
    Args:
        df: DataFrame with OHLCV data
        window: Lookback window (default: 60)
        **kwargs: Additional parameters
    
    Returns:
        Series with value factor (normalized, negative = overvalued)
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    close = df['close']
    
    # Mean price
    mean_price = close.rolling(window=window, min_periods=window).mean()
    
    # Deviation from mean (as percentage)
    value_deviation = (close - mean_price) / mean_price
    
    # Normalize (negative = overvalued, positive = undervalued)
    value_factor = normalize_series(value_deviation, method='zscore', window=window*2)
    
    return value_factor.fillna(0.0)


def compute_volatility_factor(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute volatility factor (normalized volatility).
    
    Measures current volatility relative to historical volatility.
    
    Args:
        df: DataFrame with OHLCV data
        window: Lookback window (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with volatility factor (normalized)
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    returns = calculate_returns(df, price_col='close', method='simple')
    
    # Current volatility (short window)
    vol_short = returns.rolling(window=window//2, min_periods=1).std()
    
    # Historical volatility (long window)
    vol_long = returns.rolling(window=window, min_periods=window).std()
    
    # Volatility ratio
    vol_ratio = safe_divide(vol_short, vol_long + 1e-10, fill_value=1.0)
    
    # Normalize
    vol_factor = normalize_series(vol_ratio, method='zscore', window=window*2)
    
    return vol_factor.fillna(0.0)

