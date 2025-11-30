"""
Microstructure Features
Retail-friendly microstructure features from OHLCV data
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.features.utils import calculate_mid_price, safe_divide


def compute_micro_price(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute microstructure price (bid-ask weighted).
    
    Micro price = (bid * ask_volume + ask * bid_volume) / (bid_volume + ask_volume)
    Falls back to mid-price if volume not available.
    
    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional parameters
    
    Returns:
        Series with micro price
    """
    if 'bid' in df.columns and 'ask' in df.columns:
        bid = df['bid']
        ask = df['ask']
        
        # If volume data available, use weighted average
        if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
            bid_vol = df['bid_volume']
            ask_vol = df['ask_volume']
            total_vol = bid_vol + ask_vol
            
            micro_price = safe_divide(
                bid * ask_vol + ask * bid_vol,
                total_vol,
                fill_value=(bid + ask) / 2.0
            )
            return micro_price
        else:
            # Fall back to mid-price
            return (bid + ask) / 2.0
    else:
        # Use close price as fallback
        return df.get('close', pd.Series(0.0, index=df.index))


def compute_synthetic_spread(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute synthetic bid-ask spread from high-low.
    
    Spread estimate = 2 * sqrt(ln(H/L) / (4 * ln(2)))
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window for smoothing (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with synthetic spread
    """
    if 'high' not in df.columns or 'low' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    high = df['high']
    low = df['low']
    
    # Synthetic spread from Parkinson estimator
    log_hl = np.log(high / low)
    spread = 2.0 * np.sqrt(log_hl / (4.0 * np.log(2.0)))
    
    # Smooth with rolling mean
    spread = spread.rolling(window=window, min_periods=1).mean()
    
    return spread.fillna(0.0)


def compute_relative_spread(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute relative spread (spread / mid price).
    
    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional parameters
    
    Returns:
        Series with relative spread
    """
    mid_price = calculate_mid_price(df)
    
    if 'bid' in df.columns and 'ask' in df.columns:
        spread = df['ask'] - df['bid']
    else:
        # Use high-low as proxy
        if 'high' in df.columns and 'low' in df.columns:
            spread = df['high'] - df['low']
        else:
            return pd.Series(0.0, index=df.index)
    
    relative_spread = safe_divide(spread, mid_price, fill_value=0.0)
    
    return relative_spread


def compute_wick_ratio(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute upper wick to lower wick ratio.
    
    Upper wick = high - max(open, close)
    Lower wick = min(open, close) - low
    
    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional parameters
    
    Returns:
        Series with wick ratio
    """
    required_cols = ['high', 'low', 'open', 'close']
    if not all(col in df.columns for col in required_cols):
        return pd.Series(1.0, index=df.index)
    
    high = df['high']
    low = df['low']
    open_price = df['open']
    close = df['close']
    
    # Upper and lower wicks
    upper_wick = high - np.maximum(open_price, close)
    lower_wick = np.minimum(open_price, close) - low
    
    # Ratio (add small epsilon to avoid division by zero)
    wick_ratio = safe_divide(upper_wick, lower_wick + 1e-10, fill_value=1.0)
    
    return wick_ratio


def compute_body_ratio(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute body size to total range ratio.
    
    Body = |close - open|
    Range = high - low
    
    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional parameters
    
    Returns:
        Series with body ratio (0-1)
    """
    required_cols = ['high', 'low', 'open', 'close']
    if not all(col in df.columns for col in required_cols):
        return pd.Series(0.5, index=df.index)
    
    high = df['high']
    low = df['low']
    open_price = df['open']
    close = df['close']
    
    body = np.abs(close - open_price)
    range_size = high - low
    
    body_ratio = safe_divide(body, range_size + 1e-10, fill_value=0.0)
    
    return body_ratio.clip(0.0, 1.0)


def compute_rejection_wicks(df: pd.DataFrame, threshold: float = 0.6, **kwargs) -> pd.Series:
    """
    Compute rejection wick indicator.
    
    Identifies candles with long wicks (potential rejection).
    
    Args:
        df: DataFrame with OHLCV data
        threshold: Minimum wick ratio for rejection (default: 0.6)
        **kwargs: Additional parameters
    
    Returns:
        Series with rejection indicator (1 = rejection, 0 = no rejection)
    """
    required_cols = ['high', 'low', 'open', 'close']
    if not all(col in df.columns for col in required_cols):
        return pd.Series(0.0, index=df.index)
    
    high = df['high']
    low = df['low']
    open_price = df['open']
    close = df['close']
    
    # Upper and lower wicks
    upper_wick = high - np.maximum(open_price, close)
    lower_wick = np.minimum(open_price, close) - low
    range_size = high - low
    
    # Wick ratios
    upper_wick_ratio = safe_divide(upper_wick, range_size + 1e-10, fill_value=0.0)
    lower_wick_ratio = safe_divide(lower_wick, range_size + 1e-10, fill_value=0.0)
    
    # Rejection if either wick is above threshold
    rejection = ((upper_wick_ratio > threshold) | (lower_wick_ratio > threshold)).astype(float)
    
    return rejection


def compute_standardized_range(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute range standardized by volatility.
    
    Standardized range = (high - low) / rolling_std(returns)
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window for volatility (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with standardized range
    """
    if 'high' not in df.columns or 'low' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    high = df['high']
    low = df['low']
    range_size = high - low
    
    # Compute volatility from returns
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        vol = returns.rolling(window=window, min_periods=1).std()
        
        # Standardize
        standardized = safe_divide(range_size, vol + 1e-10, fill_value=0.0)
        return standardized.fillna(0.0)
    else:
        return pd.Series(0.0, index=df.index)


def compute_volume_adv_ratio(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute volume to average daily volume ratio.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window for average volume calculation (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with volume/ADV ratio
    """
    if 'volume' not in df.columns:
        return pd.Series(1.0, index=df.index)
    
    volume = df['volume']
    
    # Average daily volume
    adv = volume.rolling(window=window, min_periods=1).mean()
    
    # Ratio
    volume_adv_ratio = safe_divide(volume, adv + 1e-10, fill_value=1.0)
    
    return volume_adv_ratio.fillna(1.0)

