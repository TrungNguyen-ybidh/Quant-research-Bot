"""
Trend Features
Trend detection and strength indicators
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.features.utils import safe_divide


def compute_sma_ratio(df: pd.DataFrame, short: int = 10, long: int = 20, **kwargs) -> pd.Series:
    """
    Compute SMA(short) / SMA(long) ratio.
    
    Args:
        df: DataFrame with OHLCV data
        short: Short SMA period (default: 10)
        long: Long SMA period (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with SMA ratio
    """
    if 'close' not in df.columns:
        return pd.Series(1.0, index=df.index)
    
    close = df['close']
    
    sma_short = close.rolling(window=short, min_periods=short).mean()
    sma_long = close.rolling(window=long, min_periods=long).mean()
    
    ratio = safe_divide(sma_short, sma_long, fill_value=1.0)
    
    return ratio.fillna(1.0)


def compute_ema_ratio(df: pd.DataFrame, short: int = 12, long: int = 26, **kwargs) -> pd.Series:
    """
    Compute EMA(short) / EMA(long) ratio.
    
    Args:
        df: DataFrame with OHLCV data
        short: Short EMA period (default: 12)
        long: Long EMA period (default: 26)
        **kwargs: Additional parameters
    
    Returns:
        Series with EMA ratio
    """
    if 'close' not in df.columns:
        return pd.Series(1.0, index=df.index)
    
    close = df['close']
    
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    
    ratio = safe_divide(ema_short, ema_long, fill_value=1.0)
    
    return ratio.fillna(1.0)


def compute_adx(df: pd.DataFrame, period: int = 14, **kwargs) -> pd.Series:
    """
    Compute Average Directional Index (ADX).
    
    Args:
        df: DataFrame with OHLCV data
        period: ADX period (default: 14)
        **kwargs: Additional parameters
    
    Returns:
        Series with ADX (0-100, higher = stronger trend)
    """
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        return pd.Series(0.0, index=df.index)
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    # Smoothed TR and DM
    atr = tr.rolling(window=period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(window=period, min_periods=1).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period, min_periods=1).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=period, min_periods=period).mean()
    
    return adx.fillna(0.0).clip(0.0, 100.0)


def compute_trend_strength(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute trend strength indicator.
    
    Measures consistency of price movement direction.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with trend strength (0-1, higher = stronger trend)
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    close = df['close']
    returns = close.pct_change()
    
    # Count positive vs negative returns
    positive_returns = (returns > 0).astype(int)
    negative_returns = (returns < 0).astype(int)
    
    # Rolling counts
    pos_count = positive_returns.rolling(window=window, min_periods=1).sum()
    neg_count = negative_returns.rolling(window=window, min_periods=1).sum()
    total_count = pos_count + neg_count
    
    # Trend strength = max(pos, neg) / total
    trend_strength = safe_divide(
        np.maximum(pos_count, neg_count),
        total_count + 1e-10,
        fill_value=0.5
    )
    
    return trend_strength.fillna(0.5)


def compute_price_position(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute price position in recent range.
    
    Position = (close - min) / (max - min)
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with price position (0-1, 0 = at low, 1 = at high)
    """
    if 'close' not in df.columns:
        return pd.Series(0.5, index=df.index)
    
    close = df['close']
    
    min_price = close.rolling(window=window, min_periods=1).min()
    max_price = close.rolling(window=window, min_periods=1).max()
    
    price_position = safe_divide(
        close - min_price,
        max_price - min_price + 1e-10,
        fill_value=0.5
    )
    
    return price_position.fillna(0.5).clip(0.0, 1.0)

