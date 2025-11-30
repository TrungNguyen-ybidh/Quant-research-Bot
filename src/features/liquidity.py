"""
Liquidity Features
Market liquidity measures
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.features.utils import calculate_mid_price, safe_divide


def compute_bid_ask_spread(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute bid-ask spread.
    
    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional parameters
    
    Returns:
        Series with bid-ask spread
    """
    if 'bid' in df.columns and 'ask' in df.columns:
        spread = df['ask'] - df['bid']
        return spread
    else:
        # Use high-low as proxy
        if 'high' in df.columns and 'low' in df.columns:
            spread = df['high'] - df['low']
            return spread
        else:
            return pd.Series(0.0, index=df.index)


def compute_spread_percentile(df: pd.DataFrame, window: int = 100, **kwargs) -> pd.Series:
    """
    Compute spread percentile in recent distribution.
    
    Measures how current spread compares to historical spread distribution.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 100)
        **kwargs: Additional parameters
    
    Returns:
        Series with spread percentile (0-1, higher = wider spread)
    """
    spread = compute_bid_ask_spread(df, **kwargs)
    
    # Rolling percentile
    def percentile_rank(series):
        if len(series) < 2:
            return 0.5
        current = series.iloc[-1]
        historical = series.iloc[:-1]
        if len(historical) == 0:
            return 0.5
        percentile = (historical <= current).sum() / len(historical)
        return percentile
    
    spread_percentile = spread.rolling(window=window, min_periods=2).apply(
        percentile_rank, raw=False
    )
    
    return spread_percentile.fillna(0.5)


def compute_liquidity_score(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute composite liquidity score.
    
    Combines spread, volume, and volatility into a single liquidity measure.
    Higher score = better liquidity.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with liquidity score (normalized, higher = better liquidity)
    """
    # Components
    spread = compute_bid_ask_spread(df, **kwargs)
    mid_price = calculate_mid_price(df)
    relative_spread = safe_divide(spread, mid_price + 1e-10, fill_value=0.0)
    
    # Volume component
    if 'volume' in df.columns:
        volume = df['volume']
        avg_volume = volume.rolling(window=window, min_periods=1).mean()
        volume_ratio = safe_divide(volume, avg_volume + 1e-10, fill_value=1.0)
    else:
        volume_ratio = pd.Series(1.0, index=df.index)
    
    # Volatility component (lower vol = better liquidity)
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        vol = returns.rolling(window=window, min_periods=1).std()
        avg_vol = vol.rolling(window=window*2, min_periods=1).mean()
        vol_ratio = safe_divide(vol, avg_vol + 1e-10, fill_value=1.0)
    else:
        vol_ratio = pd.Series(1.0, index=df.index)
    
    # Composite score: (volume_ratio) / (relative_spread * vol_ratio)
    # Higher volume, lower spread, lower vol = better liquidity
    liquidity_score = safe_divide(
        volume_ratio,
        (relative_spread + 1e-10) * (vol_ratio + 1e-10),
        fill_value=1.0
    )
    
    # Normalize to 0-1 range
    liquidity_score_normalized = (liquidity_score - liquidity_score.rolling(window=window*2, min_periods=1).min()) / (
        liquidity_score.rolling(window=window*2, min_periods=1).max() - 
        liquidity_score.rolling(window=window*2, min_periods=1).min() + 1e-10
    )
    
    return liquidity_score_normalized.fillna(0.5).clip(0.0, 1.0)

