"""
Technical Indicators
Classic technical analysis indicators (RSI, MACD, Stochastic, etc.)

These complement the existing feature modules with widely-used indicators.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from src.features.utils import safe_divide


def compute_rsi(df: pd.DataFrame, 
                window: int = 14, 
                **kwargs) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    
    RSI measures momentum on a 0-100 scale.
    - Above 70: Overbought
    - Below 30: Oversold
    
    Args:
        df: DataFrame with OHLCV data
        window: RSI period (default: 14)
        **kwargs: Additional parameters
    
    Returns:
        Series with RSI values (0-100)
    """
    if 'close' not in df.columns:
        return pd.Series(50.0, index=df.index)
    
    close = df['close']
    delta = close.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # Exponential moving average of gains and losses
    avg_gain = gains.ewm(span=window, adjust=False).mean()
    avg_loss = losses.ewm(span=window, adjust=False).mean()
    
    # RSI calculation
    rs = safe_divide(avg_gain, avg_loss, fill_value=0.0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.fillna(50.0)


def compute_rsi_normalized(df: pd.DataFrame,
                           window: int = 14,
                           **kwargs) -> pd.Series:
    """
    Compute normalized RSI (-1 to 1 scale).
    
    Centered around 0 for easier ML use.
    
    Args:
        df: DataFrame with OHLCV data
        window: RSI period (default: 14)
        **kwargs: Additional parameters
    
    Returns:
        Series with normalized RSI (-1 to 1)
    """
    rsi = compute_rsi(df, window, **kwargs)
    # Convert 0-100 to -1 to 1
    return (rsi - 50) / 50


def compute_macd(df: pd.DataFrame,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 **kwargs) -> pd.Series:
    """
    Compute MACD line (fast EMA - slow EMA).
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        **kwargs: Additional parameters
    
    Returns:
        Series with MACD line values
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    close = df['close']
    
    # EMAs
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    
    # MACD line
    macd = ema_fast - ema_slow
    
    return macd.fillna(0.0)


def compute_macd_signal(df: pd.DataFrame,
                        fast_period: int = 12,
                        slow_period: int = 26,
                        signal_period: int = 9,
                        **kwargs) -> pd.Series:
    """
    Compute MACD signal line.
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        **kwargs: Additional parameters
    
    Returns:
        Series with MACD signal line values
    """
    macd = compute_macd(df, fast_period, slow_period, signal_period, **kwargs)
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    return signal.fillna(0.0)


def compute_macd_histogram(df: pd.DataFrame,
                           fast_period: int = 12,
                           slow_period: int = 26,
                           signal_period: int = 9,
                           **kwargs) -> pd.Series:
    """
    Compute MACD histogram (MACD - Signal).
    
    Positive = bullish momentum
    Negative = bearish momentum
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)
        **kwargs: Additional parameters
    
    Returns:
        Series with MACD histogram values
    """
    macd = compute_macd(df, fast_period, slow_period, signal_period, **kwargs)
    signal = compute_macd_signal(df, fast_period, slow_period, signal_period, **kwargs)
    
    histogram = macd - signal
    
    return histogram.fillna(0.0)


def compute_stochastic_k(df: pd.DataFrame,
                         window: int = 14,
                         **kwargs) -> pd.Series:
    """
    Compute Stochastic %K (raw stochastic).
    
    Measures where close is relative to high-low range.
    - Above 80: Overbought
    - Below 20: Oversold
    
    Args:
        df: DataFrame with OHLCV data
        window: Lookback period (default: 14)
        **kwargs: Additional parameters
    
    Returns:
        Series with %K values (0-100)
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        return pd.Series(50.0, index=df.index)
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Highest high and lowest low over window
    highest_high = high.rolling(window=window, min_periods=1).max()
    lowest_low = low.rolling(window=window, min_periods=1).min()
    
    # %K calculation
    k = safe_divide(close - lowest_low, highest_high - lowest_low, fill_value=0.5) * 100
    
    return k.fillna(50.0)


def compute_stochastic_d(df: pd.DataFrame,
                         window: int = 14,
                         smooth: int = 3,
                         **kwargs) -> pd.Series:
    """
    Compute Stochastic %D (smoothed %K).
    
    Args:
        df: DataFrame with OHLCV data
        window: %K lookback period (default: 14)
        smooth: Smoothing period for %D (default: 3)
        **kwargs: Additional parameters
    
    Returns:
        Series with %D values (0-100)
    """
    k = compute_stochastic_k(df, window, **kwargs)
    d = k.rolling(window=smooth, min_periods=1).mean()
    
    return d.fillna(50.0)


def compute_stochastic_normalized(df: pd.DataFrame,
                                   window: int = 14,
                                   smooth: int = 3,
                                   **kwargs) -> pd.Series:
    """
    Compute normalized Stochastic (-1 to 1).
    
    Args:
        df: DataFrame with OHLCV data
        window: Lookback period (default: 14)
        smooth: Smoothing period (default: 3)
        **kwargs: Additional parameters
    
    Returns:
        Series with normalized stochastic (-1 to 1)
    """
    d = compute_stochastic_d(df, window, smooth, **kwargs)
    return (d - 50) / 50


def compute_cci(df: pd.DataFrame,
                window: int = 20,
                **kwargs) -> pd.Series:
    """
    Compute Commodity Channel Index (CCI).
    
    Measures deviation from statistical mean.
    - Above +100: Overbought / strong uptrend
    - Below -100: Oversold / strong downtrend
    
    Args:
        df: DataFrame with OHLCV data
        window: Lookback period (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with CCI values
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        return pd.Series(0.0, index=df.index)
    
    # Typical price
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # Simple moving average of typical price
    sma_tp = tp.rolling(window=window, min_periods=1).mean()
    
    # Mean absolute deviation
    mad = tp.rolling(window=window, min_periods=1).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    
    # CCI calculation
    cci = safe_divide(tp - sma_tp, 0.015 * mad, fill_value=0.0)
    
    return cci.fillna(0.0)


def compute_cci_normalized(df: pd.DataFrame,
                           window: int = 20,
                           **kwargs) -> pd.Series:
    """
    Compute normalized CCI (-1 to 1, clipped).
    
    Args:
        df: DataFrame with OHLCV data
        window: Lookback period (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with normalized CCI
    """
    cci = compute_cci(df, window, **kwargs)
    # Clip to -200 to 200, then normalize to -1 to 1
    cci_clipped = cci.clip(-200, 200)
    return cci_clipped / 200


def compute_williams_r(df: pd.DataFrame,
                       window: int = 14,
                       **kwargs) -> pd.Series:
    """
    Compute Williams %R.
    
    Similar to Stochastic but inverted.
    - Above -20: Overbought
    - Below -80: Oversold
    
    Args:
        df: DataFrame with OHLCV data
        window: Lookback period (default: 14)
        **kwargs: Additional parameters
    
    Returns:
        Series with Williams %R values (-100 to 0)
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        return pd.Series(-50.0, index=df.index)
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    highest_high = high.rolling(window=window, min_periods=1).max()
    lowest_low = low.rolling(window=window, min_periods=1).min()
    
    wr = safe_divide(highest_high - close, highest_high - lowest_low, fill_value=0.5) * -100
    
    return wr.fillna(-50.0)


def compute_williams_r_normalized(df: pd.DataFrame,
                                   window: int = 14,
                                   **kwargs) -> pd.Series:
    """
    Compute normalized Williams %R (-1 to 1).
    
    Args:
        df: DataFrame with OHLCV data
        window: Lookback period (default: 14)
        **kwargs: Additional parameters
    
    Returns:
        Series with normalized Williams %R
    """
    wr = compute_williams_r(df, window, **kwargs)
    # Convert -100 to 0 → -1 to 1
    return (wr + 50) / 50


def compute_roc(df: pd.DataFrame,
                window: int = 12,
                **kwargs) -> pd.Series:
    """
    Compute Rate of Change (ROC).
    
    Simple momentum as percentage change.
    
    Args:
        df: DataFrame with OHLCV data
        window: Lookback period (default: 12)
        **kwargs: Additional parameters
    
    Returns:
        Series with ROC values (percentage)
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    close = df['close']
    roc = ((close - close.shift(window)) / close.shift(window)) * 100
    
    return roc.fillna(0.0)


def compute_atr(df: pd.DataFrame,
                window: int = 14,
                **kwargs) -> pd.Series:
    """
    Compute Average True Range (ATR).
    
    Measures volatility using high, low, close.
    
    Args:
        df: DataFrame with OHLCV data
        window: ATR period (default: 14)
        **kwargs: Additional parameters
    
    Returns:
        Series with ATR values
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        return pd.Series(0.0, index=df.index)
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range components
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    # True Range = max of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Average True Range
    atr = tr.ewm(span=window, adjust=False).mean()
    
    return atr.fillna(0.0)


def compute_atr_normalized(df: pd.DataFrame,
                           window: int = 14,
                           **kwargs) -> pd.Series:
    """
    Compute ATR as percentage of price (normalized).
    
    Allows comparison across different price levels.
    
    Args:
        df: DataFrame with OHLCV data
        window: ATR period (default: 14)
        **kwargs: Additional parameters
    
    Returns:
        Series with ATR percentage
    """
    atr = compute_atr(df, window, **kwargs)
    close = df['close'] if 'close' in df.columns else pd.Series(1.0, index=df.index)
    
    atr_pct = safe_divide(atr, close, fill_value=0.0) * 100
    
    return atr_pct.fillna(0.0)


def compute_obv(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute On-Balance Volume (OBV).
    
    Cumulative volume based on price direction.
    Rising OBV = accumulation
    Falling OBV = distribution
    
    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional parameters
    
    Returns:
        Series with OBV values
    """
    if 'close' not in df.columns or 'volume' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    close = df['close']
    volume = df['volume']
    
    # Direction based on close change
    direction = np.sign(close.diff())
    
    # OBV = cumulative signed volume
    obv = (direction * volume).cumsum()
    
    return obv.fillna(0.0)


def compute_obv_normalized(df: pd.DataFrame,
                           window: int = 20,
                           **kwargs) -> pd.Series:
    """
    Compute normalized OBV (z-score over window).
    
    Args:
        df: DataFrame with OHLCV data
        window: Normalization window (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with normalized OBV
    """
    obv = compute_obv(df, **kwargs)
    
    obv_mean = obv.rolling(window=window, min_periods=1).mean()
    obv_std = obv.rolling(window=window, min_periods=1).std()
    
    obv_zscore = safe_divide(obv - obv_mean, obv_std, fill_value=0.0)
    
    return obv_zscore.fillna(0.0)


def compute_mfi(df: pd.DataFrame,
                window: int = 14,
                **kwargs) -> pd.Series:
    """
    Compute Money Flow Index (MFI).
    
    Volume-weighted RSI.
    - Above 80: Overbought
    - Below 20: Oversold
    
    Args:
        df: DataFrame with OHLCV data
        window: MFI period (default: 14)
        **kwargs: Additional parameters
    
    Returns:
        Series with MFI values (0-100)
    """
    required = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required):
        return pd.Series(50.0, index=df.index)
    
    # Typical price
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # Raw money flow
    raw_mf = tp * df['volume']
    
    # Positive and negative money flow
    tp_diff = tp.diff()
    positive_mf = raw_mf.where(tp_diff > 0, 0.0)
    negative_mf = raw_mf.where(tp_diff < 0, 0.0)
    
    # Money flow ratio
    positive_sum = positive_mf.rolling(window=window, min_periods=1).sum()
    negative_sum = negative_mf.rolling(window=window, min_periods=1).sum()
    
    mf_ratio = safe_divide(positive_sum, negative_sum, fill_value=1.0)
    
    # MFI calculation
    mfi = 100 - (100 / (1 + mf_ratio))
    
    return mfi.fillna(50.0)


def compute_mfi_normalized(df: pd.DataFrame,
                           window: int = 14,
                           **kwargs) -> pd.Series:
    """
    Compute normalized MFI (-1 to 1).
    
    Args:
        df: DataFrame with OHLCV data
        window: MFI period (default: 14)
        **kwargs: Additional parameters
    
    Returns:
        Series with normalized MFI
    """
    mfi = compute_mfi(df, window, **kwargs)
    return (mfi - 50) / 50


def compute_pivot_point(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute Pivot Point (standard).
    
    PP = (High + Low + Close) / 3
    
    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional parameters
    
    Returns:
        Series with pivot point values
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        return pd.Series(0.0, index=df.index)
    
    pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    
    return pivot.fillna(method='bfill').fillna(0.0)


def compute_pivot_distance(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute distance from pivot point (normalized).
    
    Positive = above pivot (bullish)
    Negative = below pivot (bearish)
    
    Args:
        df: DataFrame with OHLCV data
        **kwargs: Additional parameters
    
    Returns:
        Series with normalized distance from pivot
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    pivot = compute_pivot_point(df, **kwargs)
    close = df['close']
    
    # Distance as percentage
    distance = safe_divide(close - pivot, pivot, fill_value=0.0) * 100
    
    return distance.fillna(0.0)


def compute_bollinger_bandwidth(df: pd.DataFrame,
                                 window: int = 20,
                                 num_std: float = 2.0,
                                 **kwargs) -> pd.Series:
    """
    Compute Bollinger Bandwidth.
    
    Measures width of Bollinger Bands relative to middle band.
    Low bandwidth = consolidation / squeeze
    High bandwidth = expansion / trending
    
    Args:
        df: DataFrame with OHLCV data
        window: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2.0)
        **kwargs: Additional parameters
    
    Returns:
        Series with bandwidth values
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    close = df['close']
    
    # Middle band (SMA)
    middle = close.rolling(window=window, min_periods=1).mean()
    
    # Standard deviation
    std = close.rolling(window=window, min_periods=1).std()
    
    # Upper and lower bands
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    
    # Bandwidth
    bandwidth = safe_divide(upper - lower, middle, fill_value=0.0)
    
    return bandwidth.fillna(0.0)


def compute_bollinger_percent_b(df: pd.DataFrame,
                                 window: int = 20,
                                 num_std: float = 2.0,
                                 **kwargs) -> pd.Series:
    """
    Compute Bollinger %B.
    
    Measures where price is relative to bands.
    - Above 1: Above upper band
    - Below 0: Below lower band
    - 0.5: At middle band
    
    Args:
        df: DataFrame with OHLCV data
        window: Moving average period (default: 20)
        num_std: Number of standard deviations (default: 2.0)
        **kwargs: Additional parameters
    
    Returns:
        Series with %B values
    """
    if 'close' not in df.columns:
        return pd.Series(0.5, index=df.index)
    
    close = df['close']
    
    # Middle band (SMA)
    middle = close.rolling(window=window, min_periods=1).mean()
    
    # Standard deviation
    std = close.rolling(window=window, min_periods=1).std()
    
    # Upper and lower bands
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    
    # %B calculation
    percent_b = safe_divide(close - lower, upper - lower, fill_value=0.5)
    
    return percent_b.fillna(0.5)