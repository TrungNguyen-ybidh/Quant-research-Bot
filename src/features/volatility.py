"""
Volatility Features
OHLC-based volatility structure and measures
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.features.utils import calculate_returns, safe_divide


def compute_realized_volatility(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute realized volatility (annualized).
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with realized volatility (annualized)
    """
    returns = calculate_returns(df, price_col='close', method='simple')
    
    # Realized volatility: std of returns
    vol = returns.rolling(window=window, min_periods=1).std()
    
    # Annualize (assuming daily data, adjust multiplier as needed)
    # For hourly: sqrt(24 * 365) ≈ 93.6
    # For daily: sqrt(252) ≈ 15.87
    # For 1-minute: sqrt(1440 * 365) ≈ 725
    annualization_factor = np.sqrt(252)  # Default for daily, adjust based on timeframe
    vol_annualized = vol * annualization_factor
    
    return vol_annualized.fillna(0.0)


def compute_bipower_variation(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute bipower variation (jump-robust volatility estimator).
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with bipower variation
    """
    returns = calculate_returns(df, price_col='close', method='simple')
    
    # Bipower variation: sum of |r_t| * |r_{t-1}|
    abs_returns = returns.abs()
    bipower = abs_returns * abs_returns.shift(1)
    
    # Rolling sum
    bipower_var = bipower.rolling(window=window, min_periods=2).sum()
    
    # Scale by pi/2
    bipower_var = bipower_var * (np.pi / 2)
    
    return bipower_var.fillna(0.0)


def compute_jump_component(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute jump component of volatility.
    
    Jump = Realized Volatility - Bipower Variation
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with jump component
    """
    realized_vol = compute_realized_volatility(df, window, **kwargs)
    bipower_var = compute_bipower_variation(df, window, **kwargs)
    
    # Jump component (positive indicates jumps)
    jump = realized_vol - bipower_var
    
    return jump.clip(lower=0.0)  # Only positive jumps


def compute_volatility_of_volatility(df: pd.DataFrame, window: int = 20, vol_window: int = 10, **kwargs) -> pd.Series:
    """
    Compute volatility of volatility.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window for volatility calculation (default: 20)
        vol_window: Window for vol-of-vol calculation (default: 10)
        **kwargs: Additional parameters
    
    Returns:
        Series with volatility of volatility
    """
    # First compute realized volatility
    realized_vol = compute_realized_volatility(df, window, **kwargs)
    
    # Then compute volatility of that volatility
    vol_of_vol = realized_vol.rolling(window=vol_window, min_periods=1).std()
    
    return vol_of_vol.fillna(0.0)


def compute_parkinson_volatility(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute Parkinson volatility estimator (uses high-low).
    
    Args:
        df: DataFrame with OHLCV data (must have 'high' and 'low')
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with Parkinson volatility
    """
    if 'high' not in df.columns or 'low' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    high = df['high']
    low = df['low']
    
    # Parkinson estimator: (1/(4*ln(2))) * (ln(H/L))^2
    log_hl = np.log(high / low)
    parkinson = (1.0 / (4.0 * np.log(2.0))) * (log_hl ** 2)
    
    # Rolling mean
    parkinson_vol = parkinson.rolling(window=window, min_periods=1).mean()
    
    # Annualize
    annualization_factor = np.sqrt(252)
    parkinson_vol = parkinson_vol * annualization_factor
    
    return parkinson_vol.fillna(0.0)


def compute_skew(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute return skewness.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with skewness
    """
    returns = calculate_returns(df, price_col='close', method='simple')
    
    skew = returns.rolling(window=window, min_periods=window).skew()
    
    return skew.fillna(0.0)


def compute_kurtosis(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute return kurtosis.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with kurtosis
    """
    returns = calculate_returns(df, price_col='close', method='simple')
    
    kurt = returns.rolling(window=window, min_periods=window).kurt()
    
    return kurt.fillna(0.0)


def compute_variance_ratio(df: pd.DataFrame, short_window: int = 5, long_window: int = 20, **kwargs) -> pd.Series:
    """
    Compute variance ratio test statistic.
    
    Tests for mean reversion (VR < 1) or momentum (VR > 1).
    
    Args:
        df: DataFrame with OHLCV data
        short_window: Short-term window (default: 5)
        long_window: Long-term window (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with variance ratio
    """
    returns = calculate_returns(df, price_col='close', method='simple')
    
    # Short-term variance
    var_short = returns.rolling(window=short_window, min_periods=short_window).var()
    
    # Long-term variance
    var_long = returns.rolling(window=long_window, min_periods=long_window).var()
    
    # Variance ratio
    vr = safe_divide(var_short * long_window, var_long * short_window, fill_value=1.0)
    
    return vr.fillna(1.0)


def compute_hurst_exponent(df: pd.DataFrame, window: int = 100, **kwargs) -> pd.Series:
    """
    Compute Hurst exponent (long memory measure).
    
    H < 0.5: mean reverting
    H = 0.5: random walk
    H > 0.5: trending
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 100)
        **kwargs: Additional parameters
    
    Returns:
        Series with Hurst exponent (0-1)
    """
    returns = calculate_returns(df, price_col='close', method='simple')
    
    def hurst_rs(series):
        """Compute Hurst using R/S statistic."""
        if len(series) < 10:
            return 0.5
        
        # Cumulative sum (price series)
        cumsum = series.cumsum()
        
        # Mean
        mean = series.mean()
        
        # Adjusted range
        adjusted_range = cumsum - (np.arange(1, len(series) + 1) * mean)
        R = adjusted_range.max() - adjusted_range.min()
        
        # Standard deviation
        S = series.std()
        
        if S == 0 or R == 0:
            return 0.5
        
        # R/S statistic
        rs = R / S
        
        # Hurst = log(RS) / log(n)
        if rs > 0:
            H = np.log(rs) / np.log(len(series))
            return np.clip(H, 0, 1)
        else:
            return 0.5
    
    hurst = returns.rolling(window=window, min_periods=10).apply(hurst_rs, raw=True)
    
    return hurst.fillna(0.5)

