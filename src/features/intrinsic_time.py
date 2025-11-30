"""
Intrinsic-Time Features
Directional-change and intrinsic-time based features
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from src.features.utils import safe_divide


def compute_dc_return(df: pd.DataFrame, delta: float = 0.0010, **kwargs) -> pd.Series:
    """
    Compute directional-change return for threshold delta.
    
    Args:
        df: DataFrame with intrinsic-time events (must have 'dc_return' column)
        delta: Directional-change threshold (default: 0.0010)
        **kwargs: Additional parameters
    
    Returns:
        Series with DC returns
    """
    if 'dc_return' in df.columns:
        return df['dc_return'].copy()
    else:
        # If not available, return zeros
        return pd.Series(0.0, index=df.index)


def compute_overshoot_return(df: pd.DataFrame, delta: float = 0.0010, **kwargs) -> pd.Series:
    """
    Compute overshoot return after DC event.
    
    Args:
        df: DataFrame with intrinsic-time events (must have 'overshoot_return' column)
        delta: Directional-change threshold (default: 0.0010)
        **kwargs: Additional parameters
    
    Returns:
        Series with overshoot returns
    """
    if 'overshoot_return' in df.columns:
        return df['overshoot_return'].copy()
    else:
        return pd.Series(0.0, index=df.index)


def compute_overshoot_ratio(df: pd.DataFrame, delta: float = 0.0010, **kwargs) -> pd.Series:
    """
    Compute ratio of overshoot return to DC return.
    
    Args:
        df: DataFrame with intrinsic-time events
        delta: Directional-change threshold (default: 0.0010)
        **kwargs: Additional parameters
    
    Returns:
        Series with overshoot ratios
    """
    dc_return = compute_dc_return(df, delta, **kwargs)
    overshoot_return = compute_overshoot_return(df, delta, **kwargs)
    
    return safe_divide(overshoot_return, dc_return, fill_value=0.0)


def compute_event_frequency(df: pd.DataFrame, window: int = 24, **kwargs) -> pd.Series:
    """
    Compute number of DC events per unit time.
    
    Args:
        df: DataFrame with intrinsic-time events (must have 'dc_timestamp' or index)
        window: Time window in hours (default: 24)
        **kwargs: Additional parameters
    
    Returns:
        Series with event frequency
    """
    if 'dc_timestamp' in df.columns:
        timestamps = pd.to_datetime(df['dc_timestamp'])
    elif isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    else:
        return pd.Series(0.0, index=df.index)
    
    # Count events in rolling window
    if len(timestamps) == 0:
        return pd.Series(0.0, index=df.index)
    
    # Create a series of ones for each event
    event_series = pd.Series(1, index=timestamps)
    event_series = event_series.reindex(df.index, fill_value=0)
    
    # Rolling count
    window_timedelta = pd.Timedelta(hours=window)
    frequency = event_series.rolling(window=window_timedelta).sum()
    
    return frequency.fillna(0.0)


def compute_clustering(df: pd.DataFrame, window: int = 10, **kwargs) -> pd.Series:
    """
    Compute temporal clustering of DC events.
    
    Measures how clustered events are in time (variance of inter-event times).
    
    Args:
        df: DataFrame with intrinsic-time events
        window: Number of events to consider (default: 10)
        **kwargs: Additional parameters
    
    Returns:
        Series with clustering measure (lower = more clustered)
    """
    # Get timestamps - prefer dc_timestamp column, then index if DatetimeIndex, then timestamp column
    if 'dc_timestamp' in df.columns:
        timestamps = pd.to_datetime(df['dc_timestamp'])
    elif isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    elif 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
    else:
        return pd.Series(0.0, index=df.index)
    
    if len(timestamps) < 2:
        return pd.Series(0.0, index=df.index)
    
    # Calculate inter-event times
    # timestamps.diff() returns TimedeltaIndex, convert to Series first
    # Use df.index to preserve the original index type
    inter_event_times = pd.Series(timestamps.diff(), index=df.index).dt.total_seconds()
    
    # Rolling coefficient of variation (std / mean)
    mean_inter = inter_event_times.rolling(window=window, min_periods=2).mean()
    std_inter = inter_event_times.rolling(window=window, min_periods=2).std()
    
    clustering = safe_divide(std_inter, mean_inter, fill_value=0.0)
    
    # clustering already has df.index, so just fill NaN values
    return clustering.fillna(0.0)


def compute_multi_delta_agreement(df: pd.DataFrame, deltas: List[float] = [0.0005, 0.0010, 0.0020], **kwargs) -> pd.Series:
    """
    Compute agreement across multiple delta thresholds.
    
    Measures how often different thresholds agree on direction.
    
    Args:
        df: DataFrame with clock-time data (will need to load multiple intrinsic files)
        deltas: List of delta thresholds (default: [0.0005, 0.0010, 0.0020])
        **kwargs: Additional parameters
    
    Returns:
        Series with agreement score (0-1, higher = more agreement)
    """
    # This is a placeholder - in practice, would need to load multiple intrinsic files
    # and compare their directions at aligned timestamps
    
    # For now, return a simple implementation if direction column exists
    if 'direction' in df.columns:
        # Simple agreement: consistency of direction
        direction = df['direction']
        agreement = (direction == direction.shift(1)).astype(float)
        return agreement.fillna(0.0)
    else:
        return pd.Series(0.5, index=df.index)  # Neutral if no data


def compute_intrinsic_trend_strength(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute strength of trend in intrinsic time.
    
    Measures the consistency of directional changes.
    
    Args:
        df: DataFrame with intrinsic-time events
        window: Number of events to consider (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with trend strength (0-1, higher = stronger trend)
    """
    if 'direction' in df.columns:
        direction = df['direction']
        
        # Count consecutive same-direction events
        direction_changes = (direction != direction.shift(1)).astype(int)
        
        # Rolling mean of direction (absolute value)
        # Positive trend: more +1, negative trend: more -1
        trend_strength = direction.rolling(window=window, min_periods=1).mean().abs()
        
        return trend_strength.fillna(0.0)
    else:
        return pd.Series(0.0, index=df.index)

