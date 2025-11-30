"""
Session Features
Trading session-based features (London, New York, Asia)
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.features.utils import get_session_hours, ensure_timestamp_index


def compute_london_session(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute London session indicator.
    
    London session: 8:00-17:00 GMT (UTC)
    
    Args:
        df: DataFrame with timestamp index or 'timestamp' column
        **kwargs: Additional parameters
    
    Returns:
        Series with London session indicator (1 = in session, 0 = out)
    """
    df = ensure_timestamp_index(df)
    
    sessions = df.index.map(lambda ts: get_session_hours(ts)['london'])
    
    return pd.Series(sessions.astype(float), index=df.index)


def compute_new_york_session(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute New York session indicator.
    
    New York session: 13:00-22:00 GMT (UTC)
    
    Args:
        df: DataFrame with timestamp index or 'timestamp' column
        **kwargs: Additional parameters
    
    Returns:
        Series with New York session indicator (1 = in session, 0 = out)
    """
    df = ensure_timestamp_index(df)
    
    sessions = df.index.map(lambda ts: get_session_hours(ts)['new_york'])
    
    return pd.Series(sessions.astype(float), index=df.index)


def compute_asia_session(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute Asia session indicator.
    
    Asia session: 23:00-8:00 GMT (UTC) (spans midnight)
    
    Args:
        df: DataFrame with timestamp index or 'timestamp' column
        **kwargs: Additional parameters
    
    Returns:
        Series with Asia session indicator (1 = in session, 0 = out)
    """
    df = ensure_timestamp_index(df)
    
    sessions = df.index.map(lambda ts: get_session_hours(ts)['asia'])
    
    return pd.Series(sessions.astype(float), index=df.index)


def compute_session_overlap(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Compute trading session overlap indicator.
    
    Identifies periods when multiple sessions are active.
    
    Args:
        df: DataFrame with timestamp index or 'timestamp' column
        **kwargs: Additional parameters
    
    Returns:
        Series with overlap indicator (1 = overlap, 0 = no overlap)
    """
    df = ensure_timestamp_index(df)
    
    sessions = df.index.map(get_session_hours)
    
    # Count active sessions
    active_sessions = sessions.map(lambda s: sum([
        s['london'],
        s['new_york'],
        s['asia']
    ]))
    
    # Overlap = 2 or more sessions active
    overlap = (active_sessions >= 2).astype(float)
    
    return pd.Series(overlap, index=df.index)


def compute_session_volatility(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """
    Compute volatility by trading session.
    
    Calculates volatility separately for each session and assigns to current session.
    
    Args:
        df: DataFrame with OHLCV data and timestamp index
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with session-adjusted volatility
    """
    df = ensure_timestamp_index(df)
    
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    returns = df['close'].pct_change()
    
    # Get session for each timestamp
    sessions = df.index.map(get_session_hours)
    
    # Calculate volatility by session
    session_vol = pd.Series(0.0, index=df.index)
    
    for session_name in ['london', 'new_york', 'asia']:
        session_mask = sessions.map(lambda s: s[session_name])
        session_returns = returns[session_mask]
        
        if len(session_returns) > 0:
            vol = session_returns.rolling(window=window, min_periods=1).std()
            session_vol[session_mask] = vol
    
    return session_vol.fillna(0.0)

