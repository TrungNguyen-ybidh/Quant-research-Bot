"""
FX Factor Features
FX-specific factor models (carry, momentum, value, volatility)

Updated with real interest rate data from FRED API.
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.features.utils import calculate_returns, normalize_series, safe_divide
from src.utils.logger import info, error


# Cache for external data clients (lazy initialization)
_fred_client = None
_external_store = None


def _get_fred_client():
    """Lazy initialization of FRED client."""
    global _fred_client
    if _fred_client is None:
        try:
            from src.data.external.fred_client import get_fred_client
            _fred_client = get_fred_client()
        except Exception as e:
            error(f"Failed to initialize FRED client: {e}")
    return _fred_client


def _get_external_store():
    """Lazy initialization of external data store."""
    global _external_store
    if _external_store is None:
        try:
            from src.data.external.external_store import get_external_store
            _external_store = get_external_store()
        except Exception as e:
            error(f"Failed to initialize external store: {e}")
    return _external_store


def _extract_pair_from_df(df: pd.DataFrame) -> Optional[str]:
    """Extract FX pair from DataFrame."""
    if 'symbol' in df.columns:
        symbol = df['symbol'].iloc[0] if len(df) > 0 else None
        if symbol and len(str(symbol)) == 6:
            return str(symbol).upper()
    return None


def _align_external_to_df(external_series: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    Align external data series to DataFrame index.
    Works with DataFrames that have timestamp as a column (not index).
    """
    if external_series is None or len(external_series) == 0:
        return pd.Series(0.0, index=df.index)
    
    try:
        external = external_series.copy()
        
        # Ensure datetime index on external series
        if not isinstance(external.index, pd.DatetimeIndex):
            external.index = pd.to_datetime(external.index)
        
        # Remove timezone info from external series
        if external.index.tz is not None:
            external.index = external.index.tz_localize(None)
        
        # Get timestamps from df
        if 'timestamp' in df.columns:
            df_timestamps = pd.to_datetime(df['timestamp'])
        elif isinstance(df.index, pd.DatetimeIndex):
            df_timestamps = df.index.to_series()
        else:
            try:
                df_timestamps = pd.to_datetime(df.index).to_series()
            except:
                error("Cannot find timestamps in DataFrame")
                return pd.Series(0.0, index=df.index)
        
        # Remove timezone from df timestamps
        if hasattr(df_timestamps, 'dt') and df_timestamps.dt.tz is not None:
            df_timestamps = df_timestamps.dt.tz_localize(None)
        
        # Convert external series to daily dates
        external_daily = external.copy()
        external_daily.index = pd.to_datetime(external_daily.index.date)
        external_daily = external_daily[~external_daily.index.duplicated(keep='last')]
        external_daily = external_daily.sort_index()
        
        # Get dates from df timestamps
        if hasattr(df_timestamps, 'dt'):
            df_dates = pd.to_datetime(df_timestamps.dt.date)
        else:
            df_dates = pd.to_datetime(df_timestamps.date)
        
        # Create result series
        result = pd.Series(index=df.index, dtype=float)
        
        # Map each df date to the external value
        for i, date in enumerate(df_dates):
            mask = external_daily.index <= date
            if mask.any():
                result.iloc[i] = external_daily[mask].iloc[-1]
            else:
                result.iloc[i] = np.nan
        
        result = result.ffill().bfill().fillna(0.0)
        return result
        
    except Exception as e:
        error(f"Error in _align_external_to_df: {e}")
        import traceback
        traceback.print_exc()
        return pd.Series(0.0, index=df.index)


def _safe_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Safely compute z-score with level-based fallback.
    
    When values are constant (z-score would be 0), returns a scaled
    version of the raw level instead.
    """
    # Calculate global stats
    global_mean = series.mean()
    global_std = series.std()
    
    # If entire series is constant, return level-based score
    if global_std < 1e-8:
        if abs(global_mean) < 1e-8:
            return pd.Series(0.0, index=series.index)
        # Scale to roughly -1 to 1 based on typical rate ranges (0-5%)
        scaled = (global_mean / 2.5).clip(-1, 1)
        return pd.Series(scaled, index=series.index)
    
    # Normal z-score with rolling window
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    # Replace zero/tiny std with global std
    rolling_std = rolling_std.replace(0, global_std)
    rolling_std = rolling_std.fillna(global_std)
    rolling_std = rolling_std.clip(lower=global_std * 0.1)
    
    zscore = (series - rolling_mean) / rolling_std
    
    # KEY FIX: When z-score is near zero but level is significant,
    # blend in the level-based signal
    level_signal = series / 2.5  # Scale rate diff to roughly -1 to 1
    level_signal = level_signal.clip(-1, 1)
    
    # Use level signal when z-score is near zero
    zscore_abs = zscore.abs()
    blend_weight = (1 - zscore_abs.clip(0, 1))  # Higher weight when zscore near 0
    
    # Blended result: mostly z-score, but level-based when z-score is flat
    result = zscore * (1 - blend_weight * 0.5) + level_signal * (blend_weight * 0.5)
    
    # Clip extreme values
    result = result.clip(-3, 3)
    
    return result.fillna(0.0)


def compute_carry_factor(df: pd.DataFrame, 
                         pair: Optional[str] = None,
                         use_external: bool = True,
                         **kwargs) -> pd.Series:
    """
    Compute carry trade factor (interest rate differential).
    
    Uses FRED API to fetch real central bank rates when available.
    Falls back to proxy if external data unavailable.
    
    Args:
        df: DataFrame with OHLCV data
        pair: FX pair (e.g., 'EURUSD'). Auto-detected if not provided.
        use_external: Whether to try external data sources
        **kwargs: Additional parameters
    
    Returns:
        Series with carry factor (normalized)
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    if pair is None:
        pair = _extract_pair_from_df(df)
    
    carry_raw = None
    
    # Try external data first
    if use_external and pair is not None:
        try:
            store = _get_external_store()
            fred = _get_fred_client()
            
            if store is not None and fred is not None and fred.is_connected:
                info(f"Fetching carry factor for {pair} from FRED")
                
                rate_diff = store.get_rate_differential(pair, fred)
                
                if rate_diff is not None and len(rate_diff) > 0:
                    carry_raw = _align_external_to_df(rate_diff, df)
                    
                    if carry_raw.abs().sum() > 0:
                        info(f"Using real interest rate differential for {pair}")
                    else:
                        info("Aligned data is all zeros, falling back to proxy")
                        carry_raw = None
        except Exception as e:
            error(f"Failed to fetch external carry data: {e}")
            carry_raw = None
    
    # Fallback to proxy
    if carry_raw is None:
        info("Using carry proxy (forward return)")
        returns = calculate_returns(df, price_col='close', method='simple')
        carry_raw = returns.rolling(window=20, min_periods=1).mean()
    
    # Normalize using safe z-score
    carry_normalized = _safe_zscore(carry_raw, window=60)
    
    return carry_normalized


def compute_carry_raw(df: pd.DataFrame,
                      pair: Optional[str] = None,
                      **kwargs) -> pd.Series:
    """
    Compute raw carry (interest rate differential without normalization).
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    if pair is None:
        pair = _extract_pair_from_df(df)
    
    if pair is None:
        return pd.Series(0.0, index=df.index)
    
    try:
        store = _get_external_store()
        fred = _get_fred_client()
        
        if store is not None and fred is not None and fred.is_connected:
            rate_diff = store.get_rate_differential(pair, fred)
            
            if rate_diff is not None and len(rate_diff) > 0:
                aligned = _align_external_to_df(rate_diff, df)
                if aligned.abs().sum() > 0:
                    return aligned
    except Exception as e:
        error(f"Failed to fetch raw carry: {e}")
    
    return pd.Series(0.0, index=df.index)


def compute_momentum_factor(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """Compute momentum factor (past return)."""
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    close = df['close']
    momentum = (close / close.shift(window) - 1.0)
    momentum_normalized = _safe_zscore(momentum, window=window*2)
    
    return momentum_normalized.fillna(0.0)


def compute_value_factor(df: pd.DataFrame, window: int = 60, **kwargs) -> pd.Series:
    """Compute value factor (deviation from mean)."""
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    close = df['close']
    mean_price = close.rolling(window=window, min_periods=window).mean()
    value_deviation = (close - mean_price) / mean_price
    value_factor = -value_deviation  # Positive = undervalued
    value_normalized = _safe_zscore(value_factor, window=window*2)
    
    return value_normalized.fillna(0.0)


def compute_volatility_factor(df: pd.DataFrame, window: int = 20, **kwargs) -> pd.Series:
    """Compute volatility factor (normalized volatility regime)."""
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    returns = calculate_returns(df, price_col='close', method='simple')
    vol_short = returns.rolling(window=window//2, min_periods=1).std()
    vol_long = returns.rolling(window=window, min_periods=window).std()
    vol_ratio = safe_divide(vol_short, vol_long + 1e-10, fill_value=1.0)
    vol_factor = _safe_zscore(vol_ratio, window=window*2)
    
    return vol_factor.fillna(0.0)


def compute_yield_curve_factor(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Compute yield curve factor (US 10Y - 2Y spread)."""
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    try:
        fred = _get_fred_client()
        store = _get_external_store()
        
        if fred is not None and fred.is_connected and store is not None:
            us10y = store.get_or_fetch_fred('DGS10', fred)
            us2y = store.get_or_fetch_fred('DGS2', fred)
            
            if us10y is not None and us2y is not None:
                combined = pd.DataFrame({'us10y': us10y, 'us2y': us2y})
                combined = combined.ffill().bfill()
                spread = combined['us10y'] - combined['us2y']
                spread_aligned = _align_external_to_df(spread, df)
                
                if spread_aligned.abs().sum() > 0:
                    spread_normalized = _safe_zscore(spread_aligned, window=60)
                    return spread_normalized.fillna(0.0)
    except Exception as e:
        error(f"Failed to compute yield curve factor: {e}")
    
    return pd.Series(0.0, index=df.index)


def compute_rate_level_factor(df: pd.DataFrame,
                              currency: str = 'USD',
                              **kwargs) -> pd.Series:
    """Compute absolute interest rate level factor."""
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    try:
        fred = _get_fred_client()
        store = _get_external_store()
        
        if fred is not None and fred.is_connected and store is not None:
            series_id = fred.RATE_SERIES.get(currency, 'DFF')
            rate = store.get_or_fetch_fred(series_id, fred)
            
            if rate is not None and len(rate) > 0:
                rate_aligned = _align_external_to_df(rate, df)
                
                if rate_aligned.abs().sum() > 0:
                    rate_normalized = _safe_zscore(rate_aligned, window=60)
                    return rate_normalized.fillna(0.0)
    except Exception as e:
        error(f"Failed to compute rate level factor: {e}")
    
    return pd.Series(0.0, index=df.index)