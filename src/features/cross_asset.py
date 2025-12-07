"""
Cross-Asset Features
Features that incorporate relationships with other assets (DXY, SPX, VIX, Gold, Oil)

Updated with real market data from Yahoo Finance.
FIXED: Timezone alignment issues between FX data and external data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from src.features.utils import calculate_returns, safe_divide, normalize_series
from src.utils.logger import info, error


# Cache for external data clients (lazy initialization)
_yahoo_client = None
_external_store = None


def _get_yahoo_client():
    """Lazy initialization of Yahoo client."""
    global _yahoo_client
    if _yahoo_client is None:
        try:
            from src.data.external.yahoo_client import get_yahoo_client
            _yahoo_client = get_yahoo_client()
        except Exception as e:
            error(f"Failed to initialize Yahoo client: {e}")
    return _yahoo_client


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


def _strip_timezone(index_or_series):
    """
    Remove timezone info from DatetimeIndex or Series index.
    
    Args:
        index_or_series: DatetimeIndex, Series, or DataFrame
    
    Returns:
        Object with tz-naive index
    """
    if isinstance(index_or_series, pd.DatetimeIndex):
        if index_or_series.tz is not None:
            return index_or_series.tz_localize(None)
        return index_or_series
    elif isinstance(index_or_series, (pd.Series, pd.DataFrame)):
        if hasattr(index_or_series.index, 'tz') and index_or_series.index.tz is not None:
            result = index_or_series.copy()
            result.index = result.index.tz_localize(None)
            return result
        return index_or_series
    return index_or_series


def _align_external_to_df(external_series: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    Align external data series to DataFrame index.
    
    Args:
        external_series: Series with external data (usually daily)
        df: Target DataFrame (may be intraday)
    
    Returns:
        Series aligned to df.index (tz-naive)
    """
    if external_series is None or len(external_series) == 0:
        return pd.Series(np.nan, index=_strip_timezone(df).index)
    
    # Ensure datetime index and strip timezone
    external_series = _strip_timezone(external_series)
    external_series.index = pd.to_datetime(external_series.index)
    
    # Get tz-naive version of df index
    df_index = _strip_timezone(df.index) if isinstance(df.index, pd.DatetimeIndex) else df.index
    
    # Reindex with forward fill
    combined_index = external_series.index.union(df_index)
    aligned = external_series.reindex(combined_index).ffill()
    
    # Select only df timestamps
    result = aligned.reindex(df_index)
    
    return result


def _get_cross_asset_returns(df: pd.DataFrame, 
                              asset: str,
                              window: int = 20) -> Optional[pd.Series]:
    """
    Get returns for a cross-asset, aligned to df.
    
    Args:
        df: Target DataFrame
        asset: Asset symbol ('DXY', 'SPX', 'VIX', 'GOLD', 'OIL')
        window: Window for computing returns
    
    Returns:
        Series with asset returns aligned to df index (tz-naive)
    """
    try:
        yahoo = _get_yahoo_client()
        store = _get_external_store()
        
        if yahoo is None or store is None:
            return None
        
        # Fetch asset data (cached)
        asset_data = store.get_or_fetch_yahoo(asset, yahoo)
        
        if asset_data is None or 'close' not in asset_data.columns:
            return None
        
        # Compute returns
        asset_close = asset_data['close']
        asset_returns = asset_close.pct_change()
        
        # Align to df (handles timezone stripping)
        returns_aligned = _align_external_to_df(asset_returns, df)
        
        return returns_aligned
        
    except Exception as e:
        error(f"Failed to get {asset} returns: {e}")
        return None


def compute_correlation_dxy(df: pd.DataFrame,
                            window: int = 20,
                            **kwargs) -> pd.Series:
    """
    Compute rolling correlation with US Dollar Index (DXY).
    
    Positive correlation = moves with dollar strength
    Negative correlation = moves against dollar
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with rolling correlation to DXY
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    # Get FX returns and strip timezone
    fx_returns = calculate_returns(df, price_col='close', method='simple')
    fx_returns = _strip_timezone(fx_returns)
    
    # Get DXY returns (already tz-naive from _get_cross_asset_returns)
    dxy_returns = _get_cross_asset_returns(df, 'DXY', window)
    
    if dxy_returns is None:
        # Fallback: use autocorrelation
        info("DXY unavailable, using autocorrelation as proxy")
        correlation = fx_returns.rolling(window=window, min_periods=window//2).corr(fx_returns.shift(1))
        return correlation.fillna(0.0).reindex(df.index)
    
    # Compute rolling correlation (both are now tz-naive)
    correlation = fx_returns.rolling(window=window, min_periods=window//2).corr(dxy_returns)
    
    return correlation.fillna(0.0).reindex(df.index)


def compute_correlation_spx(df: pd.DataFrame,
                            window: int = 20,
                            **kwargs) -> pd.Series:
    """
    Compute rolling correlation with S&P 500 (SPX).
    
    Measures risk sentiment relationship.
    Positive = risk-on currency
    Negative = safe-haven currency
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with rolling correlation to SPX
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    fx_returns = _strip_timezone(calculate_returns(df, price_col='close', method='simple'))
    spx_returns = _get_cross_asset_returns(df, 'SPX', window)
    
    if spx_returns is None:
        info("SPX unavailable, returning zeros")
        return pd.Series(0.0, index=df.index)
    
    correlation = fx_returns.rolling(window=window, min_periods=window//2).corr(spx_returns)
    
    return correlation.fillna(0.0).reindex(df.index)


def compute_correlation_gold(df: pd.DataFrame,
                             window: int = 20,
                             **kwargs) -> pd.Series:
    """
    Compute rolling correlation with Gold.
    
    Gold is typically safe-haven / inflation hedge.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with rolling correlation to Gold
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    fx_returns = _strip_timezone(calculate_returns(df, price_col='close', method='simple'))
    gold_returns = _get_cross_asset_returns(df, 'GOLD', window)
    
    if gold_returns is None:
        info("Gold unavailable, returning zeros")
        return pd.Series(0.0, index=df.index)
    
    correlation = fx_returns.rolling(window=window, min_periods=window//2).corr(gold_returns)
    
    return correlation.fillna(0.0).reindex(df.index)


def compute_correlation_vix(df: pd.DataFrame,
                            window: int = 20,
                            **kwargs) -> pd.Series:
    """
    Compute rolling correlation with VIX (volatility index).
    
    Positive = currency strengthens when volatility rises (safe haven)
    Negative = currency weakens when volatility rises (risk currency)
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with rolling correlation to VIX
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    fx_returns = _strip_timezone(calculate_returns(df, price_col='close', method='simple'))
    vix_returns = _get_cross_asset_returns(df, 'VIX', window)
    
    if vix_returns is None:
        info("VIX unavailable, returning zeros")
        return pd.Series(0.0, index=df.index)
    
    correlation = fx_returns.rolling(window=window, min_periods=window//2).corr(vix_returns)
    
    return correlation.fillna(0.0).reindex(df.index)


def compute_correlation_oil(df: pd.DataFrame,
                            window: int = 20,
                            **kwargs) -> pd.Series:
    """
    Compute rolling correlation with Oil (WTI Crude).
    
    Important for commodity currencies (CAD, NOK, RUB).
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with rolling correlation to Oil
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    fx_returns = _strip_timezone(calculate_returns(df, price_col='close', method='simple'))
    oil_returns = _get_cross_asset_returns(df, 'OIL', window)
    
    if oil_returns is None:
        info("Oil unavailable, returning zeros")
        return pd.Series(0.0, index=df.index)
    
    correlation = fx_returns.rolling(window=window, min_periods=window//2).corr(oil_returns)
    
    return correlation.fillna(0.0).reindex(df.index)


def compute_relative_strength_vs_dxy(df: pd.DataFrame,
                                      window: int = 20,
                                      **kwargs) -> pd.Series:
    """
    Compute relative strength vs Dollar Index.
    
    Positive = outperforming dollar
    Negative = underperforming dollar
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with relative strength (normalized)
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    # Get cumulative returns (strip timezone)
    fx_returns = _strip_timezone(calculate_returns(df, price_col='close', method='simple'))
    fx_cum = (1 + fx_returns).rolling(window=window, min_periods=1).apply(lambda x: x.prod() - 1)
    
    # Get DXY cumulative returns
    try:
        yahoo = _get_yahoo_client()
        store = _get_external_store()
        
        if yahoo is not None and store is not None:
            dxy_data = store.get_or_fetch_yahoo('DXY', yahoo)
            
            if dxy_data is not None and 'close' in dxy_data.columns:
                dxy_returns = dxy_data['close'].pct_change()
                dxy_aligned = _align_external_to_df(dxy_returns, df)
                dxy_cum = (1 + dxy_aligned).rolling(window=window, min_periods=1).apply(lambda x: x.prod() - 1)
                
                # Relative strength = FX return - DXY return
                rel_strength = fx_cum - dxy_cum
                rel_strength_normalized = normalize_series(rel_strength, method='zscore', window=window*2)
                
                return rel_strength_normalized.fillna(0.0).reindex(df.index)
    except Exception as e:
        error(f"Failed to compute relative strength: {e}")
    
    # Fallback: use momentum
    momentum = fx_cum
    return normalize_series(momentum, method='zscore', window=window*2).fillna(0.0).reindex(df.index)


def compute_cross_asset_momentum(df: pd.DataFrame,
                                  window: int = 20,
                                  assets: List[str] = None,
                                  **kwargs) -> pd.Series:
    """
    Compute cross-asset momentum signal.
    
    Aggregates momentum from multiple assets to gauge overall risk sentiment.
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        assets: List of assets to include (default: SPX, GOLD)
        **kwargs: Additional parameters
    
    Returns:
        Series with aggregated cross-asset momentum
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    if assets is None:
        assets = ['SPX', 'GOLD']
    
    momentums = []
    
    # Get tz-naive index for alignment
    df_index = _strip_timezone(df.index)
    
    try:
        yahoo = _get_yahoo_client()
        store = _get_external_store()
        
        if yahoo is not None and store is not None:
            for asset in assets:
                asset_data = store.get_or_fetch_yahoo(asset, yahoo)
                
                if asset_data is not None and 'close' in asset_data.columns:
                    asset_close = asset_data['close']
                    
                    # Compute momentum
                    asset_momentum = (asset_close / asset_close.shift(window) - 1)
                    asset_aligned = _align_external_to_df(asset_momentum, df)
                    
                    momentums.append(asset_aligned)
            
            if momentums:
                # Average momentum across assets
                combined = pd.concat(momentums, axis=1)
                avg_momentum = combined.mean(axis=1)
                
                # Normalize
                avg_normalized = normalize_series(avg_momentum, method='zscore', window=window*2)
                
                return avg_normalized.fillna(0.0).reindex(df.index)
    except Exception as e:
        error(f"Failed to compute cross-asset momentum: {e}")
    
    # Fallback: use FX momentum
    fx_momentum = (df['close'] / df['close'].shift(window) - 1)
    return normalize_series(fx_momentum, method='zscore', window=window*2).fillna(0.0)


def compute_risk_sentiment(df: pd.DataFrame,
                           window: int = 20,
                           **kwargs) -> pd.Series:
    """
    Compute risk sentiment indicator from cross-asset data.
    
    Combines SPX momentum and inverse VIX to measure risk appetite.
    Positive = risk-on environment
    Negative = risk-off environment
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 20)
        **kwargs: Additional parameters
    
    Returns:
        Series with risk sentiment score (normalized)
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    try:
        yahoo = _get_yahoo_client()
        store = _get_external_store()
        
        if yahoo is not None and store is not None:
            # Get SPX momentum
            spx_data = store.get_or_fetch_yahoo('SPX', yahoo)
            vix_data = store.get_or_fetch_yahoo('VIX', yahoo)
            
            components = []
            
            if spx_data is not None and 'close' in spx_data.columns:
                spx_momentum = (spx_data['close'] / spx_data['close'].shift(window) - 1)
                spx_aligned = _align_external_to_df(spx_momentum, df)
                spx_zscore = (spx_aligned - spx_aligned.rolling(window*2).mean()) / spx_aligned.rolling(window*2).std()
                components.append(spx_zscore)
            
            if vix_data is not None and 'close' in vix_data.columns:
                # Inverse VIX z-score (low VIX = risk on)
                vix_aligned = _align_external_to_df(vix_data['close'], df)
                vix_zscore = (vix_aligned - vix_aligned.rolling(window*2).mean()) / vix_aligned.rolling(window*2).std()
                components.append(-vix_zscore)  # Invert so positive = risk on
            
            if components:
                combined = pd.concat(components, axis=1)
                risk_sentiment = combined.mean(axis=1)
                return risk_sentiment.fillna(0.0).reindex(df.index)
    except Exception as e:
        error(f"Failed to compute risk sentiment: {e}")
    
    return pd.Series(0.0, index=df.index)


def compute_correlation_regime(df: pd.DataFrame,
                               window: int = 60,
                               **kwargs) -> pd.Series:
    """
    Detect correlation regime (high vs low correlation environment).
    
    High correlation = assets moving together (risk-off or risk-on)
    Low correlation = differentiated markets
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default: 60)
        **kwargs: Additional parameters
    
    Returns:
        Series with average absolute correlation (0-1 scale)
    """
    if 'close' not in df.columns:
        return pd.Series(0.0, index=df.index)
    
    # Compute correlations with multiple assets
    corr_dxy = compute_correlation_dxy(df, window=window)
    corr_spx = compute_correlation_spx(df, window=window)
    corr_gold = compute_correlation_gold(df, window=window)
    
    # Average absolute correlation
    correlations = pd.concat([
        corr_dxy.abs(),
        corr_spx.abs(),
        corr_gold.abs()
    ], axis=1)
    
    avg_abs_corr = correlations.mean(axis=1)
    
    return avg_abs_corr.fillna(0.0).reindex(df.index)