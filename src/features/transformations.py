"""
Feature Transformation Layer
Scaling and transformation functions for feature engineering
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, List
from src.utils.logger import info, warning


def zscore_transform(
    series: pd.Series,
    window: Optional[int] = None,
    fill_value: float = 0.0
) -> pd.Series:
    """
    Z-score normalization: (x - mean) / std
    
    Args:
        series: Input series
        window: Rolling window (None = global normalization)
        fill_value: Value to use for NaN (default: 0.0)
    
    Returns:
        Normalized series
    """
    if window is None:
        # Global normalization
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(0.0, index=series.index)
        return ((series - mean) / std).fillna(fill_value)
    else:
        # Rolling normalization
        mean = series.rolling(window=window, min_periods=1).mean()
        std = series.rolling(window=window, min_periods=1).std()
        normalized = (series - mean) / (std + 1e-10)
        return normalized.fillna(fill_value)


def percentile_transform(
    series: pd.Series,
    window: Optional[int] = None,
    fill_value: float = 0.5
) -> pd.Series:
    """
    Percentile rank transformation: maps values to [0, 1] based on percentile.
    
    Args:
        series: Input series
        window: Rolling window (None = global percentile)
        fill_value: Value to use for NaN (default: 0.5)
    
    Returns:
        Percentile-ranked series (0-1)
    """
    if window is None:
        # Global percentile
        return series.rank(pct=True).fillna(fill_value)
    else:
        # Rolling percentile
        def rolling_percentile(x):
            if len(x) < 2:
                return 0.5
            current = x.iloc[-1]
            historical = x.iloc[:-1]
            if len(historical) == 0:
                return 0.5
            return (historical <= current).sum() / len(historical)
        
        percentile = series.rolling(window=window, min_periods=2).apply(
            rolling_percentile, raw=False
        )
        return percentile.fillna(fill_value)


def log_transform(
    series: pd.Series,
    offset: float = 1e-10,
    fill_value: float = 0.0
) -> pd.Series:
    """
    Log transformation: log(x + offset)
    
    Args:
        series: Input series
        offset: Offset to add before log (default: 1e-10)
        fill_value: Value to use for NaN (default: 0.0)
    
    Returns:
        Log-transformed series
    """
    # Ensure positive values
    positive_series = series + offset
    positive_series = positive_series.clip(lower=offset)
    
    log_series = np.log(positive_series)
    return log_series.fillna(fill_value)


def minmax_transform(
    series: pd.Series,
    window: Optional[int] = None,
    feature_range: tuple = (0, 1),
    fill_value: float = 0.5
) -> pd.Series:
    """
    Min-max normalization: (x - min) / (max - min) * (max_range - min_range) + min_range
    
    Args:
        series: Input series
        window: Rolling window (None = global normalization)
        feature_range: Output range (default: (0, 1))
        fill_value: Value to use for NaN (default: 0.5)
    
    Returns:
        Min-max normalized series
    """
    min_range, max_range = feature_range
    
    if window is None:
        # Global normalization
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(fill_value, index=series.index)
        normalized = (series - min_val) / (max_val - min_val)
        scaled = normalized * (max_range - min_range) + min_range
        return scaled.fillna(fill_value)
    else:
        # Rolling normalization
        min_val = series.rolling(window=window, min_periods=1).min()
        max_val = series.rolling(window=window, min_periods=1).max()
        normalized = (series - min_val) / (max_val - min_val + 1e-10)
        scaled = normalized * (max_range - min_range) + min_range
        return scaled.fillna(fill_value)


def winsorize_transform(
    series: pd.Series,
    limits: tuple = (0.05, 0.05),
    window: Optional[int] = None,
    fill_value: float = 0.0
) -> pd.Series:
    """
    Winsorization: clip extreme values to percentiles.
    
    Args:
        series: Input series
        limits: Tuple of (lower_limit, upper_limit) percentiles (default: (0.05, 0.05))
        window: Rolling window (None = global winsorization)
        fill_value: Value to use for NaN (default: 0.0)
    
    Returns:
        Winsorized series
    """
    lower_limit, upper_limit = limits
    
    if window is None:
        # Global winsorization
        lower_bound = series.quantile(lower_limit)
        upper_bound = series.quantile(1 - upper_limit)
        winsorized = series.clip(lower=lower_bound, upper=upper_bound)
        return winsorized.fillna(fill_value)
    else:
        # Rolling winsorization
        def rolling_winsorize(x):
            if len(x) < 2:
                return x.iloc[-1] if len(x) > 0 else fill_value
            lower_bound = x.quantile(lower_limit)
            upper_bound = x.quantile(1 - upper_limit)
            value = x.iloc[-1]
            return np.clip(value, lower_bound, upper_bound)
        
        winsorized = series.rolling(window=window, min_periods=2).apply(
            rolling_winsorize, raw=False
        )
        return winsorized.fillna(fill_value)


def robust_transform(
    series: pd.Series,
    window: Optional[int] = None,
    fill_value: float = 0.0
) -> pd.Series:
    """
    Robust normalization using median and MAD (Median Absolute Deviation).
    
    Args:
        series: Input series
        window: Rolling window (None = global normalization)
        fill_value: Value to use for NaN (default: 0.0)
    
    Returns:
        Robust-normalized series
    """
    if window is None:
        # Global robust normalization
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            return pd.Series(0.0, index=series.index)
        # MAD to std conversion: 1.4826
        normalized = (series - median) / (mad * 1.4826)
        return normalized.fillna(fill_value)
    else:
        # Rolling robust normalization
        median = series.rolling(window=window, min_periods=1).median()
        mad = series.rolling(window=window, min_periods=1).apply(
            lambda x: (x - x.median()).abs().median()
        )
        normalized = (series - median) / (mad * 1.4826 + 1e-10)
        return normalized.fillna(fill_value)


class FeatureTransformer:
    """
    Feature Transformer - Applies transformations to feature DataFrames.
    """
    
    TRANSFORM_MAP = {
        'zscore': zscore_transform,
        'percentile': percentile_transform,
        'log': log_transform,
        'minmax': minmax_transform,
        'winsorize': winsorize_transform,
        'robust': robust_transform
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Feature Transformer.
        
        Args:
            config: Transformation configuration dictionary
        """
        self.config = config or {}
    
    def transform_feature(
        self,
        series: pd.Series,
        transform_type: str,
        **kwargs
    ) -> pd.Series:
        """
        Transform a single feature series.
        
        Args:
            series: Input series
            transform_type: Type of transformation
            **kwargs: Additional parameters for transformation
        
        Returns:
            Transformed series
        """
        if transform_type not in self.TRANSFORM_MAP:
            warning(f"Unknown transformation type: {transform_type}, returning original")
            return series
        
        transform_func = self.TRANSFORM_MAP[transform_type]
        return transform_func(series, **kwargs)
    
    def transform_dataframe(
        self,
        df: pd.DataFrame,
        transform_config: Optional[Dict[str, Union[str, Dict]]] = None
    ) -> pd.DataFrame:
        """
        Transform entire DataFrame based on configuration.
        
        Args:
            df: Input DataFrame
            transform_config: Dictionary mapping column names to transform config
                            e.g., {'feature1': 'zscore', 'feature2': {'type': 'minmax', 'window': 20}}
        
        Returns:
            Transformed DataFrame
        """
        if transform_config is None:
            transform_config = self.config.get('transformations', {})
        
        if not transform_config:
            info("No transformation config provided, returning original DataFrame")
            return df
        
        transformed_df = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            if col in transform_config:
                config = transform_config[col]
                
                # Handle string (simple transform type)
                if isinstance(config, str):
                    transform_type = config
                    kwargs = {}
                # Handle dict (transform type + params)
                elif isinstance(config, dict):
                    transform_type = config.get('type', 'zscore')
                    kwargs = {k: v for k, v in config.items() if k != 'type'}
                else:
                    warning(f"Invalid transform config for {col}, skipping")
                    transformed_df[col] = df[col]
                    continue
                
                # Apply transformation
                transformed_df[col] = self.transform_feature(
                    df[col], transform_type, **kwargs
                )
            else:
                # No transformation specified, keep original
                transformed_df[col] = df[col]
        
        info(f"Transformed {len(transform_config)} features")
        
        return transformed_df

