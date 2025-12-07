"""
Preprocessing Pipeline
Handles feature scaling, missing values, and data preparation for ML models.
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from src.utils.logger import info, success, error, warning


class FeaturePreprocessor:
    """
    Preprocesses features for ML models.
    
    Handles:
    - Missing value imputation
    - Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
    - Saving/loading fitted scalers for inference
    """
    
    # Features that are naturally bounded [0, 100] or [-1, 1]
    BOUNDED_FEATURES = [
        'rsi', 'rsi_normalized',
        'stochastic_k', 'stochastic_d', 'stochastic_normalized',
        'mfi', 'mfi_normalized',
        'williams_r', 'williams_r_normalized',
        'cci_normalized',
        'bollinger_percent_b',
        'price_position',
        'body_ratio', 'wick_ratio',
        'london_session', 'new_york_session', 'asia_session', 'session_overlap'
    ]
    
    # Features that can have extreme outliers
    ROBUST_SCALE_FEATURES = [
        'skew', 'kurtosis',
        'obv', 'obv_normalized',
        'cci',
        'jump_component',
        'volatility_of_vol'
    ]
    
    def __init__(self, 
                 default_method: str = 'standard',
                 handle_inf: bool = True,
                 clip_outliers: bool = True,
                 outlier_std: float = 5.0):
        """
        Initialize preprocessor.
        
        Args:
            default_method: Default scaling method ('standard', 'minmax', 'robust')
            handle_inf: Replace inf values with NaN (default: True)
            clip_outliers: Clip extreme outliers before scaling (default: True)
            outlier_std: Number of std devs for outlier clipping (default: 5.0)
        """
        self.default_method = default_method
        self.handle_inf = handle_inf
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std
        
        # Fitted scalers (one per feature or group)
        self.scalers: Dict[str, object] = {}
        self.feature_stats: Dict[str, dict] = {}
        self.is_fitted = False
        
        # Track which features use which scaling method
        self.scaling_methods: Dict[str, str] = {}
    
    def _get_scaling_method(self, feature_name: str) -> str:
        """Determine scaling method for a feature."""
        if feature_name in self.BOUNDED_FEATURES:
            return 'minmax'
        elif feature_name in self.ROBUST_SCALE_FEATURES:
            return 'robust'
        else:
            return self.default_method
    
    def _create_scaler(self, method: str) -> object:
        """Create a scaler object based on method."""
        if method == 'standard':
            return StandardScaler()
        elif method == 'minmax':
            return MinMaxScaler(feature_range=(-1, 1))
        elif method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def _handle_missing_and_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values and infinities."""
        df = df.copy()
        
        # Replace inf with NaN
        if self.handle_inf:
            df = df.replace([np.inf, -np.inf], np.nan)
        
        # Count missing before
        missing_before = df.isna().sum().sum()
        
        # Forward fill, then backward fill, then fill with 0
        df = df.ffill().bfill().fillna(0)
        
        missing_after = df.isna().sum().sum()
        
        if missing_before > 0:
            info(f"Filled {missing_before} missing values")
        
        return df
    
    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clip extreme outliers based on standard deviations."""
        if not self.clip_outliers:
            return df
        
        df = df.copy()
        
        for col in df.columns:
            if col in self.BOUNDED_FEATURES:
                continue  # Don't clip bounded features
            
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:
                lower = mean - self.outlier_std * std
                upper = mean + self.outlier_std * std
                df[col] = df[col].clip(lower, upper)
        
        return df
    
    def fit(self, df: pd.DataFrame) -> 'FeaturePreprocessor':
        """
        Fit scalers on training data.
        
        Args:
            df: DataFrame with features (DatetimeIndex, feature columns)
        
        Returns:
            Self (fitted preprocessor)
        """
        info(f"Fitting preprocessor on {len(df)} samples, {len(df.columns)} features")
        
        # Handle missing/inf values
        df_clean = self._handle_missing_and_inf(df)
        
        # Clip outliers before fitting
        df_clean = self._clip_outliers(df_clean)
        
        # Fit a scaler for each feature
        for col in df_clean.columns:
            method = self._get_scaling_method(col)
            scaler = self._create_scaler(method)
            
            # Fit on column (reshape to 2D)
            values = df_clean[col].values.reshape(-1, 1)
            scaler.fit(values)
            
            # Store scaler and method
            self.scalers[col] = scaler
            self.scaling_methods[col] = method
            
            # Store feature statistics
            self.feature_stats[col] = {
                'mean': float(df_clean[col].mean()),
                'std': float(df_clean[col].std()),
                'min': float(df_clean[col].min()),
                'max': float(df_clean[col].max()),
                'null_count': int(df[col].isna().sum()),
                'scaling_method': method
            }
        
        self.is_fitted = True
        success(f"Fitted {len(self.scalers)} scalers")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scalers.
        
        Args:
            df: DataFrame with features
        
        Returns:
            Scaled DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        # Handle missing/inf values
        df_clean = self._handle_missing_and_inf(df)
        
        # Clip outliers
        df_clean = self._clip_outliers(df_clean)
        
        # Transform each column
        result = pd.DataFrame(index=df_clean.index)
        
        for col in df_clean.columns:
            if col in self.scalers:
                values = df_clean[col].values.reshape(-1, 1)
                scaled = self.scalers[col].transform(values)
                result[col] = scaled.flatten()
            else:
                warning(f"No scaler for '{col}', using raw values")
                result[col] = df_clean[col]
        
        return result
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform scaled features back to original scale.
        
        Args:
            df: Scaled DataFrame
        
        Returns:
            DataFrame in original scale
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        
        result = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            if col in self.scalers:
                values = df[col].values.reshape(-1, 1)
                original = self.scalers[col].inverse_transform(values)
                result[col] = original.flatten()
            else:
                result[col] = df[col]
        
        return result
    
    def save(self, path: str):
        """
        Save fitted preprocessor to disk.
        
        Args:
            path: Path to save (e.g., 'models/scalers/preprocessor.pkl')
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'scalers': self.scalers,
            'scaling_methods': self.scaling_methods,
            'feature_stats': self.feature_stats,
            'default_method': self.default_method,
            'handle_inf': self.handle_inf,
            'clip_outliers': self.clip_outliers,
            'outlier_std': self.outlier_std,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        success(f"Saved preprocessor to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeaturePreprocessor':
        """
        Load fitted preprocessor from disk.
        
        Args:
            path: Path to saved preprocessor
        
        Returns:
            Loaded FeaturePreprocessor
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            default_method=state['default_method'],
            handle_inf=state['handle_inf'],
            clip_outliers=state['clip_outliers'],
            outlier_std=state['outlier_std']
        )
        
        preprocessor.scalers = state['scalers']
        preprocessor.scaling_methods = state['scaling_methods']
        preprocessor.feature_stats = state['feature_stats']
        preprocessor.is_fitted = state['is_fitted']
        
        success(f"Loaded preprocessor from {path}")
        return preprocessor
    
    def get_feature_report(self) -> pd.DataFrame:
        """
        Get summary report of feature statistics.
        
        Returns:
            DataFrame with feature statistics
        """
        if not self.feature_stats:
            return pd.DataFrame()
        
        return pd.DataFrame(self.feature_stats).T


def preprocess_features(input_path: str,
                        output_path: Optional[str] = None,
                        scaler_path: Optional[str] = None,
                        method: str = 'standard') -> Tuple[pd.DataFrame, FeaturePreprocessor]:
    """
    Convenience function to preprocess a feature file.
    
    Args:
        input_path: Path to input parquet file
        output_path: Optional path to save scaled features
        scaler_path: Optional path to save/load scaler
        method: Scaling method ('standard', 'minmax', 'robust')
    
    Returns:
        Tuple of (scaled DataFrame, fitted preprocessor)
    """
    info(f"Loading features from {input_path}")
    df = pd.read_parquet(input_path)
    
    # Check if we should load existing scaler
    if scaler_path and os.path.exists(scaler_path):
        info(f"Loading existing scaler from {scaler_path}")
        preprocessor = FeaturePreprocessor.load(scaler_path)
        scaled_df = preprocessor.transform(df)
    else:
        preprocessor = FeaturePreprocessor(default_method=method)
        scaled_df = preprocessor.fit_transform(df)
        
        if scaler_path:
            preprocessor.save(scaler_path)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        scaled_df.to_parquet(output_path)
        success(f"Saved scaled features to {output_path}")
    
    return scaled_df, preprocessor


# Quick test
if __name__ == "__main__":
    # Test with sample data
    print("Testing FeaturePreprocessor...")
    
    # Load sample data
    df = pd.read_parquet('data/processed/EURUSD_1h_features.parquet')
    print(f"Loaded: {df.shape}")
    
    # Fit and transform
    preprocessor = FeaturePreprocessor()
    scaled = preprocessor.fit_transform(df)
    
    print(f"\nScaled shape: {scaled.shape}")
    print(f"Scaled mean (should be ~0): {scaled.mean().mean():.4f}")
    print(f"Scaled std (should be ~1): {scaled.std().mean():.4f}")
    
    # Save and reload
    preprocessor.save('models/scalers/test_preprocessor.pkl')
    loaded = FeaturePreprocessor.load('models/scalers/test_preprocessor.pkl')
    
    # Verify reload works
    scaled2 = loaded.transform(df)
    assert np.allclose(scaled.values, scaled2.values, equal_nan=True)
    print("\n✓ Save/load test passed!")
    
    # Print feature report
    report = preprocessor.get_feature_report()
    print(f"\nFeature Report (first 10):")
    print(report.head(10))