"""
Feature Selection Pipeline
Selects the most important features for ML models.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from src.utils.logger import info, success, error, warning


class FeatureSelector:
    """
    Selects best features using multiple methods:
    - Variance threshold (remove near-constant features)
    - Correlation filter (remove highly correlated features)
    - Importance ranking (Random Forest or Mutual Information)
    """
    
    def __init__(self,
                 variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.95,
                 n_features: Optional[int] = 30,
                 importance_method: str = 'random_forest'):
        """
        Initialize feature selector.
        
        Args:
            variance_threshold: Remove features with variance below this (default: 0.01)
            correlation_threshold: Remove features with correlation above this (default: 0.95)
            n_features: Number of top features to select (default: 30, None = keep all)
            importance_method: 'random_forest' or 'mutual_info' (default: 'random_forest')
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.n_features = n_features
        self.importance_method = importance_method
        
        # Fitted state
        self.selected_features: List[str] = []
        self.removed_low_variance: List[str] = []
        self.removed_high_correlation: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.is_fitted = False
    
    def _filter_low_variance(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with very low variance."""
        variances = df.var()
        low_var_features = variances[variances < self.variance_threshold].index.tolist()
        
        if low_var_features:
            info(f"Removing {len(low_var_features)} low-variance features")
            df = df.drop(columns=low_var_features)
        
        return df, low_var_features
    
    def _filter_high_correlation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features (keep first, remove second)."""
        corr_matrix = df.corr().abs()
        self.correlation_matrix = corr_matrix
        
        # Find pairs with high correlation
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to remove
        to_remove = set()
        for col in upper_tri.columns:
            high_corr = upper_tri[col][upper_tri[col] > self.correlation_threshold].index.tolist()
            to_remove.update(high_corr)
        
        removed = list(to_remove)
        
        if removed:
            info(f"Removing {len(removed)} highly-correlated features")
            df = df.drop(columns=removed)
        
        return df, removed
    
    def _compute_importance_rf(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute feature importance using Random Forest."""
        info("Computing feature importance with Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Handle any remaining NaN
        X_clean = X.fillna(0)
        
        rf.fit(X_clean, y)
        
        importance = dict(zip(X.columns, rf.feature_importances_))
        return importance
    
    def _compute_importance_mi(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute feature importance using Mutual Information."""
        info("Computing feature importance with Mutual Information...")
        
        # Handle any remaining NaN
        X_clean = X.fillna(0)
        
        mi_scores = mutual_info_classif(X_clean, y, random_state=42)
        
        importance = dict(zip(X.columns, mi_scores))
        return importance
    
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelector':
        """
        Fit feature selector.
        
        Args:
            df: DataFrame with features
            y: Optional target variable (needed for importance ranking)
        
        Returns:
            Self (fitted selector)
        """
        info(f"Fitting feature selector on {len(df.columns)} features")
        
        # Step 1: Remove low variance features
        df_filtered, self.removed_low_variance = self._filter_low_variance(df)
        
        # Step 2: Remove highly correlated features
        df_filtered, self.removed_high_correlation = self._filter_high_correlation(df_filtered)
        
        remaining_features = df_filtered.columns.tolist()
        info(f"After filtering: {len(remaining_features)} features remain")
        
        # Step 3: Rank by importance (if target provided and n_features set)
        if y is not None and self.n_features is not None and len(remaining_features) > self.n_features:
            
            # Compute importance
            if self.importance_method == 'random_forest':
                self.feature_importance = self._compute_importance_rf(df_filtered, y)
            else:
                self.feature_importance = self._compute_importance_mi(df_filtered, y)
            
            # Sort by importance and select top N
            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            self.selected_features = [f[0] for f in sorted_features[:self.n_features]]
            info(f"Selected top {len(self.selected_features)} features by importance")
        else:
            self.selected_features = remaining_features
            info(f"Keeping all {len(self.selected_features)} filtered features")
        
        self.is_fitted = True
        success(f"Feature selection complete: {len(self.selected_features)} features selected")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select features from DataFrame.
        
        Args:
            df: DataFrame with features
        
        Returns:
            DataFrame with only selected features
        """
        if not self.is_fitted:
            raise RuntimeError("Selector not fitted. Call fit() first.")
        
        # Check for missing features
        missing = set(self.selected_features) - set(df.columns)
        if missing:
            warning(f"Missing features in input: {missing}")
        
        # Select available features
        available = [f for f in self.selected_features if f in df.columns]
        return df[available]
    
    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, y)
        return self.transform(df)
    
    def save(self, path: str):
        """Save fitted selector to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted selector")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'variance_threshold': self.variance_threshold,
            'correlation_threshold': self.correlation_threshold,
            'n_features': self.n_features,
            'importance_method': self.importance_method,
            'selected_features': self.selected_features,
            'removed_low_variance': self.removed_low_variance,
            'removed_high_correlation': self.removed_high_correlation,
            'feature_importance': self.feature_importance,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        success(f"Saved feature selector to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeatureSelector':
        """Load fitted selector from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        selector = cls(
            variance_threshold=state['variance_threshold'],
            correlation_threshold=state['correlation_threshold'],
            n_features=state['n_features'],
            importance_method=state['importance_method']
        )
        
        selector.selected_features = state['selected_features']
        selector.removed_low_variance = state['removed_low_variance']
        selector.removed_high_correlation = state['removed_high_correlation']
        selector.feature_importance = state['feature_importance']
        selector.is_fitted = state['is_fitted']
        
        success(f"Loaded feature selector from {path}")
        return selector
    
    def get_importance_report(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        if not self.feature_importance:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {'feature': k, 'importance': v, 'selected': k in self.selected_features}
            for k, v in self.feature_importance.items()
        ])
        
        return df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    def get_selection_summary(self) -> dict:
        """Get summary of feature selection process."""
        return {
            'original_features': len(self.removed_low_variance) + 
                                len(self.removed_high_correlation) + 
                                len(self.selected_features),
            'removed_low_variance': len(self.removed_low_variance),
            'removed_high_correlation': len(self.removed_high_correlation),
            'selected_features': len(self.selected_features),
            'features': self.selected_features
        }


# Quick test
if __name__ == "__main__":
    print("Testing FeatureSelector...")
    
    # Load sample data
    df = pd.read_parquet('data/processed/EURUSD_1h_features.parquet')
    print(f"Loaded: {df.shape}")
    
    # Create dummy target (for testing)
    y = (df['momentum_factor'].shift(-1) > 0).astype(int)
    y = y.fillna(0)
    
    # Fit selector
    selector = FeatureSelector(n_features=30)
    selected_df = selector.fit_transform(df, y)
    
    print(f"\nSelected shape: {selected_df.shape}")
    print(f"\nSelection summary:")
    summary = selector.get_selection_summary()
    for k, v in summary.items():
        if k != 'features':
            print(f"  {k}: {v}")
    
    print(f"\nTop 10 features by importance:")
    importance_report = selector.get_importance_report()
    print(importance_report.head(10))
    
    # Save and reload
    selector.save('models/selectors/test_selector.pkl')
    loaded = FeatureSelector.load('models/selectors/test_selector.pkl')
    
    # Verify reload works
    selected2 = loaded.transform(df)
    assert list(selected_df.columns) == list(selected2.columns)
    print("\n✓ Save/load test passed!")