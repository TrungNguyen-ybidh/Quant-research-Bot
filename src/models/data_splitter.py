"""
Time-Series Data Splitter
Handles train/validation/test splits for time-series data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Generator
from dataclasses import dataclass
from src.utils.logger import info, success, warning


@dataclass
class DataSplit:
    """Container for train/val/test data."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    
    # Metadata
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    
    def summary(self) -> dict:
        """Get split summary."""
        return {
            'train': {
                'samples': len(self.X_train),
                'start': str(self.train_start),
                'end': str(self.train_end)
            },
            'val': {
                'samples': len(self.X_val),
                'start': str(self.val_start),
                'end': str(self.val_end)
            },
            'test': {
                'samples': len(self.X_test),
                'start': str(self.test_start),
                'end': str(self.test_end)
            }
        }


class TimeSeriesSplitter:
    """
    Splits time-series data preserving temporal order.
    
    Key principle: Never use future data to predict the past.
    """
    
    def __init__(self,
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 gap: int = 0):
        """
        Initialize splitter.
        
        Args:
            train_ratio: Fraction for training (default: 0.70)
            val_ratio: Fraction for validation (default: 0.15)
            test_ratio: Fraction for testing (default: 0.15)
            gap: Number of samples to skip between splits (prevent leakage)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
            "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.gap = gap
    
    def split(self, 
              X: pd.DataFrame, 
              y: pd.Series) -> DataSplit:
        """
        Split data into train/val/test sets.
        
        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Target Series with same index
        
        Returns:
            DataSplit object with all splits
        """
        n = len(X)
        
        # Calculate split indices
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        # Apply gap
        train_idx = slice(0, train_end)
        val_idx = slice(train_end + self.gap, val_end)
        test_idx = slice(val_end + self.gap, n)
        
        # Split data
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return DataSplit(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            train_start=X_train.index[0],
            train_end=X_train.index[-1],
            val_start=X_val.index[0],
            val_end=X_val.index[-1],
            test_start=X_test.index[0],
            test_end=X_test.index[-1]
        )
    
    def walk_forward_split(self,
                           X: pd.DataFrame,
                           y: pd.Series,
                           n_splits: int = 5,
                           train_size: Optional[int] = None,
                           test_size: Optional[int] = None) -> Generator[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], None, None]:
        """
        Generate walk-forward validation splits.
        
        Walk-forward: Train on past, test on future, slide forward.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of train/test splits (default: 5)
            train_size: Fixed training size (None = expanding window)
            test_size: Test size per split (None = auto-calculate)
        
        Yields:
            Tuple of (X_train, y_train, X_test, y_test) for each split
        """
        n = len(X)
        
        if test_size is None:
            test_size = n // (n_splits + 1)
        
        info(f"Walk-forward: {n_splits} splits, test_size={test_size}")
        
        for i in range(n_splits):
            # Test window
            test_end = n - i * test_size
            test_start = test_end - test_size
            
            # Train window
            if train_size is not None:
                train_start = max(0, test_start - self.gap - train_size)
            else:
                train_start = 0
            train_end = test_start - self.gap
            
            if train_end <= train_start or test_start >= test_end:
                warning(f"Invalid split {i}, skipping")
                continue
            
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            yield X_train, y_train, X_test, y_test
    
    def purged_kfold_split(self,
                           X: pd.DataFrame,
                           y: pd.Series,
                           n_splits: int = 5,
                           purge_gap: int = 10) -> Generator[Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], None, None]:
        """
        Generate purged K-Fold splits (no data leakage).
        
        Each fold uses future data as test, with a purge gap to prevent leakage.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of folds (default: 5)
            purge_gap: Samples to skip around test set (default: 10)
        
        Yields:
            Tuple of (X_train, y_train, X_test, y_test) for each fold
        """
        n = len(X)
        fold_size = n // n_splits
        
        info(f"Purged K-Fold: {n_splits} folds, fold_size={fold_size}, purge_gap={purge_gap}")
        
        for i in range(n_splits):
            # Test indices
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n)
            
            # Train indices (everything except test + purge gap)
            train_mask = np.ones(n, dtype=bool)
            
            # Exclude test region + purge gap
            exclude_start = max(0, test_start - purge_gap)
            exclude_end = min(n, test_end + purge_gap)
            train_mask[exclude_start:exclude_end] = False
            
            X_train = X.iloc[train_mask]
            y_train = y.iloc[train_mask]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            yield X_train, y_train, X_test, y_test


def create_train_val_test_split(X: pd.DataFrame,
                                 y: pd.Series,
                                 train_ratio: float = 0.70,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15) -> DataSplit:
    """
    Convenience function to create train/val/test split.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        train_ratio: Training fraction
        val_ratio: Validation fraction
        test_ratio: Test fraction
    
    Returns:
        DataSplit object
    """
    splitter = TimeSeriesSplitter(train_ratio, val_ratio, test_ratio)
    return splitter.split(X, y)


# Quick test
if __name__ == "__main__":
    print("Testing TimeSeriesSplitter...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='h')
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000)
    }, index=dates)
    y = pd.Series(np.random.randint(0, 2, 1000), index=dates)
    
    # Test basic split
    splitter = TimeSeriesSplitter()
    split = splitter.split(X, y)
    
    print("\nBasic Split:")
    for name, data in split.summary().items():
        print(f"  {name}: {data}")
    
    # Verify no overlap
    assert split.train_end < split.val_start, "Train/Val overlap!"
    assert split.val_end < split.test_start, "Val/Test overlap!"
    print("\n✓ No temporal overlap!")
    
    # Test walk-forward
    print("\nWalk-Forward Splits:")
    for i, (X_tr, y_tr, X_te, y_te) in enumerate(splitter.walk_forward_split(X, y, n_splits=3)):
        print(f"  Split {i}: Train={len(X_tr)} ({X_tr.index[0].date()} to {X_tr.index[-1].date()}), "
              f"Test={len(X_te)} ({X_te.index[0].date()} to {X_te.index[-1].date()})")
    
    print("\n✓ All tests passed!")