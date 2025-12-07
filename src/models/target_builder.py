"""
Target Variable Builder
Creates prediction targets (labels) for ML models.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Union
from enum import Enum
from src.utils.logger import info, success, warning


class TargetType(Enum):
    """Types of prediction targets."""
    BINARY = 'binary'           # Up (1) or Down (0)
    TERNARY = 'ternary'         # Up (2), Flat (1), Down (0)
    REGRESSION = 'regression'   # Actual return value
    DIRECTION = 'direction'     # +1, 0, -1


class TargetBuilder:
    """
    Creates target variables for ML models.
    
    Supports:
    - Binary classification (up/down)
    - Ternary classification (up/flat/down)
    - Regression (actual returns)
    - Configurable lookahead and thresholds
    """
    
    def __init__(self,
                 target_type: Union[TargetType, str] = TargetType.BINARY,
                 lookahead: int = 1,
                 threshold: float = 0.0,
                 use_log_returns: bool = False):
        """
        Initialize target builder.
        
        Args:
            target_type: Type of target ('binary', 'ternary', 'regression', 'direction')
            lookahead: Number of bars to look ahead (default: 1)
            threshold: Threshold for flat zone in ternary classification (default: 0.0)
                      For binary, this determines the neutral zone to skip
            use_log_returns: Use log returns instead of simple returns (default: False)
        """
        if isinstance(target_type, str):
            target_type = TargetType(target_type)
        
        self.target_type = target_type
        self.lookahead = lookahead
        self.threshold = threshold
        self.use_log_returns = use_log_returns
        
        # Statistics
        self.target_stats: dict = {}
    
    def _compute_returns(self, prices: pd.Series) -> pd.Series:
        """Compute forward returns."""
        if self.use_log_returns:
            future_price = prices.shift(-self.lookahead)
            returns = np.log(future_price / prices)
        else:
            returns = prices.pct_change(self.lookahead).shift(-self.lookahead)
        
        return returns
    
    def _build_binary_target(self, returns: pd.Series) -> pd.Series:
        """
        Build binary target (1 = up, 0 = down).
        
        If threshold > 0, samples within threshold are set to NaN (excluded).
        """
        if self.threshold > 0:
            # Exclude samples near zero
            target = pd.Series(np.nan, index=returns.index)
            target[returns > self.threshold] = 1
            target[returns < -self.threshold] = 0
        else:
            target = (returns > 0).astype(int)
        
        return target
    
    def _build_ternary_target(self, returns: pd.Series) -> pd.Series:
        """
        Build ternary target (2 = up, 1 = flat, 0 = down).
        """
        target = pd.Series(1, index=returns.index)  # Default: flat
        target[returns > self.threshold] = 2   # Up
        target[returns < -self.threshold] = 0  # Down
        
        return target
    
    def _build_direction_target(self, returns: pd.Series) -> pd.Series:
        """
        Build direction target (+1, 0, -1).
        """
        target = pd.Series(0, index=returns.index)
        target[returns > self.threshold] = 1
        target[returns < -self.threshold] = -1
        
        return target
    
    def build(self, 
              df: pd.DataFrame,
              price_col: str = 'close') -> pd.Series:
        """
        Build target variable from DataFrame.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for prices (default: 'close')
        
        Returns:
            Series with target values
        """
        if price_col not in df.columns:
            # Try to find close price
            if 'micro_price' in df.columns:
                price_col = 'micro_price'
                info(f"Using '{price_col}' as price column")
            else:
                raise ValueError(f"Price column '{price_col}' not found")
        
        prices = df[price_col]
        
        # Compute forward returns
        returns = self._compute_returns(prices)
        
        # Build target based on type
        if self.target_type == TargetType.BINARY:
            target = self._build_binary_target(returns)
        elif self.target_type == TargetType.TERNARY:
            target = self._build_ternary_target(returns)
        elif self.target_type == TargetType.DIRECTION:
            target = self._build_direction_target(returns)
        elif self.target_type == TargetType.REGRESSION:
            target = returns
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")
        
        # Compute statistics
        self._compute_stats(target, returns)
        
        target.name = f'target_{self.target_type.value}_h{self.lookahead}'
        
        return target
    
    def _compute_stats(self, target: pd.Series, returns: pd.Series):
        """Compute target statistics."""
        valid_target = target.dropna()
        
        self.target_stats = {
            'total_samples': len(target),
            'valid_samples': len(valid_target),
            'nan_samples': target.isna().sum(),
            'lookahead': self.lookahead,
            'threshold': self.threshold,
            'target_type': self.target_type.value
        }
        
        if self.target_type in [TargetType.BINARY, TargetType.TERNARY, TargetType.DIRECTION]:
            value_counts = valid_target.value_counts().to_dict()
            self.target_stats['class_distribution'] = value_counts
            
            # Class balance
            if len(value_counts) > 0:
                total = sum(value_counts.values())
                self.target_stats['class_balance'] = {
                    k: f"{v/total*100:.1f}%" for k, v in value_counts.items()
                }
        
        if self.target_type == TargetType.REGRESSION:
            self.target_stats['mean_return'] = float(returns.mean())
            self.target_stats['std_return'] = float(returns.std())
    
    def get_stats(self) -> dict:
        """Get target statistics."""
        return self.target_stats
    
    def print_stats(self):
        """Print target statistics."""
        print("\nTarget Statistics:")
        print("-" * 40)
        for k, v in self.target_stats.items():
            print(f"  {k}: {v}")


def build_target(df: pd.DataFrame,
                 target_type: str = 'binary',
                 lookahead: int = 1,
                 threshold: float = 0.0,
                 price_col: str = 'close') -> Tuple[pd.Series, dict]:
    """
    Convenience function to build target variable.
    
    Args:
        df: DataFrame with features and price
        target_type: 'binary', 'ternary', 'regression', or 'direction'
        lookahead: Bars to look ahead
        threshold: Threshold for classification
        price_col: Price column name
    
    Returns:
        Tuple of (target Series, statistics dict)
    """
    builder = TargetBuilder(
        target_type=target_type,
        lookahead=lookahead,
        threshold=threshold
    )
    
    target = builder.build(df, price_col)
    stats = builder.get_stats()
    
    return target, stats


def build_multiple_targets(df: pd.DataFrame,
                           lookaheads: list = [1, 4, 8, 24],
                           target_type: str = 'binary',
                           threshold: float = 0.0,
                           price_col: str = 'close') -> pd.DataFrame:
    """
    Build targets for multiple lookahead periods.
    
    Args:
        df: DataFrame with price data
        lookaheads: List of lookahead periods
        target_type: Target type
        threshold: Classification threshold
        price_col: Price column name
    
    Returns:
        DataFrame with all targets
    """
    targets = pd.DataFrame(index=df.index)
    
    for h in lookaheads:
        builder = TargetBuilder(
            target_type=target_type,
            lookahead=h,
            threshold=threshold
        )
        target = builder.build(df, price_col)
        targets[f'target_h{h}'] = target
    
    info(f"Built {len(lookaheads)} targets: {list(targets.columns)}")
    
    return targets


# Quick test
if __name__ == "__main__":
    print("Testing TargetBuilder...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='h')
    df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
        'feature1': np.random.randn(1000)
    }, index=dates)
    
    print(f"\nSample data: {df.shape}")
    
    # Test binary target
    print("\n--- Binary Target (h=1) ---")
    builder = TargetBuilder(target_type='binary', lookahead=1)
    y_binary = builder.build(df)
    builder.print_stats()
    
    # Test ternary target with threshold
    print("\n--- Ternary Target (h=4, threshold=0.001) ---")
    builder = TargetBuilder(target_type='ternary', lookahead=4, threshold=0.001)
    y_ternary = builder.build(df)
    builder.print_stats()
    
    # Test regression target
    print("\n--- Regression Target (h=24) ---")
    builder = TargetBuilder(target_type='regression', lookahead=24)
    y_reg = builder.build(df)
    builder.print_stats()
    
    # Test multiple targets
    print("\n--- Multiple Targets ---")
    targets = build_multiple_targets(df, lookaheads=[1, 4, 8, 24])
    print(f"Shape: {targets.shape}")
    print(targets.head())
    
    print("\n✓ All tests passed!")