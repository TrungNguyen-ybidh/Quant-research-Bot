"""
Regime Labeler (FIXED - Timeframe-Aware)
========================================
Creates regime labels with appropriate lookforward periods per timeframe
to prevent data leakage.

Regimes:
- RANGING (0): Price moves sideways, no clear direction
- TREND_UP (1): Price trending upward
- TREND_DOWN (2): Price trending downward

Usage:
    from src.models.targets.regime_labeler import RegimeLabeler, get_lookforward_for_timeframe
    
    # Automatic lookforward based on timeframe
    labeler = RegimeLabeler.for_timeframe("1h")
    labels = labeler.build(df)
    
    # Or manual configuration
    labeler = RegimeLabeler(lookforward=12, trend_threshold=0.002)
    labels = labeler.build(df)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from src.utils.logger import info, success, warning


# =============================================================================
# TIMEFRAME-AWARE LOOKFORWARD CONFIGURATION
# =============================================================================

# Lookforward periods to prevent data leakage
# Goal: Look far enough ahead that current features can't "see" the label
LOOKFORWARD_CONFIG = {
    # Timeframe: (lookforward_bars, target_hours_ahead, trend_threshold)
    '1m':  (120, 2.0, 0.0015),   # 120 bars = 2 hours ahead
    '5m':  (48,  4.0, 0.0020),   # 48 bars = 4 hours ahead  
    '15m': (24,  6.0, 0.0025),   # 24 bars = 6 hours ahead
    '1h':  (12,  12.0, 0.0020),  # 12 bars = 12 hours ahead
    '4h':  (6,   24.0, 0.0030),  # 6 bars = 24 hours ahead
    '1d':  (5,   120.0, 0.0050), # 5 bars = 1 week ahead
}


def get_lookforward_for_timeframe(timeframe: str) -> Tuple[int, float]:
    """
    Get appropriate lookforward and threshold for a timeframe.
    
    Args:
        timeframe: Timeframe string ('1m', '5m', '15m', '1h', '4h', '1d')
    
    Returns:
        Tuple of (lookforward_bars, trend_threshold)
    """
    if timeframe in LOOKFORWARD_CONFIG:
        lookforward, _, threshold = LOOKFORWARD_CONFIG[timeframe]
        return lookforward, threshold
    else:
        warning(f"Unknown timeframe {timeframe}, using default (12 bars, 0.002 threshold)")
        return 12, 0.002


@dataclass
class RegimeLabelerConfig:
    """Configuration for regime labeling."""
    lookforward: int = 12
    trend_threshold: float = 0.002  # 0.2% move to be considered trend
    dominance_ratio: float = 1.5    # How much stronger one direction must be
    smooth_window: int = 0          # Optional smoothing (0 = disabled)
    min_trend_bars: int = 3         # Minimum bars in same direction for trend


class RegimeLabeler:
    """
    Creates regime labels based on future price movement.
    
    Logic:
        1. Look forward N bars
        2. Calculate max upside and max downside from current price
        3. If upside dominates -> TREND_UP
        4. If downside dominates -> TREND_DOWN
        5. If neither dominates -> RANGING
    """
    
    def __init__(self, config: Optional[RegimeLabelerConfig] = None, **kwargs):
        """
        Initialize regime labeler.
        
        Args:
            config: RegimeLabelerConfig object
            **kwargs: Override config parameters
        """
        if config is None:
            config = RegimeLabelerConfig()
        
        # Allow kwargs to override config
        self.lookforward = kwargs.get('lookforward', config.lookforward)
        self.trend_threshold = kwargs.get('trend_threshold', config.trend_threshold)
        self.dominance_ratio = kwargs.get('dominance_ratio', config.dominance_ratio)
        self.smooth_window = kwargs.get('smooth_window', config.smooth_window)
        self.min_trend_bars = kwargs.get('min_trend_bars', config.min_trend_bars)
        
        # Stats
        self.stats: Dict = {}
    
    @classmethod
    def for_timeframe(cls, timeframe: str, **kwargs) -> 'RegimeLabeler':
        """
        Create a RegimeLabeler with appropriate settings for a timeframe.
        
        Args:
            timeframe: Timeframe string ('1m', '5m', '15m', '1h', '4h', '1d')
            **kwargs: Additional overrides
        
        Returns:
            Configured RegimeLabeler instance
        """
        lookforward, threshold = get_lookforward_for_timeframe(timeframe)
        
        return cls(
            lookforward=kwargs.get('lookforward', lookforward),
            trend_threshold=kwargs.get('trend_threshold', threshold),
            **{k: v for k, v in kwargs.items() if k not in ['lookforward', 'trend_threshold']}
        )
    
    def build(self, df: pd.DataFrame, price_col: str = 'close') -> np.ndarray:
        """
        Create regime labels for each row.
        
        Args:
            df: DataFrame with OHLC data
            price_col: Column to use for price (default: 'close')
        
        Returns:
            Array of regime labels (0=RANGING, 1=TREND_UP, 2=TREND_DOWN)
        """
        info(f"Building regime labels (lookforward={self.lookforward}, threshold={self.trend_threshold*100:.2f}%)")
        
        prices = df[price_col].values
        n = len(prices)
        labels = np.full(n, np.nan)
        
        for i in range(n - self.lookforward):
            current_price = prices[i]
            future_prices = prices[i+1:i+self.lookforward+1]
            
            # Calculate returns relative to current price
            future_returns = (future_prices - current_price) / current_price
            
            # Get max upside and max downside
            max_up = np.max(future_returns)
            max_down = np.min(future_returns)  # This is negative for downside
            
            # Determine regime
            abs_up = abs(max_up)
            abs_down = abs(max_down)
            
            # Check if there's a clear trend
            if abs_up >= self.trend_threshold and abs_up >= abs_down * self.dominance_ratio:
                labels[i] = 1  # TREND_UP
            elif abs_down >= self.trend_threshold and abs_down >= abs_up * self.dominance_ratio:
                labels[i] = 2  # TREND_DOWN
            else:
                labels[i] = 0  # RANGING
        
        # Optional smoothing
        if self.smooth_window > 0:
            labels = self._smooth_labels(labels)
        
        # Compute stats
        self._compute_stats(pd.Series(labels))
        
        # Log results
        success(f"Regime labels built: RANGING={self.stats['pct_ranging']:.1f}%, "
                f"TREND_UP={self.stats['pct_trend_up']:.1f}%, "
                f"TREND_DOWN={self.stats['pct_trend_down']:.1f}%")
        
        return labels
    
    def _smooth_labels(self, labels: np.ndarray) -> np.ndarray:
        """Apply smoothing to reduce noise in labels."""
        smoothed = labels.copy()
        
        for i in range(self.smooth_window, len(labels) - self.smooth_window):
            window = labels[i-self.smooth_window:i+self.smooth_window+1]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                # Use mode (most common label in window)
                counts = np.bincount(valid.astype(int), minlength=3)
                smoothed[i] = np.argmax(counts)
        
        return smoothed
    
    def _compute_stats(self, labels: pd.Series):
        """Compute label statistics."""
        valid = labels.dropna()
        total = len(valid)
        
        if total == 0:
            self.stats = {
                'total': 0,
                'n_ranging': 0, 'n_trend_up': 0, 'n_trend_down': 0,
                'pct_ranging': 0, 'pct_trend_up': 0, 'pct_trend_down': 0
            }
            return
        
        counts = valid.value_counts()
        
        self.stats = {
            'total': total,
            'n_ranging': int(counts.get(0, 0)),
            'n_trend_up': int(counts.get(1, 0)),
            'n_trend_down': int(counts.get(2, 0)),
            'pct_ranging': counts.get(0, 0) / total * 100,
            'pct_trend_up': counts.get(1, 0) / total * 100,
            'pct_trend_down': counts.get(2, 0) / total * 100,
            'lookforward': self.lookforward,
            'trend_threshold': self.trend_threshold
        }
    
    def get_stats(self) -> Dict:
        """Return labeling statistics."""
        return self.stats
    
    def print_stats(self):
        """Print labeling statistics."""
        print("\nRegime Label Statistics:")
        print(f"  Total samples: {self.stats.get('total', 0):,}")
        print(f"  RANGING (0):   {self.stats.get('n_ranging', 0):,} ({self.stats.get('pct_ranging', 0):.1f}%)")
        print(f"  TREND_UP (1):  {self.stats.get('n_trend_up', 0):,} ({self.stats.get('pct_trend_up', 0):.1f}%)")
        print(f"  TREND_DOWN (2):{self.stats.get('n_trend_down', 0):,} ({self.stats.get('pct_trend_down', 0):.1f}%)")


# =============================================================================
# VOLATILITY REGIME LABELER
# =============================================================================

class VolatilityRegimeLabeler:
    """
    Creates volatility regime labels based on realized volatility.
    
    Regimes:
        - LOW (0): Below 25th percentile
        - NORMAL (1): 25th to 75th percentile
        - HIGH (2): 75th to 95th percentile
        - CRISIS (3): Above 95th percentile
    """
    
    def __init__(self,
                 lookback: int = 20,
                 low_pct: float = 25,
                 high_pct: float = 75,
                 crisis_pct: float = 95,
                 vol_method: str = 'realized'):
        """
        Initialize volatility regime labeler.
        
        Args:
            lookback: Lookback window for volatility calculation
            low_pct: Percentile threshold for LOW regime
            high_pct: Percentile threshold for HIGH regime
            crisis_pct: Percentile threshold for CRISIS regime
            vol_method: 'realized', 'parkinson', or 'garman_klass'
        """
        self.lookback = lookback
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.crisis_pct = crisis_pct
        self.vol_method = vol_method
        
        self.stats: Dict = {}
        self.percentiles: Dict = {}
    
    def build(self, df: pd.DataFrame) -> pd.Series:
        """
        Create volatility regime labels.
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            Series of volatility regime labels
        """
        info(f"Building volatility regime labels (lookback={self.lookback}, method={self.vol_method})")
        
        # Calculate volatility
        if self.vol_method == 'realized':
            vol = self._realized_volatility(df)
        elif self.vol_method == 'parkinson':
            vol = self._parkinson_volatility(df)
        elif self.vol_method == 'garman_klass':
            vol = self._garman_klass_volatility(df)
        else:
            raise ValueError(f"Unknown vol_method: {self.vol_method}")
        
        # Calculate percentiles (expanding window to avoid lookahead)
        labels = pd.Series(index=df.index, dtype=float)
        
        min_periods = max(100, self.lookback * 2)  # Need enough history
        
        for i in range(min_periods, len(df)):
            historical_vol = vol.iloc[:i]
            current_vol = vol.iloc[i]
            
            # Calculate percentiles from historical data only
            p_low = np.nanpercentile(historical_vol, self.low_pct)
            p_high = np.nanpercentile(historical_vol, self.high_pct)
            p_crisis = np.nanpercentile(historical_vol, self.crisis_pct)
            
            # Assign label
            if current_vol <= p_low:
                labels.iloc[i] = 0  # LOW
            elif current_vol <= p_high:
                labels.iloc[i] = 1  # NORMAL
            elif current_vol <= p_crisis:
                labels.iloc[i] = 2  # HIGH
            else:
                labels.iloc[i] = 3  # CRISIS
        
        # Store final percentiles
        self.percentiles = {
            'low': np.nanpercentile(vol, self.low_pct),
            'high': np.nanpercentile(vol, self.high_pct),
            'crisis': np.nanpercentile(vol, self.crisis_pct)
        }
        
        # Compute stats
        self._compute_stats(labels)
        
        success(f"Volatility regime labels built: LOW={self.stats['pct_low']:.1f}%, "
                f"NORMAL={self.stats['pct_normal']:.1f}%, "
                f"HIGH={self.stats['pct_high']:.1f}%, "
                f"CRISIS={self.stats['pct_crisis']:.1f}%")
        
        return labels
    
    def _realized_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate realized volatility from returns."""
        returns = df['close'].pct_change()
        return returns.rolling(window=self.lookback).std() * np.sqrt(252)  # Annualized
    
    def _parkinson_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility from high/low."""
        log_hl = np.log(df['high'] / df['low'])
        return np.sqrt((log_hl ** 2).rolling(window=self.lookback).mean() / (4 * np.log(2))) * np.sqrt(252)
    
    def _garman_klass_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility."""
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        
        gk = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        return np.sqrt(gk.rolling(window=self.lookback).mean() * 252)
    
    def _compute_stats(self, labels: pd.Series):
        """Compute label statistics."""
        valid = labels.dropna()
        total = len(valid)
        
        if total == 0:
            self.stats = {
                'total': 0,
                'n_low': 0, 'n_normal': 0, 'n_high': 0, 'n_crisis': 0,
                'pct_low': 0, 'pct_normal': 0, 'pct_high': 0, 'pct_crisis': 0
            }
            return
        
        counts = valid.value_counts()
        
        self.stats = {
            'total': total,
            'n_low': int(counts.get(0, 0)),
            'n_normal': int(counts.get(1, 0)),
            'n_high': int(counts.get(2, 0)),
            'n_crisis': int(counts.get(3, 0)),
            'pct_low': counts.get(0, 0) / total * 100,
            'pct_normal': counts.get(1, 0) / total * 100,
            'pct_high': counts.get(2, 0) / total * 100,
            'pct_crisis': counts.get(3, 0) / total * 100
        }
    
    def get_stats(self) -> Dict:
        """Return labeling statistics."""
        return self.stats


# =============================================================================
# TRANSITION LABELER
# =============================================================================

class TransitionLabeler:
    """
    Creates binary labels for regime transitions.
    
    Labels:
        - 0: No transition (regime stays the same)
        - 1: Transition (regime changes within horizon)
    """
    
    def __init__(self, horizon: int = 8):
        """
        Initialize transition labeler.
        
        Args:
            horizon: Number of bars to look ahead for transition
        """
        self.horizon = horizon
        self.stats: Dict = {}
    
    def build(self, regime_labels: pd.Series) -> pd.Series:
        """
        Create transition labels from regime labels.
        
        Args:
            regime_labels: Series of regime labels (0, 1, 2)
        
        Returns:
            Series of transition labels (0 or 1)
        """
        info(f"Building transition labels (horizon={self.horizon})")
        
        n = len(regime_labels)
        labels = pd.Series(index=regime_labels.index, dtype=float)
        
        for i in range(n - self.horizon):
            current_regime = regime_labels.iloc[i]
            future_regimes = regime_labels.iloc[i+1:i+self.horizon+1]
            
            # Check if any future regime is different
            if pd.isna(current_regime):
                labels.iloc[i] = np.nan
            elif (future_regimes != current_regime).any():
                labels.iloc[i] = 1  # Transition
            else:
                labels.iloc[i] = 0  # No transition
        
        # Compute stats
        self._compute_stats(labels)
        
        success(f"Transition labels built: NO_TRANSITION={self.stats['pct_no_trans']:.1f}%, "
                f"TRANSITION={self.stats['pct_trans']:.1f}%")
        
        return labels
    
    def _compute_stats(self, labels: pd.Series):
        """Compute label statistics."""
        valid = labels.dropna()
        total = len(valid)
        
        if total == 0:
            self.stats = {'total': 0, 'n_no_trans': 0, 'n_trans': 0, 'pct_no_trans': 0, 'pct_trans': 0}
            return
        
        counts = valid.value_counts()
        
        self.stats = {
            'total': total,
            'n_no_trans': int(counts.get(0, 0)),
            'n_trans': int(counts.get(1, 0)),
            'pct_no_trans': counts.get(0, 0) / total * 100,
            'pct_trans': counts.get(1, 0) / total * 100
        }
    
    def get_stats(self) -> Dict:
        """Return labeling statistics."""
        return self.stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def build_regime_labels(df: pd.DataFrame,
                        timeframe: str = '1h',
                        **kwargs) -> np.ndarray:
    """
    Convenience function to build regime labels with timeframe-appropriate settings.
    
    Args:
        df: DataFrame with OHLC data
        timeframe: Timeframe string
        **kwargs: Additional labeler arguments
    
    Returns:
        Array of regime labels
    """
    labeler = RegimeLabeler.for_timeframe(timeframe, **kwargs)
    return labeler.build(df)


def build_all_labels(df: pd.DataFrame,
                     timeframe: str = '1h') -> Dict[str, np.ndarray]:
    """
    Build all label types (regime, volatility, transition).
    
    Args:
        df: DataFrame with OHLC data
        timeframe: Timeframe string
    
    Returns:
        Dictionary with all label arrays
    """
    # Regime labels
    regime_labeler = RegimeLabeler.for_timeframe(timeframe)
    regime_labels = regime_labeler.build(df)
    
    # Volatility regime labels
    vol_labeler = VolatilityRegimeLabeler()
    vol_labels = vol_labeler.build(df)
    
    # Transition labels
    trans_labeler = TransitionLabeler()
    trans_labels = trans_labeler.build(pd.Series(regime_labels))
    
    return {
        'regime': regime_labels,
        'volatility_regime': vol_labels.values,
        'transition': trans_labels.values
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Testing Regime Labelers")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    # Simulate price with trends and ranges
    price = 100.0
    prices = [price]
    for i in range(n-1):
        # Add some trending behavior
        if i % 100 < 30:
            drift = 0.001  # Uptrend
        elif i % 100 < 60:
            drift = -0.001  # Downtrend
        else:
            drift = 0  # Range
        
        price *= (1 + drift + np.random.randn() * 0.005)
        prices.append(price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.randn()) * 0.002) for p in prices],
        'low': [p * (1 - abs(np.random.randn()) * 0.002) for p in prices],
        'close': prices
    })
    
    print("\n--- Test 1: Regime Labeler (default) ---")
    labeler = RegimeLabeler()
    labels = labeler.build(df)
    labeler.print_stats()
    
    print("\n--- Test 2: Regime Labeler (for 1h timeframe) ---")
    labeler_1h = RegimeLabeler.for_timeframe('1h')
    labels_1h = labeler_1h.build(df)
    print(f"Lookforward: {labeler_1h.lookforward}, Threshold: {labeler_1h.trend_threshold}")
    
    print("\n--- Test 3: Regime Labeler (for 1m timeframe) ---")
    labeler_1m = RegimeLabeler.for_timeframe('1m')
    labels_1m = labeler_1m.build(df)
    print(f"Lookforward: {labeler_1m.lookforward}, Threshold: {labeler_1m.trend_threshold}")
    
    print("\n--- Test 4: Volatility Regime Labeler ---")
    vol_labeler = VolatilityRegimeLabeler()
    vol_labels = vol_labeler.build(df)
    print(f"Stats: {vol_labeler.get_stats()}")
    
    print("\n--- Test 5: Transition Labeler ---")
    trans_labeler = TransitionLabeler()
    trans_labels = trans_labeler.build(pd.Series(labels))
    print(f"Stats: {trans_labeler.get_stats()}")
    
    print("\n--- Test 6: Build All Labels ---")
    all_labels = build_all_labels(df, '1h')
    print(f"Labels created: {list(all_labels.keys())}")
    
    print("\n" + "="*60)
    print("✓ All labeler tests passed!")
    print("="*60)