"""
Target Labelers
===============
Modules for creating ground truth labels for supervised learning.

Available labelers:
- RegimeLabeler: Creates RANGING/TREND_UP/TREND_DOWN labels
- VolatilityRegimeLabeler: Creates LOW_VOL/NORMAL_VOL/HIGH_VOL/CRISIS labels
- TransitionLabeler: Creates binary labels for regime transitions

Usage:
    from src.models.targets import RegimeLabeler, VolatilityRegimeLabeler
    
    # Regime labels
    labeler = RegimeLabeler(lookforward=12, threshold=0.002)
    labels = labeler.build(df)
    
    # Volatility regime labels
    vol_labeler = VolatilityRegimeLabeler(lookforward=12)
    vol_labels = vol_labeler.build(df)
"""

from .regime_labeler import (
    RegimeLabeler,
    RegimeLabelerConfig,
    RegimeLabel,
    VolatilityRegimeLabeler,
    TransitionLabeler,
    build_regime_labels,
    build_all_labels
)

__all__ = [
    # Classes
    'RegimeLabeler',
    'RegimeLabelerConfig',
    'RegimeLabel',
    'VolatilityRegimeLabeler',
    'TransitionLabeler',
    
    # Functions
    'build_regime_labels',
    'build_all_labels'
]