"""
State Models Module
===================
Models for detecting market states and regimes.

Available models:
- RegimeClassifier: Detects RANGING/TREND_UP/TREND_DOWN states
- VolatilityRegimeClassifier: Detects LOW/NORMAL/HIGH/CRISIS volatility (coming soon)
- RegimeTransitionDetector: Predicts regime transitions (coming soon)

Example:
    >>> from src.models.state import RegimeClassifier
    >>> 
    >>> classifier = RegimeClassifier(
    ...     symbol="EURUSD",
    ...     timeframe="1h",
    ...     config=config
    ... )
    >>> 
    >>> # Full pipeline
    >>> classifier.prepare_data()
    >>> classifier.train()
    >>> metrics = classifier.evaluate()
    >>> classifier.save()
    >>> 
    >>> # Inference
    >>> regime, confidence = classifier.predict(features)
"""

from .regime_classifier import RegimeClassifier

__all__ = [
    'RegimeClassifier',
]