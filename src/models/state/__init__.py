"""
State Detection Models
======================
Models for detecting and analyzing market state conditions.

This is a RESEARCH TOOL for understanding market behavior,
NOT a trading signal generator.

Available Classes:
    - RegimeClassifier: Detects RANGING/TREND_UP/TREND_DOWN
    - VolatilityRegimeClassifier: Detects LOW/NORMAL/HIGH/CRISIS
    - TransitionDetector: Predicts upcoming regime changes
    - MarketStateAnalyzer: Combines all models for comprehensive analysis
    - MultiPairAnalyzer: Analyzes multiple pairs simultaneously
    - MarketState: Dataclass holding market state information

Usage:
    from src.models.state import MarketStateAnalyzer
    
    analyzer = MarketStateAnalyzer(symbol="EURUSD", timeframe="1h")
    analyzer.load_models()
    state = analyzer.get_market_state(features_df)
    print(state.summary())
"""

from .regime_classifier import RegimeClassifier
from .volatility_regime import VolatilityRegimeClassifier
from .transition_detector import TransitionDetector
from .state_aggregator import (
    MarketStateAnalyzer,
    MultiPairAnalyzer,
    MarketState,
    # Backward compatibility aliases
    StateAggregator,
    MultiPairAggregator
)

__all__ = [
    # Core classifiers
    'RegimeClassifier',
    'VolatilityRegimeClassifier',
    'TransitionDetector',
    
    # Market state analysis (research tool)
    'MarketStateAnalyzer',
    'MultiPairAnalyzer',
    'MarketState',
    
    # Backward compatibility aliases
    'StateAggregator',
    'MultiPairAggregator',
]