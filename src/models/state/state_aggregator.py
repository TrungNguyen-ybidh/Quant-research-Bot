"""
Market State Analyzer
=====================
Combines regime, volatility, and transition models to provide
a comprehensive description of current market conditions.

This is a RESEARCH TOOL for understanding market state,
NOT a trading signal generator.

Usage:
    from src.models.state.state_aggregator import MarketStateAnalyzer
    
    analyzer = MarketStateAnalyzer(symbol="EURUSD", timeframe="1h")
    analyzer.load_models()
    
    state = analyzer.get_market_state(features_df)
    print(state.summary())
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.logger import info, success, warning, error


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarketState:
    """
    Comprehensive market state description.
    Focuses on WHAT the market is doing, not WHAT TO DO.
    
    Note: Fields without defaults must come before fields with defaults.
    """
    # Required fields (no defaults) - must come first
    symbol: str
    timeframe: str
    timestamp: datetime
    regime: str  # RANGING, TREND_UP, TREND_DOWN
    regime_confidence: float
    volatility_regime: str  # LOW, NORMAL, HIGH, CRISIS
    volatility_confidence: float
    is_transitioning: bool
    transition_probability: float
    
    # Optional fields (with defaults) - must come after required fields
    regime_probabilities: Dict[str, float] = field(default_factory=dict)
    volatility_probabilities: Dict[str, float] = field(default_factory=dict)
    models_available: Dict[str, bool] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate human-readable summary of market state."""
        lines = [
            f"\n{'='*60}",
            f"MARKET STATE: {self.symbol} {self.timeframe}",
            f"{'='*60}",
            f"Timestamp: {self.timestamp}",
            f"",
            f"REGIME ANALYSIS:",
            f"  Current Regime: {self.regime}",
            f"  Confidence: {self.regime_confidence:.1%}",
        ]
        
        if self.regime_probabilities:
            lines.append(f"  Probabilities:")
            for regime, prob in sorted(self.regime_probabilities.items()):
                lines.append(f"    {regime}: {prob:.1%}")
        
        lines.extend([
            f"",
            f"VOLATILITY ANALYSIS:",
            f"  Current State: {self.volatility_regime}",
            f"  Confidence: {self.volatility_confidence:.1%}",
        ])
        
        if self.volatility_probabilities:
            lines.append(f"  Probabilities:")
            for vol, prob in sorted(self.volatility_probabilities.items()):
                lines.append(f"    {vol}: {prob:.1%}")
        
        lines.extend([
            f"",
            f"TRANSITION ANALYSIS:",
            f"  Regime Change Expected: {'Yes' if self.is_transitioning else 'No'}",
            f"  Transition Probability: {self.transition_probability:.1%}",
            f"",
            f"MODELS LOADED:",
        ])
        
        for model, available in self.models_available.items():
            status = '✓' if available else '✗'
            lines.append(f"  {model}: {status}")
        
        lines.append(f"{'='*60}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat(),
            'regime': self.regime,
            'regime_confidence': float(self.regime_confidence),
            'regime_probabilities': self.regime_probabilities,
            'volatility_regime': self.volatility_regime,
            'volatility_confidence': float(self.volatility_confidence),
            'volatility_probabilities': self.volatility_probabilities,
            'is_transitioning': self.is_transitioning,
            'transition_probability': float(self.transition_probability),
            'models_available': self.models_available
        }
    
    @property
    def state_code(self) -> str:
        """
        Generate compact state code for analysis.
        Format: REGIME_VOLATILITY_TRANSITION
        Example: TREND_UP_HIGH_STABLE
        """
        trans = "TRANS" if self.is_transitioning else "STABLE"
        return f"{self.regime}_{self.volatility_regime}_{trans}"


# =============================================================================
# MARKET STATE ANALYZER
# =============================================================================

class MarketStateAnalyzer:
    """
    Analyzes market state by combining multiple classification models.
    
    This is a RESEARCH TOOL that describes market conditions.
    It does NOT generate trading signals or recommendations.
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        version: str = 'v1'
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.version = version
        
        # Model references (loaded later)
        self.regime_classifier = None
        self.volatility_classifier = None
        self.transition_detector = None
        
        # Configuration
        self.transition_threshold = 0.5  # Probability threshold for "transitioning"
        
        success(f"Initialized MarketStateAnalyzer for {symbol} {timeframe}")
    
    def load_models(self) -> bool:
        """Load all available models."""
        info(f"Loading state models for {self.symbol} {self.timeframe}...")
        
        models_loaded = 0
        
        # Load regime classifier
        # Note: load() is a classmethod that returns a new instance
        try:
            from src.models.state.regime_classifier import RegimeClassifier
            self.regime_classifier = RegimeClassifier.load(
                symbol=self.symbol,
                timeframe=self.timeframe,
                version=self.version
            )
            success("  ✓ Regime model loaded")
            models_loaded += 1
        except Exception as e:
            warning(f"  ✗ Regime model not available: {e}")
            self.regime_classifier = None
        
        # Load volatility classifier
        try:
            from src.models.state.volatility_regime import VolatilityRegimeClassifier
            self.volatility_classifier = VolatilityRegimeClassifier.load(
                symbol=self.symbol,
                timeframe=self.timeframe,
                version=self.version
            )
            success("  ✓ Volatility model loaded")
            models_loaded += 1
        except Exception as e:
            warning(f"  ✗ Volatility model not available: {e}")
            self.volatility_classifier = None
        
        # Load transition detector
        try:
            from src.models.state.transition_detector import TransitionDetector
            self.transition_detector = TransitionDetector.load(
                symbol=self.symbol,
                timeframe=self.timeframe,
                version=self.version
            )
            success("  ✓ Transition model loaded")
            models_loaded += 1
        except Exception as e:
            warning(f"  ✗ Transition model not available: {e}")
            self.transition_detector = None
        
        if models_loaded > 0:
            success(f"Loaded {models_loaded}/3 models for {self.symbol} {self.timeframe}")
            return True
        else:
            error(f"No models loaded for {self.symbol} {self.timeframe}")
            return False
    
    def get_market_state(self, features: pd.DataFrame) -> MarketState:
        """
        Analyze current market state from features.
        
        Args:
            features: DataFrame with feature values (single row or last row used)
        
        Returns:
            MarketState object describing current conditions
        """
        # Use last row if multiple provided
        if len(features) > 1:
            features = features.tail(1)
        
        # Initialize state components
        regime = "UNKNOWN"
        regime_confidence = 0.0
        regime_probs = {}
        
        volatility = "UNKNOWN"
        volatility_confidence = 0.0
        volatility_probs = {}
        
        is_transitioning = False
        transition_prob = 0.0
        
        # Get regime prediction
        # predict() returns tuple: (predictions_array, confidence_array)
        if self.regime_classifier is not None:
            try:
                preds, confs = self.regime_classifier.predict(features)
                regime_map = {0: "RANGING", 1: "TREND_UP", 2: "TREND_DOWN"}
                regime = regime_map.get(int(preds[0]), "UNKNOWN")
                regime_confidence = float(confs[0])
            except Exception as e:
                warning(f"Regime prediction failed: {e}")
        
        # Get volatility prediction
        if self.volatility_classifier is not None:
            try:
                preds, confs = self.volatility_classifier.predict(features)
                vol_map = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "CRISIS"}
                volatility = vol_map.get(int(preds[0]), "UNKNOWN")
                volatility_confidence = float(confs[0])
            except Exception as e:
                warning(f"Volatility prediction failed: {e}")
        
        # Get transition prediction
        if self.transition_detector is not None:
            try:
                preds, confs = self.transition_detector.predict(features)
                # For binary classification, confidence of class 1 is transition probability
                transition_prob = float(confs[0])
                is_transitioning = int(preds[0]) == 1
            except Exception as e:
                warning(f"Transition prediction failed: {e}")
        
        # Build market state - note the order matches the dataclass definition
        state = MarketState(
            symbol=self.symbol,
            timeframe=self.timeframe,
            timestamp=datetime.now(),
            regime=regime,
            regime_confidence=regime_confidence,
            volatility_regime=volatility,
            volatility_confidence=volatility_confidence,
            is_transitioning=is_transitioning,
            transition_probability=transition_prob,
            regime_probabilities=regime_probs,
            volatility_probabilities=volatility_probs,
            models_available={
                'regime': self.regime_classifier is not None,
                'volatility': self.volatility_classifier is not None,
                'transition': self.transition_detector is not None
            }
        )
        
        return state


# =============================================================================
# MULTI-PAIR ANALYZER
# =============================================================================

class MultiPairAnalyzer:
    """
    Analyze market state across multiple currency pairs.
    Useful for understanding cross-market conditions.
    """
    
    def __init__(
        self,
        symbols: List[str],
        timeframe: str,
        version: str = 'v1'
    ):
        self.symbols = symbols
        self.timeframe = timeframe
        self.version = version
        
        # Initialize analyzers for each symbol
        self.analyzers: Dict[str, MarketStateAnalyzer] = {}
        for symbol in symbols:
            self.analyzers[symbol] = MarketStateAnalyzer(
                symbol=symbol,
                timeframe=timeframe,
                version=version
            )
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load models for all pairs."""
        results = {}
        for symbol, analyzer in self.analyzers.items():
            results[symbol] = analyzer.load_models()
        return results
    
    def get_all_states(
        self,
        features_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, MarketState]:
        """
        Get market state for all pairs.
        
        Args:
            features_dict: Dict mapping symbol to features DataFrame
        
        Returns:
            Dict mapping symbol to MarketState
        """
        states = {}
        for symbol, analyzer in self.analyzers.items():
            if symbol in features_dict:
                states[symbol] = analyzer.get_market_state(features_dict[symbol])
        return states
    
    def get_market_summary(
        self,
        states: Dict[str, MarketState]
    ) -> dict:
        """
        Generate summary statistics across all pairs.
        
        Returns dict with:
        - regime_distribution: Count of pairs in each regime
        - volatility_distribution: Count of pairs in each vol state
        - transition_count: Number of pairs expecting regime change
        - dominant_regime: Most common regime
        - dominant_volatility: Most common volatility state
        """
        if not states:
            return {}
        
        regimes = [s.regime for s in states.values()]
        volatilities = [s.volatility_regime for s in states.values()]
        transitioning = [s.is_transitioning for s in states.values()]
        
        # Count distributions
        regime_dist = {}
        for r in regimes:
            regime_dist[r] = regime_dist.get(r, 0) + 1
        
        vol_dist = {}
        for v in volatilities:
            vol_dist[v] = vol_dist.get(v, 0) + 1
        
        # Find dominant states
        dominant_regime = max(regime_dist, key=regime_dist.get) if regime_dist else "UNKNOWN"
        dominant_vol = max(vol_dist, key=vol_dist.get) if vol_dist else "UNKNOWN"
        
        return {
            'n_pairs': len(states),
            'regime_distribution': regime_dist,
            'volatility_distribution': vol_dist,
            'dominant_regime': dominant_regime,
            'dominant_volatility': dominant_vol,
            'n_transitioning': sum(transitioning),
            'pct_transitioning': 100 * sum(transitioning) / len(states) if states else 0,
            'avg_regime_confidence': np.mean([s.regime_confidence for s in states.values()]),
            'avg_volatility_confidence': np.mean([s.volatility_confidence for s in states.values()])
        }
    
    def print_dashboard(self, states: Dict[str, MarketState]):
        """Print a research dashboard showing all pair states."""
        
        print(f"\n{'='*80}")
        print(f"{'MARKET STATE DASHBOARD':^80}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Header
        print(f"\n{'Symbol':<10} {'Regime':<12} {'Conf':<8} {'Volatility':<10} {'Conf':<8} {'Trans?':<8} {'State Code':<25}")
        print(f"{'-'*90}")
        
        # Sort by regime for easier reading
        sorted_states = sorted(states.items(), key=lambda x: x[1].regime)
        
        for symbol, state in sorted_states:
            trans = '⚠ Yes' if state.is_transitioning else 'No'
            print(f"{symbol:<10} {state.regime:<12} {state.regime_confidence:.0%}     "
                  f"{state.volatility_regime:<10} {state.volatility_confidence:.0%}     "
                  f"{trans:<8} {state.state_code}")
        
        # Summary
        summary = self.get_market_summary(states)
        
        print(f"\n{'='*80}")
        print("MARKET SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nDominant Regime: {summary['dominant_regime']}")
        print(f"Dominant Volatility: {summary['dominant_volatility']}")
        print(f"Pairs Transitioning: {summary['n_transitioning']} ({summary['pct_transitioning']:.0f}%)")
        
        print(f"\nRegime Distribution:")
        for regime, count in sorted(summary['regime_distribution'].items()):
            bar = '█' * count
            print(f"  {regime:<12} {count:>2} {bar}")
        
        print(f"\nVolatility Distribution:")
        for vol, count in sorted(summary['volatility_distribution'].items()):
            bar = '█' * count
            print(f"  {vol:<10} {count:>2} {bar}")


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# Keep old names working
StateAggregator = MarketStateAnalyzer
MultiPairAggregator = MultiPairAnalyzer