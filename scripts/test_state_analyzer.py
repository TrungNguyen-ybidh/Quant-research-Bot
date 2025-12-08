"""
Test Market State Analyzer
==========================
Test script to verify the MarketStateAnalyzer works correctly
by loading all three model types and analyzing market conditions.

This is a RESEARCH TOOL - it describes market state,
NOT trading recommendations.

Usage:
    python scripts/test_state_analyzer.py --symbol EURUSD --timeframe 1h
    python scripts/test_state_analyzer.py --all --timeframe 1h
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.utils.logger import info, success, warning, error


# =============================================================================
# CONSTANTS
# =============================================================================

MAJOR_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
ALL_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
    'AUDJPY', 'CHFJPY', 'EURJPY', 'GBPJPY', 'NZDUSD', 'USDCHF'
]


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_single_pair(symbol: str, timeframe: str, version: str = 'v1'):
    """Test MarketStateAnalyzer for a single pair."""
    
    # Import here to avoid circular imports
    from src.models.state.state_aggregator import MarketStateAnalyzer
    
    print(f"\n{'='*80}")
    print(f"ANALYZING MARKET STATE: {symbol} {timeframe}")
    print(f"{'='*80}")
    
    # Initialize analyzer
    try:
        analyzer = MarketStateAnalyzer(
            symbol=symbol,
            timeframe=timeframe,
            version=version
        )
    except Exception as e:
        error(f"Failed to initialize: {e}")
        return None
    
    # Load models
    print(f"\n{'-'*40}")
    print("Loading Models")
    print(f"{'-'*40}")
    
    load_success = analyzer.load_models()
    
    # Check which models are loaded
    has_regime = analyzer.regime_classifier is not None
    has_volatility = analyzer.volatility_classifier is not None
    has_transition = analyzer.transition_detector is not None
    
    print(f"\nModel Status:")
    print(f"  Regime Model:     {'✓ Loaded' if has_regime else '✗ Not available'}")
    print(f"  Volatility Model: {'✓ Loaded' if has_volatility else '✗ Not available'}")
    print(f"  Transition Model: {'✓ Loaded' if has_transition else '✗ Not available'}")
    
    if not any([has_regime, has_volatility, has_transition]):
        error("No models loaded! Cannot analyze state.")
        return None
    
    # Load test data
    print(f"\n{'-'*40}")
    print("Loading Feature Data")
    print(f"{'-'*40}")
    
    features_path = project_root / f"data/processed/{symbol}_{timeframe}_features.parquet"
    
    if not features_path.exists():
        error(f"Features file not found: {features_path}")
        return None
    
    df = pd.read_parquet(features_path)
    info(f"Loaded {len(df)} samples from {features_path.name}")
    
    # Analyze last N samples
    n_samples = 10
    test_data = df.tail(n_samples)
    
    print(f"\n{'-'*40}")
    print(f"Analyzing Last {n_samples} Bars")
    print(f"{'-'*40}")
    
    states = []
    for i, (idx, row) in enumerate(test_data.iterrows()):
        features = row.to_frame().T
        state = analyzer.get_market_state(features)
        states.append(state)
        
        # Print state summary
        print(f"\n[Bar {i+1}] {idx}")
        print(f"  Regime:      {state.regime} ({state.regime_confidence:.0%} conf)")
        print(f"  Volatility:  {state.volatility_regime} ({state.volatility_confidence:.0%} conf)")
        print(f"  Transition:  {'Expected' if state.is_transitioning else 'Stable'} ({state.transition_probability:.0%} prob)")
        print(f"  State Code:  {state.state_code}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    regimes = [s.regime for s in states]
    volatilities = [s.volatility_regime for s in states]
    transitions = [s.is_transitioning for s in states]
    
    print(f"\nRegime Distribution (last {n_samples} bars):")
    for reg in set(regimes):
        count = regimes.count(reg)
        pct = count / len(regimes)
        bar = '█' * int(pct * 20)
        print(f"  {reg:<12} {count:>2} ({pct:.0%}) {bar}")
    
    print(f"\nVolatility Distribution:")
    for vol in set(volatilities):
        count = volatilities.count(vol)
        pct = count / len(volatilities)
        bar = '█' * int(pct * 20)
        print(f"  {vol:<10} {count:>2} ({pct:.0%}) {bar}")
    
    trans_count = sum(transitions)
    print(f"\nTransition Alerts: {trans_count}/{n_samples} ({trans_count/n_samples:.0%})")
    
    # Current state (most recent)
    current = states[-1]
    print(f"\n{'='*80}")
    print("CURRENT MARKET STATE")
    print(f"{'='*80}")
    print(current.summary())
    
    return states


def test_all_pairs(timeframe: str, version: str = 'v1'):
    """Analyze market state across all pairs."""
    
    # Import here to avoid circular imports
    from src.models.state.state_aggregator import MultiPairAnalyzer
    
    print(f"\n{'='*80}")
    print(f"MULTI-PAIR MARKET ANALYSIS - {timeframe}")
    print(f"{'='*80}")
    
    # Initialize MultiPairAnalyzer
    multi_analyzer = MultiPairAnalyzer(
        symbols=ALL_PAIRS,
        timeframe=timeframe,
        version=version
    )
    
    # Load all models
    print(f"\n{'-'*40}")
    print("Loading Models for All Pairs")
    print(f"{'-'*40}")
    
    load_results = multi_analyzer.load_all_models()
    
    print(f"\nLoad Results:")
    print(f"{'Symbol':<10} {'Status':<15}")
    print(f"{'-'*25}")
    
    loaded_count = 0
    for symbol, success_status in load_results.items():
        status = '✓ Ready' if success_status else '✗ Failed'
        print(f"{symbol:<10} {status}")
        if success_status:
            loaded_count += 1
    
    print(f"\nLoaded: {loaded_count}/{len(ALL_PAIRS)} pairs")
    
    # Load feature data for each pair
    print(f"\n{'-'*40}")
    print("Loading Feature Data")
    print(f"{'-'*40}")
    
    all_features = {}
    for symbol in ALL_PAIRS:
        features_path = project_root / f"data/processed/{symbol}_{timeframe}_features.parquet"
        if features_path.exists():
            df = pd.read_parquet(features_path)
            all_features[symbol] = df.tail(1)  # Latest bar only
            info(f"  {symbol}: {len(df)} bars available")
        else:
            warning(f"  {symbol}: No data found")
    
    # Get states for all pairs
    states = multi_analyzer.get_all_states(all_features)
    
    # Print dashboard
    multi_analyzer.print_dashboard(states)
    
    # Additional research insights
    print(f"\n{'='*80}")
    print("RESEARCH INSIGHTS")
    print(f"{'='*80}")
    
    # Group pairs by regime
    regime_groups = {}
    for symbol, state in states.items():
        if state.regime not in regime_groups:
            regime_groups[state.regime] = []
        regime_groups[state.regime].append(symbol)
    
    print(f"\nPairs by Regime:")
    for regime, pairs in sorted(regime_groups.items()):
        print(f"  {regime}: {', '.join(pairs)}")
    
    # Group pairs by volatility
    vol_groups = {}
    for symbol, state in states.items():
        if state.volatility_regime not in vol_groups:
            vol_groups[state.volatility_regime] = []
        vol_groups[state.volatility_regime].append(symbol)
    
    print(f"\nPairs by Volatility:")
    for vol, pairs in sorted(vol_groups.items()):
        print(f"  {vol}: {', '.join(pairs)}")
    
    # Transition warnings
    transitioning = [(s, st) for s, st in states.items() if st.is_transitioning]
    if transitioning:
        print(f"\n⚠️  Regime Transitions Expected:")
        for symbol, state in transitioning:
            print(f"  {symbol}: {state.regime} → ? (prob: {state.transition_probability:.0%})")
    else:
        print(f"\n✓ No regime transitions expected across pairs")
    
    # State diversity metric
    unique_states = len(set(s.state_code for s in states.values()))
    print(f"\nState Diversity: {unique_states} unique states across {len(states)} pairs")
    
    return states


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test Market State Analyzer')
    
    parser.add_argument('--symbol', type=str, help='Single symbol to analyze')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe')
    parser.add_argument('--version', type=str, default='v1', help='Model version')
    parser.add_argument('--all', action='store_true', help='Analyze all pairs')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"{'MARKET STATE ANALYZER - RESEARCH TOOL':^80}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"\nNote: This tool describes market conditions for research purposes.")
    print(f"      It does NOT generate trading signals or recommendations.")
    
    if args.all:
        test_all_pairs(args.timeframe, args.version)
    elif args.symbol:
        test_single_pair(args.symbol, args.timeframe, args.version)
    else:
        # Default: analyze EURUSD
        test_single_pair("EURUSD", args.timeframe, args.version)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()