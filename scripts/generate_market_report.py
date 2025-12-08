"""
Market State Report Generator
=============================
Generate comprehensive market state reports for all FX pairs.
Outputs analysis to console and optionally saves to file.

This is a RESEARCH TOOL - it describes market conditions,
NOT trading recommendations.

Usage:
    python scripts/generate_market_report.py --timeframe 1h
    python scripts/generate_market_report.py --timeframe 1h --output report.json
    python scripts/generate_market_report.py --timeframe 1h --pairs EURUSD GBPUSD
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.utils.logger import info, success, warning, error


# =============================================================================
# CONSTANTS
# =============================================================================

ALL_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
    'AUDJPY', 'CHFJPY', 'EURJPY', 'GBPJPY', 'NZDUSD', 'USDCHF'
]

MAJOR_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']


# =============================================================================
# REPORT GENERATION
# =============================================================================

def load_latest_features(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load the most recent features for a symbol."""
    features_path = project_root / f"data/processed/{symbol}_{timeframe}_features.parquet"
    
    if not features_path.exists():
        warning(f"Features not found: {features_path}")
        return None
    
    df = pd.read_parquet(features_path)
    return df.tail(1)


def generate_report(pairs: List[str], timeframe: str, version: str = 'v1'):
    """
    Generate market state report for multiple pairs.
    
    Returns:
        Tuple of (report_dict, states_dict, analyzer)
    """
    # Import here to avoid circular imports
    from src.models.state.state_aggregator import MultiPairAnalyzer
    
    report = {}
    
    # Initialize multi-pair analyzer
    multi_analyzer = MultiPairAnalyzer(
        symbols=pairs,
        timeframe=timeframe,
        version=version
    )
    
    # Load all models
    info("Loading models...")
    load_results = multi_analyzer.load_all_models()
    
    loaded_count = sum(1 for s in load_results.values() if s)
    info(f"Loaded models for {loaded_count}/{len(pairs)} pairs")
    
    # Load features for each pair
    all_features = {}
    for symbol in pairs:
        features = load_latest_features(symbol, timeframe)
        if features is not None:
            all_features[symbol] = features
    
    info(f"Loaded features for {len(all_features)}/{len(pairs)} pairs")
    
    # Generate states
    states = multi_analyzer.get_all_states(all_features)
    
    # Convert states to report format
    for symbol, state in states.items():
        report[symbol] = state.to_dict()
    
    return report, states, multi_analyzer


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_report(states: Dict, analyzer):
    """Print comprehensive market state report."""
    
    print(f"\n{'='*100}")
    print(f"{'MARKET STATE REPORT':^100}")
    print(f"{'='*100}")
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Purpose: Research analysis of current market conditions")
    print(f"Note: This report describes market state - it is NOT trading advice")
    
    # Main state table
    print(f"\n{'='*100}")
    print("CURRENT MARKET CONDITIONS")
    print(f"{'='*100}")
    
    print(f"\n{'Symbol':<10} {'Regime':<12} {'R.Conf':<8} {'Volatility':<10} {'V.Conf':<8} {'Transition':<12} {'State Code':<25}")
    print(f"{'-'*95}")
    
    # Sort by state code for grouping
    sorted_states = sorted(states.items(), key=lambda x: (x[1].regime, x[1].volatility_regime))
    
    for symbol, state in sorted_states:
        trans_status = f"⚠ {state.transition_probability:.0%}" if state.is_transitioning else "Stable"
        print(f"{symbol:<10} {state.regime:<12} {state.regime_confidence:.0%}      "
              f"{state.volatility_regime:<10} {state.volatility_confidence:.0%}      "
              f"{trans_status:<12} {state.state_code}")
    
    # Summary statistics
    summary = analyzer.get_market_summary(states)
    
    print(f"\n{'='*100}")
    print("MARKET SUMMARY")
    print(f"{'='*100}")
    
    print(f"\nOverall Conditions:")
    print(f"  Pairs Analyzed: {summary['n_pairs']}")
    print(f"  Dominant Regime: {summary['dominant_regime']}")
    print(f"  Dominant Volatility: {summary['dominant_volatility']}")
    print(f"  Avg Regime Confidence: {summary['avg_regime_confidence']:.1%}")
    print(f"  Avg Volatility Confidence: {summary['avg_volatility_confidence']:.1%}")
    
    # Regime breakdown
    print(f"\n{'='*100}")
    print("REGIME ANALYSIS")
    print(f"{'='*100}")
    
    print(f"\nDistribution:")
    for regime in ['TREND_UP', 'TREND_DOWN', 'RANGING', 'UNKNOWN']:
        count = summary['regime_distribution'].get(regime, 0)
        if count > 0:
            pct = count / summary['n_pairs']
            bar = '█' * int(pct * 30)
            pairs_in_regime = [s for s, st in states.items() if st.regime == regime]
            print(f"  {regime:<12} {count:>2} ({pct:>5.0%}) {bar}")
            print(f"              Pairs: {', '.join(pairs_in_regime)}")
    
    # Volatility breakdown
    print(f"\n{'='*100}")
    print("VOLATILITY ANALYSIS")
    print(f"{'='*100}")
    
    print(f"\nDistribution:")
    for vol in ['LOW', 'NORMAL', 'HIGH', 'CRISIS', 'UNKNOWN']:
        count = summary['volatility_distribution'].get(vol, 0)
        if count > 0:
            pct = count / summary['n_pairs']
            bar = '█' * int(pct * 30)
            pairs_in_vol = [s for s, st in states.items() if st.volatility_regime == vol]
            print(f"  {vol:<10} {count:>2} ({pct:>5.0%}) {bar}")
            print(f"            Pairs: {', '.join(pairs_in_vol)}")
    
    # Transition analysis
    print(f"\n{'='*100}")
    print("TRANSITION ANALYSIS")
    print(f"{'='*100}")
    
    transitioning = [(s, st) for s, st in states.items() if st.is_transitioning]
    stable = [(s, st) for s, st in states.items() if not st.is_transitioning]
    
    print(f"\nTransitioning: {len(transitioning)} pairs ({summary['pct_transitioning']:.0f}%)")
    print(f"Stable: {len(stable)} pairs")
    
    if transitioning:
        print(f"\n⚠️  Pairs Expecting Regime Change:")
        for symbol, state in sorted(transitioning, key=lambda x: -x[1].transition_probability):
            print(f"  {symbol:<10} Current: {state.regime:<12} Transition Prob: {state.transition_probability:.0%}")
    
    # State code analysis
    print(f"\n{'='*100}")
    print("STATE CODE ANALYSIS")
    print(f"{'='*100}")
    
    state_codes = {}
    for symbol, state in states.items():
        code = state.state_code
        if code not in state_codes:
            state_codes[code] = []
        state_codes[code].append(symbol)
    
    print(f"\nUnique Market States: {len(state_codes)}")
    print(f"\nState Code Distribution:")
    for code, pairs in sorted(state_codes.items(), key=lambda x: -len(x[1])):
        print(f"  {code:<30} ({len(pairs)} pairs): {', '.join(pairs)}")
    
    # Research observations
    print(f"\n{'='*100}")
    print("RESEARCH OBSERVATIONS")
    print(f"{'='*100}")
    
    # Market regime consensus
    regime_counts = summary['regime_distribution']
    max_regime_count = max(regime_counts.values()) if regime_counts else 0
    regime_consensus = max_regime_count / summary['n_pairs'] if summary['n_pairs'] > 0 else 0
    
    print(f"\n1. Regime Consensus: {regime_consensus:.0%}")
    if regime_consensus > 0.6:
        print(f"   → Strong agreement across pairs on {summary['dominant_regime']} regime")
    elif regime_consensus > 0.4:
        print(f"   → Moderate agreement, {summary['dominant_regime']} slightly dominant")
    else:
        print(f"   → Mixed regimes across pairs - Loss of correlation")
    
    # Volatility environment
    vol_counts = summary['volatility_distribution']
    high_vol_count = vol_counts.get('HIGH', 0) + vol_counts.get('CRISIS', 0)
    high_vol_pct = high_vol_count / summary['n_pairs'] if summary['n_pairs'] > 0 else 0
    
    print(f"\n2. Volatility Environment:")
    if high_vol_pct > 0.5:
        print(f"   → Elevated volatility across {high_vol_pct:.0%} of pairs")
    elif vol_counts.get('LOW', 0) / summary['n_pairs'] > 0.5 if summary['n_pairs'] > 0 else False:
        print(f"   → Low volatility environment")
    else:
        print(f"   → Normal volatility conditions")
    
    # Stability assessment
    print(f"\n3. Market Stability:")
    if summary['pct_transitioning'] > 30:
        print(f"   → Unstable - {summary['pct_transitioning']:.0f}% of pairs expecting regime change")
    elif summary['pct_transitioning'] > 10:
        print(f"   → Some instability - watch transitioning pairs")
    else:
        print(f"   → Stable conditions across most pairs")


def save_report(report: Dict, states: Dict, analyzer, output_path: Path):
    """Save report to JSON file."""
    
    summary = analyzer.get_market_summary(states)
    
    output_data = {
        'generated_at': datetime.now().isoformat(),
        'report_type': 'market_state_analysis',
        'disclaimer': 'This is a research tool describing market conditions, NOT trading advice',
        'summary': summary,
        'pair_states': report
    }
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    success(f"Report saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate Market State Report')
    
    parser.add_argument('--timeframe', type=str, default='1h',
                        help='Timeframe (default: 1h)')
    parser.add_argument('--pairs', nargs='+', default=None,
                        help='Specific pairs to analyze (default: all)')
    parser.add_argument('--majors', action='store_true',
                        help='Only analyze major pairs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (JSON)')
    parser.add_argument('--version', type=str, default='v1',
                        help='Model version')
    
    args = parser.parse_args()
    
    # Determine which pairs to analyze
    if args.pairs:
        pairs = args.pairs
    elif args.majors:
        pairs = MAJOR_PAIRS
    else:
        pairs = ALL_PAIRS
    
    print(f"\n{'='*100}")
    print(f"{'MARKET STATE REPORT GENERATOR':^100}")
    print(f"{'='*100}")
    print(f"\nTimeframe: {args.timeframe}")
    print(f"Pairs: {len(pairs)} ({', '.join(pairs[:5])}{'...' if len(pairs) > 5 else ''})")
    print(f"\n⚠️  RESEARCH TOOL - Describes market conditions, NOT trading advice")
    
    # Generate report
    report, states, analyzer = generate_report(pairs, args.timeframe, args.version)
    
    if not report:
        error("No report generated!")
        return
    
    # Print report
    print_report(states, analyzer)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        save_report(report, states, analyzer, output_path)
    
    print(f"\n{'='*100}")
    print(f"Report generation complete - {len(report)} pairs analyzed")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()