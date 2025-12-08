"""
Analyze Feature Importance for Regime Classification
=====================================================
Examines which features matter most for successful vs unsuccessful pairs.

Usage: python scripts/analyze_regime_features.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from src.utils.logger import info, success, warning


def load_selector(symbol: str, timeframe: str, version: str = "v1"):
    """Load saved feature selector."""
    path = f"models/trained/regime/{symbol}_{timeframe}_regime_{version}_selector.pkl"
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        warning(f"Selector not found: {path}")
        return None


def load_metrics(symbol: str, timeframe: str, version: str = "v1"):
    """Load saved metrics."""
    path = f"outputs/models/regime/{symbol}/{timeframe}/{symbol}_{timeframe}_regime_{version}_metrics.json"
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        warning(f"Metrics not found: {path}")
        return None


def analyze_features():
    """Analyze feature importance across all trained pairs."""
    
    print("="*80)
    print("FEATURE IMPORTANCE ANALYSIS FOR REGIME CLASSIFICATION")
    print("="*80)
    
    # Pairs we trained
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    timeframe = "1h"
    
    # Results storage
    results = {}
    all_features = defaultdict(list)  # feature -> list of (symbol, importance, rank)
    
    print("\n" + "="*80)
    print("INDIVIDUAL PAIR ANALYSIS")
    print("="*80)
    
    for symbol in pairs:
        selector = load_selector(symbol, timeframe)
        metrics = load_metrics(symbol, timeframe)
        
        if selector is None or metrics is None:
            continue
        
        test_acc = metrics.get('test_metrics', {}).get('accuracy', 0)
        
        results[symbol] = {
            'accuracy': test_acc,
            'selected_features': selector.selected_features,
            'feature_importance': selector.feature_importance
        }
        
        # Track feature usage across pairs
        for rank, feat in enumerate(selector.selected_features, 1):
            importance = selector.feature_importance.get(feat, 0)
            all_features[feat].append({
                'symbol': symbol,
                'rank': rank,
                'importance': importance,
                'accuracy': test_acc
            })
        
        # Print individual pair analysis
        print(f"\n{'='*60}")
        print(f"{symbol} - Test Accuracy: {test_acc:.2%}")
        status = "✅ ABOVE BASELINE" if test_acc > 0.33 else "⚠️ AT/BELOW BASELINE"
        print(f"Status: {status}")
        print(f"{'='*60}")
        print(f"\nTop 15 Features (by importance):")
        print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12}")
        print("-"*50)
        
        # Sort by importance
        sorted_feats = sorted(
            [(f, selector.feature_importance.get(f, 0)) for f in selector.selected_features],
            key=lambda x: x[1],
            reverse=True
        )
        
        for rank, (feat, imp) in enumerate(sorted_feats[:15], 1):
            print(f"{rank:<6} {feat:<30} {imp:.4f}")
    
    # Cross-pair analysis
    print("\n" + "="*80)
    print("CROSS-PAIR FEATURE ANALYSIS")
    print("="*80)
    
    # Features that appear in ALL pairs
    common_features = [f for f, usages in all_features.items() if len(usages) == len(pairs)]
    
    print(f"\n📊 Features Selected in ALL {len(pairs)} Pairs ({len(common_features)} features):")
    print("-"*70)
    
    # Calculate average rank and importance for common features
    common_stats = []
    for feat in common_features:
        usages = all_features[feat]
        avg_rank = np.mean([u['rank'] for u in usages])
        avg_importance = np.mean([u['importance'] for u in usages])
        common_stats.append((feat, avg_rank, avg_importance))
    
    # Sort by average importance
    common_stats.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Feature':<30} {'Avg Rank':<12} {'Avg Importance':<15}")
    print("-"*60)
    for feat, avg_rank, avg_imp in common_stats[:20]:
        print(f"{feat:<30} {avg_rank:<12.1f} {avg_imp:<15.4f}")
    
    # Features in HIGH-performing pairs only
    print("\n" + "="*80)
    print("FEATURES UNIQUE TO HIGH-PERFORMING PAIRS (>33% accuracy)")
    print("="*80)
    
    high_performers = [s for s, r in results.items() if r['accuracy'] > 0.33]
    low_performers = [s for s, r in results.items() if r['accuracy'] <= 0.33]
    
    print(f"\nHigh performers: {high_performers}")
    print(f"Low performers: {low_performers}")
    
    # Find features that are in high performers but not in low performers
    high_features = set()
    for symbol in high_performers:
        high_features.update(results[symbol]['selected_features'])
    
    low_features = set()
    for symbol in low_performers:
        low_features.update(results[symbol]['selected_features'])
    
    unique_to_high = high_features - low_features
    unique_to_low = low_features - high_features
    
    print(f"\n🌟 Features ONLY in high-performing pairs ({len(unique_to_high)}):")
    for feat in sorted(unique_to_high):
        print(f"  - {feat}")
    
    print(f"\n⚠️ Features ONLY in low-performing pairs ({len(unique_to_low)}):")
    for feat in sorted(unique_to_low):
        print(f"  - {feat}")
    
    # Feature category analysis
    print("\n" + "="*80)
    print("FEATURE CATEGORY BREAKDOWN")
    print("="*80)
    
    # Define feature categories
    categories = {
        'volatility': ['realized_vol', 'bipower_variation', 'parkinson', 'skew', 'kurtosis', 
                      'variance_ratio', 'hurst', 'volatility_of_vol', 'jump', 'atr'],
        'trend': ['sma_ratio', 'ema_ratio', 'adx', 'trend_strength', 'price_position', 'macd', 'roc'],
        'microstructure': ['micro_price', 'spread', 'wick_ratio', 'body_ratio', 'rejection', 'range'],
        'fx_factors': ['carry', 'momentum_factor', 'value_factor', 'yield_curve', 'rate_level'],
        'indicators': ['rsi', 'stochastic', 'cci', 'williams', 'obv', 'mfi', 'pivot', 'bollinger'],
        'session': ['london', 'new_york', 'asia', 'session_overlap', 'session_volatility'],
        'cross_asset': ['correlation', 'relative_strength', 'risk_sentiment']
    }
    
    def categorize_feature(feat_name):
        feat_lower = feat_name.lower()
        for cat, keywords in categories.items():
            for kw in keywords:
                if kw in feat_lower:
                    return cat
        return 'other'
    
    # Count categories for high vs low performers
    high_cats = defaultdict(int)
    low_cats = defaultdict(int)
    
    for symbol in high_performers:
        for feat in results[symbol]['selected_features']:
            high_cats[categorize_feature(feat)] += 1
    
    for symbol in low_performers:
        for feat in results[symbol]['selected_features']:
            low_cats[categorize_feature(feat)] += 1
    
    # Normalize by number of pairs
    n_high = len(high_performers) if high_performers else 1
    n_low = len(low_performers) if low_performers else 1
    
    print(f"\n{'Category':<20} {'High Perf (avg)':<18} {'Low Perf (avg)':<18} {'Difference':<12}")
    print("-"*70)
    
    all_cats = set(high_cats.keys()) | set(low_cats.keys())
    cat_diffs = []
    for cat in sorted(all_cats):
        high_avg = high_cats[cat] / n_high
        low_avg = low_cats[cat] / n_low
        diff = high_avg - low_avg
        cat_diffs.append((cat, high_avg, low_avg, diff))
        marker = "✅" if diff > 0 else "❌" if diff < 0 else ""
        print(f"{cat:<20} {high_avg:<18.1f} {low_avg:<18.1f} {diff:>+8.1f} {marker}")
    
    # Top differentiating features
    print("\n" + "="*80)
    print("TOP DIFFERENTIATING FEATURES")
    print("="*80)
    print("(Features with highest importance in high-performing pairs)")
    
    # Calculate importance difference
    feature_diff = []
    for feat, usages in all_features.items():
        high_imp = np.mean([u['importance'] for u in usages if u['accuracy'] > 0.33]) if any(u['accuracy'] > 0.33 for u in usages) else 0
        low_imp = np.mean([u['importance'] for u in usages if u['accuracy'] <= 0.33]) if any(u['accuracy'] <= 0.33 for u in usages) else 0
        diff = high_imp - low_imp
        feature_diff.append((feat, high_imp, low_imp, diff))
    
    # Sort by difference
    feature_diff.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\n{'Feature':<30} {'High Imp':<12} {'Low Imp':<12} {'Diff':<10}")
    print("-"*70)
    print("\n🌟 Features MORE important in high performers:")
    for feat, high, low, diff in feature_diff[:10]:
        if diff > 0:
            print(f"{feat:<30} {high:<12.4f} {low:<12.4f} {diff:>+10.4f}")
    
    print("\n⚠️ Features MORE important in low performers:")
    for feat, high, low, diff in feature_diff[-10:]:
        if diff < 0:
            print(f"{feat:<30} {high:<12.4f} {low:<12.4f} {diff:>+10.4f}")
    
    # Summary recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("""
Based on the analysis:

1. VOLATILITY FEATURES are crucial:
   - hurst_exponent, realized_vol, parkinson_volatility
   - These capture market regime characteristics well

2. FX-SPECIFIC FACTORS matter:
   - yield_curve_factor, carry indicators
   - These are fundamental drivers in FX

3. TREND INDICATORS work:
   - adx, macd_signal, trend_strength
   - But need to be combined with vol features

4. POTENTIAL IMPROVEMENTS:
   - For USDCAD: Try different feature subsets
   - For EURUSD: May need more FX-factor emphasis
   - Consider: Session-based regime detection

5. NEXT STEPS:
   - Train with LSTM to capture temporal patterns
   - Build volatility regime classifier
   - Test regime-conditional strategies
""")
    
    print("="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    analyze_features()