"""
Full Dataset Market State Analysis (Optimized)
==============================================
Analyzes entire dataset using batch predictions for speed.

Usage:
    python scripts/analyze_full_dataset.py --symbol EURUSD --timeframe 1h
    python scripts/analyze_full_dataset.py --symbol EURUSD --timeframe 1h --save
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.models.state.regime_classifier import RegimeClassifier
from src.models.state.volatility_regime import VolatilityRegimeClassifier
from src.models.state.transition_detector import TransitionDetector
from src.utils.logger import info, success, warning, error


def batch_predict(classifier, features_df, batch_size=1000):
    """
    Run predictions in batches for efficiency.
    
    Returns:
        Tuple of (predictions, confidences) as numpy arrays
    """
    n_samples = len(features_df)
    all_preds = []
    all_confs = []
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = features_df.iloc[start_idx:end_idx]
        
        # Preprocess
        X_preprocessed = classifier.preprocessor.transform(batch)
        X_selected = classifier.selector.transform(X_preprocessed)
        
        if isinstance(X_selected, pd.DataFrame):
            X_selected = X_selected.values
        
        X_tensor = torch.FloatTensor(X_selected)
        
        # Predict
        classifier.model.eval()
        with torch.no_grad():
            logits = classifier.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confs = torch.max(probs, dim=1)[0]
        
        all_preds.append(preds.numpy())
        all_confs.append(confs.numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_confs)


def batch_predict_transition(detector, features_df, batch_size=1000):
    """
    Run transition predictions in batches.
    
    Returns:
        Tuple of (predictions, transition_probabilities) as numpy arrays
    """
    n_samples = len(features_df)
    all_preds = []
    all_probs = []
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = features_df.iloc[start_idx:end_idx]
        
        # Preprocess
        X_preprocessed = detector.preprocessor.transform(batch)
        X_selected = detector.selector.transform(X_preprocessed)
        
        if isinstance(X_selected, pd.DataFrame):
            X_selected = X_selected.values
        
        X_tensor = torch.FloatTensor(X_selected)
        
        # Predict
        detector.model.eval()
        with torch.no_grad():
            logits = detector.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            trans_prob = probs[:, 1]  # Probability of transition
        
        all_preds.append(preds.numpy())
        all_probs.append(trans_prob.numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_probs)


def analyze_full_dataset(symbol: str, timeframe: str, save_results: bool = False):
    """
    Analyze entire dataset with batch predictions.
    """
    
    print("=" * 80)
    print(f"FULL DATASET ANALYSIS: {symbol} {timeframe}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load models
    print("\n📦 Loading Models...")
    print("-" * 40)
    
    try:
        regime_clf = RegimeClassifier.load(symbol, timeframe, version='v1')
        success("  ✓ Regime classifier loaded")
    except Exception as e:
        error(f"  ✗ Regime classifier failed: {e}")
        regime_clf = None
    
    try:
        vol_clf = VolatilityRegimeClassifier.load(symbol, timeframe, version='v1')
        success("  ✓ Volatility classifier loaded")
    except Exception as e:
        error(f"  ✗ Volatility classifier failed: {e}")
        vol_clf = None
    
    try:
        trans_det = TransitionDetector.load(symbol, timeframe, version='v1')
        success("  ✓ Transition detector loaded")
    except Exception as e:
        error(f"  ✗ Transition detector failed: {e}")
        trans_det = None
    
    # Load feature data
    print("\n📂 Loading Data...")
    print("-" * 40)
    
    feature_path = Path(f"data/processed/{symbol}_{timeframe}_features.parquet")
    if not feature_path.exists():
        error(f"Feature file not found: {feature_path}")
        return None
    
    df = pd.read_parquet(feature_path)
    n_samples = len(df)
    info(f"  Loaded {n_samples:,} samples")
    
    # Initialize results
    results_df = pd.DataFrame(index=df.index)
    
    # Run batch predictions
    print("\n🔮 Running Predictions...")
    print("-" * 40)
    
    # Regime predictions
    if regime_clf is not None:
        print("  Processing regime classification...")
        regime_preds, regime_confs = batch_predict(regime_clf, df)
        regime_map = {0: "RANGING", 1: "TREND_UP", 2: "TREND_DOWN"}
        results_df['regime'] = [regime_map.get(p, "UNKNOWN") for p in regime_preds]
        results_df['regime_confidence'] = regime_confs
        success(f"    ✓ Regime: {n_samples:,} predictions")
    else:
        results_df['regime'] = "UNKNOWN"
        results_df['regime_confidence'] = 0.0
    
    # Volatility predictions
    if vol_clf is not None:
        print("  Processing volatility classification...")
        vol_preds, vol_confs = batch_predict(vol_clf, df)
        vol_map = {0: "LOW", 1: "NORMAL", 2: "HIGH", 3: "CRISIS"}
        results_df['volatility'] = [vol_map.get(p, "UNKNOWN") for p in vol_preds]
        results_df['volatility_confidence'] = vol_confs
        success(f"    ✓ Volatility: {n_samples:,} predictions")
    else:
        results_df['volatility'] = "UNKNOWN"
        results_df['volatility_confidence'] = 0.0
    
    # Transition predictions
    if trans_det is not None:
        print("  Processing transition detection...")
        trans_preds, trans_probs = batch_predict_transition(trans_det, df)
        results_df['is_transitioning'] = trans_preds == 1
        results_df['transition_prob'] = trans_probs
        success(f"    ✓ Transition: {n_samples:,} predictions")
    else:
        results_df['is_transitioning'] = False
        results_df['transition_prob'] = 0.0
    
    # Create state code
    trans_str = results_df['is_transitioning'].map({True: "TRANS", False: "STABLE"})
    results_df['state_code'] = results_df['regime'] + "_" + results_df['volatility'] + "_" + trans_str
    
    # Generate statistics
    print("\n" + "=" * 80)
    print("📊 ANALYSIS RESULTS")
    print("=" * 80)
    
    # Regime distribution
    print("\n🎯 REGIME DISTRIBUTION")
    print("-" * 40)
    regime_counts = results_df['regime'].value_counts()
    regime_pcts = results_df['regime'].value_counts(normalize=True) * 100
    for regime in ['RANGING', 'TREND_UP', 'TREND_DOWN', 'UNKNOWN']:
        if regime in regime_counts.index:
            count = regime_counts[regime]
            pct = regime_pcts[regime]
            bar = '█' * int(pct / 2)
            print(f"  {regime:<12} {count:>6,} ({pct:>5.1f}%) {bar}")
    
    # Volatility distribution
    print("\n📈 VOLATILITY DISTRIBUTION")
    print("-" * 40)
    vol_counts = results_df['volatility'].value_counts()
    vol_pcts = results_df['volatility'].value_counts(normalize=True) * 100
    for vol in ['LOW', 'NORMAL', 'HIGH', 'CRISIS', 'UNKNOWN']:
        if vol in vol_counts.index:
            count = vol_counts[vol]
            pct = vol_pcts[vol]
            bar = '█' * int(pct / 2)
            print(f"  {vol:<10} {count:>6,} ({pct:>5.1f}%) {bar}")
    
    # Transition statistics
    print("\n🔄 TRANSITION STATISTICS")
    print("-" * 40)
    trans_count = results_df['is_transitioning'].sum()
    trans_pct = trans_count / len(results_df) * 100
    print(f"  Transitions Expected:  {trans_count:,} ({trans_pct:.1f}%)")
    print(f"  Avg Transition Prob:   {results_df['transition_prob'].mean():.1%}")
    print(f"  Median Transition Prob:{results_df['transition_prob'].median():.1%}")
    print(f"  Max Transition Prob:   {results_df['transition_prob'].max():.1%}")
    
    # Confidence statistics
    print("\n📏 CONFIDENCE STATISTICS")
    print("-" * 40)
    print(f"  Regime Confidence:")
    print(f"    Mean:   {results_df['regime_confidence'].mean():.1%}")
    print(f"    Median: {results_df['regime_confidence'].median():.1%}")
    print(f"    Std:    {results_df['regime_confidence'].std():.1%}")
    print(f"  Volatility Confidence:")
    print(f"    Mean:   {results_df['volatility_confidence'].mean():.1%}")
    print(f"    Median: {results_df['volatility_confidence'].median():.1%}")
    print(f"    Std:    {results_df['volatility_confidence'].std():.1%}")
    
    # State code distribution (top 10)
    print("\n🏷️ TOP 10 STATE COMBINATIONS")
    print("-" * 40)
    state_counts = results_df['state_code'].value_counts().head(10)
    state_pcts = results_df['state_code'].value_counts(normalize=True).head(10) * 100
    for state_code, count in state_counts.items():
        pct = state_pcts[state_code]
        print(f"  {state_code:<30} {count:>6,} ({pct:>5.1f}%)")
    
    # Regime changes over time
    print("\n📅 REGIME DYNAMICS")
    print("-" * 40)
    regime_changes = (results_df['regime'] != results_df['regime'].shift(1)).sum() - 1
    vol_changes = (results_df['volatility'] != results_df['volatility'].shift(1)).sum() - 1
    
    print(f"  Total regime changes:     {regime_changes:,}")
    print(f"  Avg bars per regime:      {n_samples / max(regime_changes, 1):.1f}")
    print(f"  Total volatility changes: {vol_changes:,}")
    print(f"  Avg bars per vol regime:  {n_samples / max(vol_changes, 1):.1f}")
    
    # Time-based analysis (if datetime index)
    if isinstance(df.index, pd.DatetimeIndex):
        print("\n📆 TIME RANGE")
        print("-" * 40)
        print(f"  Start: {df.index[0]}")
        print(f"  End:   {df.index[-1]}")
        print(f"  Span:  {(df.index[-1] - df.index[0]).days} days")
    
    # Save results if requested
    if save_results:
        output_dir = Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        output_path = output_dir / f"{symbol}_{timeframe}_state_analysis.csv"
        results_df.to_csv(output_path)
        success(f"\n💾 Saved detailed results to {output_path}")
        
        # Save summary
        import json
        summary = {
            'symbol': symbol,
            'timeframe': timeframe,
            'n_samples': int(n_samples),
            'analysis_date': datetime.now().isoformat(),
            'regime_distribution': {k: int(v) for k, v in regime_counts.to_dict().items()},
            'volatility_distribution': {k: int(v) for k, v in vol_counts.to_dict().items()},
            'avg_regime_confidence': float(results_df['regime_confidence'].mean()),
            'avg_volatility_confidence': float(results_df['volatility_confidence'].mean()),
            'avg_transition_prob': float(results_df['transition_prob'].mean()),
            'transition_rate_pct': float(trans_pct),
            'regime_changes': int(regime_changes),
            'volatility_changes': int(vol_changes),
            'top_state_codes': {k: int(v) for k, v in state_counts.to_dict().items()}
        }
        
        summary_path = output_dir / f"{symbol}_{timeframe}_state_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        success(f"💾 Saved summary to {summary_path}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Analyze full dataset with state models")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Currency pair")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe")
    parser.add_argument("--save", action="store_true", help="Save results to CSV")
    
    args = parser.parse_args()
    
    analyze_full_dataset(
        symbol=args.symbol,
        timeframe=args.timeframe,
        save_results=args.save
    )


if __name__ == "__main__":
    main()