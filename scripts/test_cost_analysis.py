"""
Test Cost Analysis
==================
Runs baseline prediction + cost analysis together.

Usage:
    python scripts/test_cost_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from src.models.baseline_predictor import BaselinePredictor, BaselineConfig
from src.models.cost_model import CostModel, get_cost_summary_all_pairs


def run_full_analysis(symbol: str = "EURUSD", timeframe: str = "1h"):
    """Run baseline + cost analysis."""
    
    print("=" * 70)
    print(f"FULL ANALYSIS: {symbol} {timeframe}")
    print("=" * 70)
    
    # Step 1: Run baseline prediction
    print("\n[STEP 1] Running baseline prediction...")
    print("-" * 70)
    
    baseline_config = BaselineConfig(
        lookahead_bars=4,
        target_threshold=0.0001,
        processed_data_dir="data/processed"
    )
    
    predictor = BaselinePredictor(baseline_config)
    baseline_results = predictor.run_baseline_test(symbol, timeframe, save_results=False)
    
    # Check if signal exists
    signal_assessment = baseline_results["signal_assessment"]
    
    if not signal_assessment["has_signal"]:
        print("\n" + "=" * 70)
        print("ANALYSIS STOPPED: No predictive signal detected")
        print("Recommendation: Revisit feature engineering before cost analysis")
        print("=" * 70)
        return None
    
    # Step 2: Get predictions from best model
    print("\n[STEP 2] Extracting predictions from best model...")
    print("-" * 70)
    
    best_model = signal_assessment["best_model"]
    print(f"Best model: {best_model}")
    print(f"OOS Accuracy: {signal_assessment['best_oos_accuracy']:.3f}")
    print(f"OOS AUC: {signal_assessment['best_oos_auc']:.3f}")
    
    # Re-run prediction to get actual predictions
    # (In production, you'd save these from baseline run)
    df = predictor.load_data(symbol, timeframe)
    
    # Build target
    from src.models.target_builder import TargetBuilder
    target_builder = TargetBuilder(
        target_type='binary',
        lookahead=baseline_config.lookahead_bars,
        threshold=baseline_config.target_threshold
    )
    df['target'] = target_builder.build(df, price_col='close')
    
    # Split
    from src.models.data_splitter import TimeSeriesSplitter
    splitter = TimeSeriesSplitter(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # Get feature columns (exclude non-features)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Handle nulls
    df_clean = df.dropna(subset=['target'])
    X = df_clean[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df_clean['target']
    prices = df_clean['close'].values
    
    # Split
    split = splitter.split(X, y)
    
    # Train best model on train+val, predict on test
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Combine train + val for final model
    X_trainval = pd.concat([split.X_train, split.X_val])
    y_trainval = pd.concat([split.y_train, split.y_val])
    
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    X_test_scaled = scaler.transform(split.X_test)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_trainval_scaled, y_trainval)
    
    # Get test predictions
    test_predictions = model.predict(X_test_scaled)
    
    # Get test prices (aligned with predictions)
    test_start_idx = len(split.X_train) + len(split.X_val)
    test_prices = prices[test_start_idx:test_start_idx + len(test_predictions)]
    
    print(f"Test predictions: {len(test_predictions)}")
    print(f"Test prices: {len(test_prices)}")
    
    # Step 3: Run cost analysis
    print("\n[STEP 3] Running cost analysis...")
    print("-" * 70)
    
    cost_model = CostModel()
    cost_results = cost_model.analyze(
        predictions=test_predictions,
        prices=test_prices,
        symbol=symbol,
        timeframe=timeframe,
        lookahead=baseline_config.lookahead_bars,
        session="london"
    )
    
    cost_model.print_report(cost_results)
    cost_model.save_results(cost_results)
    
    # Step 4: Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    print(f"\n  Signal Detected:     {signal_assessment['has_signal']}")
    print(f"  Signal Strength:     {signal_assessment['signal_strength']}")
    print(f"  OOS Accuracy:        {signal_assessment['best_oos_accuracy']:.3f}")
    print(f"  Gross Sharpe:        {cost_results['gross']['sharpe']:.3f}")
    print(f"  Net Sharpe:          {cost_results['net']['sharpe']:.3f}")
    print(f"  Survives Costs:      {cost_results['survives_costs']}")
    
    print(f"\n  RECOMMENDATION: {cost_results['recommendation']}")
    print("=" * 70)
    
    return {
        "baseline": baseline_results,
        "costs": cost_results
    }


def main():
    # Show cost summary first
    print("\nCost Summary (All Pairs):")
    print("-" * 50)
    cost_df = get_cost_summary_all_pairs()
    print(cost_df.to_string(index=False))
    print()
    
    # Run full analysis
    results = run_full_analysis("EURUSD", "1h")
    
    if results:
        # Quick summary
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        
        if results["costs"]["survives_costs"]:
            print("""
    ✓ Signal survives costs! Next steps:
    
    1. Run temporal_stability.py - Check if signal persists across time
    2. Run cross_validation.py - Check if signal works on other pairs
    3. Run factor_attribution.py - Verify it's true alpha, not factor exposure
    4. Paper trade for 1-3 months before live capital
            """)
        else:
            print("""
    ✗ Signal does not survive costs. Options:
    
    1. Reduce trading frequency (longer holding period)
    2. Focus on lower-cost pairs (EURUSD, USDJPY)
    3. Improve signal accuracy (need >55% for thin edges)
    4. Combine with regime filter (only trade favorable conditions)
            """)


if __name__ == "__main__":
    main()