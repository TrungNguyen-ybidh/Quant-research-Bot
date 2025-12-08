"""
Test Multiple Holding Periods
=============================
Tests if longer holding periods improve net edge after costs.

Hypothesis: If signal predicts direction correctly, longer holds capture more movement
while paying costs only once.

Usage:
    python scripts/test_holding_periods.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.models.target_builder import TargetBuilder
from src.models.data_splitter import TimeSeriesSplitter
from src.models.cost_model import CostModel
from src.utils.logger import info, success, warning


def load_and_prepare_data(symbol: str, timeframe: str):
    """Load and merge features with raw data."""
    from src.models.baseline_predictor import BaselinePredictor, BaselineConfig
    
    config = BaselineConfig(processed_data_dir="data/processed")
    predictor = BaselinePredictor(config)
    df = predictor.load_data(symbol, timeframe)
    
    return df


def test_single_holding_period(
    df: pd.DataFrame,
    lookahead: int,
    symbol: str,
    timeframe: str
) -> dict:
    """Test a single holding period."""
    
    # Build target for this lookahead
    target_builder = TargetBuilder(
        target_type='binary',
        lookahead=lookahead,
        threshold=0.0001
    )
    target = target_builder.build(df, price_col='close')
    
    # Remove NaN targets
    valid_mask = target.notna()
    df_valid = df[valid_mask].copy()
    y = target[valid_mask]
    prices = df_valid['close'].values
    
    if len(df_valid) < 500:
        return {"error": f"Not enough samples: {len(df_valid)}"}
    
    # Get feature columns
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    feature_cols = [c for c in df_valid.columns if c not in exclude_cols]
    X = df_valid[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Split
    splitter = TimeSeriesSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    split = splitter.split(X, y)
    
    # Train model
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
    
    # Predictions
    test_predictions = model.predict(X_test_scaled)
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Accuracy
    accuracy = np.mean(test_predictions == split.y_test.values)
    
    # Get test prices
    test_start_idx = len(split.X_train) + len(split.X_val)
    test_prices = prices[test_start_idx:test_start_idx + len(test_predictions)]
    
    # Cost analysis
    cost_model = CostModel()
    cost_results = cost_model.analyze(
        predictions=test_predictions,
        prices=test_prices,
        symbol=symbol,
        timeframe=timeframe,
        lookahead=lookahead,
        session="london"
    )
    
    return {
        "lookahead": lookahead,
        "lookahead_hours": lookahead,  # For 1h data
        "test_samples": len(test_predictions),
        "accuracy": accuracy,
        "num_trades": cost_results["num_trades"],
        "trades_per_day": cost_results["trades_per_day"],
        "gross_sharpe": cost_results["gross"]["sharpe"],
        "net_sharpe": cost_results["net"]["sharpe"],
        "gross_return_pct": cost_results["gross"]["total_return_pct"],
        "net_return_pct": cost_results["net"]["total_return_pct"],
        "cost_per_trade_bps": cost_results["costs"]["cost_per_trade_bps"],
        "edge_consumed_pct": cost_results["breakeven"]["edge_consumed_by_costs_pct"],
        "survives_costs": cost_results["survives_costs"],
        "recommendation": cost_results["recommendation"]
    }


def test_multiple_holding_periods(
    symbol: str = "EURUSD",
    timeframe: str = "1h",
    lookaheads: list = [4, 8, 12, 24, 48]
):
    """Test multiple holding periods and compare."""
    
    print("=" * 80)
    print(f"HOLDING PERIOD ANALYSIS: {symbol} {timeframe}")
    print("=" * 80)
    
    # Load data once
    print("\nLoading data...")
    df = load_and_prepare_data(symbol, timeframe)
    print(f"Loaded {len(df)} rows")
    
    # Test each holding period
    results = []
    
    for lookahead in lookaheads:
        print(f"\n{'─' * 80}")
        print(f"Testing lookahead = {lookahead} hours...")
        print(f"{'─' * 80}")
        
        try:
            result = test_single_holding_period(df, lookahead, symbol, timeframe)
            results.append(result)
            
            if "error" not in result:
                status = "✓ SURVIVES" if result["survives_costs"] else "✗ FAILS"
                print(f"  Accuracy: {result['accuracy']:.1%}")
                print(f"  Trades/day: {result['trades_per_day']:.1f}")
                print(f"  Gross Sharpe: {result['gross_sharpe']:.2f}")
                print(f"  Net Sharpe: {result['net_sharpe']:.2f}")
                print(f"  Edge consumed: {result['edge_consumed_pct']:.1f}%")
                print(f"  {status}")
            else:
                print(f"  Error: {result['error']}")
                
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"lookahead": lookahead, "error": str(e)})
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: HOLDING PERIOD COMPARISON")
    print("=" * 80)
    
    valid_results = [r for r in results if "error" not in r]
    
    if valid_results:
        print(f"\n{'Lookahead':<12} {'Accuracy':<10} {'Trades/Day':<12} {'Gross SR':<10} "
              f"{'Net SR':<10} {'Edge Lost':<12} {'Status':<15}")
        print("-" * 90)
        
        for r in valid_results:
            status = "SURVIVES ✓" if r["survives_costs"] else "FAILS ✗"
            print(f"{r['lookahead']:>4}h       "
                  f"{r['accuracy']:<10.1%} "
                  f"{r['trades_per_day']:<12.1f} "
                  f"{r['gross_sharpe']:<10.2f} "
                  f"{r['net_sharpe']:<10.2f} "
                  f"{r['edge_consumed_pct']:<12.1f}% "
                  f"{status:<15}")
        
        # Find best
        survivors = [r for r in valid_results if r["survives_costs"]]
        
        print("\n" + "-" * 80)
        
        if survivors:
            best = max(survivors, key=lambda x: x["net_sharpe"])
            print(f"\n✓ BEST HOLDING PERIOD: {best['lookahead']} hours")
            print(f"  Net Sharpe: {best['net_sharpe']:.2f}")
            print(f"  Accuracy: {best['accuracy']:.1%}")
            print(f"  Trades/day: {best['trades_per_day']:.1f}")
            print(f"  Edge consumed by costs: {best['edge_consumed_pct']:.1f}%")
        else:
            # Find the one closest to surviving
            best_attempt = min(valid_results, key=lambda x: x["edge_consumed_pct"])
            print(f"\n✗ NO HOLDING PERIOD SURVIVES COSTS")
            print(f"\n  Closest to viable: {best_attempt['lookahead']} hours")
            print(f"  Edge consumed: {best_attempt['edge_consumed_pct']:.1f}%")
            print(f"  Need to reduce costs or improve accuracy")
            
            # Calculate required accuracy
            if best_attempt["edge_consumed_pct"] > 0:
                needed_improvement = (best_attempt["edge_consumed_pct"] - 70) / 100
                current_acc = best_attempt["accuracy"]
                print(f"\n  Options:")
                print(f"  1. Improve accuracy from {current_acc:.1%} to ~{current_acc + 0.03:.1%}")
                print(f"  2. Trade only high-confidence predictions (probability threshold)")
                print(f"  3. Add regime filter to avoid unfavorable conditions")
                print(f"  4. Try different timeframe (15m or 1d)")
    
    print("\n" + "=" * 80)
    
    return results


def main():
    results = test_multiple_holding_periods(
        symbol="EURUSD",
        timeframe="1h",
        lookaheads=[4, 8, 12, 24, 48]
    )


if __name__ == "__main__":
    main()