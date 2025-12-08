"""
Alternative Signal Research
===========================
Three alternative approaches to FX market prediction:

1. VOLATILITY PREDICTION - Predict if next period will be high/low volatility
2. REGIME CLASSIFICATION - Identify trending vs ranging markets
3. RISK SIGNALS - Predict when NOT to trade (adverse conditions)

These are often more stable and actionable than direction prediction.

Usage:
    python scripts/test_alternative_signals.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent if Path(__file__).parent.name == 'scripts' else Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.cluster import KMeans

from src.utils.logger import info, success, warning, error


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AlternativeSignalConfig:
    """Configuration for alternative signal testing."""
    
    # Data settings
    timeframe: str = "1h"
    
    # Volatility prediction settings
    vol_lookahead: int = 24  # Predict 24-hour forward volatility
    vol_threshold_percentile: float = 70  # High vol = top 30%
    
    # Regime settings
    regime_window: int = 48  # Window for regime detection
    n_regimes: int = 3  # Number of regimes (trending_up, trending_down, ranging)
    
    # Risk signal settings
    risk_lookahead: int = 24  # Predict 24-hour forward risk
    drawdown_threshold: float = 0.005  # 0.5% drawdown = risky period
    
    # Model settings
    n_estimators: int = 100
    max_depth: int = 5
    random_state: int = 42
    
    # Validation
    train_ratio: float = 0.7
    test_ratio: float = 0.3


# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """Loads data for alternative signal research."""
    
    def __init__(self, processed_dir: str = "data/processed", raw_dir: str = "data/raw/clock"):
        self.processed_dir = Path(processed_dir)
        self.raw_dir = Path(raw_dir)
    
    def load(self, symbol: str, timeframe: str = "1h") -> Optional[pd.DataFrame]:
        """Load and merge features with raw OHLCV."""
        features_path = self.processed_dir / f"{symbol}_{timeframe}_features.parquet"
        raw_path = self.raw_dir / f"{symbol}_{timeframe}.parquet"
        
        if not features_path.exists() or not raw_path.exists():
            warning(f"Data not found for {symbol} {timeframe}")
            return None
        
        features_df = pd.read_parquet(features_path)
        raw_df = pd.read_parquet(raw_path)
        
        min_len = min(len(features_df), len(raw_df))
        features_df = features_df.iloc[:min_len]
        raw_df = raw_df.iloc[:min_len]
        
        df = raw_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        for col in features_df.columns:
            df[col] = features_df[col].values
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns."""
        exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'returns']
        return [c for c in df.columns if c not in exclude]


# =============================================================================
# 1. VOLATILITY PREDICTION
# =============================================================================

class VolatilityPredictor:
    """
    Predicts whether future volatility will be HIGH or LOW.
    
    Why this works better than direction:
    - Volatility clusters (high vol follows high vol)
    - Your volatility features (Hurst, RV, Parkinson) are designed for this
    - Actionable: size positions based on expected vol
    """
    
    def __init__(self, config: AlternativeSignalConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
    
    def build_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Build volatility target.
        
        Target = 1 if forward realized vol > threshold, else 0
        """
        # Calculate forward realized volatility
        returns = df['close'].pct_change()
        forward_vol = returns.rolling(self.config.vol_lookahead).std().shift(-self.config.vol_lookahead)
        
        # Annualize
        forward_vol_ann = forward_vol * np.sqrt(252 * 24)  # Hourly data
        
        # Threshold based on historical percentile
        threshold = forward_vol_ann.quantile(self.config.vol_threshold_percentile / 100)
        
        # Binary target: 1 = high vol, 0 = low vol
        target = (forward_vol_ann > threshold).astype(int)
        
        return target, forward_vol_ann, threshold
    
    def train_and_evaluate(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Train and evaluate volatility prediction model."""
        
        info("Building volatility target...")
        target, actual_vol, threshold = self.build_target(df)
        
        # Prepare data
        valid_mask = target.notna()
        X = df.loc[valid_mask, feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = target[valid_mask]
        actual = actual_vol[valid_mask]
        
        # Split
        split_idx = int(len(X) * self.config.train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        actual_test = actual.iloc[split_idx:]
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train
        info("Training volatility model...")
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        predictions = self.model.predict(X_test_scaled)
        probas = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probas)
        
        # Feature importance
        importance = dict(zip(feature_cols, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Practical value: when we predict high vol, is actual vol higher?
        pred_high_vol = predictions == 1
        pred_low_vol = predictions == 0
        
        avg_vol_when_pred_high = actual_test[pred_high_vol].mean() if pred_high_vol.sum() > 0 else 0
        avg_vol_when_pred_low = actual_test[pred_low_vol].mean() if pred_low_vol.sum() > 0 else 0
        
        vol_separation = avg_vol_when_pred_high / avg_vol_when_pred_low if avg_vol_when_pred_low > 0 else 0
        
        return {
            "accuracy": accuracy,
            "auc": auc,
            "threshold": threshold,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "class_balance": y_test.mean(),
            "avg_vol_pred_high": avg_vol_when_pred_high,
            "avg_vol_pred_low": avg_vol_when_pred_low,
            "vol_separation_ratio": vol_separation,
            "top_features": top_features,
            "is_useful": accuracy > 0.55 and vol_separation > 1.3
        }


# =============================================================================
# 2. REGIME CLASSIFICATION
# =============================================================================

class RegimeClassifier:
    """
    Classifies market regimes: TRENDING_UP, TRENDING_DOWN, RANGING
    
    Why this works better than direction:
    - Different strategies work in different regimes
    - Regimes are more persistent than price direction
    - Actionable: use trend-following in trends, mean-reversion in ranges
    """
    
    def __init__(self, config: AlternativeSignalConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
    
    def build_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Build regime labels using multiple indicators.
        
        Regimes:
        - 0: RANGING (low directional movement)
        - 1: TRENDING_UP (sustained upward movement)
        - 2: TRENDING_DOWN (sustained downward movement)
        """
        window = self.config.regime_window
        
        # Calculate regime indicators
        returns = df['close'].pct_change()
        
        # Cumulative return over window
        cum_return = returns.rolling(window).sum()
        
        # Volatility over window
        volatility = returns.rolling(window).std()
        
        # ADX-like directional strength (if available, else calculate)
        if 'adx' in df.columns:
            adx = df['adx']
        else:
            # Simple directional strength proxy
            high_low_range = (df['high'] - df['low']).rolling(window).mean()
            close_range = df['close'].diff().abs().rolling(window).mean()
            adx = (close_range / high_low_range).fillna(0)
        
        # Hurst exponent (if available)
        if 'hurst_exponent' in df.columns:
            hurst = df['hurst_exponent']
        else:
            hurst = pd.Series(0.5, index=df.index)
        
        # Define regimes
        # Trending: strong directional move + high ADX + Hurst > 0.5
        # Ranging: low directional move OR low ADX OR Hurst < 0.5
        
        regime = pd.Series(0, index=df.index)  # Default: ranging
        
        # Thresholds
        return_threshold = cum_return.quantile(0.7)  # Top 30% moves
        adx_threshold = adx.quantile(0.6)  # Top 40% directional
        
        # Trending up
        trending_up = (cum_return > return_threshold) & (adx > adx_threshold)
        regime[trending_up] = 1
        
        # Trending down
        trending_down = (cum_return < -return_threshold) & (adx > adx_threshold)
        regime[trending_down] = 2
        
        return regime
    
    def build_forward_regime(self, df: pd.DataFrame) -> pd.Series:
        """Build forward-looking regime for prediction."""
        current_regime = self.build_target(df)
        # Shift to get future regime
        forward_regime = current_regime.shift(-self.config.regime_window)
        return forward_regime
    
    def train_and_evaluate(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Train and evaluate regime classification model."""
        
        info("Building regime targets...")
        target = self.build_forward_regime(df)
        
        # Prepare data
        valid_mask = target.notna()
        X = df.loc[valid_mask, feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = target[valid_mask].astype(int)
        
        # Split
        split_idx = int(len(X) * self.config.train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train
        info("Training regime model...")
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        predictions = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, predictions)
        
        # Per-class accuracy
        regime_names = ['RANGING', 'TREND_UP', 'TREND_DOWN']
        class_accuracies = {}
        for i, name in enumerate(regime_names):
            mask = y_test == i
            if mask.sum() > 0:
                class_acc = (predictions[mask] == y_test.values[mask]).mean()
                class_accuracies[name] = class_acc
        
        # Feature importance
        importance = dict(zip(feature_cols, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Class distribution
        class_dist = y_test.value_counts(normalize=True).to_dict()
        
        return {
            "accuracy": accuracy,
            "class_accuracies": class_accuracies,
            "class_distribution": {regime_names[int(k)]: v for k, v in class_dist.items()},
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "top_features": top_features,
            "is_useful": accuracy > 0.45 and len(class_accuracies) == 3  # Better than 33% random
        }


# =============================================================================
# 3. RISK SIGNALS (When NOT to Trade)
# =============================================================================

class RiskSignalPredictor:
    """
    Predicts periods of HIGH RISK - when NOT to trade.
    
    Why this works better than direction:
    - Avoiding losses is easier than picking winners
    - Risk events cluster (volatility clustering)
    - Actionable: reduce position size or stay flat
    """
    
    def __init__(self, config: AlternativeSignalConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
    
    def build_target(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Build risk target.
        
        Target = 1 if forward period has significant drawdown, else 0
        """
        prices = df['close']
        lookahead = self.config.risk_lookahead
        
        # Calculate forward max drawdown
        forward_drawdowns = pd.Series(0.0, index=df.index)
        
        for i in range(len(df) - lookahead):
            future_prices = prices.iloc[i:i+lookahead]
            peak = future_prices.expanding().max()
            drawdown = (future_prices - peak) / peak
            max_dd = drawdown.min()  # Most negative = worst drawdown
            forward_drawdowns.iloc[i] = abs(max_dd)
        
        # Binary target: 1 = risky period (large drawdown ahead)
        target = (forward_drawdowns > self.config.drawdown_threshold).astype(int)
        
        return target, forward_drawdowns
    
    def train_and_evaluate(self, df: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Train and evaluate risk signal model."""
        
        info("Building risk targets...")
        target, actual_dd = self.build_target(df)
        
        # Prepare data
        valid_mask = (target.notna()) & (target.index.isin(df.index))
        X = df.loc[valid_mask, feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        y = target[valid_mask]
        actual = actual_dd[valid_mask]
        
        # Split
        split_idx = int(len(X) * self.config.train_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        actual_test = actual.iloc[split_idx:]
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train
        info("Training risk model...")
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        predictions = self.model.predict(X_test_scaled)
        probas = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probas) if len(np.unique(y_test)) > 1 else 0.5
        
        # Key metrics for risk prediction
        # True Positive Rate (catching risky periods)
        risky_periods = y_test == 1
        if risky_periods.sum() > 0:
            true_positive_rate = (predictions[risky_periods] == 1).mean()
        else:
            true_positive_rate = 0
        
        # False Positive Rate (false alarms)
        safe_periods = y_test == 0
        if safe_periods.sum() > 0:
            false_positive_rate = (predictions[safe_periods] == 1).mean()
        else:
            false_positive_rate = 0
        
        # Practical value: average drawdown avoided
        pred_risky = predictions == 1
        pred_safe = predictions == 0
        
        avg_dd_when_pred_risky = actual_test[pred_risky].mean() if pred_risky.sum() > 0 else 0
        avg_dd_when_pred_safe = actual_test[pred_safe].mean() if pred_safe.sum() > 0 else 0
        
        dd_reduction = avg_dd_when_pred_risky - avg_dd_when_pred_safe
        
        # Feature importance
        importance = dict(zip(feature_cols, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "accuracy": accuracy,
            "auc": auc,
            "true_positive_rate": true_positive_rate,  # Catching risky periods
            "false_positive_rate": false_positive_rate,  # False alarms
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "risky_period_ratio": y_test.mean(),
            "avg_dd_when_pred_risky": avg_dd_when_pred_risky * 100,  # As percentage
            "avg_dd_when_pred_safe": avg_dd_when_pred_safe * 100,
            "dd_reduction_pct": dd_reduction * 100,
            "top_features": top_features,
            "is_useful": true_positive_rate > 0.5 and dd_reduction > 0.001
        }


# =============================================================================
# COMBINED SIGNAL SYSTEM
# =============================================================================

class AlternativeSignalResearcher:
    """
    Researches all three alternative signal approaches.
    """
    
    def __init__(self, config: Optional[AlternativeSignalConfig] = None):
        self.config = config or AlternativeSignalConfig()
        self.loader = DataLoader()
        
        self.vol_predictor = VolatilityPredictor(self.config)
        self.regime_classifier = RegimeClassifier(self.config)
        self.risk_predictor = RiskSignalPredictor(self.config)
    
    def research(self, symbol: str = "EURUSD") -> Dict:
        """Run all alternative signal research."""
        
        print("=" * 80)
        print(f"ALTERNATIVE SIGNAL RESEARCH: {symbol}")
        print("=" * 80)
        
        # Load data
        df = self.loader.load(symbol, self.config.timeframe)
        if df is None:
            return {"error": "Data not found"}
        
        feature_cols = self.loader.get_feature_columns(df)
        print(f"\nData: {len(df)} rows, {len(feature_cols)} features")
        
        results = {
            "symbol": symbol,
            "timeframe": self.config.timeframe,
            "timestamp": datetime.now().isoformat(),
            "data_rows": len(df),
            "features": len(feature_cols)
        }
        
        # 1. Volatility Prediction
        print("\n" + "─" * 80)
        print("1. VOLATILITY PREDICTION")
        print("   Predict: Will next 24h be HIGH or LOW volatility?")
        print("─" * 80)
        
        vol_results = self.vol_predictor.train_and_evaluate(df, feature_cols)
        results["volatility"] = vol_results
        
        status = "✓ USEFUL" if vol_results["is_useful"] else "✗ NOT USEFUL"
        print(f"\n  Accuracy: {vol_results['accuracy']:.1%}")
        print(f"  AUC: {vol_results['auc']:.3f}")
        print(f"  Vol when predict HIGH: {vol_results['avg_vol_pred_high']*100:.2f}%")
        print(f"  Vol when predict LOW: {vol_results['avg_vol_pred_low']*100:.2f}%")
        print(f"  Separation Ratio: {vol_results['vol_separation_ratio']:.2f}x")
        print(f"  {status}")
        
        print("\n  Top 5 Features:")
        for i, (name, imp) in enumerate(vol_results["top_features"][:5], 1):
            print(f"    {i}. {name}: {imp:.4f}")
        
        # 2. Regime Classification
        print("\n" + "─" * 80)
        print("2. REGIME CLASSIFICATION")
        print("   Predict: TRENDING_UP, TRENDING_DOWN, or RANGING?")
        print("─" * 80)
        
        regime_results = self.regime_classifier.train_and_evaluate(df, feature_cols)
        results["regime"] = regime_results
        
        status = "✓ USEFUL" if regime_results["is_useful"] else "✗ NOT USEFUL"
        print(f"\n  Overall Accuracy: {regime_results['accuracy']:.1%}")
        print(f"  (Random baseline: 33.3%)")
        
        print("\n  Per-Class Accuracy:")
        for regime, acc in regime_results["class_accuracies"].items():
            print(f"    {regime}: {acc:.1%}")
        
        print("\n  Class Distribution:")
        for regime, pct in regime_results["class_distribution"].items():
            print(f"    {regime}: {pct:.1%}")
        
        print(f"\n  {status}")
        
        print("\n  Top 5 Features:")
        for i, (name, imp) in enumerate(regime_results["top_features"][:5], 1):
            print(f"    {i}. {name}: {imp:.4f}")
        
        # 3. Risk Signals
        print("\n" + "─" * 80)
        print("3. RISK SIGNALS (When NOT to Trade)")
        print("   Predict: Will next 24h have significant drawdown?")
        print("─" * 80)
        
        risk_results = self.risk_predictor.train_and_evaluate(df, feature_cols)
        results["risk"] = risk_results
        
        status = "✓ USEFUL" if risk_results["is_useful"] else "✗ NOT USEFUL"
        print(f"\n  Accuracy: {risk_results['accuracy']:.1%}")
        print(f"  AUC: {risk_results['auc']:.3f}")
        print(f"  True Positive Rate: {risk_results['true_positive_rate']:.1%} (catching risky periods)")
        print(f"  False Positive Rate: {risk_results['false_positive_rate']:.1%} (false alarms)")
        print(f"  Avg Drawdown when RISKY: {risk_results['avg_dd_when_pred_risky']:.2f}%")
        print(f"  Avg Drawdown when SAFE: {risk_results['avg_dd_when_pred_safe']:.2f}%")
        print(f"  {status}")
        
        print("\n  Top 5 Features:")
        for i, (name, imp) in enumerate(risk_results["top_features"][:5], 1):
            print(f"    {i}. {name}: {imp:.4f}")
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        
        useful_signals = []
        if vol_results["is_useful"]:
            useful_signals.append("Volatility Prediction")
        if regime_results["is_useful"]:
            useful_signals.append("Regime Classification")
        if risk_results["is_useful"]:
            useful_signals.append("Risk Signals")
        
        results["summary"] = {
            "useful_signals": useful_signals,
            "n_useful": len(useful_signals),
            "recommendation": self._get_recommendation(vol_results, regime_results, risk_results)
        }
        
        print(f"\n  Useful Signals: {len(useful_signals)}/3")
        for sig in useful_signals:
            print(f"    ✓ {sig}")
        
        not_useful = [s for s in ["Volatility", "Regime", "Risk"] 
                      if s + " Prediction" not in useful_signals and s + " Classification" not in useful_signals and s + " Signals" not in useful_signals]
        
        print(f"\n  RECOMMENDATION: {results['summary']['recommendation']}")
        
        print("\n" + "=" * 80)
        
        return results
    
    def _get_recommendation(self, vol: Dict, regime: Dict, risk: Dict) -> str:
        """Generate recommendation based on results."""
        
        useful = []
        if vol["is_useful"]:
            useful.append("vol")
        if regime["is_useful"]:
            useful.append("regime")
        if risk["is_useful"]:
            useful.append("risk")
        
        if len(useful) >= 2:
            return "STRONG - Multiple alternative signals work. Build combined system."
        elif len(useful) == 1:
            if "vol" in useful:
                return "MODERATE - Volatility prediction works. Use for position sizing."
            elif "regime" in useful:
                return "MODERATE - Regime detection works. Use for strategy selection."
            else:
                return "MODERATE - Risk signals work. Use for trade filtering."
        else:
            return "WEAK - No strong alternative signals. Consider different features or timeframe."
    
    def save_results(self, results: Dict, output_dir: str = "outputs/alternative_signals"):
        """Save results to JSON."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Clean for JSON
        clean_results = self._clean_for_json(results)
        
        filename = f"{results['symbol']}_alternative_signals.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            import json
            json.dump(clean_results, f, indent=2, default=str)
        
        success(f"Results saved to {filepath}")
    
    def _clean_for_json(self, obj):
        """Clean object for JSON."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_json(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj


# =============================================================================
# CROSS-PAIR VALIDATION
# =============================================================================

def validate_across_pairs(pairs: List[str] = None) -> Dict:
    """Validate alternative signals across multiple pairs."""
    
    if pairs is None:
        pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    
    print("=" * 80)
    print("CROSS-PAIR ALTERNATIVE SIGNAL VALIDATION")
    print("=" * 80)
    
    researcher = AlternativeSignalResearcher()
    all_results = {}
    
    for symbol in pairs:
        print(f"\n{'─' * 80}")
        print(f"Testing {symbol}...")
        print(f"{'─' * 80}")
        
        try:
            results = researcher.research(symbol)
            all_results[symbol] = results
        except Exception as e:
            print(f"  Error: {e}")
            all_results[symbol] = {"error": str(e)}
    
    # Summary across pairs
    print("\n" + "=" * 80)
    print("CROSS-PAIR SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Pair':<10} {'Volatility':<15} {'Regime':<15} {'Risk':<15}")
    print("-" * 55)
    
    vol_pass = 0
    regime_pass = 0
    risk_pass = 0
    
    for symbol, res in all_results.items():
        if "error" in res:
            print(f"{symbol:<10} ERROR")
            continue
        
        vol_status = "✓ USEFUL" if res["volatility"]["is_useful"] else "✗"
        regime_status = "✓ USEFUL" if res["regime"]["is_useful"] else "✗"
        risk_status = "✓ USEFUL" if res["risk"]["is_useful"] else "✗"
        
        if res["volatility"]["is_useful"]:
            vol_pass += 1
        if res["regime"]["is_useful"]:
            regime_pass += 1
        if res["risk"]["is_useful"]:
            risk_pass += 1
        
        print(f"{symbol:<10} {vol_status:<15} {regime_status:<15} {risk_status:<15}")
    
    n_pairs = len([r for r in all_results.values() if "error" not in r])
    
    print("-" * 55)
    print(f"{'TOTAL':<10} {vol_pass}/{n_pairs:<13} {regime_pass}/{n_pairs:<13} {risk_pass}/{n_pairs:<13}")
    
    print("\n" + "=" * 80)
    
    # Overall recommendation
    if vol_pass >= n_pairs * 0.6:
        print("✓ VOLATILITY PREDICTION generalizes across pairs!")
    if regime_pass >= n_pairs * 0.6:
        print("✓ REGIME CLASSIFICATION generalizes across pairs!")
    if risk_pass >= n_pairs * 0.6:
        print("✓ RISK SIGNALS generalize across pairs!")
    
    print("=" * 80)
    
    return all_results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_alternative_research(symbol: str = "EURUSD") -> Dict:
    """Run alternative signal research."""
    researcher = AlternativeSignalResearcher()
    results = researcher.research(symbol)
    researcher.save_results(results)
    return results


if __name__ == "__main__":
    # Single pair research
    results = run_alternative_research("EURUSD")
    
    # Optional: Cross-pair validation
    # all_results = validate_across_pairs()