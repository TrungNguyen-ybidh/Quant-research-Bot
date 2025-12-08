"""
Module 1: Baseline Predictor
============================
Purpose: Answer the fundamental question - "Can my features predict anything?"

This module:
1. Loads processed feature data
2. Builds a binary target (4-bar forward return)
3. Trains simple models (Logistic Regression, Random Forest, XGBoost)
4. Evaluates on out-of-sample data with proper time-series split
5. Reports whether predictive signal exists

Usage:
    from src.models.baseline_predictor import BaselinePredictor
    
    predictor = BaselinePredictor()
    results = predictor.run_baseline_test("EURUSD", "1h")
    
    if results["has_signal"]:
        print("Signal detected! Proceed to cost analysis.")
    else:
        print("No signal. Revisit feature engineering.")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import json

# ML imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# Try importing XGBoost (optional but recommended)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using Random Forest as alternative.")

warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BaselineConfig:
    """Configuration for baseline prediction tests."""
    
    # Target configuration
    lookahead_bars: int = 4  # Predict 4 bars ahead
    target_threshold: float = 0.0001  # 0.01% minimum move (1 pip for most pairs)
    
    # Train/test split
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Model parameters
    random_state: int = 42
    n_estimators: int = 100  # For RF and XGBoost
    max_depth: int = 5  # Prevent overfitting
    
    # Thresholds for signal detection
    min_accuracy: float = 0.51  # Better than random
    min_auc: float = 0.52  # Better than random
    max_overfit_ratio: float = 0.70  # OOS should be within 70% of IS
    
    # Features to exclude (known to have issues)
    exclude_features: List[str] = field(default_factory=lambda: [
        'correlation_dxy', 'correlation_spx', 'correlation_gold',
        'correlation_vix', 'correlation_oil', 'relative_strength_vs_dxy',
        'cross_asset_momentum', 'risk_sentiment',  # All null in your data
        'timestamp', 'open', 'high', 'low', 'close', 'volume',  # Raw price data
        'target', 'returns'  # Target columns
    ])
    
    # Paths
    processed_data_dir: str = "data/processed"
    output_dir: str = "outputs/baseline"


# =============================================================================
# TARGET BUILDER
# =============================================================================

class TargetBuilder:
    """Builds prediction targets from price data."""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
    
    def build_binary_target(self, df: pd.DataFrame, price_col: str = 'close') -> pd.Series:
        """
        Build binary target: 1 if price goes up by threshold, 0 if down.
        
        Logic:
            future_return = price[t + lookahead] / price[t] - 1
            target = 1 if future_return > threshold else 0
        """
        # Calculate forward return
        future_price = df[price_col].shift(-self.config.lookahead_bars)
        forward_return = (future_price / df[price_col]) - 1
        
        # Binary classification
        # 1 = up move, 0 = down move
        # Using threshold to filter noise
        target = (forward_return > self.config.target_threshold).astype(int)
        
        # Handle edge case: exactly at threshold (rare)
        # Assign to the direction of the move
        at_threshold = np.abs(forward_return) <= self.config.target_threshold
        target[at_threshold] = (forward_return[at_threshold] >= 0).astype(int)
        
        return target
    
    def get_target_stats(self, target: pd.Series) -> Dict:
        """Get statistics about the target distribution."""
        valid_target = target.dropna()
        
        return {
            "total_samples": len(valid_target),
            "positive_class": int(valid_target.sum()),
            "negative_class": int(len(valid_target) - valid_target.sum()),
            "positive_ratio": float(valid_target.mean()),
            "negative_ratio": float(1 - valid_target.mean()),
            "is_balanced": 0.4 <= valid_target.mean() <= 0.6
        }


# =============================================================================
# TIME SERIES SPLITTER
# =============================================================================

class TimeSeriesSplitter:
    """Handles chronological train/val/test splits."""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
    
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically.
        
        Returns:
            train_df, val_df, test_df
        """
        n = len(df)
        
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        return train_df, val_df, test_df
    
    def get_split_info(self, train_df, val_df, test_df) -> Dict:
        """Get information about the splits."""
        return {
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "train_ratio": len(train_df) / (len(train_df) + len(val_df) + len(test_df)),
            "val_ratio": len(val_df) / (len(train_df) + len(val_df) + len(test_df)),
            "test_ratio": len(test_df) / (len(train_df) + len(val_df) + len(test_df)),
        }


# =============================================================================
# FEATURE PROCESSOR
# =============================================================================

class FeatureProcessor:
    """Prepares features for ML models."""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify valid feature columns."""
        all_cols = df.columns.tolist()
        
        # Exclude known problematic columns
        feature_cols = [
            col for col in all_cols 
            if col not in self.config.exclude_features
            and not col.startswith('_')  # Internal columns
        ]
        
        # Exclude columns with too many nulls (>10%)
        valid_cols = []
        for col in feature_cols:
            null_ratio = df[col].isnull().sum() / len(df)
            if null_ratio < 0.10:
                valid_cols.append(col)
        
        return valid_cols
    
    def prepare_features(
        self, 
        df: pd.DataFrame, 
        target_col: str = 'target',
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and target vector.
        
        Returns:
            X, y, feature_names
        """
        # Get feature columns on first call
        if not self.feature_columns:
            self.feature_columns = self.get_feature_columns(df)
        
        # Extract features
        X = df[self.feature_columns].copy()
        y = df[target_col].copy()
        
        # Handle remaining nulls
        X = X.fillna(0)  # Simple imputation for now
        
        # Handle infinities
        X = X.replace([np.inf, -np.inf], 0)
        
        # Drop rows where target is null
        valid_idx = ~y.isnull()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y.values, self.feature_columns


# =============================================================================
# MODEL TRAINER
# =============================================================================

class ModelTrainer:
    """Trains and evaluates baseline models."""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.models = self._initialize_models()
    
    def _initialize_models(self) -> Dict:
        """Initialize all models to test."""
        models = {
            "logistic_regression": LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
        }
        
        if HAS_XGBOOST:
            models["xgboost"] = XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
        
        return models
    
    def train_and_evaluate(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Dict]:
        """
        Train all models and evaluate on validation and test sets.
        
        Returns:
            Dictionary with results for each model
        """
        results = {}
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            # Probabilities for AUC
            y_train_prob = model.predict_proba(X_train)[:, 1]
            y_val_prob = model.predict_proba(X_val)[:, 1]
            y_test_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            results[name] = {
                "train": self._calculate_metrics(y_train, y_train_pred, y_train_prob),
                "val": self._calculate_metrics(y_val, y_val_pred, y_val_prob),
                "test": self._calculate_metrics(y_test, y_test_pred, y_test_prob),
                "feature_importance": self._get_feature_importance(model, feature_names),
                "overfit_ratio": None  # Calculated below
            }
            
            # Calculate overfit ratio (OOS accuracy / IS accuracy)
            is_acc = results[name]["train"]["accuracy"]
            oos_acc = results[name]["test"]["accuracy"]
            results[name]["overfit_ratio"] = oos_acc / is_acc if is_acc > 0 else 0
        
        return results
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """Calculate classification metrics."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "auc_roc": float(roc_auc_score(y_true, y_prob)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
    
    def _get_feature_importance(
        self, 
        model, 
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Extract feature importance from model."""
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_[0])
        else:
            return {}
        
        # Create sorted dictionary
        importance_dict = dict(zip(feature_names, importances))
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance


# =============================================================================
# SIGNAL DETECTOR
# =============================================================================

class SignalDetector:
    """Determines if predictive signal exists based on results."""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
    
    def evaluate_results(self, results: Dict[str, Dict]) -> Dict:
        """
        Evaluate model results and determine if signal exists.
        
        Returns:
            Dictionary with signal assessment
        """
        assessment = {
            "has_signal": False,
            "best_model": None,
            "best_oos_accuracy": 0,
            "best_oos_auc": 0,
            "signal_strength": "none",
            "model_assessments": {},
            "recommendation": ""
        }
        
        for model_name, model_results in results.items():
            test_metrics = model_results["test"]
            
            # Assess this model
            model_assessment = {
                "accuracy": test_metrics["accuracy"],
                "auc": test_metrics["auc_roc"],
                "overfit_ratio": model_results["overfit_ratio"],
                "passes_accuracy": test_metrics["accuracy"] > self.config.min_accuracy,
                "passes_auc": test_metrics["auc_roc"] > self.config.min_auc,
                "passes_overfit": model_results["overfit_ratio"] >= self.config.max_overfit_ratio,
                "is_valid": False
            }
            
            # Model is valid if passes all checks
            model_assessment["is_valid"] = (
                model_assessment["passes_accuracy"] and
                model_assessment["passes_auc"] and
                model_assessment["passes_overfit"]
            )
            
            assessment["model_assessments"][model_name] = model_assessment
            
            # Track best model
            if test_metrics["accuracy"] > assessment["best_oos_accuracy"]:
                assessment["best_oos_accuracy"] = test_metrics["accuracy"]
                assessment["best_oos_auc"] = test_metrics["auc_roc"]
                assessment["best_model"] = model_name
        
        # Determine overall signal
        valid_models = [
            name for name, assess in assessment["model_assessments"].items()
            if assess["is_valid"]
        ]
        
        if len(valid_models) >= 2:
            assessment["has_signal"] = True
            assessment["signal_strength"] = "strong"
            assessment["recommendation"] = "PROCEED - Multiple models show valid signal"
        elif len(valid_models) == 1:
            assessment["has_signal"] = True
            assessment["signal_strength"] = "moderate"
            assessment["recommendation"] = "PROCEED WITH CAUTION - Single model shows signal"
        elif assessment["best_oos_accuracy"] > 0.51:
            assessment["has_signal"] = False
            assessment["signal_strength"] = "weak"
            assessment["recommendation"] = "INVESTIGATE - Weak signal, may need feature refinement"
        else:
            assessment["has_signal"] = False
            assessment["signal_strength"] = "none"
            assessment["recommendation"] = "STOP - No predictive signal detected"
        
        return assessment


# =============================================================================
# MAIN PREDICTOR CLASS
# =============================================================================

class BaselinePredictor:
    """
    Main class for running baseline prediction tests.
    
    This is the entry point for Module 1 of the ML pipeline.
    """
    
    def __init__(self, config: Optional[BaselineConfig] = None):
        self.config = config or BaselineConfig()
        self.target_builder = TargetBuilder(self.config)
        self.splitter = TimeSeriesSplitter(self.config)
        self.feature_processor = FeatureProcessor(self.config)
        self.trainer = ModelTrainer(self.config)
        self.signal_detector = SignalDetector(self.config)
    
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load processed feature data and merge with raw OHLCV."""
        # Map timeframe formats
        tf_map = {
            "1 hour": "1h", "1hour": "1h", "1h": "1h",
            "15 mins": "15m", "15m": "15m",
            "5 mins": "5m", "5m": "5m",
            "1 min": "1m", "1m": "1m",
            "1 day": "1d", "1d": "1d"
        }
        tf_normalized = tf_map.get(timeframe, timeframe.replace(" ", ""))
        
        # Paths
        features_path = Path(self.config.processed_data_dir) / f"{symbol}_{tf_normalized}_features.parquet"
        raw_path = Path("data/raw/clock") / f"{symbol}_{tf_normalized}.parquet"
        
        # Load features
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        features_df = pd.read_parquet(features_path)
        print(f"  Loaded {len(features_df)} rows of features from {features_path}")
        
        # Load raw OHLCV for target building
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
        raw_df = pd.read_parquet(raw_path)
        print(f"  Loaded {len(raw_df)} rows of raw data from {raw_path}")
        
        # Verify alignment
        if len(features_df) != len(raw_df):
            print(f"  WARNING: Row count mismatch! Features: {len(features_df)}, Raw: {len(raw_df)}")
            # Use minimum length
            min_len = min(len(features_df), len(raw_df))
            features_df = features_df.iloc[:min_len]
            raw_df = raw_df.iloc[:min_len]
        
        # Merge: take OHLCV + timestamp from raw, features from processed
        df = raw_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Add all features
        for col in features_df.columns:
            df[col] = features_df[col].values
        
        print(f"  Merged data: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def run_baseline_test(
        self, 
        symbol: str, 
        timeframe: str,
        save_results: bool = True
    ) -> Dict:
        """
        Run complete baseline prediction test.
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "1h")
            save_results: Whether to save results to file
            
        Returns:
            Complete results dictionary
        """
        print(f"\n{'='*60}")
        print(f"BASELINE PREDICTION TEST: {symbol} {timeframe}")
        print(f"{'='*60}\n")
        
        # Step 1: Load data
        print("Step 1: Loading data...")
        df = self.load_data(symbol, timeframe)
        
        # Step 2: Build target
        print("Step 2: Building target...")
        df['target'] = self.target_builder.build_binary_target(df)
        target_stats = self.target_builder.get_target_stats(df['target'])
        print(f"  Target distribution: {target_stats['positive_ratio']:.1%} up / "
              f"{target_stats['negative_ratio']:.1%} down")
        
        # Step 3: Split data
        print("Step 3: Splitting data (chronological)...")
        train_df, val_df, test_df = self.splitter.split(df)
        split_info = self.splitter.get_split_info(train_df, val_df, test_df)
        print(f"  Train: {split_info['train_size']} | "
              f"Val: {split_info['val_size']} | "
              f"Test: {split_info['test_size']}")
        
        # Step 4: Prepare features
        print("Step 4: Preparing features...")
        X_train, y_train, feature_names = self.feature_processor.prepare_features(
            train_df, fit_scaler=True
        )
        X_val, y_val, _ = self.feature_processor.prepare_features(
            val_df, fit_scaler=False
        )
        X_test, y_test, _ = self.feature_processor.prepare_features(
            test_df, fit_scaler=False
        )
        print(f"  Features: {len(feature_names)}")
        print(f"  Train samples: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        
        # Step 5: Train and evaluate models
        print("Step 5: Training models...")
        model_results = self.trainer.train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names
        )
        
        # Step 6: Assess signal
        print("Step 6: Assessing signal...")
        signal_assessment = self.signal_detector.evaluate_results(model_results)
        
        # Compile results
        results = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "lookahead_bars": self.config.lookahead_bars,
                "target_threshold": self.config.target_threshold,
                "train_ratio": self.config.train_ratio
            },
            "target_stats": target_stats,
            "split_info": split_info,
            "feature_count": len(feature_names),
            "features_used": feature_names,
            "model_results": model_results,
            "signal_assessment": signal_assessment
        }
        
        # Print summary
        self._print_summary(results)
        
        # Save results
        if save_results:
            self._save_results(results, symbol, timeframe)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print a summary of the results."""
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}\n")
        
        assessment = results["signal_assessment"]
        
        # Model performance table
        print("Model Performance (Test Set):")
        print("-" * 50)
        print(f"{'Model':<25} {'Accuracy':>10} {'AUC':>10} {'Overfit':>10}")
        print("-" * 50)
        
        for model_name, model_results in results["model_results"].items():
            test = model_results["test"]
            overfit = model_results["overfit_ratio"]
            print(f"{model_name:<25} {test['accuracy']:>10.3f} "
                  f"{test['auc_roc']:>10.3f} {overfit:>10.2f}")
        
        print("-" * 50)
        
        # Signal assessment
        print(f"\nSignal Detected: {'YES' if assessment['has_signal'] else 'NO'}")
        print(f"Signal Strength: {assessment['signal_strength'].upper()}")
        print(f"Best Model: {assessment['best_model']}")
        print(f"Best OOS Accuracy: {assessment['best_oos_accuracy']:.3f}")
        print(f"Best OOS AUC: {assessment['best_oos_auc']:.3f}")
        
        # Top features
        if assessment['best_model']:
            print("\nTop 10 Features:")
            importance = results["model_results"][assessment['best_model']]["feature_importance"]
            for i, (feat, imp) in enumerate(list(importance.items())[:10]):
                print(f"  {i+1}. {feat}: {imp:.4f}")
        
        # Recommendation
        print(f"\n{'='*60}")
        print(f"RECOMMENDATION: {assessment['recommendation']}")
        print(f"{'='*60}\n")
    
    def _save_results(self, results: Dict, symbol: str, timeframe: str):
        """Save results to JSON file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{symbol}_{timeframe}_baseline_results.json"
        filepath = output_dir / filename
        
        # Convert numpy/python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):  # Handle numpy and Python bools
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(i) for i in obj]
            return obj
        
        results_clean = convert_types(results)
        
        with open(filepath, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        print(f"Results saved to: {filepath}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_quick_test(symbol: str = "EURUSD", timeframe: str = "1h") -> Dict:
    """Quick test function for single asset."""
    predictor = BaselinePredictor()
    return predictor.run_baseline_test(symbol, timeframe)


def run_multi_asset_test(
    symbols: List[str] = None,
    timeframe: str = "1h"
) -> Dict[str, Dict]:
    """Run baseline test on multiple assets."""
    if symbols is None:
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
    
    predictor = BaselinePredictor()
    results = {}
    
    for symbol in symbols:
        try:
            results[symbol] = predictor.run_baseline_test(symbol, timeframe)
        except Exception as e:
            print(f"Error testing {symbol}: {e}")
            results[symbol] = {"error": str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("MULTI-ASSET SUMMARY")
    print("="*60)
    
    signals_found = []
    for symbol, result in results.items():
        if "error" not in result:
            has_signal = result["signal_assessment"]["has_signal"]
            accuracy = result["signal_assessment"]["best_oos_accuracy"]
            print(f"{symbol}: {'SIGNAL' if has_signal else 'NO SIGNAL'} "
                  f"(accuracy: {accuracy:.3f})")
            if has_signal:
                signals_found.append(symbol)
    
    print(f"\nSignals found in {len(signals_found)}/{len(symbols)} pairs")
    
    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run single test
    results = run_quick_test("EURUSD", "1h")