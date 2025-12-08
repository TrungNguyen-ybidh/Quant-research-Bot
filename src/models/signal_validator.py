"""
Signal Validator
================
Rigorous validation of trading signals across multiple dimensions:
1. Walk-forward validation (time-series cross-validation)
2. Cross-pair generalization
3. Temporal stability (year-by-year)
4. Feature importance analysis
5. Realistic P&L calculation

Usage:
    python scripts/validate_signal.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent if Path(__file__).parent.name == 'scripts' else Path(__file__).parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from src.models.target_builder import TargetBuilder
from src.models.cost_model import CostModel, CostConfig
from src.utils.logger import info, success, warning, error


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for signal validation."""
    lookahead: int = 24  # Hours
    target_threshold: float = 0.0001
    
    # Walk-forward settings
    n_splits: int = 5
    train_pct: float = 0.60
    gap_bars: int = 24  # Gap between train and test to prevent leakage
    
    # Model settings
    n_estimators: int = 100
    max_depth: int = 5
    random_state: int = 42
    
    # Pairs to test
    pairs: List[str] = None
    
    def __post_init__(self):
        if self.pairs is None:
            self.pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]


# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """Loads and prepares data for validation."""
    
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
        
        # Align lengths
        min_len = min(len(features_df), len(raw_df))
        features_df = features_df.iloc[:min_len]
        raw_df = raw_df.iloc[:min_len]
        
        # Merge
        df = raw_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        for col in features_df.columns:
            df[col] = features_df[col].values
        
        # Set timestamp as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns (exclude OHLCV and target)."""
        exclude = ['open', 'high', 'low', 'close', 'volume', 'target', 'returns']
        return [c for c in df.columns if c not in exclude]


# =============================================================================
# WALK-FORWARD VALIDATOR
# =============================================================================

class WalkForwardValidator:
    """
    Walk-forward validation for time-series.
    
    Trains on past, tests on future, slides forward.
    More realistic than single train/test split.
    """
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def generate_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward train/test indices."""
        splits = []
        
        # Calculate sizes
        test_size = n_samples // (self.config.n_splits + 1)
        
        for i in range(self.config.n_splits):
            # Test window (from end, moving backwards)
            test_end = n_samples - i * test_size
            test_start = test_end - test_size
            
            # Train window (everything before test, minus gap)
            train_end = test_start - self.config.gap_bars
            train_start = 0
            
            if train_end <= train_start + 100:  # Need minimum training data
                continue
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def validate(self, X: np.ndarray, y: np.ndarray, prices: np.ndarray) -> Dict:
        """
        Run walk-forward validation.
        
        Returns metrics for each fold and aggregate statistics.
        """
        splits = self.generate_splits(len(X))
        
        fold_results = []
        all_predictions = []
        all_actuals = []
        all_returns = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            prices_test = prices[test_idx]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                class_weight='balanced',
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Predict
            predictions = model.predict(X_test_scaled)
            probas = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, predictions)
            try:
                auc = roc_auc_score(y_test, probas)
            except:
                auc = 0.5
            
            # Calculate actual returns
            returns = self._calculate_returns(predictions, prices_test)
            
            fold_results.append({
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "accuracy": accuracy,
                "auc": auc,
                "mean_return": np.mean(returns),
                "total_return": np.sum(returns),
                "win_rate": np.mean(returns > 0)
            })
            
            all_predictions.extend(predictions)
            all_actuals.extend(y_test)
            all_returns.extend(returns)
        
        # Aggregate statistics
        accuracies = [f["accuracy"] for f in fold_results]
        returns = [f["total_return"] for f in fold_results]
        
        return {
            "n_folds": len(fold_results),
            "fold_results": fold_results,
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "min_accuracy": np.min(accuracies),
            "max_accuracy": np.max(accuracies),
            "mean_return_per_fold": np.mean(returns),
            "total_return": np.sum(returns),
            "consistency": np.sum(np.array(accuracies) > 0.5) / len(accuracies),
            "all_predictions": all_predictions,
            "all_actuals": all_actuals,
            "all_returns": all_returns
        }
    
    def _calculate_returns(self, predictions: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """Calculate per-trade returns."""
        n = len(predictions)
        returns = np.zeros(n)
        
        lookahead = min(self.config.lookahead, n - 1)
        
        for i in range(n - lookahead):
            # Position: 1 for long (predict up), -1 for short (predict down)
            position = 1 if predictions[i] == 1 else -1
            
            # Forward return
            forward_ret = (prices[i + lookahead] - prices[i]) / prices[i]
            
            # Strategy return
            returns[i] = position * forward_ret
        
        return returns


# =============================================================================
# CROSS-PAIR VALIDATOR
# =============================================================================

class CrossPairValidator:
    """Tests signal generalization across currency pairs."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.loader = DataLoader()
    
    def validate(self) -> Dict:
        """Test signal on multiple pairs."""
        results = {}
        
        for symbol in self.config.pairs:
            info(f"Testing {symbol}...")
            
            df = self.loader.load(symbol, "1h")
            if df is None:
                results[symbol] = {"error": "Data not found"}
                continue
            
            try:
                result = self._test_single_pair(df, symbol)
                results[symbol] = result
                
                status = "✓" if result["accuracy"] > 0.52 else "✗"
                print(f"  {status} {symbol}: Accuracy={result['accuracy']:.1%}, "
                      f"Sharpe={result['net_sharpe']:.2f}")
                
            except Exception as e:
                results[symbol] = {"error": str(e)}
                print(f"  ✗ {symbol}: Error - {e}")
        
        # Summary
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if valid_results:
            accuracies = [v["accuracy"] for v in valid_results.values()]
            passing = sum(1 for a in accuracies if a > 0.52)
            
            results["_summary"] = {
                "pairs_tested": len(self.config.pairs),
                "pairs_valid": len(valid_results),
                "pairs_passing": passing,
                "pass_rate": passing / len(valid_results) if valid_results else 0,
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "generalizes": passing >= len(valid_results) * 0.6  # 60% must pass
            }
        
        return results
    
    def _test_single_pair(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Test on a single pair with simple train/test split."""
        # Build target
        target_builder = TargetBuilder(
            target_type='binary',
            lookahead=self.config.lookahead,
            threshold=self.config.target_threshold
        )
        target = target_builder.build(df.reset_index(), price_col='close')
        
        # Clean data
        valid_mask = target.notna()
        df_valid = df[valid_mask.values].copy()
        y = target[valid_mask].values
        prices = df_valid['close'].values
        
        # Features
        feature_cols = self.loader.get_feature_columns(df_valid)
        X = df_valid[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        
        # Simple 70/30 split
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        prices_test = prices[split_idx:]
        
        # Scale and train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        
        # Cost analysis
        cost_model = CostModel()
        cost_results = cost_model.analyze(
            predictions=predictions,
            prices=prices_test,
            symbol=symbol,
            timeframe="1h",
            lookahead=self.config.lookahead
        )
        
        return {
            "accuracy": accuracy,
            "test_samples": len(y_test),
            "gross_sharpe": cost_results["gross"]["sharpe"],
            "net_sharpe": cost_results["net"]["sharpe"],
            "survives_costs": cost_results["survives_costs"]
        }


# =============================================================================
# TEMPORAL STABILITY VALIDATOR
# =============================================================================

class TemporalStabilityValidator:
    """Tests if signal is stable across different time periods."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.loader = DataLoader()
    
    def validate(self, symbol: str = "EURUSD") -> Dict:
        """Test signal stability across years."""
        df = self.loader.load(symbol, "1h")
        if df is None:
            return {"error": "Data not found"}
        
        # Build target
        target_builder = TargetBuilder(
            target_type='binary',
            lookahead=self.config.lookahead,
            threshold=self.config.target_threshold
        )
        target = target_builder.build(df.reset_index(), price_col='close')
        
        # Clean
        valid_mask = target.notna()
        df_valid = df[valid_mask.values].copy()
        y = target[valid_mask]
        
        # Get years in data
        if not isinstance(df_valid.index, pd.DatetimeIndex):
            df_valid.index = pd.to_datetime(df_valid.index)
        
        years = df_valid.index.year.unique()
        
        results = {}
        
        for year in sorted(years):
            year_mask = df_valid.index.year == year
            
            if year_mask.sum() < 500:  # Need minimum samples
                continue
            
            info(f"Testing {year}...")
            
            df_year = df_valid[year_mask]
            y_year = y[year_mask]
            
            try:
                result = self._test_single_period(df_year, y_year.values, symbol)
                results[str(year)] = result
                
                status = "✓" if result["accuracy"] > 0.52 else "✗"
                print(f"  {status} {year}: Accuracy={result['accuracy']:.1%}, "
                      f"Samples={result['test_samples']}")
                
            except Exception as e:
                results[str(year)] = {"error": str(e)}
        
        # Summary
        valid_years = {k: v for k, v in results.items() if "error" not in v and k.isdigit()}
        
        if valid_years:
            accuracies = [v["accuracy"] for v in valid_years.values()]
            passing = sum(1 for a in accuracies if a > 0.52)
            
            results["_summary"] = {
                "years_tested": len(valid_years),
                "years_passing": passing,
                "mean_accuracy": np.mean(accuracies),
                "std_accuracy": np.std(accuracies),
                "min_accuracy": np.min(accuracies),
                "max_accuracy": np.max(accuracies),
                "is_stable": passing >= len(valid_years) * 0.7,  # 70% of years must pass
                "decay": self._calculate_decay(valid_years)
            }
        
        return results
    
    def _test_single_period(self, df: pd.DataFrame, y: np.ndarray, symbol: str) -> Dict:
        """Test on a single time period."""
        feature_cols = self.loader.get_feature_columns(df)
        X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
        prices = df['close'].values
        
        # 70/30 split within period
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        prices_test = prices[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)
        
        return {
            "accuracy": accuracy,
            "train_samples": len(y_train),
            "test_samples": len(y_test)
        }
    
    def _calculate_decay(self, year_results: Dict) -> float:
        """Calculate signal decay over time."""
        years = sorted([int(y) for y in year_results.keys()])
        if len(years) < 2:
            return 0.0
        
        # Compare first half vs second half
        mid = len(years) // 2
        early_years = years[:mid]
        recent_years = years[mid:]
        
        early_acc = np.mean([year_results[str(y)]["accuracy"] for y in early_years])
        recent_acc = np.mean([year_results[str(y)]["accuracy"] for y in recent_years])
        
        decay = (early_acc - recent_acc) / early_acc if early_acc > 0 else 0
        return decay


# =============================================================================
# FEATURE IMPORTANCE ANALYZER
# =============================================================================

class FeatureImportanceAnalyzer:
    """Analyzes which features drive the signal."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.loader = DataLoader()
    
    def analyze(self, symbol: str = "EURUSD") -> Dict:
        """Analyze feature importance for the signal."""
        df = self.loader.load(symbol, "1h")
        if df is None:
            return {"error": "Data not found"}
        
        # Build target
        target_builder = TargetBuilder(
            target_type='binary',
            lookahead=self.config.lookahead,
            threshold=self.config.target_threshold
        )
        target = target_builder.build(df.reset_index(), price_col='close')
        
        # Clean
        valid_mask = target.notna()
        df_valid = df[valid_mask.values]
        y = target[valid_mask].values
        
        # Features
        feature_cols = self.loader.get_feature_columns(df_valid)
        X = df_valid[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Train model on all data to get importance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_scaled, y)
        
        # Get importance
        importance = dict(zip(feature_cols, model.feature_importances_))
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize top features
        categories = self._categorize_features(sorted_importance[:20])
        
        return {
            "top_20_features": sorted_importance[:20],
            "feature_categories": categories,
            "total_features": len(feature_cols)
        }
    
    def _categorize_features(self, features: List[Tuple[str, float]]) -> Dict:
        """Categorize features by type."""
        categories = {
            "trend": [],
            "volatility": [],
            "momentum": [],
            "macro": [],
            "microstructure": [],
            "session": [],
            "other": []
        }
        
        for name, importance in features:
            name_lower = name.lower()
            
            if any(x in name_lower for x in ['sma', 'ema', 'trend', 'adx', 'macd']):
                categories["trend"].append((name, importance))
            elif any(x in name_lower for x in ['vol', 'atr', 'hurst', 'variance']):
                categories["volatility"].append((name, importance))
            elif any(x in name_lower for x in ['rsi', 'stoch', 'momentum', 'roc', 'cci']):
                categories["momentum"].append((name, importance))
            elif any(x in name_lower for x in ['yield', 'carry', 'rate', 'factor']):
                categories["macro"].append((name, importance))
            elif any(x in name_lower for x in ['spread', 'wick', 'body', 'micro']):
                categories["microstructure"].append((name, importance))
            elif any(x in name_lower for x in ['session', 'london', 'asia', 'york']):
                categories["session"].append((name, importance))
            else:
                categories["other"].append((name, importance))
        
        # Calculate category importance
        category_totals = {}
        for cat, feats in categories.items():
            if feats:
                category_totals[cat] = sum(imp for _, imp in feats)
        
        return {
            "by_category": categories,
            "category_totals": category_totals
        }


# =============================================================================
# REALISTIC P&L CALCULATOR
# =============================================================================

class RealisticPnLCalculator:
    """Calculates realistic trade-by-trade P&L."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.cost_config = CostConfig()
    
    def calculate(
        self,
        predictions: np.ndarray,
        prices: np.ndarray,
        symbol: str,
        position_size: float = 10000  # $10k per trade
    ) -> Dict:
        """Calculate realistic P&L with costs."""
        
        n = len(predictions)
        lookahead = self.config.lookahead
        
        # Get cost per trade
        pip_value = self.cost_config.PIP_VALUES.get(symbol, 0.0001)
        spread_pips = self.cost_config.SPREAD_PIPS.get(symbol, 0.5)
        spread_cost = spread_pips * pip_value * position_size * 2  # Round trip
        
        commission = self.cost_config.commission_pct * position_size * 2
        slippage = self.cost_config.slippage_pips * pip_value * position_size * 2
        
        total_cost_per_trade = spread_cost + commission + slippage
        
        # Calculate trade-by-trade P&L
        trades = []
        equity_curve = [0]
        
        for i in range(n - lookahead):
            # Position
            direction = 1 if predictions[i] == 1 else -1
            
            # Entry and exit prices
            entry_price = prices[i]
            exit_price = prices[i + lookahead]
            
            # Gross P&L
            price_change = (exit_price - entry_price) / entry_price
            gross_pnl = direction * price_change * position_size
            
            # Net P&L (after costs)
            net_pnl = gross_pnl - total_cost_per_trade
            
            trades.append({
                "entry_idx": i,
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl,
                "costs": total_cost_per_trade,
                "net_pnl": net_pnl
            })
            
            equity_curve.append(equity_curve[-1] + net_pnl)
        
        # Statistics
        net_pnls = [t["net_pnl"] for t in trades]
        gross_pnls = [t["gross_pnl"] for t in trades]
        
        wins = sum(1 for p in net_pnls if p > 0)
        losses = sum(1 for p in net_pnls if p < 0)
        
        avg_win = np.mean([p for p in net_pnls if p > 0]) if wins > 0 else 0
        avg_loss = np.mean([p for p in net_pnls if p < 0]) if losses > 0 else 0
        
        # Drawdown
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = running_max - equity
        max_drawdown = np.max(drawdown)
        
        return {
            "num_trades": len(trades),
            "position_size": position_size,
            "cost_per_trade": total_cost_per_trade,
            "total_costs": total_cost_per_trade * len(trades),
            
            "gross_pnl": sum(gross_pnls),
            "net_pnl": sum(net_pnls),
            "total_return_pct": sum(net_pnls) / position_size * 100,
            
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(trades) if trades else 0,
            
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss != 0 else 0,
            
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown / position_size * 100,
            
            "equity_curve": equity_curve,
            "trades": trades[:10]  # First 10 trades for inspection
        }


# =============================================================================
# MAIN VALIDATOR
# =============================================================================

class SignalValidator:
    """Main class orchestrating all validation."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.loader = DataLoader()
        self.walk_forward = WalkForwardValidator(self.config)
        self.cross_pair = CrossPairValidator(self.config)
        self.temporal = TemporalStabilityValidator(self.config)
        self.feature_analyzer = FeatureImportanceAnalyzer(self.config)
        self.pnl_calc = RealisticPnLCalculator(self.config)
    
    def run_full_validation(self, symbol: str = "EURUSD") -> Dict:
        """Run complete validation suite."""
        
        print("=" * 80)
        print(f"COMPREHENSIVE SIGNAL VALIDATION: {symbol} | Lookahead: {self.config.lookahead}h")
        print("=" * 80)
        
        results = {
            "symbol": symbol,
            "lookahead": self.config.lookahead,
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. Walk-Forward Validation
        print("\n" + "─" * 80)
        print("1. WALK-FORWARD VALIDATION")
        print("─" * 80)
        
        df = self.loader.load(symbol, "1h")
        if df is not None:
            target_builder = TargetBuilder(
                target_type='binary',
                lookahead=self.config.lookahead,
                threshold=self.config.target_threshold
            )
            target = target_builder.build(df.reset_index(), price_col='close')
            
            valid_mask = target.notna()
            df_valid = df[valid_mask.values]
            y = target[valid_mask].values
            prices = df_valid['close'].values
            
            feature_cols = self.loader.get_feature_columns(df_valid)
            X = df_valid[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
            
            wf_results = self.walk_forward.validate(X, y, prices)
            results["walk_forward"] = wf_results
            
            print(f"  Folds: {wf_results['n_folds']}")
            print(f"  Mean Accuracy: {wf_results['mean_accuracy']:.1%} ± {wf_results['std_accuracy']:.1%}")
            print(f"  Min/Max Accuracy: {wf_results['min_accuracy']:.1%} / {wf_results['max_accuracy']:.1%}")
            print(f"  Consistency (>50%): {wf_results['consistency']:.0%}")
        
        # 2. Cross-Pair Validation
        print("\n" + "─" * 80)
        print("2. CROSS-PAIR GENERALIZATION")
        print("─" * 80)
        
        cp_results = self.cross_pair.validate()
        results["cross_pair"] = cp_results
        
        if "_summary" in cp_results:
            summary = cp_results["_summary"]
            print(f"\n  Pairs Passing: {summary['pairs_passing']}/{summary['pairs_valid']}")
            print(f"  Mean Accuracy: {summary['mean_accuracy']:.1%}")
            print(f"  Generalizes: {'YES ✓' if summary['generalizes'] else 'NO ✗'}")
        
        # 3. Temporal Stability
        print("\n" + "─" * 80)
        print("3. TEMPORAL STABILITY")
        print("─" * 80)
        
        ts_results = self.temporal.validate(symbol)
        results["temporal"] = ts_results
        
        if "_summary" in ts_results:
            summary = ts_results["_summary"]
            print(f"\n  Years Passing: {summary['years_passing']}/{summary['years_tested']}")
            print(f"  Mean Accuracy: {summary['mean_accuracy']:.1%} ± {summary['std_accuracy']:.1%}")
            print(f"  Signal Decay: {summary['decay']*100:.1f}%")
            print(f"  Is Stable: {'YES ✓' if summary['is_stable'] else 'NO ✗'}")
        
        # 4. Feature Importance
        print("\n" + "─" * 80)
        print("4. FEATURE IMPORTANCE ANALYSIS")
        print("─" * 80)
        
        fi_results = self.feature_analyzer.analyze(symbol)
        results["features"] = fi_results
        
        if "top_20_features" in fi_results:
            print("\n  Top 10 Features:")
            for i, (name, imp) in enumerate(fi_results["top_20_features"][:10], 1):
                print(f"    {i:2}. {name}: {imp:.4f}")
            
            print("\n  Category Breakdown:")
            for cat, total in sorted(fi_results["feature_categories"]["category_totals"].items(), 
                                    key=lambda x: x[1], reverse=True):
                print(f"    {cat}: {total:.3f}")
        
        # 5. Realistic P&L
        print("\n" + "─" * 80)
        print("5. REALISTIC P&L CALCULATION")
        print("─" * 80)
        
        if df is not None:
            # Get predictions from walk-forward
            predictions = np.array(wf_results["all_predictions"])
            # Use corresponding prices (simplified - use last portion of data)
            test_prices = prices[-len(predictions):]
            
            pnl_results = self.pnl_calc.calculate(predictions, test_prices, symbol)
            results["pnl"] = pnl_results
            
            print(f"\n  Trades: {pnl_results['num_trades']}")
            print(f"  Win Rate: {pnl_results['win_rate']:.1%}")
            print(f"  Profit Factor: {pnl_results['profit_factor']:.2f}")
            print(f"  Gross P&L: ${pnl_results['gross_pnl']:,.2f}")
            print(f"  Total Costs: ${pnl_results['total_costs']:,.2f}")
            print(f"  Net P&L: ${pnl_results['net_pnl']:,.2f}")
            print(f"  Max Drawdown: ${pnl_results['max_drawdown']:,.2f} ({pnl_results['max_drawdown_pct']:.1f}%)")
        
        # 6. Final Verdict
        print("\n" + "=" * 80)
        print("FINAL VERDICT")
        print("=" * 80)
        
        verdict = self._generate_verdict(results)
        results["verdict"] = verdict
        
        print(f"\n  Walk-Forward: {'PASS ✓' if verdict['walk_forward_pass'] else 'FAIL ✗'}")
        print(f"  Cross-Pair: {'PASS ✓' if verdict['cross_pair_pass'] else 'FAIL ✗'}")
        print(f"  Temporal: {'PASS ✓' if verdict['temporal_pass'] else 'FAIL ✗'}")
        print(f"  P&L Positive: {'PASS ✓' if verdict['pnl_pass'] else 'FAIL ✗'}")
        
        print(f"\n  OVERALL: {verdict['overall']}")
        print(f"  CONFIDENCE: {verdict['confidence']}")
        
        print("\n" + "=" * 80)
        
        return results
    
    def _generate_verdict(self, results: Dict) -> Dict:
        """Generate final verdict from all validations."""
        
        # Walk-forward pass: consistency > 60% and mean accuracy > 52%
        wf = results.get("walk_forward", {})
        wf_pass = (wf.get("consistency", 0) >= 0.6 and 
                   wf.get("mean_accuracy", 0) > 0.52)
        
        # Cross-pair pass: generalizes to 60%+ of pairs
        cp = results.get("cross_pair", {}).get("_summary", {})
        cp_pass = cp.get("generalizes", False)
        
        # Temporal pass: stable across 70%+ of years
        ts = results.get("temporal", {}).get("_summary", {})
        ts_pass = ts.get("is_stable", False)
        
        # P&L pass: positive net P&L
        pnl = results.get("pnl", {})
        pnl_pass = pnl.get("net_pnl", 0) > 0
        
        # Count passes
        passes = sum([wf_pass, cp_pass, ts_pass, pnl_pass])
        
        if passes == 4:
            overall = "STRONG SIGNAL - All validations pass"
            confidence = "HIGH"
        elif passes == 3:
            overall = "MODERATE SIGNAL - Most validations pass"
            confidence = "MEDIUM"
        elif passes == 2:
            overall = "WEAK SIGNAL - Mixed results"
            confidence = "LOW"
        else:
            overall = "NO RELIABLE SIGNAL - Most validations fail"
            confidence = "VERY LOW"
        
        return {
            "walk_forward_pass": wf_pass,
            "cross_pair_pass": cp_pass,
            "temporal_pass": ts_pass,
            "pnl_pass": pnl_pass,
            "passes": passes,
            "total_tests": 4,
            "overall": overall,
            "confidence": confidence
        }
    
    def save_results(self, results: Dict, output_dir: str = "outputs/validation"):
        """Save validation results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Remove non-serializable items
        clean_results = self._clean_for_json(results)
        
        filename = f"{results['symbol']}_{results['lookahead']}h_validation.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        success(f"Results saved to {filepath}")
    
    def _clean_for_json(self, obj):
        """Clean object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items() 
                    if k not in ['equity_curve', 'all_predictions', 'all_actuals', 'all_returns', 'trades']}
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
# MAIN ENTRY POINT
# =============================================================================

def run_validation(symbol: str = "EURUSD", lookahead: int = 24):
    """Run full validation."""
    config = ValidationConfig(lookahead=lookahead)
    validator = SignalValidator(config)
    results = validator.run_full_validation(symbol)
    validator.save_results(results)
    return results


if __name__ == "__main__":
    results = run_validation("EURUSD", lookahead=24)