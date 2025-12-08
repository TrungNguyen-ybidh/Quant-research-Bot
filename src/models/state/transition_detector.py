"""
Regime Transition Detector
==========================
Predicts when market regime is about to change.

This is useful for:
- Early warning system for regime shifts
- Adjusting positions before volatility spikes
- Timing strategy switches

Labels:
    - 0: No transition (regime stable)
    - 1: Transition coming (regime will change within horizon)

Usage:
    from src.models.state import TransitionDetector
    
    detector = TransitionDetector(symbol="EURUSD", timeframe="1h")
    detector.prepare_data()
    detector.train()
    detector.evaluate()
    detector.save()
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

import torch

from src.data.loader import load_parquet_file
from src.models.preprocessing import FeaturePreprocessor
from src.models.feature_selection import FeatureSelector
from src.models.data_splitter import TimeSeriesSplitter
from src.models.targets.regime_labeler import RegimeLabeler, TransitionLabeler
from src.models.nn import create_model, TimeSeriesDataModule, Trainer, TrainerConfig
from src.utils.logger import info, success, error, warning


class TransitionDetector:
    """
    Binary classifier to predict regime transitions.
    
    Workflow:
        1. Load data
        2. Create regime labels first
        3. Create transition labels from regime labels
        4. Preprocess and select features
        5. Train binary classifier
        6. Evaluate (focus on recall - don't miss transitions!)
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        architecture: str = "mlp",
        config: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize transition detector.
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "1h")
            architecture: Model type ("mlp" or "lstm")
            config: Configuration dictionary
            random_state: Random seed
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.architecture = architecture
        self.config = config or {}
        self.random_state = random_state
        
        # Set seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Components
        self.preprocessor = None
        self.selector = None
        self.regime_labeler = None
        self.transition_labeler = None
        self.splitter = None
        self.model = None
        self.trainer = None
        
        # Data containers
        self.df = None
        self.df_raw = None
        self.X = None
        self.y = None
        self.regime_labels = None
        self.selected_features = None
        
        # Results
        self.train_history = None
        self.test_metrics = None
        
        info(f"Initialized TransitionDetector for {symbol} {timeframe} with {architecture}")
    
    def prepare_data(self, transition_horizon: int = 8) -> None:
        """
        Load data, create labels, preprocess, and split.
        
        Args:
            transition_horizon: Bars to look ahead for transition
        """
        
        info("=" * 80)
        info("Step 1: Loading Data")
        info("=" * 80)
        
        # Load raw OHLC data
        raw_path = f"data/raw/clock/{self.symbol}_{self.timeframe}.parquet"
        self.df_raw = load_parquet_file(raw_path)
        if self.df_raw is None:
            raise FileNotFoundError(f"Failed to load raw data: {raw_path}")
        
        # Load processed features
        processed_path = f"data/processed/{self.symbol}_{self.timeframe}_features.parquet"
        self.df = load_parquet_file(processed_path)
        if self.df is None:
            raise FileNotFoundError(f"Failed to load processed data: {processed_path}")
        
        info(f"Loaded {len(self.df)} bars")
        info(f"Features: {self.df.shape[1]} columns")
        
        # Step 2: Create regime labels first
        info("\n" + "=" * 80)
        info("Step 2: Creating Regime Labels")
        info("=" * 80)
        
        self.regime_labeler = RegimeLabeler.for_timeframe(self.timeframe)
        self.regime_labels = self.regime_labeler.build(self.df_raw)
        
        # Step 3: Create transition labels from regime labels
        info("\n" + "=" * 80)
        info("Step 3: Creating Transition Labels")
        info("=" * 80)
        
        self.transition_labeler = TransitionLabeler(horizon=transition_horizon)
        self.y = self.transition_labeler.build(pd.Series(self.regime_labels))
        
        # Log distribution
        stats = self.transition_labeler.get_stats()
        info("\nTransition Distribution:")
        info(f"  NO_TRANSITION (0): {stats['n_no_trans']:6d} ({stats['pct_no_trans']:5.1f}%)")
        info(f"  TRANSITION (1):    {stats['n_trans']:6d} ({stats['pct_trans']:5.1f}%)")
        
        # Align and clean data
        info("\n" + "=" * 80)
        info("Step 4: Aligning Data")
        info("=" * 80)
        
        self.df = self.df.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)
        
        valid_mask = ~self.y.isna()
        self.df = self.df[valid_mask].reset_index(drop=True)
        self.y = self.y[valid_mask].reset_index(drop=True)
        
        info(f"After removing NaN labels: {len(self.df)} samples")
        
        # Preprocessing
        info("\n" + "=" * 80)
        info("Step 5: Preprocessing Features")
        info("=" * 80)
        
        self.preprocessor = FeaturePreprocessor(
            default_method='robust',
            handle_inf=True,
            clip_outliers=True,
            outlier_std=3.0
        )
        
        X_preprocessed = self.preprocessor.fit_transform(self.df)
        info(f"Preprocessed features: {X_preprocessed.shape}")
        
        # Feature Selection
        info("\n" + "=" * 80)
        info("Step 6: Feature Selection")
        info("=" * 80)
        
        self.selector = FeatureSelector(
            variance_threshold=0.01,
            correlation_threshold=0.95,
            n_features=30
        )
        
        self.X = self.selector.fit_transform(X_preprocessed, self.y)
        self.selected_features = self.selector.selected_features
        
        info(f"Selected {len(self.selected_features)} features:")
        for i, feat in enumerate(self.selected_features[:10], 1):
            info(f"  {i:2d}. {feat}")
        if len(self.selected_features) > 10:
            info(f"  ... and {len(self.selected_features) - 10} more")
        
        # Train/Val/Test Split
        info("\n" + "=" * 80)
        info("Step 7: Train/Val/Test Split")
        info("=" * 80)
        
        self.splitter = TimeSeriesSplitter(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        splits = self.splitter.split(self.X, self.y)
        self.X_train = splits.X_train
        self.X_val = splits.X_val
        self.X_test = splits.X_test
        self.y_train = splits.y_train
        self.y_val = splits.y_val
        self.y_test = splits.y_test
        
        info(f"Train: {len(self.X_train)} samples")
        info(f"Val:   {len(self.X_val)} samples")
        info(f"Test:  {len(self.X_test)} samples")
        
        # Check class balance
        train_trans = (self.y_train == 1).sum()
        train_no_trans = (self.y_train == 0).sum()
        info(f"\nTrain set: {train_no_trans} no-transition, {train_trans} transition")
        
        info("\n✓ Data preparation complete")
    
    def train(self) -> Dict[str, Any]:
        """Train the transition detection model."""
        
        if self.X_train is None:
            raise ValueError("Must call prepare_data() before train()")
        
        info("\n" + "=" * 80)
        info("Step 8: Training Neural Network")
        info("=" * 80)
        
        nn_config = self.config.get('models', {}).get('nn', {})
        
        input_dim = self.X_train.shape[1]
        n_classes = 2  # Binary: transition or not
        
        # Create model
        if self.architecture == "mlp":
            model_kwargs = {
                'input_dim': input_dim,
                'hidden_dims': [128, 64, 32],
                'n_classes': n_classes,
                'dropout': 0.3,
                'batch_norm': True
            }
        else:
            model_kwargs = {
                'input_dim': input_dim,
                'hidden_dim': 64,
                'num_layers': 2,
                'n_classes': n_classes,
                'dropout': 0.3
            }
        
        self.model = create_model(self.architecture, **model_kwargs)
        info(f"Created {self.architecture.upper()} model:")
        info(f"  Input dim: {input_dim}")
        info(f"  Output classes: {n_classes} (binary)")
        info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create data module
        data_module = TimeSeriesDataModule(
            X=self.X,
            y=self.y,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=nn_config.get('batch_size', 64),
            seq_length=None if self.architecture == "mlp" else 24,
            scale_features=False
        )
        
        # Training config - use class weights for imbalanced data
        trainer_config = TrainerConfig(
            epochs=nn_config.get('default_epochs', 100),
            lr=nn_config.get('learning_rate', 0.001),
            weight_decay=nn_config.get('weight_decay', 0.0001),
            patience=nn_config.get('early_stopping_patience', 10),
            use_class_weights=True,  # Important for imbalanced transition data
            device=nn_config.get('device', 'cpu')
        )
        
        info(f"\nTraining Configuration:")
        info(f"  Max epochs: {trainer_config.epochs}")
        info(f"  Learning rate: {trainer_config.lr}")
        info(f"  Class weights: Enabled (for imbalanced data)")
        
        # Train
        self.trainer = Trainer(self.model, trainer_config)
        self.train_history = self.trainer.fit(
            data_module.train_dataloader(),
            data_module.val_dataloader()
        )
        
        info("\n✓ Training complete")
        info(f"Best epoch: {self.trainer.best_metrics.get('epoch', 'N/A')}")
        info(f"Best val loss: {self.trainer.best_metrics.get('val_loss', 0):.4f}")
        
        return self.train_history
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the trained model on test set."""
        
        if self.trainer is None:
            raise ValueError("Must call train() before evaluate()")
        
        info("\n" + "=" * 80)
        info("Step 9: Evaluation on Test Set")
        info("=" * 80)
        
        data_module = TimeSeriesDataModule(
            X=self.X,
            y=self.y,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=64,
            seq_length=None if self.architecture == "mlp" else 24,
            scale_features=False
        )
        
        self.test_metrics = self.trainer.evaluate(data_module.test_dataloader())
        
        info("\nTest Set Results:")
        info(f"  Loss:     {self.test_metrics['loss']:.4f}")
        info(f"  Accuracy: {self.test_metrics['accuracy']:.2%}")
        
        # Per-class accuracy (important for binary)
        info("\nPer-Class Accuracy:")
        info(f"  NO_TRANSITION (0): {self.test_metrics['class_accuracy'].get(0, 0):.2%}")
        info(f"  TRANSITION (1):    {self.test_metrics['class_accuracy'].get(1, 0):.2%}")
        
        # Calculate additional metrics for binary classification
        preds = self.test_metrics.get('predictions', [])
        targets = self.test_metrics.get('targets', [])
        
        if len(preds) > 0 and len(targets) > 0:
            # True positives, false positives, etc.
            tp = ((preds == 1) & (targets == 1)).sum()
            fp = ((preds == 1) & (targets == 0)).sum()
            fn = ((preds == 0) & (targets == 1)).sum()
            tn = ((preds == 0) & (targets == 0)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            info(f"\nBinary Classification Metrics:")
            info(f"  Precision: {precision:.2%} (of predicted transitions, how many were real)")
            info(f"  Recall:    {recall:.2%} (of real transitions, how many did we catch)")
            info(f"  F1 Score:  {f1:.2%}")
            
            self.test_metrics['precision'] = precision
            self.test_metrics['recall'] = recall
            self.test_metrics['f1'] = f1
        
        info("\n✓ Evaluation complete")
        
        return self.test_metrics
    
    def save(self, version: str = "v1") -> None:
        """Save model, preprocessor, selector, and metrics."""
        
        info("\n" + "=" * 80)
        info("Step 10: Saving Model and Artifacts")
        info("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.symbol}_{self.timeframe}_transition_{version}"
        
        # Create directories
        trained_dir = Path("models/trained/transition")
        trained_dir.mkdir(parents=True, exist_ok=True)
        
        outputs_dir = Path("outputs/models/transition") / self.symbol / self.timeframe
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model checkpoint
        model_path = trained_dir / f"{base_name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.architecture,
            'model_kwargs': self.model.get_config() if hasattr(self.model, 'get_config') else {},
            'input_dim': self.X_train.shape[1],
            'n_classes': 2,
            'selected_features': self.selected_features,
            'timestamp': timestamp
        }, model_path)
        info(f"Saved model: {model_path}")
        
        # Save preprocessor
        preprocessor_path = trained_dir / f"{base_name}_preprocessor.pkl"
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        info(f"Saved preprocessor: {preprocessor_path}")
        
        # Save selector
        selector_path = trained_dir / f"{base_name}_selector.pkl"
        with open(selector_path, 'wb') as f:
            pickle.dump(self.selector, f)
        info(f"Saved selector: {selector_path}")
        
        # Save config
        config_data = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'architecture': self.architecture,
            'input_dim': self.X_train.shape[1],
            'n_classes': 2,
            'selected_features': self.selected_features,
            'train_samples': len(self.X_train),
            'val_samples': len(self.X_val),
            'test_samples': len(self.X_test),
            'timestamp': timestamp,
            'version': version
        }
        
        config_path = outputs_dir / f"{base_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        info(f"Saved config: {config_path}")
        
        # Save metrics (JSON-safe)
        test_metrics_json = {
            'loss': float(self.test_metrics['loss']),
            'accuracy': float(self.test_metrics['accuracy']),
            'class_accuracy': {str(k): float(v) for k, v in self.test_metrics['class_accuracy'].items()},
            'n_samples': int(self.test_metrics['n_samples']),
            'precision': float(self.test_metrics.get('precision', 0)),
            'recall': float(self.test_metrics.get('recall', 0)),
            'f1': float(self.test_metrics.get('f1', 0))
        }
        
        metrics_data = {
            'train_history': self.train_history,
            'test_metrics': test_metrics_json,
            'timestamp': timestamp
        }
        
        metrics_path = outputs_dir / f"{base_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        info(f"Saved metrics: {metrics_path}")
        
        info("\n✓ All artifacts saved successfully")
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict transition probability for new data."""
        
        if self.model is None:
            raise ValueError("Must train or load a model before predict()")
        
        X_preprocessed = self.preprocessor.transform(features)
        X_selected = self.selector.transform(X_preprocessed)
        X_tensor = torch.FloatTensor(X_selected)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            # Return probability of transition (class 1)
            transition_prob = probs[:, 1]
        
        return preds.numpy(), transition_prob.numpy()
    
    @classmethod
    def load(cls, symbol: str, timeframe: str, version: str = "v1") -> 'TransitionDetector':
        """Load a saved transition detector."""
        
        base_name = f"{symbol}_{timeframe}_transition_{version}"
        trained_dir = Path("models/trained/transition")
        outputs_dir = Path("outputs/models/transition") / symbol / timeframe
        
        # Load config
        config_path = outputs_dir / f"{base_name}_config.json"
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create instance
        detector = cls(
            symbol=symbol,
            timeframe=timeframe,
            architecture=config_data['architecture']
        )
        
        # Load preprocessor
        with open(trained_dir / f"{base_name}_preprocessor.pkl", 'rb') as f:
            detector.preprocessor = pickle.load(f)
        
        # Load selector
        with open(trained_dir / f"{base_name}_selector.pkl", 'rb') as f:
            detector.selector = pickle.load(f)
        
        # Load model
        checkpoint = torch.load(trained_dir / f"{base_name}.pt", map_location='cpu')
        detector.model = create_model(
            checkpoint['model_type'],
            **checkpoint.get('model_kwargs', {
                'input_dim': checkpoint['input_dim'],
                'n_classes': checkpoint['n_classes']
            })
        )
        detector.model.load_state_dict(checkpoint['model_state_dict'])
        detector.selected_features = checkpoint['selected_features']
        
        info(f"Loaded transition model for {symbol} {timeframe}")
        
        return detector