"""
Volatility Regime Classifier
============================
Classifies market volatility into regimes: LOW, NORMAL, HIGH, CRISIS

Regimes:
    - LOW (0): Below 25th percentile - quiet markets
    - NORMAL (1): 25th to 75th percentile - typical conditions
    - HIGH (2): 75th to 95th percentile - elevated volatility
    - CRISIS (3): Above 95th percentile - extreme volatility

Usage:
    from src.models.state import VolatilityRegimeClassifier
    
    classifier = VolatilityRegimeClassifier(symbol="EURUSD", timeframe="1h")
    classifier.prepare_data()
    classifier.train()
    classifier.evaluate()
    classifier.save()
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
from src.models.targets.regime_labeler import VolatilityRegimeLabeler
from src.models.nn import create_model, TimeSeriesDataModule, Trainer, TrainerConfig
from src.utils.logger import info, success, error, warning


class VolatilityRegimeClassifier:
    """
    Volatility regime classification pipeline.
    
    Workflow:
        1. Load processed features + raw OHLC
        2. Create volatility regime labels
        3. Preprocess and select features
        4. Split data (time-series aware)
        5. Train neural network
        6. Evaluate on test set
        7. Save model and metrics
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
        Initialize volatility regime classifier.
        
        Args:
            symbol: Currency pair (e.g., "EURUSD")
            timeframe: Timeframe (e.g., "1h", "1d")
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
        self.labeler = None
        self.splitter = None
        self.model = None
        self.trainer = None
        
        # Data containers
        self.df = None
        self.df_raw = None
        self.X = None
        self.y = None
        self.selected_features = None
        
        # Results
        self.train_history = None
        self.test_metrics = None
        
        info(f"Initialized VolatilityRegimeClassifier for {symbol} {timeframe} with {architecture}")
    
    def prepare_data(self) -> None:
        """Load data, create labels, preprocess, and split."""
        
        info("=" * 80)
        info("Step 1: Loading Data")
        info("=" * 80)
        
        # Load raw OHLC data for volatility calculation
        raw_path = f"data/raw/clock/{self.symbol}_{self.timeframe}.parquet"
        self.df_raw = load_parquet_file(raw_path)
        if self.df_raw is None:
            raise FileNotFoundError(f"Failed to load raw data: {raw_path}")
        
        # Load processed features
        processed_path = f"data/processed/{self.symbol}_{self.timeframe}_features.parquet"
        self.df = load_parquet_file(processed_path)
        if self.df is None:
            raise FileNotFoundError(f"Failed to load processed data: {processed_path}")
        
        info(f"Loaded {len(self.df)} bars from {processed_path}")
        info(f"Features: {self.df.shape[1]} columns")
        
        # Create volatility regime labels
        info("\n" + "=" * 80)
        info("Step 2: Creating Volatility Regime Labels")
        info("=" * 80)
        
        vol_config = self.config.get('models', {}).get('volatility', {})
        self.labeler = VolatilityRegimeLabeler(
            lookback=vol_config.get('lookback', 20),
            low_pct=vol_config.get('low_pct', 25),
            high_pct=vol_config.get('high_pct', 75),
            crisis_pct=vol_config.get('crisis_pct', 95),
            vol_method=vol_config.get('vol_method', 'realized')
        )
        
        self.y = self.labeler.build(self.df_raw)
        info(f"Created {len(self.y)} volatility regime labels")
        
        # Log distribution
        stats = self.labeler.get_stats()
        info("\nVolatility Regime Distribution:")
        info(f"  LOW (0):    {stats['n_low']:6d} ({stats['pct_low']:5.1f}%)")
        info(f"  NORMAL (1): {stats['n_normal']:6d} ({stats['pct_normal']:5.1f}%)")
        info(f"  HIGH (2):   {stats['n_high']:6d} ({stats['pct_high']:5.1f}%)")
        info(f"  CRISIS (3): {stats['n_crisis']:6d} ({stats['pct_crisis']:5.1f}%)")
        
        # Align data and remove NaN labels
        info("\n" + "=" * 80)
        info("Step 3: Aligning Data")
        info("=" * 80)
        
        # Reset indices
        self.df = self.df.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)
        
        # Filter valid samples
        valid_mask = ~self.y.isna()
        self.df = self.df[valid_mask].reset_index(drop=True)
        self.y = self.y[valid_mask].reset_index(drop=True)
        
        info(f"After removing NaN labels: {len(self.df)} samples")
        
        # Preprocessing
        info("\n" + "=" * 80)
        info("Step 4: Preprocessing Features")
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
        info("Step 5: Feature Selection")
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
        info("Step 6: Train/Val/Test Split")
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
        
        info("\n✓ Data preparation complete")
    
    def train(self) -> Dict[str, Any]:
        """Train the volatility regime classification model."""
        
        if self.X_train is None:
            raise ValueError("Must call prepare_data() before train()")
        
        info("\n" + "=" * 80)
        info("Step 7: Training Neural Network")
        info("=" * 80)
        
        nn_config = self.config.get('models', {}).get('nn', {})
        
        input_dim = self.X_train.shape[1]
        n_classes = 4  # LOW, NORMAL, HIGH, CRISIS
        
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
        info(f"  Output classes: {n_classes}")
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
        
        # Training config
        trainer_config = TrainerConfig(
            epochs=nn_config.get('default_epochs', 100),
            lr=nn_config.get('learning_rate', 0.001),
            weight_decay=nn_config.get('weight_decay', 0.0001),
            patience=nn_config.get('early_stopping_patience', 10),
            use_class_weights=True,
            device=nn_config.get('device', 'cpu')
        )
        
        info(f"\nTraining Configuration:")
        info(f"  Max epochs: {trainer_config.epochs}")
        info(f"  Learning rate: {trainer_config.lr}")
        info(f"  Early stopping patience: {trainer_config.patience}")
        
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
        info("Step 8: Evaluation on Test Set")
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
        
        # Per-class accuracy
        info("\nPer-Class Accuracy:")
        class_names = ['LOW', 'NORMAL', 'HIGH', 'CRISIS']
        for i, name in enumerate(class_names):
            acc = self.test_metrics['class_accuracy'].get(i, 0)
            info(f"  {name}: {acc:.2%}")
        
        info("\n✓ Evaluation complete")
        
        return self.test_metrics
    
    def save(self, version: str = "v1") -> None:
        """Save model, preprocessor, selector, and metrics."""
        
        info("\n" + "=" * 80)
        info("Step 9: Saving Model and Artifacts")
        info("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.symbol}_{self.timeframe}_volatility_{version}"
        
        # Create directories
        trained_dir = Path("models/trained/volatility")
        trained_dir.mkdir(parents=True, exist_ok=True)
        
        outputs_dir = Path("outputs/models/volatility") / self.symbol / self.timeframe
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model checkpoint
        model_path = trained_dir / f"{base_name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.architecture,
            'model_kwargs': self.model.get_config() if hasattr(self.model, 'get_config') else {},
            'input_dim': self.X_train.shape[1],
            'n_classes': 4,
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
            'n_classes': 4,
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
            'n_samples': int(self.test_metrics['n_samples'])
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
        """Predict volatility regime for new data."""
        
        if self.model is None:
            raise ValueError("Must train or load a model before predict()")
        
        # Preprocess (returns DataFrame)
        X_preprocessed = self.preprocessor.transform(features)
        
        # Select features (expects DataFrame, returns DataFrame)
        X_selected = self.selector.transform(X_preprocessed)
        
        # NOW convert to numpy array for the model
        if isinstance(X_selected, pd.DataFrame):
            X_selected = X_selected.values
        
        X_tensor = torch.FloatTensor(X_selected)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1)[0]
        
        return preds.numpy(), confidence.numpy()
    
    @classmethod
    def load(cls, symbol: str, timeframe: str, version: str = "v1") -> 'VolatilityRegimeClassifier':
        """Load a saved volatility regime classifier."""
        
        base_name = f"{symbol}_{timeframe}_volatility_{version}"
        trained_dir = Path("models/trained/volatility")
        outputs_dir = Path("outputs/models/volatility") / symbol / timeframe
        
        # Load config
        config_path = outputs_dir / f"{base_name}_config.json"
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create instance
        classifier = cls(
            symbol=symbol,
            timeframe=timeframe,
            architecture=config_data['architecture']
        )
        
        # Load preprocessor
        with open(trained_dir / f"{base_name}_preprocessor.pkl", 'rb') as f:
            classifier.preprocessor = pickle.load(f)
        
        # Load selector
        with open(trained_dir / f"{base_name}_selector.pkl", 'rb') as f:
            classifier.selector = pickle.load(f)
        
        # Load model
        checkpoint = torch.load(trained_dir / f"{base_name}.pt", map_location='cpu')
        classifier.model = create_model(
            checkpoint['model_type'],
            **checkpoint.get('model_kwargs', {
                'input_dim': checkpoint['input_dim'],
                'n_classes': checkpoint['n_classes']
            })
        )
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        classifier.selected_features = checkpoint['selected_features']
        
        info(f"Loaded volatility model for {symbol} {timeframe}")
        
        return classifier