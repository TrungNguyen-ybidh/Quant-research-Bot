"""
Regime Classifier
=================
End-to-end pipeline for training regime classification models.

Integrates:
- Data loading (existing loader.py)
- Preprocessing (existing preprocessing.py)
- Feature selection (existing feature_selection.py)
- Target creation (new regime_labeler.py)
- Model training (new nn module)
- Validation (existing data_splitter.py)

Usage:
    >>> from src.models.state import RegimeClassifier
    >>> 
    >>> classifier = RegimeClassifier(
    ...     symbol="EURUSD",
    ...     timeframe="1h",
    ...     architecture="mlp",  # or "lstm"
    ...     config=config
    ... )
    >>> 
    >>> classifier.prepare_data()
    >>> classifier.train()
    >>> metrics = classifier.evaluate()
    >>> classifier.save()
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
from src.models.targets.regime_labeler import RegimeLabeler
from src.models.nn import create_model, TimeSeriesDataModule, Trainer, TrainerConfig
from src.utils.logger import info, success, error, warning


class RegimeClassifier:
    """
    Complete regime classification pipeline.
    
    Workflow:
        1. Load processed features
        2. Create regime labels
        3. Preprocess and select features
        4. Split data (time-series aware)
        5. Train neural network
        6. Evaluate with walk-forward validation
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
        Initialize regime classifier.
        
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
        
        # Components (removed path_builder and data_loader - using functions instead)
        self.preprocessor = None
        self.selector = None
        self.labeler = None
        self.splitter = None
        self.model = None
        self.trainer = None
        
        # Data containers
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.selected_features = None
        
        # Results
        self.train_history = None
        self.test_metrics = None
        
        info(f"Initialized RegimeClassifier for {symbol} {timeframe} with {architecture}")
    
    def prepare_data(self) -> None:
        """
        Load data, create labels, preprocess, and split.
        
        Steps:
            1. Load processed features
            2. Create regime labels
            3. Remove NaN labels
            4. Preprocess features
            5. Select features
            6. Split into train/val/test
        """
        info("="*80)
        info("Step 1: Loading Data")
        info("="*80)
        
        # Load RAW OHLC data for regime labeling
        raw_path = f"data/raw/clock/{self.symbol}_{self.timeframe}.parquet"
        df_raw = load_parquet_file(raw_path)
        if df_raw is None:
            raise FileNotFoundError(f"Failed to load raw OHLC data: {raw_path}")
        
        # Load processed features (for training)
        processed_path = f"data/processed/{self.symbol}_{self.timeframe}_features.parquet"
        self.df = load_parquet_file(processed_path)
        if self.df is None:
            raise FileNotFoundError(f"Failed to load processed data: {processed_path}")
        info(f"Loaded {len(self.df)} bars from {processed_path}")
        info(f"Features: {self.df.shape[1]} columns")
        
        # Create regime labels
        info("\n" + "="*80)
        info("Step 2: Creating Regime Labels")
        info("="*80)
        
        regime_config = self.config.get('models', {}).get('regime', {}).get('labeling', {})
        self.labeler = RegimeLabeler(
            trend_threshold=regime_config.get('trend_threshold', 0.002),
            lookforward=regime_config.get('lookforward', 12),
            dominance_ratio=regime_config.get('dominance_ratio', 1.5)
        )
        
        # Create labels from RAW data (has OHLC columns)
        self.y = self.labeler.build(df_raw)
        info(f"Created {len(self.y)} regime labels")
        
        # Log regime distribution
        regime_counts = pd.Series(self.y).value_counts()
        regime_pcts = pd.Series(self.y).value_counts(normalize=True) * 100
        info("\nRegime Distribution:")
        info(f"  RANGING (0):   {regime_counts.get(0, 0):6d} ({regime_pcts.get(0, 0):5.1f}%)")
        info(f"  TREND_UP (1):  {regime_counts.get(1, 0):6d} ({regime_pcts.get(1, 0):5.1f}%)")
        info(f"  TREND_DOWN (2):{regime_counts.get(2, 0):6d} ({regime_pcts.get(2, 0):5.1f}%)")
        
        # Remove rows with NaN labels
        # Match indices between raw data (used for labels) and processed data
        valid_idx = ~np.isnan(self.y)
        
        # Reset indices to align
        self.df = self.df.reset_index(drop=True)
        self.y = pd.Series(self.y).reset_index(drop=True)
        
        # Now filter both by valid indices
        valid_mask = ~self.y.isna()
        self.df = self.df[valid_mask].reset_index(drop=True)
        self.y = self.y[valid_mask].reset_index(drop=True)
        info(f"\nAfter removing NaN labels: {len(self.df)} samples")
        
        # Preprocessing
        info("\n" + "="*80)
        info("Step 3: Preprocessing Features")
        info("="*80)
        
        self.preprocessor = FeaturePreprocessor(
            default_method='robust',
            handle_inf=True,
            clip_outliers=True,
            outlier_std=3.0
        )
        
        X_preprocessed = self.preprocessor.fit_transform(self.df)
        info(f"Preprocessed features: {X_preprocessed.shape}")
        
        # Feature Selection
        info("\n" + "="*80)
        info("Step 4: Feature Selection")
        info("="*80)
        
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
        info("\n" + "="*80)
        info("Step 5: Train/Val/Test Split")
        info("="*80)
        
        self.splitter = TimeSeriesSplitter(
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        splits = self.splitter.split(self.X, self.y)
        self.X_train, self.X_val, self.X_test = splits.X_train, splits.X_val, splits.X_test
        self.y_train, self.y_val, self.y_test = splits.y_train, splits.y_val, splits.y_test
        
        info(f"Train: {len(self.X_train)} samples")
        info(f"Val:   {len(self.X_val)} samples")
        info(f"Test:  {len(self.X_test)} samples")
        
        # Check class balance in train set
        train_counts = pd.Series(self.y_train).value_counts()
        train_pcts = pd.Series(self.y_train).value_counts(normalize=True) * 100
        info("\nTrain Set Distribution:")
        info(f"  RANGING (0):   {train_counts.get(0, 0):6d} ({train_pcts.get(0, 0):5.1f}%)")
        info(f"  TREND_UP (1):  {train_counts.get(1, 0):6d} ({train_pcts.get(1, 0):5.1f}%)")
        info(f"  TREND_DOWN (2):{train_counts.get(2, 0):6d} ({train_pcts.get(2, 0):5.1f}%)")
        
        info("\n✓ Data preparation complete")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the regime classification model.
        
        Returns:
            Training history with losses and metrics
        """
        if self.X_train is None:
            raise ValueError("Must call prepare_data() before train()")
        
        info("\n" + "="*80)
        info("Step 6: Training Neural Network")
        info("="*80)
        
        # Model config
        nn_config = self.config.get('models', {}).get('nn', {})
        regime_config = self.config.get('models', {}).get('regime', {})
        
        input_dim = self.X_train.shape[1]
        n_classes = 3  # RANGING, TREND_UP, TREND_DOWN
        
        # Create model
        if self.architecture == "mlp":
            model_kwargs = {
                'input_dim': input_dim,
                'hidden_dims': regime_config.get('hidden_dims', [128, 64, 32]),
                'n_classes': n_classes,
                'dropout': regime_config.get('dropout', 0.3),
                'batch_norm': True
            }
        elif self.architecture == "lstm":
            model_kwargs = {
                'input_dim': input_dim,
                'hidden_dim': regime_config.get('hidden_dim', 64),
                'num_layers': regime_config.get('num_layers', 2),
                'n_classes': n_classes,
                'dropout': regime_config.get('dropout', 0.3)
            }
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        self.model = create_model(self.architecture, **model_kwargs)
        info(f"Created {self.architecture.upper()} model:")
        info(f"  Input dim: {input_dim}")
        info(f"  Output classes: {n_classes}")
        info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create data module (it will do the splitting internally)
        # Note: Features are already preprocessed/scaled, so set scale_features=False
        data_module = TimeSeriesDataModule(
            X=self.X,
            y=self.y,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=nn_config.get('batch_size', 64),
            seq_length=None if self.architecture == "mlp" else 24,
            scale_features=False  # Already scaled by FeaturePreprocessor
        )
        
        # Training config
        training_config = self.config.get('training', {})
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
        info(f"  Batch size: {data_module.batch_size}")
        info(f"  Early stopping patience: {trainer_config.patience}")
        info(f"  Device: {trainer_config.device}")
        
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
        """
        Evaluate the trained model on test set.
        
        Returns:
            Dictionary of test metrics
        """
        if self.trainer is None:
            raise ValueError("Must call train() before evaluate()")
        
        info("\n" + "="*80)
        info("Step 7: Evaluation on Test Set")
        info("="*80)
        
        data_module = TimeSeriesDataModule(
            X=self.X,
            y=self.y,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            batch_size=64,
            seq_length=None if self.architecture == "mlp" else 24,
            scale_features=False  # Already scaled by FeaturePreprocessor
        )
        
        self.test_metrics = self.trainer.evaluate(data_module.test_dataloader())
        
        info("\nTest Set Results:")
        info(f"  Loss:     {self.test_metrics['loss']:.4f}")
        info(f"  Accuracy: {self.test_metrics['accuracy']:.2%}")
        
        info("\n✓ Evaluation complete")
        
        return self.test_metrics
    
    def save(self, version: str = "v1") -> None:
        """
        Save model, preprocessor, selector, and metrics.
        
        Args:
            version: Version string for saved files
        """
        info("\n" + "="*80)
        info("Step 8: Saving Model and Artifacts")
        info("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.symbol}_{self.timeframe}_regime_{version}"
        
        # Create directories
        trained_dir = Path("models/trained/regime")
        trained_dir.mkdir(parents=True, exist_ok=True)
        
        outputs_dir = Path("outputs/models/regime") / self.symbol / self.timeframe
        outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model checkpoint
        model_path = trained_dir / f"{base_name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.architecture,
            'model_kwargs': self.model.get_config() if hasattr(self.model, 'get_config') else {},
            'input_dim': self.X_train.shape[1],
            'n_classes': 3,
            'selected_features': self.selected_features,
            'timestamp': timestamp
        }, model_path)
        info(f"Saved model checkpoint: {model_path}")
        
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
            'n_classes': 3,
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
        
        # Save metrics
        # Create a JSON-safe copy of test_metrics (exclude numpy arrays)
        test_metrics_safe = {
            k: v for k, v in self.test_metrics.items() 
            if k not in ['predictions', 'targets', 'probabilities', 'confusion_matrix']
        }
        # Add confusion matrix as list (it's already a list from evaluate())
        if 'confusion_matrix' in self.test_metrics:
            test_metrics_safe['confusion_matrix'] = self.test_metrics['confusion_matrix']
        
        metrics_data = {
            'train_history': self.train_history,
            'test_metrics': test_metrics_safe,
            'regime_distribution': {
                'train': pd.Series(self.y_train).value_counts().to_dict(),
                'val': pd.Series(self.y_val).value_counts().to_dict(),
                'test': pd.Series(self.y_test).value_counts().to_dict()
            },
            'timestamp': timestamp
        }
        
        metrics_path = outputs_dir / f"{base_name}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        info(f"Saved metrics: {metrics_path}")
        
        info("\n✓ All artifacts saved successfully")
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regime for new data.
        
        Args:
            features: DataFrame with same features as training data
        
        Returns:
            Tuple of (predicted_labels, confidence_scores)
        """
        if self.model is None:
            raise ValueError("Must train or load a model before predict()")
        
        # Preprocess (returns DataFrame)
        X_preprocessed = self.preprocessor.transform(features)
        
        # Select features (expects DataFrame, returns DataFrame)
        X_selected = self.selector.transform(X_preprocessed)
        
        # NOW convert to numpy array for the model
        if isinstance(X_selected, pd.DataFrame):
            X_selected = X_selected.values
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_selected)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1)[0]
        
        return preds.numpy(), confidence.numpy()
    
    @classmethod
    def load(cls, symbol: str, timeframe: str, version: str = "v1") -> 'RegimeClassifier':
        """
        Load a saved regime classifier.
        
        Args:
            symbol: Currency pair
            timeframe: Timeframe
            version: Model version
        
        Returns:
            Loaded RegimeClassifier instance
        """
        base_name = f"{symbol}_{timeframe}_regime_{version}"
        trained_dir = Path("models/trained/regime")
        
        # Load config
        outputs_dir = Path("outputs/models/regime") / symbol / timeframe
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
        preprocessor_path = trained_dir / f"{base_name}_preprocessor.pkl"
        with open(preprocessor_path, 'rb') as f:
            classifier.preprocessor = pickle.load(f)
        
        # Load selector
        selector_path = trained_dir / f"{base_name}_selector.pkl"
        with open(selector_path, 'rb') as f:
            classifier.selector = pickle.load(f)
        
        # Load model
        model_path = trained_dir / f"{base_name}.pt"
        checkpoint = torch.load(model_path, map_location='cpu')
        
        classifier.model = create_model(
            checkpoint['model_type'],
            **checkpoint.get('model_kwargs', {
                'input_dim': checkpoint['input_dim'],
                'n_classes': checkpoint['n_classes']
            })
        )
        classifier.model.load_state_dict(checkpoint['model_state_dict'])
        classifier.selected_features = checkpoint['selected_features']
        
        info(f"Loaded model from {model_path}")
        
        return classifier