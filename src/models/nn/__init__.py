"""
Neural Network Module
=====================
PyTorch-based neural networks for quantitative research.

Submodules:
- architectures: Model architectures (MLP, LSTM, GRU)
- data: Dataset classes and data modules
- training: Training loop and utilities

Example:
    >>> from src.models.nn.architectures import MLPClassifier, create_model
    >>> from src.models.nn.data import TabularDataset, TimeSeriesDataModule
    >>> from src.models.nn.training import Trainer, TrainerConfig
    >>> 
    >>> # Create model
    >>> model = create_model('mlp', input_dim=30, n_classes=3)
    >>> 
    >>> # Prepare data
    >>> dm = TimeSeriesDataModule(X_train, y_train, X_val, y_val)
    >>> 
    >>> # Train
    >>> config = TrainerConfig(max_epochs=100, learning_rate=0.001)
    >>> trainer = Trainer(model, config)
    >>> history = trainer.fit(dm.train_dataloader(), dm.val_dataloader())
"""

# Import key classes for convenience
from .architectures import (
    MLPClassifier,
    LSTMClassifier,
    GRUClassifier,
    create_model
)

from .data import (
    TabularDataset,
    SequenceDataset,
    TimeSeriesDataModule
)

from .training import (
    TrainerConfig,
    EarlyStopping,
    Trainer
)

__all__ = [
    # Architectures
    'MLPClassifier',
    'LSTMClassifier',
    'GRUClassifier',
    'create_model',
    
    # Data
    'TabularDataset',
    'SequenceDataset',
    'TimeSeriesDataModule',
    
    # Training
    'TrainerConfig',
    'EarlyStopping',
    'Trainer',
]