"""
Training Module
===============
Neural network training infrastructure with early stopping and checkpointing.

Available classes:
- TrainerConfig: Configuration dataclass for training
- EarlyStopping: Early stopping callback
- Trainer: Complete training loop with validation and checkpointing

Example:
    >>> from src.models.nn.training import Trainer, TrainerConfig
    >>> 
    >>> config = TrainerConfig(
    ...     max_epochs=100,
    ...     learning_rate=0.001,
    ...     early_stopping_patience=10
    ... )
    >>> 
    >>> trainer = Trainer(model, config)
    >>> history = trainer.fit(train_loader, val_loader)
    >>> 
    >>> # Evaluate
    >>> test_metrics = trainer.evaluate(test_loader)
"""

from .trainer import (
    TrainerConfig,
    EarlyStopping,
    Trainer
)

__all__ = [
    'TrainerConfig',
    'EarlyStopping',
    'Trainer'
]