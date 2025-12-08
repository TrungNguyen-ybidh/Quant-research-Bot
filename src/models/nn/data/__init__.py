"""
Dataset Module
==============
PyTorch Dataset classes and data modules for time-series ML.

Available classes:
- TabularDataset: For MLP models (single timestep per sample)
- SequenceDataset: For LSTM/GRU models (sequence of timesteps)
- TimeSeriesDataModule: Complete data pipeline with train/val/test splits

Example:
    >>> from src.models.nn.data import TabularDataset, TimeSeriesDataModule
    >>> 
    >>> # For MLP
    >>> dataset = TabularDataset(X, y)
    >>> 
    >>> # Complete pipeline
    >>> dm = TimeSeriesDataModule(
    ...     X_train, y_train,
    ...     X_val, y_val,
    ...     X_test, y_test,
    ...     batch_size=64
    ... )
    >>> train_loader = dm.train_dataloader()
"""

from .dataset import (
    TabularDataset,
    SequenceDataset,
    TimeSeriesDataModule
)

__all__ = [
    'TabularDataset',
    'SequenceDataset',
    'TimeSeriesDataModule'
]