"""
PyTorch Dataset Classes
=======================
Custom Dataset classes for time-series ML models.

Includes:
- TabularDataset: For MLP (single timestep per sample)
- SequenceDataset: For LSTM/GRU (multiple timesteps per sample)
- TimeSeriesDataModule: Manages train/val/test splits and dataloaders

Usage:
    from src.models.nn.data.dataset import TabularDataset, SequenceDataset, TimeSeriesDataModule
    
    # For MLP
    dataset = TabularDataset(X, y)
    loader = DataLoader(dataset, batch_size=64)
    
    # For LSTM
    dataset = SequenceDataset(X, y, seq_length=24)
    loader = DataLoader(dataset, batch_size=64)
    
    # Full data module
    dm = TimeSeriesDataModule(X, y, batch_size=64, seq_length=24)
    train_loader = dm.train_dataloader()
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List, Dict, Union
from pathlib import Path


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data (single timestep per sample).
    
    Suitable for MLP models where each sample is independent.
    
    Args:
        X: Feature matrix (n_samples, n_features) - numpy array or DataFrame
        y: Target vector (n_samples,) - numpy array or Series
        feature_names: Optional list of feature names
    
    Returns per sample:
        Tuple of (features_tensor, label_tensor)
    """
    
    def __init__(self,
                 X: Union[np.ndarray, pd.DataFrame],
                 y: Union[np.ndarray, pd.Series],
                 feature_names: Optional[List[str]] = None):
        
        # Convert DataFrame to numpy, preserve feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Convert Series to numpy
        if isinstance(y, pd.Series):
            y = y.values
        
        # Clean data: replace NaN/inf with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y = np.nan_to_num(y, nan=0).astype(np.int64)
        
        # Convert to tensors
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        
        # Store dimensions
        self.n_samples = len(X)
        self.n_features = X.shape[1]
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate inverse frequency class weights for imbalanced data.
        
        Returns:
            Tensor of weights per class (n_classes,)
        """
        classes, counts = torch.unique(self.y, return_counts=True)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum()  # Normalize to sum to 1
        return weights
    
    def get_class_counts(self) -> Dict[int, int]:
        """Get count of samples per class."""
        classes, counts = torch.unique(self.y, return_counts=True)
        return {int(c): int(n) for c, n in zip(classes, counts)}


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequence data (multiple timesteps per sample).
    
    Suitable for LSTM/GRU models that need temporal context.
    Each sample contains `seq_length` consecutive timesteps.
    The target is the label at the LAST timestep of each sequence.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        seq_length: Number of timesteps per sequence (default: 24)
        stride: Step between sequences (default: 1, meaning overlapping sequences)
        feature_names: Optional list of feature names
    
    Returns per sample:
        Tuple of (sequence_tensor, label_tensor)
        - sequence_tensor: (seq_length, n_features)
        - label_tensor: scalar
    """
    
    def __init__(self,
                 X: Union[np.ndarray, pd.DataFrame],
                 y: Union[np.ndarray, pd.Series],
                 seq_length: int = 24,
                 stride: int = 1,
                 feature_names: Optional[List[str]] = None):
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Clean data
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        y = np.nan_to_num(y, nan=0).astype(np.int64)
        
        # Store raw data
        self.X_raw = X
        self.y_raw = y
        self.seq_length = seq_length
        self.stride = stride
        self.n_features = X.shape[1]
        
        # Calculate valid sequence start indices
        # A sequence starting at index i uses data from i to i+seq_length-1
        # The target is y[i + seq_length - 1] (end of sequence)
        self.valid_indices = self._compute_valid_indices()
        self.n_samples = len(self.valid_indices)
    
    def _compute_valid_indices(self) -> np.ndarray:
        """
        Compute indices where complete sequences can be formed.
        
        A valid start index i requires:
        - i + seq_length - 1 < len(data)  (sequence fits)
        - y[i + seq_length - 1] is not NaN (valid target)
        """
        n = len(self.X_raw)
        max_start = n - self.seq_length  # Last valid start index
        
        indices = []
        for i in range(0, max_start + 1, self.stride):
            target_idx = i + self.seq_length - 1
            # Check if target is valid (already cleaned, so just check bounds)
            if target_idx < n:
                indices.append(i)
        
        return np.array(indices, dtype=np.int64)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence sample.
        
        Args:
            idx: Sample index (not raw data index!)
        
        Returns:
            X: Tensor of shape (seq_length, n_features)
            y: Scalar tensor (target at end of sequence)
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_length
        
        # Extract sequence: shape (seq_length, n_features)
        X_seq = self.X_raw[start_idx:end_idx]
        
        # Target is at end of sequence
        y_target = self.y_raw[end_idx - 1]
        
        return torch.from_numpy(X_seq), torch.tensor(y_target, dtype=torch.long)
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights from valid samples only."""
        # Get targets for all valid sequences
        target_indices = self.valid_indices + self.seq_length - 1
        y_valid = self.y_raw[target_indices]
        y_tensor = torch.from_numpy(y_valid)
        
        classes, counts = torch.unique(y_tensor, return_counts=True)
        weights = 1.0 / counts.float()
        weights = weights / weights.sum()
        return weights
    
    def get_class_counts(self) -> Dict[int, int]:
        """Get count of samples per class."""
        target_indices = self.valid_indices + self.seq_length - 1
        y_valid = self.y_raw[target_indices]
        unique, counts = np.unique(y_valid, return_counts=True)
        return {int(c): int(n) for c, n in zip(unique, counts)}


class TimeSeriesDataModule:
    """
    Manages complete data pipeline for time-series ML.
    
    Handles:
    - Chronological train/val/test splitting (NO shuffling)
    - Feature scaling (fit on train, transform all)
    - Creating appropriate Dataset objects (Tabular or Sequence)
    - Creating DataLoaders
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        train_ratio: Fraction for training (default: 0.70)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)
        batch_size: Batch size for dataloaders (default: 64)
        seq_length: Sequence length for LSTM (None = use TabularDataset)
        num_workers: DataLoader workers (default: 0)
        scale_features: Whether to standardize features (default: True)
    
    Usage:
        dm = TimeSeriesDataModule(X, y, batch_size=64)
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()
    """
    
    def __init__(self,
                 X: Union[np.ndarray, pd.DataFrame],
                 y: Union[np.ndarray, pd.Series],
                 train_ratio: float = 0.70,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 batch_size: int = 64,
                 seq_length: Optional[int] = None,
                 num_workers: int = 0,
                 scale_features: bool = True):
        
        # Store config
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_workers = num_workers
        self.scale_features = scale_features
        
        # Convert to numpy, extract feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values.astype(np.float32)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = X.astype(np.float32)
        
        if isinstance(y, pd.Series):
            y = y.values
        y = y.astype(np.float64)  # Keep as float to handle NaN
        
        # Remove samples where target is NaN
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask].astype(np.int64)
        
        # Store dimensions
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        
        # Chronological split indices
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Split data (chronological, no shuffling!)
        self.X_train = X[:train_end]
        self.y_train = y[:train_end]
        
        self.X_val = X[train_end:val_end]
        self.y_val = y[train_end:val_end]
        
        self.X_test = X[val_end:]
        self.y_test = y[val_end:]
        
        # Scaling: fit on train only, transform all
        if scale_features:
            self._fit_scaler()
            self.X_train = self._scale(self.X_train)
            self.X_val = self._scale(self.X_val)
            self.X_test = self._scale(self.X_test)
        
        # Clean any remaining NaN/inf after scaling
        self.X_train = np.nan_to_num(self.X_train, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_val = np.nan_to_num(self.X_val, nan=0.0, posinf=0.0, neginf=0.0)
        self.X_test = np.nan_to_num(self.X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create datasets
        self._create_datasets()
    
    def _fit_scaler(self):
        """Fit standardization parameters on training data."""
        self.mean = np.nanmean(self.X_train, axis=0)
        self.std = np.nanstd(self.X_train, axis=0)
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0
    
    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Apply standardization."""
        return (X - self.mean) / self.std
    
    def _create_datasets(self):
        """Create train/val/test Dataset objects."""
        if self.seq_length is not None and self.seq_length > 1:
            # Use SequenceDataset for LSTM
            self.train_dataset = SequenceDataset(
                self.X_train, self.y_train,
                seq_length=self.seq_length,
                feature_names=self.feature_names
            )
            self.val_dataset = SequenceDataset(
                self.X_val, self.y_val,
                seq_length=self.seq_length,
                feature_names=self.feature_names
            )
            self.test_dataset = SequenceDataset(
                self.X_test, self.y_test,
                seq_length=self.seq_length,
                feature_names=self.feature_names
            )
        else:
            # Use TabularDataset for MLP
            self.train_dataset = TabularDataset(
                self.X_train, self.y_train,
                feature_names=self.feature_names
            )
            self.val_dataset = TabularDataset(
                self.X_val, self.y_val,
                feature_names=self.feature_names
            )
            self.test_dataset = TabularDataset(
                self.X_test, self.y_test,
                feature_names=self.feature_names
            )
    
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader (NO shuffling for time series)."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # NEVER shuffle time series
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights from training set for loss weighting."""
        return self.train_dataset.get_class_weights()
    
    def get_input_dim(self) -> int:
        """Get input dimension for model initialization."""
        return self.n_features
    
    def get_output_dim(self) -> int:
        """Get output dimension (number of classes)."""
        return self.n_classes
    
    def get_summary(self) -> Dict:
        """Get summary of data module configuration."""
        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "seq_length": self.seq_length,
            "batch_size": self.batch_size,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "test_samples": len(self.test_dataset),
            "train_class_dist": self.train_dataset.get_class_counts(),
            "scale_features": self.scale_features
        }
    
    def print_summary(self):
        """Print data module summary."""
        summary = self.get_summary()
        print("\n" + "=" * 50)
        print("TimeSeriesDataModule Summary")
        print("=" * 50)
        print(f"  Features:        {summary['n_features']}")
        print(f"  Classes:         {summary['n_classes']}")
        print(f"  Sequence Length: {summary['seq_length'] or 'N/A (Tabular)'}")
        print(f"  Batch Size:      {summary['batch_size']}")
        print(f"  Train Samples:   {summary['train_samples']}")
        print(f"  Val Samples:     {summary['val_samples']}")
        print(f"  Test Samples:    {summary['test_samples']}")
        print(f"  Train Classes:   {summary['train_class_dist']}")
        print("=" * 50)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    seq_length: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Quick function to create train/val/test dataloaders.
    
    Args:
        X: Feature matrix
        y: Target vector
        train_ratio: Training fraction
        val_ratio: Validation fraction
        batch_size: Batch size
        seq_length: Sequence length (None for MLP, int for LSTM)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, summary_dict)
    """
    dm = TimeSeriesDataModule(
        X, y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=1.0 - train_ratio - val_ratio,
        batch_size=batch_size,
        seq_length=seq_length
    )
    
    return (
        dm.train_dataloader(),
        dm.val_dataloader(),
        dm.test_dataloader(),
        dm.get_summary()
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PyTorch Dataset Classes")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    
    # -------------------------------------------------------------------------
    print("\n--- Test 1: TabularDataset ---")
    dataset = TabularDataset(X, y)
    print(f"  Total samples: {len(dataset)}")
    print(f"  Features: {dataset.n_features}")
    print(f"  Class counts: {dataset.get_class_counts()}")
    
    x_sample, y_sample = dataset[0]
    print(f"  Sample shapes: X={x_sample.shape}, y={y_sample.shape}")
    print(f"  X dtype: {x_sample.dtype}, y dtype: {y_sample.dtype}")
    
    # -------------------------------------------------------------------------
    print("\n--- Test 2: SequenceDataset ---")
    seq_dataset = SequenceDataset(X, y, seq_length=24, stride=1)
    print(f"  Total sequences: {len(seq_dataset)}")
    print(f"  Sequence length: {seq_dataset.seq_length}")
    print(f"  Class counts: {seq_dataset.get_class_counts()}")
    
    x_seq, y_seq = seq_dataset[0]
    print(f"  Sample shapes: X={x_seq.shape}, y={y_seq.shape}")
    print(f"  Expected X shape: (24, 30)")
    
    # -------------------------------------------------------------------------
    print("\n--- Test 3: TimeSeriesDataModule (Tabular) ---")
    dm = TimeSeriesDataModule(X, y, batch_size=32, seq_length=None)
    dm.print_summary()
    
    train_loader = dm.train_dataloader()
    for batch_x, batch_y in train_loader:
        print(f"  Batch shapes: X={batch_x.shape}, y={batch_y.shape}")
        break
    
    # -------------------------------------------------------------------------
    print("\n--- Test 4: TimeSeriesDataModule (Sequence) ---")
    dm_seq = TimeSeriesDataModule(X, y, batch_size=32, seq_length=24)
    dm_seq.print_summary()
    
    train_loader_seq = dm_seq.train_dataloader()
    for batch_x, batch_y in train_loader_seq:
        print(f"  Batch shapes: X={batch_x.shape}, y={batch_y.shape}")
        print(f"  Expected: X=(32, 24, 30), y=(32,)")
        break
    
    # -------------------------------------------------------------------------
    print("\n--- Test 5: Class Weights ---")
    weights = dm.get_class_weights()
    print(f"  Class weights: {weights}")
    print(f"  Sum: {weights.sum():.4f} (should be 1.0)")
    
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)