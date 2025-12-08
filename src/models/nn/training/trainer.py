"""
Neural Network Trainer
======================
Training loop with early stopping, checkpointing, and metrics tracking.

Features:
- Early stopping based on validation loss or accuracy
- Model checkpointing (save best model)
- Learning rate scheduling
- Class-weighted loss for imbalanced data
- Training history tracking
- GPU/CPU automatic detection

Usage:
    from src.models.nn.training.trainer import Trainer, TrainerConfig
    
    config = TrainerConfig(epochs=100, patience=10, lr=0.001)
    trainer = Trainer(model, config)
    
    history = trainer.fit(train_loader, val_loader)
    
    # Evaluate
    metrics = trainer.evaluate(test_loader)
    
    # Save/Load
    trainer.save_checkpoint('model.pt')
    trainer.load_checkpoint('model.pt')
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
import time
from datetime import datetime


@dataclass
class TrainerConfig:
    """Configuration for model training."""
    
    # Training parameters
    epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 0.0001
    
    # Early stopping
    patience: int = 10
    min_delta: float = 0.0001
    monitor: str = 'val_loss'  # 'val_loss' or 'val_accuracy'
    mode: str = 'min'  # 'min' for loss, 'max' for accuracy
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    min_lr: float = 1e-6
    
    # Class weighting for imbalanced data
    use_class_weights: bool = True
    
    # Checkpointing
    save_best: bool = True
    checkpoint_dir: str = 'outputs/models'
    
    # Logging
    verbose: int = 1  # 0=silent, 1=progress, 2=detailed
    log_interval: int = 10  # Log every N batches (if verbose=2)
    
    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    
    # Gradient clipping
    clip_grad_norm: Optional[float] = 1.0
    
    # Random seed
    seed: int = 42


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Stops training when monitored metric stops improving.
    """
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0001,
                 mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' (for loss) or 'max' (for accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            epoch: Current epoch number
        
        Returns:
            True if should stop, False otherwise
        """
        if self.mode == 'min':
            is_improvement = self.best_score is None or score < self.best_score - self.min_delta
        else:
            is_improvement = self.best_score is None or score > self.best_score + self.min_delta
        
        if is_improvement:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


class Trainer:
    """
    Neural network trainer with early stopping and checkpointing.
    
    Handles:
    - Training loop
    - Validation
    - Early stopping
    - Learning rate scheduling
    - Model checkpointing
    - Metrics tracking
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: Optional[TrainerConfig] = None,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            config: Training configuration
            class_weights: Optional class weights for imbalanced data
        """
        self.config = config or TrainerConfig()
        
        # Set random seed
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Setup device
        self.device = self._setup_device()
        
        # Model
        self.model = model.to(self.device)
        
        # Loss function
        if class_weights is not None and self.config.use_class_weights:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = None
        if self.config.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.mode,
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                min_lr=self.config.min_lr
            )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode=self.config.mode
        )
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'lr': []
        }
        
        # Best model state
        self.best_model_state = None
        self.best_metrics = {}
    
    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        if self.config.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config.device)
        
        if self.config.verbose > 0:
            print(f"Using device: {device}")
        
        return device
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.clip_grad_norm
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)
            
            # Logging
            if self.config.verbose == 2 and batch_idx % self.config.log_interval == 0:
                print(f"    Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        
        Returns:
            Training history dictionary
        """
        start_time = time.time()
        
        if self.config.verbose > 0:
            print("\n" + "=" * 60)
            print("Starting Training")
            print("=" * 60)
            print(f"  Epochs: {self.config.epochs}")
            print(f"  Learning Rate: {self.config.lr}")
            print(f"  Early Stopping Patience: {self.config.patience}")
            print(f"  Device: {self.device}")
            print("=" * 60 + "\n")
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Get metric for early stopping/scheduler
            if self.config.monitor == 'val_loss':
                monitor_metric = val_loss
            else:
                monitor_metric = val_acc
            
            # Learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step(monitor_metric)
            
            # Check for best model
            is_best = False
            if self.config.mode == 'min':
                is_best = self.best_metrics.get('val_loss', float('inf')) > val_loss
            else:
                is_best = self.best_metrics.get('val_accuracy', 0) < val_acc
            
            if is_best:
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.best_metrics = {
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc
                }
            
            # Logging
            epoch_time = time.time() - epoch_start
            if self.config.verbose > 0:
                best_marker = " *" if is_best else ""
                print(f"Epoch {epoch:3d}/{self.config.epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s{best_marker}")
            
            # Early stopping check
            if self.early_stopping(monitor_metric, epoch):
                if self.config.verbose > 0:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best epoch was {self.early_stopping.best_epoch} with "
                          f"{self.config.monitor}={self.early_stopping.best_score:.4f}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.config.verbose > 0:
                print(f"\nLoaded best model from epoch {self.best_metrics['epoch']}")
        
        total_time = time.time() - start_time
        if self.config.verbose > 0:
            print(f"\nTraining completed in {total_time/60:.1f} minutes")
            print(f"Best Val Loss: {self.best_metrics.get('val_loss', 0):.4f}")
            print(f"Best Val Accuracy: {self.best_metrics.get('val_accuracy', 0):.4f}")
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                total_loss += loss.item() * data.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = (all_preds == all_targets).mean()
        avg_loss = total_loss / len(all_targets)
        
        # Per-class accuracy
        n_classes = all_probs.shape[1]
        class_accuracy = {}
        for c in range(n_classes):
            mask = all_targets == c
            if mask.sum() > 0:
                class_accuracy[c] = (all_preds[mask] == all_targets[mask]).mean()
            else:
                class_accuracy[c] = 0.0
        
        # Confusion matrix
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        for pred, target in zip(all_preds, all_targets):
            confusion[target, pred] += 1
        
        results = {
            'loss': float(avg_loss),
            'accuracy': float(accuracy),
            'class_accuracy': class_accuracy,
            'confusion_matrix': confusion.tolist(),
            'n_samples': len(all_targets),
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
        
        if self.config.verbose > 0:
            print("\n" + "=" * 60)
            print("Evaluation Results")
            print("=" * 60)
            print(f"  Loss:     {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Samples:  {len(all_targets)}")
            print("\n  Per-Class Accuracy:")
            for c, acc in class_accuracy.items():
                print(f"    Class {c}: {acc:.4f}")
            print("=" * 60)
        
        return results
    
    def save_checkpoint(self, path: str, include_optimizer: bool = True):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            include_optimizer: Whether to save optimizer state
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config() if hasattr(self.model, 'get_config') else {},
            'history': self.history,
            'best_metrics': self.best_metrics,
            'config': {
                'epochs': self.config.epochs,
                'lr': self.config.lr,
                'patience': self.config.patience
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        
        if self.config.verbose > 0:
            print(f"Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = False):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_metrics = checkpoint.get('best_metrics', {})
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.config.verbose > 0:
            print(f"Checkpoint loaded from: {path}")
            print(f"  Best metrics: {self.best_metrics}")
    
    def get_predictions(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions for a dataset.
        
        Args:
            loader: Data loader
        
        Returns:
            Tuple of (predictions, probabilities, targets)
        """
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(target.numpy())
        
        return np.array(all_preds), np.array(all_probs), np.array(all_targets)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int = 100,
                lr: float = 0.001,
                patience: int = 10,
                class_weights: Optional[torch.Tensor] = None,
                verbose: int = 1) -> Tuple[nn.Module, Dict]:
    """
    Quick function to train a model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        patience: Early stopping patience
        class_weights: Optional class weights
        verbose: Verbosity level
    
    Returns:
        Tuple of (trained_model, history)
    """
    config = TrainerConfig(
        epochs=epochs,
        lr=lr,
        patience=patience,
        verbose=verbose
    )
    
    trainer = Trainer(model, config, class_weights)
    history = trainer.fit(train_loader, val_loader)
    
    return model, history


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Trainer")
    print("=" * 60)
    
    # Import model
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from architectures.mlp import MLPClassifier
    from data.dataset import TimeSeriesDataModule
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 30
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Create data module
    dm = TimeSeriesDataModule(
        X, y,
        batch_size=32,
        seq_length=None  # Tabular for MLP
    )
    dm.print_summary()
    
    # Create model
    model = MLPClassifier(
        input_dim=n_features,
        hidden_dims=[64, 32],
        n_classes=n_classes,
        dropout=0.3
    )
    print(f"\n{model.summary()}")
    
    # Create trainer
    config = TrainerConfig(
        epochs=20,
        lr=0.001,
        patience=5,
        verbose=1
    )
    
    trainer = Trainer(
        model,
        config,
        class_weights=dm.get_class_weights()
    )
    
    # Train
    print("\n--- Training ---")
    history = trainer.fit(
        dm.train_dataloader(),
        dm.val_dataloader()
    )
    
    # Evaluate
    print("\n--- Evaluation ---")
    results = trainer.evaluate(dm.test_dataloader())
    
    # Save checkpoint
    print("\n--- Save/Load Test ---")
    trainer.save_checkpoint('outputs/models/test_checkpoint.pt')
    trainer.load_checkpoint('outputs/models/test_checkpoint.pt')
    
    print("\n" + "=" * 60)
    print("✓ Trainer tests passed!")
    print("=" * 60)