"""
Multi-Layer Perceptron (MLP) Architectures
==========================================
Feed-forward neural networks for classification and regression.

Includes:
- MLPClassifier: Standard MLP for multi-class classification
- MLPBinaryClassifier: Optimized for binary classification
- MLPRegressor: MLP for regression tasks

Usage:
    from src.models.nn.architectures.mlp import MLPClassifier
    
    model = MLPClassifier(
        input_dim=30,
        hidden_dims=[128, 64, 32],
        n_classes=3,
        dropout=0.3
    )
    
    # Forward pass
    logits = model(x)  # (batch_size, n_classes)
    
    # Get probabilities
    probs = model.predict_proba(x)  # (batch_size, n_classes)
    
    # Get predictions
    preds = model.predict(x)  # (batch_size,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
import numpy as np


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron for multi-class classification.
    
    Architecture:
        Input -> [Linear -> BatchNorm -> Activation -> Dropout] x N -> Linear -> Output
    
    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer sizes (default: [128, 64, 32])
        n_classes: Number of output classes (default: 3)
        dropout: Dropout probability (default: 0.3)
        batch_norm: Whether to use batch normalization (default: True)
        activation: Activation function - 'relu', 'gelu', 'leaky_relu', 'selu' (default: 'relu')
    
    Example:
        model = MLPClassifier(input_dim=30, hidden_dims=[128, 64], n_classes=3)
        logits = model(x)  # x: (batch_size, 30) -> logits: (batch_size, 3)
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = None,
                 n_classes: int = 3,
                 dropout: float = 0.3,
                 batch_norm: bool = True,
                 activation: str = 'relu'):
        super().__init__()
        
        # Default hidden dims
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        # Store config
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.dropout_rate = dropout
        self.use_batch_norm = batch_norm
        self.activation_name = activation
        
        # Build backbone layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (before activation)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            layers.append(self._get_activation(activation))
            
            # Dropout (after activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Linear(prev_dim, n_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation module by name."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'selu': nn.SELU(inplace=True),
            'elu': nn.ELU(inplace=True),
            'tanh': nn.Tanh(),
        }
        return activations.get(name.lower(), nn.ReLU(inplace=True))
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Logits of shape (batch_size, n_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Probabilities of shape (batch_size, n_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Predicted classes of shape (batch_size,)
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone (before classification head).
        
        Useful for visualization, clustering, or transfer learning.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Features of shape (batch_size, hidden_dims[-1])
        """
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
        return features
    
    def get_config(self) -> Dict:
        """Get model configuration for saving/loading."""
        return {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'n_classes': self.n_classes,
            'dropout': self.dropout_rate,
            'batch_norm': self.use_batch_norm,
            'activation': self.activation_name
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'MLPClassifier':
        """Create model from configuration dict."""
        return cls(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            n_classes=config['n_classes'],
            dropout=config.get('dropout', 0.3),
            batch_norm=config.get('batch_norm', True),
            activation=config.get('activation', 'relu')
        )
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Get model summary string."""
        lines = [
            f"MLPClassifier(",
            f"  input_dim={self.input_dim},",
            f"  hidden_dims={self.hidden_dims},",
            f"  n_classes={self.n_classes},",
            f"  dropout={self.dropout_rate},",
            f"  batch_norm={self.use_batch_norm},",
            f"  activation={self.activation_name},",
            f"  parameters={self.count_parameters():,}",
            f")"
        ]
        return "\n".join(lines)


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for sequence data.
    
    Architecture:
        Input -> LSTM -> [Linear -> Activation -> Dropout] -> Linear -> Output
    
    Args:
        input_dim: Number of input features per timestep
        hidden_dim: LSTM hidden dimension (default: 64)
        num_layers: Number of LSTM layers (default: 2)
        n_classes: Number of output classes (default: 3)
        dropout: Dropout probability (default: 0.3)
        bidirectional: Use bidirectional LSTM (default: False)
        fc_dims: List of fully connected layer sizes after LSTM (default: [32])
    
    Input shape: (batch_size, seq_length, input_dim)
    Output shape: (batch_size, n_classes)
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 n_classes: int = 3,
                 dropout: float = 0.3,
                 bidirectional: bool = False,
                 fc_dims: List[int] = None):
        super().__init__()
        
        if fc_dims is None:
            fc_dims = [32]
        
        # Store config
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.fc_dims = fc_dims
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output dimension
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Fully connected layers after LSTM
        fc_layers = []
        prev_dim = lstm_out_dim
        
        for fc_dim in fc_dims:
            fc_layers.append(nn.Linear(prev_dim, fc_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = fc_dim
        
        self.fc = nn.Sequential(*fc_layers) if fc_layers else nn.Identity()
        
        # Classification head
        self.classifier = nn.Linear(prev_dim, n_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
        
        Returns:
            Logits of shape (batch_size, n_classes)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state
        if self.bidirectional:
            # Concatenate final states from both directions
            final_state = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final_state = h_n[-1]
        
        # FC layers
        features = self.fc(final_state)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return probs
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'n_classes': self.n_classes,
            'dropout': self.dropout_rate,
            'bidirectional': self.bidirectional,
            'fc_dims': self.fc_dims
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'LSTMClassifier':
        """Create model from configuration."""
        return cls(**config)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self) -> str:
        """Get model summary."""
        direction = "Bidirectional" if self.bidirectional else "Unidirectional"
        return (
            f"LSTMClassifier(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  direction={direction},\n"
            f"  n_classes={self.n_classes},\n"
            f"  fc_dims={self.fc_dims},\n"
            f"  parameters={self.count_parameters():,}\n"
            f")"
        )


class GRUClassifier(nn.Module):
    """
    GRU-based classifier for sequence data.
    
    Similar to LSTM but with fewer parameters and often faster training.
    
    Args:
        input_dim: Number of input features per timestep
        hidden_dim: GRU hidden dimension (default: 64)
        num_layers: Number of GRU layers (default: 2)
        n_classes: Number of output classes (default: 3)
        dropout: Dropout probability (default: 0.3)
        bidirectional: Use bidirectional GRU (default: False)
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 n_classes: int = 3,
                 dropout: float = 0.3,
                 bidirectional: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_classes = n_classes
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension
        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(gru_out_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        gru_out, h_n = self.gru(x)
        
        if self.bidirectional:
            final_state = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final_state = h_n[-1]
        
        logits = self.classifier(final_state)
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.predict_proba(x), dim=-1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_model(model_type: str,
                 input_dim: int,
                 n_classes: int = 3,
                 **kwargs) -> nn.Module:
    """
    Factory function to create models by type.
    
    Args:
        model_type: 'mlp', 'lstm', or 'gru'
        input_dim: Number of input features
        n_classes: Number of output classes
        **kwargs: Additional model-specific arguments
    
    Returns:
        Initialized model
    """
    model_type = model_type.lower()
    
    if model_type == 'mlp':
        return MLPClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dims=kwargs.get('hidden_dims', [128, 64, 32]),
            dropout=kwargs.get('dropout', 0.3),
            batch_norm=kwargs.get('batch_norm', True),
            activation=kwargs.get('activation', 'relu')
        )
    
    elif model_type == 'lstm':
        return LSTMClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dim=kwargs.get('hidden_dim', 64),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.3),
            bidirectional=kwargs.get('bidirectional', False),
            fc_dims=kwargs.get('fc_dims', [32])
        )
    
    elif model_type == 'gru':
        return GRUClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dim=kwargs.get('hidden_dim', 64),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.3),
            bidirectional=kwargs.get('bidirectional', False)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'mlp', 'lstm', or 'gru'.")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Neural Network Architectures")
    print("=" * 60)
    
    # Test parameters
    batch_size = 32
    input_dim = 30
    seq_length = 24
    n_classes = 3
    
    # Create sample inputs
    x_tabular = torch.randn(batch_size, input_dim)
    x_sequence = torch.randn(batch_size, seq_length, input_dim)
    
    # -------------------------------------------------------------------------
    print("\n--- Test 1: MLPClassifier ---")
    mlp = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=[128, 64, 32],
        n_classes=n_classes,
        dropout=0.3
    )
    print(mlp.summary())
    
    logits = mlp(x_tabular)
    probs = mlp.predict_proba(x_tabular)
    preds = mlp.predict(x_tabular)
    
    print(f"  Input shape:  {x_tabular.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probs shape:  {probs.shape}")
    print(f"  Preds shape:  {preds.shape}")
    print(f"  Probs sum:    {probs[0].sum():.4f} (should be 1.0)")
    
    # -------------------------------------------------------------------------
    print("\n--- Test 2: LSTMClassifier ---")
    lstm = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        n_classes=n_classes,
        dropout=0.3
    )
    print(lstm.summary())
    
    logits = lstm(x_sequence)
    probs = lstm.predict_proba(x_sequence)
    preds = lstm.predict(x_sequence)
    
    print(f"  Input shape:  {x_sequence.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probs shape:  {probs.shape}")
    print(f"  Preds shape:  {preds.shape}")
    
    # -------------------------------------------------------------------------
    print("\n--- Test 3: GRUClassifier ---")
    gru = GRUClassifier(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        n_classes=n_classes
    )
    print(f"  Parameters: {gru.count_parameters():,}")
    
    logits = gru(x_sequence)
    print(f"  Input shape:  {x_sequence.shape}")
    print(f"  Logits shape: {logits.shape}")
    
    # -------------------------------------------------------------------------
    print("\n--- Test 4: Bidirectional LSTM ---")
    bi_lstm = LSTMClassifier(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        n_classes=n_classes,
        bidirectional=True
    )
    print(bi_lstm.summary())
    
    logits = bi_lstm(x_sequence)
    print(f"  Logits shape: {logits.shape}")
    
    # -------------------------------------------------------------------------
    print("\n--- Test 5: Factory Function ---")
    for model_type in ['mlp', 'lstm', 'gru']:
        model = create_model(model_type, input_dim=input_dim, n_classes=n_classes)
        print(f"  {model_type.upper()}: {model.count_parameters():,} parameters")
    
    # -------------------------------------------------------------------------
    print("\n--- Test 6: Config Save/Load ---")
    config = mlp.get_config()
    print(f"  Config: {config}")
    
    mlp_loaded = MLPClassifier.from_config(config)
    print(f"  Loaded model parameters: {mlp_loaded.count_parameters():,}")
    
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("✓ All architecture tests passed!")
    print("=" * 60)