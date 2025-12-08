"""
Neural Network Architectures
============================
PyTorch model architectures for classification and regression.

Available models:
- MLPClassifier: Multi-layer perceptron for tabular data
- LSTMClassifier: LSTM-based sequence classifier
- GRUClassifier: GRU-based sequence classifier

Factory function:
- create_model: Create any model by type string

Usage:
    from src.models.nn.architectures import MLPClassifier, LSTMClassifier, create_model
    
    # Direct instantiation
    mlp = MLPClassifier(input_dim=30, hidden_dims=[128, 64], n_classes=3)
    lstm = LSTMClassifier(input_dim=30, hidden_dim=64, n_classes=3)
    
    # Factory function
    model = create_model('mlp', input_dim=30, n_classes=3)
    model = create_model('lstm', input_dim=30, n_classes=3, hidden_dim=64)
"""

from .mlp import (
    MLPClassifier,
    LSTMClassifier,
    GRUClassifier,
    create_model
)

__all__ = [
    # Models
    'MLPClassifier',
    'LSTMClassifier',
    'GRUClassifier',
    
    # Factory
    'create_model'
]