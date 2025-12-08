from src.models.state import RegimeClassifier

# Create classifier
classifier = RegimeClassifier(
    symbol="EURUSD",
    timeframe="1h",
    architecture="mlp",  # or "lstm"
    config=config
)

# Full pipeline
classifier.prepare_data()      # Load, label, preprocess, split
classifier.train()             # Train neural network
metrics = classifier.evaluate() # Evaluate on test set
classifier.save(version="v1")  # Save everything

# Later: Load and predict
loaded = RegimeClassifier.load("EURUSD", "1h", "v1")
regimes, confidence = loaded.predict(new_features)