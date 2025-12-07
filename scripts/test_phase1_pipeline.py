"""
Test Phase 1: Complete Preprocessing Pipeline
Run this to verify all Phase 1 components work together.

Usage: python scripts/test_phase1_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import os


def test_full_pipeline():
    """Test the complete Phase 1 pipeline."""
    
    print("=" * 70)
    print("PHASE 1: PREPROCESSING PIPELINE TEST")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n[1/5] Loading processed features...")
    
    input_path = 'data/processed/EURUSD_1h_features.parquet'
    
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        print("Run process_all_features.py first!")
        return False
    
    df = pd.read_parquet(input_path)
    print(f"  ✓ Loaded: {df.shape[0]} rows, {df.shape[1]} features")
    print(f"  ✓ Date range: {df.index[0]} to {df.index[-1]}")
    
    # =========================================================================
    # Step 2: Build Target Variable
    # =========================================================================
    print("\n[2/5] Building target variable...")
    
    from src.models.target_builder import TargetBuilder
    
    # Binary classification: predict if price goes up in next 4 hours
    target_builder = TargetBuilder(
        target_type='binary',
        lookahead=4,
        threshold=0.0001  # 0.01% threshold
    )
    
    y = target_builder.build(df, price_col='micro_price')
    target_builder.print_stats()
    
    # Remove rows where target is NaN (end of series)
    valid_mask = y.notna()
    df_valid = df[valid_mask]
    y_valid = y[valid_mask]
    
    print(f"  ✓ Valid samples after removing NaN targets: {len(df_valid)}")
    
    # =========================================================================
    # Step 3: Feature Scaling
    # =========================================================================
    print("\n[3/5] Scaling features...")
    
    from src.models.preprocessing import FeaturePreprocessor
    
    preprocessor = FeaturePreprocessor(
        default_method='standard',
        clip_outliers=True,
        outlier_std=5.0
    )
    
    df_scaled = preprocessor.fit_transform(df_valid)
    
    print(f"  ✓ Scaled shape: {df_scaled.shape}")
    print(f"  ✓ Mean (should be ~0): {df_scaled.mean().mean():.4f}")
    print(f"  ✓ Std (should be ~1): {df_scaled.std().mean():.4f}")
    
    # Save scaler
    os.makedirs('models/scalers', exist_ok=True)
    preprocessor.save('models/scalers/preprocessor.pkl')
    print("  ✓ Saved scaler to models/scalers/preprocessor.pkl")
    
    # =========================================================================
    # Step 4: Feature Selection
    # =========================================================================
    print("\n[4/5] Selecting best features...")
    
    from src.models.feature_selection import FeatureSelector
    
    selector = FeatureSelector(
        variance_threshold=0.01,
        correlation_threshold=0.95,
        n_features=30,
        importance_method='random_forest'
    )
    
    df_selected = selector.fit_transform(df_scaled, y_valid)
    
    summary = selector.get_selection_summary()
    print(f"  ✓ Original features: {summary['original_features']}")
    print(f"  ✓ Removed (low variance): {summary['removed_low_variance']}")
    print(f"  ✓ Removed (high correlation): {summary['removed_high_correlation']}")
    print(f"  ✓ Selected features: {summary['selected_features']}")
    
    # Show top features
    print("\n  Top 10 features by importance:")
    importance_report = selector.get_importance_report()
    for i, row in importance_report.head(10).iterrows():
        print(f"    {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    # Save selector
    os.makedirs('models/selectors', exist_ok=True)
    selector.save('models/selectors/feature_selector.pkl')
    print("\n  ✓ Saved selector to models/selectors/feature_selector.pkl")
    
    # =========================================================================
    # Step 5: Train/Val/Test Split
    # =========================================================================
    print("\n[5/5] Creating train/val/test split...")
    
    from src.models.data_splitter import TimeSeriesSplitter
    
    splitter = TimeSeriesSplitter(
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    split = splitter.split(df_selected, y_valid)
    
    print(f"\n  Split Summary:")
    for name, data in split.summary().items():
        print(f"    {name.upper()}:")
        print(f"      Samples: {data['samples']}")
        print(f"      Period: {data['start'][:10]} to {data['end'][:10]}")
    
    # Verify no leakage
    assert split.train_end < split.val_start, "Train/Val overlap detected!"
    assert split.val_end < split.test_start, "Val/Test overlap detected!"
    print("\n  ✓ No temporal leakage detected!")
    
    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE - SUMMARY")
    print("=" * 70)
    
    print(f"""
  Input:    {df.shape[0]} samples × {df.shape[1]} features
  Output:   {df_selected.shape[0]} samples × {df_selected.shape[1]} features
  
  Target:   Binary (up/down), lookahead=4 hours
  Scaling:  StandardScaler (with outlier clipping)
  Selected: Top {len(selector.selected_features)} features
  
  Train:    {len(split.X_train)} samples ({len(split.X_train)/len(df_selected)*100:.1f}%)
  Val:      {len(split.X_val)} samples ({len(split.X_val)/len(df_selected)*100:.1f}%)
  Test:     {len(split.X_test)} samples ({len(split.X_test)/len(df_selected)*100:.1f}%)
  
  Saved:
    - models/scalers/preprocessor.pkl
    - models/selectors/feature_selector.pkl
    """)
    
    print("=" * 70)
    print("✓ Phase 1 Pipeline Test PASSED!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)