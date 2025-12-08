"""
Preprocess All Data - Run Phase 1 pipeline on all symbols and timeframes
Creates scalers, selectors, and generates summary report.

Usage: python scripts/preprocess_all_data.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

from src.utils.logger import info, success, error, warning
from src.models.preprocessing import FeaturePreprocessor
from src.models.feature_selection import FeatureSelector
from src.models.target_builder import TargetBuilder
from src.models.data_splitter import TimeSeriesSplitter


def discover_processed_files(data_dir: str = "data/processed") -> List[dict]:
    """Discover all processed feature files."""
    files = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        error(f"Directory not found: {data_dir}")
        return files
    
    for f in data_path.glob("*_features.parquet"):
        # Parse filename: SYMBOL_TIMEFRAME_features.parquet
        name = f.stem.replace("_features", "")
        parts = name.rsplit("_", 1)
        
        if len(parts) == 2:
            symbol, tf = parts
            files.append({
                "path": str(f),
                "symbol": symbol,
                "timeframe": tf,
                "filename": f.name
            })
    
    return sorted(files, key=lambda x: (x["symbol"], x["timeframe"]))


def preprocess_single_file(
    file_info: dict,
    output_dir: str = "models",
    lookahead: int = 4,
    n_features: int = 30
) -> Optional[dict]:
    """
    Run full preprocessing pipeline on a single file.
    
    Returns dict with results or None if failed.
    """
    symbol = file_info["symbol"]
    tf = file_info["timeframe"]
    path = file_info["path"]
    
    info(f"Processing {symbol} ({tf})...")
    
    try:
        # 1. Load data
        df = pd.read_parquet(path)
        
        if df.empty:
            warning(f"Empty file: {path}")
            return None
        
        # 2. Build target
        target_builder = TargetBuilder(
            target_type='binary',
            lookahead=lookahead,
            threshold=0.0001
        )
        
        # Find price column
        price_col = 'micro_price' if 'micro_price' in df.columns else 'close'
        if price_col not in df.columns:
            # Use first available price-like column
            for col in ['pivot_point', 'sma_ratio_10_20']:
                if col in df.columns:
                    price_col = col
                    break
        
        y = target_builder.build(df, price_col=price_col)
        
        # Remove NaN targets
        valid_mask = y.notna()
        df_valid = df[valid_mask]
        y_valid = y[valid_mask]
        
        if len(df_valid) < 100:
            warning(f"Not enough valid samples: {len(df_valid)}")
            return None
        
        # 3. Scale features
        preprocessor = FeaturePreprocessor(
            default_method='standard',
            clip_outliers=True,
            outlier_std=5.0
        )
        df_scaled = preprocessor.fit_transform(df_valid)
        
        # Save scaler
        scaler_dir = f"{output_dir}/scalers/{symbol}"
        os.makedirs(scaler_dir, exist_ok=True)
        scaler_path = f"{scaler_dir}/{tf}_scaler.pkl"
        preprocessor.save(scaler_path)
        
        # 4. Select features
        selector = FeatureSelector(
            variance_threshold=0.01,
            correlation_threshold=0.95,
            n_features=n_features,
            importance_method='random_forest'
        )
        df_selected = selector.fit_transform(df_scaled, y_valid)
        
        # Save selector
        selector_dir = f"{output_dir}/selectors/{symbol}"
        os.makedirs(selector_dir, exist_ok=True)
        selector_path = f"{selector_dir}/{tf}_selector.pkl"
        selector.save(selector_path)
        
        # 5. Get split info
        splitter = TimeSeriesSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        split = splitter.split(df_selected, y_valid)
        
        # 6. Collect results
        target_stats = target_builder.get_stats()
        selection_summary = selector.get_selection_summary()
        importance_report = selector.get_importance_report()
        
        # Top 5 features
        top_features = importance_report.head(5)['feature'].tolist() if len(importance_report) > 0 else []
        
        result = {
            "symbol": symbol,
            "timeframe": tf,
            "total_rows": len(df),
            "valid_rows": len(df_valid),
            "date_start": str(df_valid.index[0]),
            "date_end": str(df_valid.index[-1]),
            "original_features": selection_summary.get('original_features', 78),
            "selected_features": len(selector.selected_features),
            "removed_low_var": selection_summary.get('removed_low_variance', 0),
            "removed_high_corr": selection_summary.get('removed_high_correlation', 0),
            "class_balance": target_stats.get('class_balance', {}),
            "train_samples": len(split.X_train),
            "val_samples": len(split.X_val),
            "test_samples": len(split.X_test),
            "top_5_features": top_features,
            "scaler_path": scaler_path,
            "selector_path": selector_path
        }
        
        success(f"  ✓ {symbol} {tf}: {len(df_valid)} samples, {len(selector.selected_features)} features")
        
        return result
        
    except Exception as e:
        error(f"  ✗ {symbol} {tf}: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_summary_report(results: List[dict], output_path: str = "outputs/reports/phase1_summary.md"):
    """Generate markdown summary report."""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Calculate totals
    total_rows = sum(r['valid_rows'] for r in results)
    total_files = len(results)
    symbols = list(set(r['symbol'] for r in results))
    timeframes = list(set(r['timeframe'] for r in results))
    
    # Find most common top features
    all_top_features = []
    for r in results:
        all_top_features.extend(r.get('top_5_features', []))
    
    feature_counts = {}
    for f in all_top_features:
        feature_counts[f] = feature_counts.get(f, 0) + 1
    
    top_global_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Build report
    report = f"""# Phase 1: Preprocessing Pipeline Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary

| Metric | Value |
|--------|-------|
| Total Files Processed | {total_files} |
| Total Samples | {total_rows:,} |
| Symbols | {len(symbols)} |
| Timeframes | {len(timeframes)} |

### Symbols
{', '.join(sorted(symbols))}

### Timeframes
{', '.join(sorted(timeframes))}

---

## Top Predictive Features (Across All Assets)

| Rank | Feature | Frequency |
|------|---------|-----------|
"""
    
    for i, (feat, count) in enumerate(top_global_features, 1):
        report += f"| {i} | {feat} | {count}/{total_files} |\n"
    
    report += """
---

## Per-Asset Results

| Symbol | Timeframe | Samples | Features | Train | Val | Test |
|--------|-----------|---------|----------|-------|-----|------|
"""
    
    for r in results:
        report += f"| {r['symbol']} | {r['timeframe']} | {r['valid_rows']:,} | {r['selected_features']} | {r['train_samples']:,} | {r['val_samples']:,} | {r['test_samples']:,} |\n"
    
    report += """
---

## Saved Artifacts

### Scalers
```
models/scalers/{SYMBOL}/{TIMEFRAME}_scaler.pkl
```

### Selectors
```
models/selectors/{SYMBOL}/{TIMEFRAME}_selector.pkl
```

---

## Next Steps

- **Phase 2:** Regime Analysis (HMM)
- **Phase 3:** Strategy Testing
- **Phase 4:** Research Reports per Asset

"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    success(f"Report saved to {output_path}")
    return report


def main():
    """Run preprocessing on all data."""
    
    print("=" * 70)
    print("PHASE 1: PREPROCESS ALL DATA")
    print("=" * 70)
    
    # Discover files
    info("Discovering processed feature files...")
    files = discover_processed_files("data/processed")
    
    if not files:
        error("No processed feature files found!")
        error("Run 'python scripts/process_all_features.py' first.")
        return
    
    info(f"Found {len(files)} files to process")
    
    # Process each file
    results = []
    failed = []
    
    for i, file_info in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] ", end="")
        
        result = preprocess_single_file(
            file_info,
            output_dir="models",
            lookahead=4,
            n_features=30
        )
        
        if result:
            results.append(result)
        else:
            failed.append(file_info)
    
    # Summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    
    print(f"\n  Successful: {len(results)}/{len(files)}")
    print(f"  Failed: {len(failed)}/{len(files)}")
    
    if failed:
        print(f"\n  Failed files:")
        for f in failed:
            print(f"    - {f['symbol']} {f['timeframe']}")
    
    # Generate report
    if results:
        print("\n" + "-" * 70)
        report = generate_summary_report(results)
        
        # Print quick summary
        total_rows = sum(r['valid_rows'] for r in results)
        print(f"\n  Total samples processed: {total_rows:,}")
        print(f"  Report: outputs/reports/phase1_summary.md")
    
    print("\n" + "=" * 70)
    print("✓ Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()