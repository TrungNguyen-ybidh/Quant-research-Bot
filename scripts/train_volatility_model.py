"""
Train Volatility Regime Classification Models
==============================================
Command-line script for training volatility regime classifiers.

Usage:
    # Single model
    python scripts/train_volatility_model.py --symbol EURUSD --timeframe 1h
    
    # Multiple pairs
    python scripts/train_volatility_model.py --symbols EURUSD,GBPUSD,USDJPY --timeframe 1h
    
    # Batch training (majors)
    python scripts/train_volatility_model.py --batch majors --timeframe 1h
    
    # All models
    python scripts/train_volatility_model.py --batch all
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.models.state.volatility_regime import VolatilityRegimeClassifier
from src.utils.logger import info, success, error, warning


# Symbol groups
MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
ALL_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", 
             "AUDJPY", "CHFJPY", "EURJPY", "GBPJPY", "NZDUSD", "USDCHF"]
TIMEFRAMES = ["1m", "5m", "15m", "1h", "1d"]


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        warning(f"Config not found: {config_path}, using defaults")
        return {}


def train_single_model(
    symbol: str,
    timeframe: str,
    architecture: str = "mlp",
    config: Optional[Dict] = None,
    version: str = "v1"
) -> Dict:
    """Train a single volatility regime model."""
    
    info("=" * 100)
    info(f"TRAINING VOLATILITY REGIME CLASSIFIER: {symbol} {timeframe} ({architecture.upper()})")
    info("=" * 100)
    info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Create classifier
        classifier = VolatilityRegimeClassifier(
            symbol=symbol,
            timeframe=timeframe,
            architecture=architecture,
            config=config
        )
        
        # Run pipeline
        classifier.prepare_data()
        train_history = classifier.train()
        test_metrics = classifier.evaluate()
        classifier.save(version=version)
        
        elapsed = time.time() - start_time
        
        # Summary
        info("\n" + "=" * 100)
        info("TRAINING SUMMARY")
        info("=" * 100)
        info(f"Symbol:           {symbol}")
        info(f"Timeframe:        {timeframe}")
        info(f"Architecture:     {architecture.upper()}")
        info(f"Total epochs:     {len(train_history['train_loss'])}")
        info(f"Best epoch:       {classifier.trainer.best_metrics.get('epoch', 'N/A')}")
        info(f"Best val loss:    {classifier.trainer.best_metrics.get('val_loss', 0):.4f}")
        info(f"Test accuracy:    {test_metrics['accuracy']:.2%}")
        info(f"Test loss:        {test_metrics['loss']:.4f}")
        info(f"Training time:    {elapsed:.1f}s ({elapsed/60:.1f}m)")
        info(f"Completed at:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        info("=" * 100)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'success': True,
            'test_accuracy': test_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'elapsed': elapsed
        }
        
    except Exception as e:
        elapsed = time.time() - start_time
        
        error("\n" + "=" * 100)
        error(f"ERROR: Failed to train {symbol} {timeframe}")
        error("=" * 100)
        error(f"Error message: {str(e)}")
        error(f"Error type: {type(e).__name__}")
        
        import traceback
        error(f"\nFull traceback:")
        error(traceback.format_exc())
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'success': False,
            'error': str(e),
            'elapsed': elapsed
        }


def train_batch(
    symbols: List[str],
    timeframes: List[str],
    architecture: str = "mlp",
    config: Optional[Dict] = None,
    version: str = "v1"
) -> List[Dict]:
    """Train multiple models in batch."""
    
    total = len(symbols) * len(timeframes)
    results = []
    
    info("\n" + "=" * 100)
    info(f"BATCH TRAINING: {len(symbols)} symbols × {len(timeframes)} timeframes = {total} models")
    info("=" * 100)
    
    batch_start = time.time()
    
    for i, symbol in enumerate(symbols):
        for j, tf in enumerate(timeframes):
            idx = i * len(timeframes) + j + 1
            info(f"\n[{idx}/{total}] Training {symbol} {tf}...")
            
            result = train_single_model(
                symbol=symbol,
                timeframe=tf,
                architecture=architecture,
                config=config,
                version=version
            )
            results.append(result)
            
            # Progress
            completed = len(results)
            avg_time = sum(r['elapsed'] for r in results) / completed
            remaining = (total - completed) * avg_time
            info(f"\nProgress: {completed}/{total} models completed")
            info(f"Estimated time remaining: {remaining/60:.1f} minutes")
    
    batch_elapsed = time.time() - batch_start
    
    # Batch summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    info("\n" + "=" * 100)
    info("BATCH TRAINING SUMMARY")
    info("=" * 100)
    info(f"Total models:     {total}")
    info(f"Successful:       {len(successful)}")
    info(f"Failed:           {len(failed)}")
    info(f"Total time:       {batch_elapsed/60:.1f} minutes")
    info(f"Avg time/model:   {batch_elapsed/total:.1f}s")
    
    if successful:
        avg_acc = sum(r['test_accuracy'] for r in successful) / len(successful)
        info(f"Avg test accuracy: {avg_acc:.2%}")
    
    info(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info("=" * 100)
    
    # Individual results
    info("\n" + "=" * 100)
    info("INDIVIDUAL RESULTS")
    info("=" * 100)
    info(f"{'Symbol':<10} {'Timeframe':<10} {'Success':<10} {'Test Acc':<12} {'Time':<10}")
    info("-" * 100)
    
    for r in results:
        status = "✓" if r['success'] else "✗"
        acc = f"{r.get('test_accuracy', 0):.2%}" if r['success'] else "N/A"
        info(f"{r['symbol']:<10} {r['timeframe']:<10} {status:<10} {acc:<12} {r['elapsed']:.1f}s")
    
    info("=" * 100)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train Volatility Regime Models')
    
    # Single model
    parser.add_argument('--symbol', type=str, help='Single symbol to train')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe')
    
    # Multiple models
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--timeframes', type=str, help='Comma-separated timeframes')
    
    # Batch
    parser.add_argument('--batch', type=str, choices=['majors', 'all'], help='Batch mode')
    
    # Options
    parser.add_argument('--arch', type=str, default='mlp', choices=['mlp', 'lstm'], help='Architecture')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    parser.add_argument('--version', type=str, default='v1', help='Model version')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine what to train
    if args.batch:
        if args.batch == 'majors':
            symbols = MAJORS
        else:
            symbols = ALL_PAIRS
        
        if args.timeframes:
            timeframes = args.timeframes.split(',')
        elif args.timeframe:
            timeframes = [args.timeframe]
        else:
            timeframes = TIMEFRAMES
        
        train_batch(symbols, timeframes, args.arch, config, args.version)
    
    elif args.symbols:
        symbols = args.symbols.split(',')
        timeframes = args.timeframes.split(',') if args.timeframes else [args.timeframe]
        train_batch(symbols, timeframes, args.arch, config, args.version)
    
    elif args.symbol:
        train_single_model(args.symbol, args.timeframe, args.arch, config, args.version)
    
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/train_volatility_model.py --symbol EURUSD --timeframe 1h")
        print("  python scripts/train_volatility_model.py --batch majors --timeframe 1h")
        print("  python scripts/train_volatility_model.py --batch all")


if __name__ == "__main__":
    main()