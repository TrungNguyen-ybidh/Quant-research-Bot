"""
Train Regime Classification Models
===================================
Command-line script to train regime classifiers for FX pairs.

Usage:
    # Train single model
    python scripts/train_regime_model.py --symbol EURUSD --timeframe 1h
    
    # Train with LSTM
    python scripts/train_regime_model.py --symbol EURUSD --timeframe 1h --arch lstm
    
    # Train multiple pairs
    python scripts/train_regime_model.py --symbols EURUSD,GBPUSD,USDJPY --timeframe 1h
    
    # Train all timeframes for a pair
    python scripts/train_regime_model.py --symbol EURUSD --timeframes 1h,4h,1d
    
    # Batch train (all majors, all timeframes)
    python scripts/train_regime_model.py --batch majors
    
    # Custom config
    python scripts/train_regime_model.py --symbol EURUSD --timeframe 1h --config config/custom.yaml

Examples:
    # Quick test on EURUSD 1h with MLP
    python scripts/train_regime_model.py --symbol EURUSD --timeframe 1h
    
    # Production training with LSTM
    python scripts/train_regime_model.py --symbol EURUSD --timeframe 1h --arch lstm --epochs 200
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import List, Dict
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.state import RegimeClassifier
from src.utils.logger import info, success, error, warning


# Define currency groups
MAJOR_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
ALL_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
    'AUDJPY', 'CHFJPY', 'EURJPY', 'GBPJPY', 'NZDUSD', 'USDCHF'
]
ALL_TIMEFRAMES = ['1m', '5m', '15m', '1h', '1d']


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_single_model(
    symbol: str,
    timeframe: str,
    architecture: str,
    config: Dict,
    version: str = "v1",
    max_epochs: int = None
) -> Dict:
    """
    Train a single regime classification model.
    
    Args:
        symbol: Currency pair
        timeframe: Timeframe
        architecture: Model architecture ('mlp' or 'lstm')
        config: Configuration dictionary
        version: Model version string
        max_epochs: Override max epochs from config
    
    Returns:
        Dictionary with training results
    """
    info("="*100)
    info(f"TRAINING REGIME CLASSIFIER: {symbol} {timeframe} ({architecture.upper()})")
    info("="*100)
    info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Override epochs if specified
        if max_epochs is not None:
            if 'models' not in config:
                config['models'] = {}
            if 'nn' not in config['models']:
                config['models']['nn'] = {}
            config['models']['nn']['default_epochs'] = max_epochs
        
        # Create classifier
        classifier = RegimeClassifier(
            symbol=symbol,
            timeframe=timeframe,
            architecture=architecture,
            config=config
        )
        
        # Full pipeline
        classifier.prepare_data()
        train_history = classifier.train()
        test_metrics = classifier.evaluate()
        classifier.save(version=version)
        
        elapsed_time = time.time() - start_time
        
        # Summary
        info("\n" + "="*100)
        info("TRAINING SUMMARY")
        info("="*100)
        info(f"Symbol:           {symbol}")
        info(f"Timeframe:        {timeframe}")
        info(f"Architecture:     {architecture.upper()}")
        info(f"Total epochs:     {len(train_history['train_loss'])}")
        info(f"Best epoch:       {classifier.trainer.best_metrics.get('epoch', 'N/A')}")
        info(f"Best val loss:    {classifier.trainer.best_metrics.get('val_loss', 0):.4f}")
        info(f"Test accuracy:    {test_metrics['accuracy']:.2%}")
        info(f"Test loss:        {test_metrics['loss']:.4f}")
        info(f"Training time:    {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
        info(f"Completed at:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        info("="*100)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'architecture': architecture,
            'success': True,
            'test_accuracy': test_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'best_val_loss': classifier.trainer.best_metrics.get('val_loss', 0),
            'epochs': len(train_history['train_loss']),
            'training_time_seconds': elapsed_time
        }
        
    except Exception as e:
        error(f"\n{'='*100}")
        error(f"ERROR: Failed to train {symbol} {timeframe}")
        error(f"{'='*100}")
        error(f"Error message: {str(e)}")
        error(f"Error type: {type(e).__name__}")
        
        import traceback
        error("\nFull traceback:")
        error(traceback.format_exc())
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'architecture': architecture,
            'success': False,
            'error': str(e)
        }


def train_batch(
    symbols: List[str],
    timeframes: List[str],
    architecture: str,
    config: Dict,
    version: str = "v1",
    max_epochs: int = None
) -> List[Dict]:
    """
    Train multiple models in batch.
    
    Args:
        symbols: List of currency pairs
        timeframes: List of timeframes
        architecture: Model architecture
        config: Configuration dictionary
        version: Model version
        max_epochs: Override max epochs
    
    Returns:
        List of training results
    """
    total_models = len(symbols) * len(timeframes)
    
    info("="*100)
    info(f"BATCH TRAINING: {total_models} models")
    info("="*100)
    info(f"Symbols:     {', '.join(symbols)}")
    info(f"Timeframes:  {', '.join(timeframes)}")
    info(f"Architecture: {architecture.upper()}")
    info(f"Started at:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info("="*100 + "\n")
    
    results = []
    batch_start_time = time.time()
    
    for i, symbol in enumerate(symbols, 1):
        for j, timeframe in enumerate(timeframes, 1):
            model_num = (i - 1) * len(timeframes) + j
            
            info(f"\n{'='*100}")
            info(f"MODEL {model_num}/{total_models}: {symbol} {timeframe}")
            info(f"{'='*100}")
            
            result = train_single_model(
                symbol=symbol,
                timeframe=timeframe,
                architecture=architecture,
                config=config,
                version=version,
                max_epochs=max_epochs
            )
            results.append(result)
            
            # Progress update
            completed = model_num
            remaining = total_models - completed
            avg_time = (time.time() - batch_start_time) / completed
            est_remaining = avg_time * remaining
            
            info(f"\nProgress: {completed}/{total_models} models completed")
            info(f"Estimated time remaining: {est_remaining/60:.1f} minutes\n")
    
    # Batch summary
    batch_elapsed = time.time() - batch_start_time
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    info("\n" + "="*100)
    info("BATCH TRAINING SUMMARY")
    info("="*100)
    info(f"Total models:     {total_models}")
    info(f"Successful:       {successful}")
    info(f"Failed:           {failed}")
    info(f"Total time:       {batch_elapsed/60:.1f} minutes")
    info(f"Avg time/model:   {batch_elapsed/total_models:.1f}s")
    
    if successful > 0:
        successful_results = [r for r in results if r['success']]
        avg_accuracy = sum(r['test_accuracy'] for r in successful_results) / len(successful_results)
        info(f"Avg test accuracy: {avg_accuracy:.2%}")
    
    info(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    info("="*100)
    
    # Print individual results
    info("\n" + "="*100)
    info("INDIVIDUAL RESULTS")
    info("="*100)
    info(f"{'Symbol':<10} {'Timeframe':<10} {'Success':<10} {'Test Acc':<12} {'Time':<10}")
    info("-"*100)
    
    for r in results:
        status = "✓" if r['success'] else "✗"
        acc = f"{r.get('test_accuracy', 0):.2%}" if r['success'] else "N/A"
        time_str = f"{r.get('training_time_seconds', 0):.1f}s" if r['success'] else "N/A"
        info(f"{r['symbol']:<10} {r['timeframe']:<10} {status:<10} {acc:<12} {time_str:<10}")
    
    info("="*100)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train regime classification models for FX pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single model
  python scripts/train_regime_model.py --symbol EURUSD --timeframe 1h
  
  # Train with LSTM
  python scripts/train_regime_model.py --symbol EURUSD --timeframe 1h --arch lstm
  
  # Train multiple pairs
  python scripts/train_regime_model.py --symbols EURUSD,GBPUSD --timeframe 1h
  
  # Train multiple timeframes
  python scripts/train_regime_model.py --symbol EURUSD --timeframes 1h,4h,1d
  
  # Batch train majors
  python scripts/train_regime_model.py --batch majors --timeframe 1h
  
  # Batch train all pairs and timeframes
  python scripts/train_regime_model.py --batch all
        """
    )
    
    # Single model arguments
    parser.add_argument('--symbol', type=str, help='Currency pair (e.g., EURUSD)')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--timeframe', type=str, help='Timeframe (e.g., 1h, 1d)')
    parser.add_argument('--timeframes', type=str, help='Comma-separated list of timeframes')
    
    # Batch training
    parser.add_argument('--batch', type=str, choices=['majors', 'all'],
                       help='Batch train: majors (5 pairs) or all (11 pairs)')
    
    # Model configuration
    parser.add_argument('--arch', type=str, default='mlp', choices=['mlp', 'lstm'],
                       help='Model architecture (default: mlp)')
    parser.add_argument('--epochs', type=int, help='Max training epochs (overrides config)')
    parser.add_argument('--version', type=str, default='v1', help='Model version (default: v1)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file (default: config/config.yaml)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine symbols and timeframes
    if args.batch:
        symbols = MAJOR_PAIRS if args.batch == 'majors' else ALL_PAIRS
        timeframes = [args.timeframe] if args.timeframe else ALL_TIMEFRAMES
    else:
        # Single or multiple models
        if args.symbols:
            symbols = [s.strip() for s in args.symbols.split(',')]
        elif args.symbol:
            symbols = [args.symbol]
        else:
            parser.error("Must specify --symbol, --symbols, or --batch")
        
        if args.timeframes:
            timeframes = [t.strip() for t in args.timeframes.split(',')]
        elif args.timeframe:
            timeframes = [args.timeframe]
        else:
            parser.error("Must specify --timeframe, --timeframes, or --batch")
    
    # Train models
    if len(symbols) * len(timeframes) == 1:
        # Single model
        result = train_single_model(
            symbol=symbols[0],
            timeframe=timeframes[0],
            architecture=args.arch,
            config=config,
            version=args.version,
            max_epochs=args.epochs
        )
        
        # Exit with error code if failed
        sys.exit(0 if result['success'] else 1)
    else:
        # Batch training
        results = train_batch(
            symbols=symbols,
            timeframes=timeframes,
            architecture=args.arch,
            config=config,
            version=args.version,
            max_epochs=args.epochs
        )
        
        # Exit with error code if any failed
        failed = sum(1 for r in results if not r['success'])
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()