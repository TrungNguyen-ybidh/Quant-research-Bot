#!/usr/bin/env python
"""
Setup and Test External Data Sources

This script:
1. Verifies dependencies (fredapi, yfinance)
2. Tests FRED API connection
3. Tests Yahoo Finance connection
4. Pre-fetches common data to cache
5. Shows system status

Run: python scripts/setup_external_data.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import info, success, error


def check_dependencies():
    """Check if required packages are installed."""
    info("Checking dependencies...")
    
    missing = []
    
    try:
        import fredapi
        success("✓ fredapi installed")
    except ImportError:
        error("✗ fredapi not installed")
        missing.append("fredapi")
    
    try:
        import yfinance
        success("✓ yfinance installed")
    except ImportError:
        error("✗ yfinance not installed")
        missing.append("yfinance")
    
    if missing:
        print(f"\nInstall missing packages with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True


def check_fred_api_key():
    """Check if FRED API key is configured."""
    info("Checking FRED API key...")
    
    api_key = os.environ.get('FRED_API_KEY')
    
    if api_key:
        success(f"✓ FRED_API_KEY found in environment (length: {len(api_key)})")
        return api_key
    else:
        error("✗ FRED_API_KEY not found in environment")
        print("\nSet your FRED API key:")
        print("  export FRED_API_KEY='your_api_key_here'")
        print("\nOr add to your shell profile (~/.bashrc, ~/.zshrc):")
        print("  echo 'export FRED_API_KEY=\"your_api_key\"' >> ~/.bashrc")
        return None


def test_fred_connection(api_key):
    """Test FRED API connection."""
    info("Testing FRED API connection...")
    
    try:
        from src.data.external.fred_client import FREDClient
        
        client = FREDClient(api_key=api_key)
        
        if not client.is_connected:
            error("✗ Failed to connect to FRED")
            return False
        
        # Try fetching a simple series
        data = client.fetch_series('DFF', start_date='2024-01-01')
        
        if data is not None and len(data) > 0:
            success(f"✓ FRED connection successful - fetched {len(data)} Fed Funds Rate observations")
            return True
        else:
            error("✗ FRED connected but no data returned")
            return False
            
    except Exception as e:
        error(f"✗ FRED test failed: {e}")
        return False


def test_yahoo_connection():
    """Test Yahoo Finance connection."""
    info("Testing Yahoo Finance connection...")
    
    try:
        from src.data.external.yahoo_client import YahooClient
        
        client = YahooClient()
        
        if not client.is_connected:
            error("✗ Failed to initialize Yahoo client")
            return False
        
        # Use the built-in test method
        if client.test_connection():
            success("✓ Yahoo Finance connection working")
            return True
        
        # If test_connection fails, try fetching SPY directly
        info("Trying alternative test with SPY...")
        data = client.fetch_ohlcv('SPX', start_date='2024-01-01')
        
        if data is not None and len(data) > 0:
            success(f"✓ Yahoo connection successful - fetched {len(data)} observations")
            return True
        else:
            error("✗ Yahoo connected but no data returned")
            return False
            
    except Exception as e:
        error(f"✗ Yahoo test failed: {e}")
        return False


def prefetch_common_data():
    """Pre-fetch commonly used external data to cache."""
    info("Pre-fetching common data to cache...")
    
    try:
        from src.data.external import get_fred_client, get_yahoo_client, get_external_store
        
        store = get_external_store()
        fred = get_fred_client()
        yahoo = get_yahoo_client()
        
        # FRED: Interest rates
        if fred.is_connected:
            info("Fetching interest rates from FRED...")
            currencies = ['USD', 'EUR', 'GBP', 'JPY']
            for currency in currencies:
                rate = store.get_or_fetch_fred(
                    fred.RATE_SERIES.get(currency, 'DFF'),
                    fred
                )
                if rate is not None:
                    success(f"  ✓ {currency} rate: {len(rate)} observations")
            
            # Yield curve
            info("Fetching yield curve data...")
            for name, series in [('US2Y', 'DGS2'), ('US10Y', 'DGS10')]:
                data = store.get_or_fetch_fred(series, fred)
                if data is not None:
                    success(f"  ✓ {name}: {len(data)} observations")
        
        # Yahoo: Cross-asset data
        if yahoo.is_connected:
            info("Fetching cross-asset data from Yahoo...")
            assets = ['DXY', 'SPX', 'VIX', 'GOLD', 'OIL']
            for asset in assets:
                data = store.get_or_fetch_yahoo(asset, yahoo)
                if data is not None:
                    success(f"  ✓ {asset}: {len(data)} observations")
        
        return True
        
    except Exception as e:
        error(f"Pre-fetch failed: {e}")
        return False


def show_cache_status():
    """Show current cache status."""
    info("Cache status:")
    
    try:
        from src.data.external import get_external_store
        
        store = get_external_store()
        status = store.get_cache_status()
        
        print(f"\n  FRED series cached: {status['fred_count']}")
        if status['fred_series']:
            for s in status['fred_series'][:5]:
                print(f"    - {s}")
            if len(status['fred_series']) > 5:
                print(f"    ... and {len(status['fred_series']) - 5} more")
        
        print(f"\n  Yahoo symbols cached: {status['yahoo_count']}")
        if status['yahoo_symbols']:
            for s in status['yahoo_symbols'][:5]:
                print(f"    - {s}")
            if len(status['yahoo_symbols']) > 5:
                print(f"    ... and {len(status['yahoo_symbols']) - 5} more")
        
        print(f"\n  Total files: {status['total_files']}")
        
    except Exception as e:
        error(f"Failed to get cache status: {e}")


def main():
    print("=" * 60)
    print("External Data Setup & Test")
    print("=" * 60)
    print()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Fix missing dependencies first")
        return 1
    
    print()
    
    # Step 2: Check FRED API key
    api_key = check_fred_api_key()
    fred_ok = False
    
    print()
    
    # Step 3: Test FRED (if key available)
    if api_key:
        fred_ok = test_fred_connection(api_key)
    else:
        print("Skipping FRED test (no API key)")
    
    print()
    
    # Step 4: Test Yahoo
    yahoo_ok = test_yahoo_connection()
    
    print()
    
    # Step 5: Pre-fetch data (if both work)
    if fred_ok or yahoo_ok:
        prefetch_common_data()
    
    print()
    
    # Step 6: Show cache status
    show_cache_status()
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  FRED API:      {'✓ OK' if fred_ok else '✗ Not available'}")
    print(f"  Yahoo Finance: {'✓ OK' if yahoo_ok else '✗ Not available'}")
    
    if fred_ok and yahoo_ok:
        print("\n✓ All external data sources ready!")
        return 0
    elif yahoo_ok:
        print("\n⚠ Yahoo working, but FRED not available")
        print("  Some features (carry factor, yield curve) will use proxies")
        return 0
    else:
        print("\n❌ External data sources not working")
        return 1


if __name__ == "__main__":
    sys.exit(main())