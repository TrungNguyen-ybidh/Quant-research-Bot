"""
Parquet Engines Diagnostic Script
Checks pyarrow and fastparquet availability and functionality
"""

import sys
import tempfile
import os
from pathlib import Path

def check_pyarrow():
    """Check pyarrow availability and version"""
    try:
        import pyarrow
        import pyarrow.parquet as pq
        print(f"✓ PyArrow version: {pyarrow.__version__}")
        return True, pyarrow.__version__
    except ImportError as e:
        print(f"✗ PyArrow not available: {e}")
        return False, None

def check_fastparquet():
    """Check fastparquet availability and version"""
    try:
        import fastparquet
        print(f"✓ Fastparquet version: {fastparquet.__version__}")
        return True, fastparquet.__version__
    except ImportError as e:
        print(f"✗ Fastparquet not available: {e}")
        return False, None

def test_pyarrow_roundtrip():
    """Test pyarrow write/read round-trip"""
    try:
        import pandas as pd
        import pyarrow.parquet as pq
        
        # Create test data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
            'price': [100.0 + i * 0.1 for i in range(10)],
            'volume': [1000 + i * 100 for i in range(10)]
        })
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write
            df.to_parquet(tmp_path, engine='pyarrow', index=False)
            
            # Read
            df_read = pd.read_parquet(tmp_path, engine='pyarrow')
            
            # Verify
            if len(df_read) == len(df) and list(df_read.columns) == list(df.columns):
                print("✓ PyArrow round-trip test: PASSED")
                return True
            else:
                print("✗ PyArrow round-trip test: FAILED (data mismatch)")
                return False
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        print(f"✗ PyArrow round-trip test: FAILED ({e})")
        return False

def test_fastparquet_roundtrip():
    """Test fastparquet write/read round-trip"""
    try:
        import pandas as pd
        import fastparquet
        
        # Create test data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
            'price': [100.0 + i * 0.1 for i in range(10)],
            'volume': [1000 + i * 100 for i in range(10)]
        })
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write
            df.to_parquet(tmp_path, engine='fastparquet', index=False)
            
            # Read
            df_read = pd.read_parquet(tmp_path, engine='fastparquet')
            
            # Verify
            if len(df_read) == len(df) and list(df_read.columns) == list(df.columns):
                print("✓ Fastparquet round-trip test: PASSED")
                return True
            else:
                print("✗ Fastparquet round-trip test: FAILED (data mismatch)")
                return False
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except ImportError:
        print("✗ Fastparquet round-trip test: SKIPPED (not installed)")
        return None
    except Exception as e:
        print(f"✗ Fastparquet round-trip test: FAILED ({e})")
        return False

def main():
    """Run all diagnostic checks"""
    print("=" * 60)
    print("Parquet Engines Diagnostic")
    print("=" * 60)
    print()
    
    # Check versions
    print("1. Checking engine availability:")
    print("-" * 60)
    pyarrow_ok, pyarrow_ver = check_pyarrow()
    fastparquet_ok, fastparquet_ver = check_fastparquet()
    print()
    
    # Check dependencies
    print("2. Checking dependencies:")
    print("-" * 60)
    try:
        import pandas as pd
        import numpy as np
        print(f"✓ Pandas version: {pd.__version__}")
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
    print()
    
    # Test round-trips
    print("3. Testing round-trip functionality:")
    print("-" * 60)
    pyarrow_test = test_pyarrow_roundtrip() if pyarrow_ok else None
    fastparquet_test = test_fastparquet_roundtrip() if fastparquet_ok else None
    print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print("-" * 60)
    print(f"PyArrow:      {'✓ Available' if pyarrow_ok else '✗ Not available'} {f'({pyarrow_ver})' if pyarrow_ver else ''}")
    print(f"Fastparquet:  {'✓ Available' if fastparquet_ok else '✗ Not available'} {f'({fastparquet_ver})' if fastparquet_ver else ''}")
    print(f"PyArrow test: {'✓ PASSED' if pyarrow_test else '✗ FAILED' if pyarrow_test is False else 'SKIPPED'}")
    print(f"Fastparquet test: {'✓ PASSED' if fastparquet_test else '✗ FAILED' if fastparquet_test is False else 'SKIPPED'}")
    print()
    
    # Exit code
    if pyarrow_ok and (pyarrow_test or fastparquet_test):
        print("✓ System is ready for Parquet operations")
        sys.exit(0)
    else:
        print("✗ System may have issues with Parquet operations")
        print("  Please check installation and dependencies")
        sys.exit(1)

if __name__ == "__main__":
    main()

