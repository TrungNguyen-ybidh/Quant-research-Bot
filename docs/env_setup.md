# Environment Setup Guide

## Recommended Installation Order (Windows)

For best compatibility with Parquet file handling and fastparquet fallback support, install dependencies in this order:

```bash
pip install --upgrade pip setuptools wheel
pip install "numpy>=1.24,<2.0"
pip install "pyarrow>=14,<16"
pip install python-snappy zstandard
pip install "fastparquet>=2024.5.0"
pip install "pandas>=2,<2.2"
pip install -r requirements.txt
```

### Why This Order?

1. **pip/setuptools/wheel**: Ensures latest build tools
2. **numpy**: Core dependency, must be installed first
3. **pyarrow**: Parquet support, needs numpy
4. **python-snappy/zstandard**: Compression libraries needed by fastparquet
5. **fastparquet**: Fallback Parquet engine, needs compression libs
6. **pandas**: Depends on numpy and benefits from pyarrow/fastparquet
7. **requirements.txt**: Installs remaining dependencies

### Troubleshooting

If you encounter issues:

- **NumPy compatibility**: Ensure NumPy < 2.0.0 (required for pyarrow compatibility)
- **fastparquet import errors**: Install python-snappy and zstandard first
- **Parquet read errors**: Use the diagnostic script: `python scripts/check_parquet_engines.py`

