"""
External Data Store
Caches external data locally to minimize API calls
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from src.utils.logger import info, success, error


class ExternalDataStore:
    """
    Local cache for external data (FRED, Yahoo Finance).
    
    Stores data as Parquet files with metadata tracking.
    Implements smart refresh logic to minimize API calls.
    """
    
    def __init__(self, cache_dir: str = "data/external"):
        """
        Initialize external data store.
        
        Args:
            cache_dir: Directory for cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        self.fred_dir = self.cache_dir / "fred"
        self.yahoo_dir = self.cache_dir / "yahoo"
        self.fred_dir.mkdir(exist_ok=True)
        self.yahoo_dir.mkdir(exist_ok=True)
        
        # Metadata file
        self.metadata_path = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                error(f"Failed to load metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            error(f"Failed to save metadata: {e}")
    
    def _get_cache_path(self, source: str, key: str) -> Path:
        """
        Get cache file path for a data key.
        
        Args:
            source: 'fred' or 'yahoo'
            key: Data identifier (e.g., 'DFF', 'DXY')
        
        Returns:
            Path to cache file
        """
        if source == 'fred':
            return self.fred_dir / f"{key}.parquet"
        elif source == 'yahoo':
            return self.yahoo_dir / f"{key}.parquet"
        else:
            return self.cache_dir / f"{source}_{key}.parquet"
    
    def _is_cache_valid(self, source: str, key: str, max_age_hours: int = 24) -> bool:
        """
        Check if cached data is still valid.
        
        Args:
            source: Data source
            key: Data identifier
            max_age_hours: Maximum age in hours before refresh needed
        
        Returns:
            True if cache is valid, False if refresh needed
        """
        cache_key = f"{source}_{key}"
        
        if cache_key not in self.metadata:
            return False
        
        last_update = self.metadata[cache_key].get('last_update')
        if not last_update:
            return False
        
        # Parse last update time
        try:
            last_update_dt = datetime.fromisoformat(last_update)
            age = datetime.now() - last_update_dt
            return age < timedelta(hours=max_age_hours)
        except:
            return False
    
    def save(self, 
             data: pd.DataFrame | pd.Series, 
             source: str, 
             key: str,
             extra_metadata: Optional[Dict] = None):
        """
        Save data to cache.
        
        Args:
            data: DataFrame or Series to cache
            source: Data source ('fred', 'yahoo')
            key: Data identifier
            extra_metadata: Additional metadata to store
        """
        # Convert Series to DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=key)
        
        # Save to Parquet
        cache_path = self._get_cache_path(source, key)
        data.to_parquet(cache_path)
        
        # Update metadata
        cache_key = f"{source}_{key}"
        self.metadata[cache_key] = {
            'source': source,
            'key': key,
            'last_update': datetime.now().isoformat(),
            'rows': len(data),
            'columns': list(data.columns),
            'start_date': str(data.index.min()) if len(data) > 0 else None,
            'end_date': str(data.index.max()) if len(data) > 0 else None,
            **(extra_metadata or {})
        }
        self._save_metadata()
        
        success(f"Cached {source}/{key}: {len(data)} rows")
    
    def load(self, 
             source: str, 
             key: str,
             max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Load data from cache if valid.
        
        Args:
            source: Data source
            key: Data identifier
            max_age_hours: Maximum age before considering stale
        
        Returns:
            DataFrame if cache valid, None if needs refresh
        """
        cache_path = self._get_cache_path(source, key)
        
        if not cache_path.exists():
            info(f"Cache miss: {source}/{key} not found")
            return None
        
        if not self._is_cache_valid(source, key, max_age_hours):
            info(f"Cache stale: {source}/{key} older than {max_age_hours}h")
            return None
        
        try:
            data = pd.read_parquet(cache_path)
            info(f"Cache hit: {source}/{key} ({len(data)} rows)")
            return data
        except Exception as e:
            error(f"Failed to load cache {source}/{key}: {e}")
            return None
    
    def get_or_fetch_fred(self,
                          series_id: str,
                          fred_client,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          max_age_hours: int = 24) -> Optional[pd.Series]:
        """
        Get FRED data from cache or fetch if needed.
        
        Args:
            series_id: FRED series ID
            fred_client: FREDClient instance
            start_date: Start date
            end_date: End date
            max_age_hours: Cache validity period
        
        Returns:
            Series with FRED data
        """
        # Try cache first
        cached = self.load('fred', series_id, max_age_hours)
        if cached is not None:
            return cached[series_id] if series_id in cached.columns else cached.iloc[:, 0]
        
        # Fetch from FRED
        data = fred_client.fetch_series(series_id, start_date, end_date)
        
        if data is not None:
            self.save(data, 'fred', series_id)
            return data
        
        return None
    
    def get_or_fetch_yahoo(self,
                           symbol: str,
                           yahoo_client,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           interval: str = '1d',
                           max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Get Yahoo data from cache or fetch if needed.
        
        Args:
            symbol: Yahoo symbol or alias
            yahoo_client: YahooClient instance
            start_date: Start date
            end_date: End date
            interval: Data interval
            max_age_hours: Cache validity period
        
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{interval}"
        
        # Try cache first
        cached = self.load('yahoo', cache_key, max_age_hours)
        if cached is not None:
            return cached
        
        # Fetch from Yahoo
        data = yahoo_client.fetch_ohlcv(symbol, start_date, end_date, interval)
        
        if data is not None:
            self.save(data, 'yahoo', cache_key)
            return data
        
        return None
    
    def get_rate_differential(self,
                              pair: str,
                              fred_client,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              max_age_hours: int = 24) -> Optional[pd.Series]:
        """
        Get interest rate differential for FX pair.
        
        Args:
            pair: FX pair (e.g., 'EURUSD')
            fred_client: FREDClient instance
            start_date: Start date
            end_date: End date
            max_age_hours: Cache validity period
        
        Returns:
            Series with rate differential
        """
        cache_key = f"{pair}_rate_diff"
        
        # Try cache first
        cached = self.load('fred', cache_key, max_age_hours)
        if cached is not None:
            return cached.iloc[:, 0]
        
        # Compute from FRED
        if len(pair) != 6:
            error(f"Invalid pair: {pair}")
            return None
        
        base = pair[:3].upper()
        quote = pair[3:].upper()
        
        diff = fred_client.compute_rate_differential(base, quote, start_date, end_date)
        
        if diff is not None:
            self.save(diff, 'fred', cache_key)
            return diff
        
        return None
    
    def get_cross_asset_data(self,
                             yahoo_client,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             max_age_hours: int = 24) -> pd.DataFrame:
        """
        Get cross-asset data (DXY, SPX, VIX, GOLD, OIL).
        
        Args:
            yahoo_client: YahooClient instance
            start_date: Start date
            end_date: End date
            max_age_hours: Cache validity period
        
        Returns:
            DataFrame with cross-asset prices
        """
        assets = ['DXY', 'SPX', 'VIX', 'GOLD', 'OIL']
        data = {}
        
        for asset in assets:
            df = self.get_or_fetch_yahoo(
                asset, yahoo_client, start_date, end_date, '1d', max_age_hours
            )
            if df is not None and 'close' in df.columns:
                data[asset] = df['close']
        
        if not data:
            return pd.DataFrame()
        
        result = pd.DataFrame(data)
        result = result.sort_index().ffill()
        
        return result
    
    def clear_cache(self, source: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            source: If specified, only clear that source ('fred', 'yahoo')
        """
        if source == 'fred':
            for f in self.fred_dir.glob("*.parquet"):
                f.unlink()
            info("Cleared FRED cache")
        elif source == 'yahoo':
            for f in self.yahoo_dir.glob("*.parquet"):
                f.unlink()
            info("Cleared Yahoo cache")
        else:
            for f in self.fred_dir.glob("*.parquet"):
                f.unlink()
            for f in self.yahoo_dir.glob("*.parquet"):
                f.unlink()
            info("Cleared all external data cache")
        
        # Clear relevant metadata
        keys_to_remove = []
        for key in self.metadata:
            if source is None or key.startswith(f"{source}_"):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.metadata[key]
        
        self._save_metadata()
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Get status of cached data.
        
        Returns:
            Dict with cache statistics
        """
        fred_files = list(self.fred_dir.glob("*.parquet"))
        yahoo_files = list(self.yahoo_dir.glob("*.parquet"))
        
        return {
            'fred_count': len(fred_files),
            'yahoo_count': len(yahoo_files),
            'total_files': len(fred_files) + len(yahoo_files),
            'fred_series': [f.stem for f in fred_files],
            'yahoo_symbols': [f.stem for f in yahoo_files],
            'metadata_entries': len(self.metadata)
        }


def get_external_store(cache_dir: str = "data/external") -> ExternalDataStore:
    """Factory function for external data store."""
    return ExternalDataStore(cache_dir)