"""
Metadata Tracker - Track data updates and versions
Stores metadata about Parquet files for tracking and versioning
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from src.utils.logger import info, success, error
from src.data.path_builder import sanitize_symbol
from src.data.timeframes import timeframe_to_suffix


class MetadataTracker:
    """
    Track metadata for Parquet files
    """
    
    def __init__(self, metadata_file: str = "data/raw/metadata.json"):
        """
        Initialize metadata tracker
        
        Args:
            metadata_file: Path to metadata JSON file
        """
        self.metadata_file = Path(metadata_file)
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load existing metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                error(f"Failed to load metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            error(f"Failed to save metadata: {e}")
    
    def record_update(
        self,
        symbol: str,
        timeframe: str,
        rows: int,
        last_ts,
        source: str = "IBKR"
    ):
        """
        Record update metadata for a symbol/timeframe pair.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            rows: Number of rows
            last_ts: Last timestamp (pandas Timestamp or string)
            source: Data source (default: "IBKR")
        """
        suffix = timeframe_to_suffix(timeframe)
        key = f"{sanitize_symbol(symbol)}_{suffix}"
        
        # Convert timestamp to string if needed
        if hasattr(last_ts, 'strftime'):
            last_ts_str = last_ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            last_ts_str = str(last_ts)
        
        self.metadata[key] = {
            "rows": rows,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_timestamp": last_ts_str,
            "source": source,
            "timeframe": suffix,
            "symbol": sanitize_symbol(symbol)
        }
        
        self._save_metadata()
        info(f"Recorded metadata update for {key}")
    
    def update(
        self,
        symbol: str,
        timeframe: str,
        rows: int,
        source: str = "IBKR"
    ):
        """
        Update metadata for a symbol/timeframe pair (legacy method).
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            rows: Number of rows
            source: Data source (default: "IBKR")
        """
        suffix = timeframe_to_suffix(timeframe)
        key = f"{sanitize_symbol(symbol)}_{suffix}"
        
        self.metadata[key] = {
            "rows": rows,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "timeframe": suffix,
            "symbol": sanitize_symbol(symbol)
        }
        
        self._save_metadata()
        info(f"Updated metadata for {key}")
    
    def get(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """
        Get metadata for a symbol/timeframe pair
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        
        Returns:
            Metadata dictionary or None
        """
        suffix = timeframe_to_suffix(timeframe)
        key = f"{sanitize_symbol(symbol)}_{suffix}"
        return self.metadata.get(key)
    
    def get_last_updated(self, symbol: str, timeframe: str) -> Optional[str]:
        """
        Get last updated timestamp for a symbol/timeframe pair.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
        
        Returns:
            Last updated timestamp string, or None if not found
        """
        metadata = self.get(symbol, timeframe)
        if metadata:
            return metadata.get("last_update")
        return None
    
    def list_all(self) -> Dict:
        """
        Get all metadata
        
        Returns:
            Dictionary of all metadata
        """
        return self.metadata


# Global metadata tracker instance
_metadata_tracker: Optional[MetadataTracker] = None


def get_metadata_tracker(metadata_file: str = "data/raw/metadata.json") -> MetadataTracker:
    """
    Get global metadata tracker instance
    
    Args:
        metadata_file: Path to metadata file
    
    Returns:
        MetadataTracker instance
    """
    global _metadata_tracker
    if _metadata_tracker is None:
        _metadata_tracker = MetadataTracker(metadata_file)
    return _metadata_tracker

