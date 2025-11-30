"""
Intrinsic Metadata - Track intrinsic-time data updates
Stores metadata about intrinsic-time Parquet files
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from src.utils.logger import info, success, error
from src.data.path_builder import sanitize_symbol
from src.intrinsic.path_builder import delta_to_suffix


class IntrinsicMetadataTracker:
    """
    Track metadata for intrinsic-time Parquet files
    
    Metadata schema:
    {
      "EURUSD_dc010": {
        "threshold": 0.0010,
        "events": <int>,
        "last_update": "<timestamp>",
        "dc_start": "...",
        "dc_end": "...",
        "source_clock_file": "EURUSD_1m.parquet"
      }
    }
    """
    
    def __init__(self, metadata_file: str = "data/raw/intrinsic/metadata.json"):
        """
        Initialize intrinsic metadata tracker
        
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
                error(f"Failed to load intrinsic metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            error(f"Failed to save intrinsic metadata: {e}")
    
    def record_update(
        self,
        symbol: str,
        delta: float,
        events_count: int,
        dc_start,
        dc_end,
        source_clock_file: str = None
    ):
        """
        Record update metadata for a symbol/delta pair.
        
        Args:
            symbol: Trading symbol
            delta: Directional-change threshold
            events_count: Number of events
            dc_start: First DC timestamp (pandas Timestamp or string)
            dc_end: Last DC timestamp (pandas Timestamp or string)
            source_clock_file: Source clock-time file name
        """
        suffix = delta_to_suffix(delta)
        key = f"{sanitize_symbol(symbol)}_{suffix}"
        
        # Convert timestamps to string if needed
        if hasattr(dc_start, 'strftime'):
            dc_start_str = dc_start.strftime("%Y-%m-%d %H:%M:%S")
        else:
            dc_start_str = str(dc_start) if dc_start is not None else ""
        
        if hasattr(dc_end, 'strftime'):
            dc_end_str = dc_end.strftime("%Y-%m-%d %H:%M:%S")
        else:
            dc_end_str = str(dc_end) if dc_end is not None else ""
        
        # Default source clock file
        if source_clock_file is None:
            source_clock_file = f"{sanitize_symbol(symbol)}_1m.parquet"
        
        self.metadata[key] = {
            "threshold": delta,
            "events": events_count,
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dc_start": dc_start_str,
            "dc_end": dc_end_str,
            "source_clock_file": source_clock_file
        }
        
        self._save_metadata()
        info(f"Recorded intrinsic metadata update for {key}")
    
    def get(self, symbol: str, delta: float) -> Optional[Dict]:
        """
        Get metadata for a symbol/delta pair
        
        Args:
            symbol: Trading symbol
            delta: Directional-change threshold
        
        Returns:
            Metadata dictionary or None
        """
        suffix = delta_to_suffix(delta)
        key = f"{sanitize_symbol(symbol)}_{suffix}"
        return self.metadata.get(key)
    
    def get_last_updated(self, symbol: str, delta: float) -> Optional[str]:
        """
        Get last updated timestamp for a symbol/delta pair.
        
        Args:
            symbol: Trading symbol
            delta: Directional-change threshold
        
        Returns:
            Last updated timestamp string, or None if not found
        """
        metadata = self.get(symbol, delta)
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
_intrinsic_metadata_tracker: Optional[IntrinsicMetadataTracker] = None


def get_intrinsic_metadata_tracker(metadata_file: str = "data/raw/intrinsic/metadata.json") -> IntrinsicMetadataTracker:
    """
    Get global intrinsic metadata tracker instance
    
    Args:
        metadata_file: Path to metadata file
    
    Returns:
        IntrinsicMetadataTracker instance
    """
    global _intrinsic_metadata_tracker
    if _intrinsic_metadata_tracker is None:
        _intrinsic_metadata_tracker = IntrinsicMetadataTracker(metadata_file)
    return _intrinsic_metadata_tracker
