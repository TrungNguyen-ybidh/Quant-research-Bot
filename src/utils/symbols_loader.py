"""
Symbols Loader - Load FX pairs from symbols.yaml
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any
from src.utils.logger import info, error


def load_symbols(symbols_file: str = "config/symbols.yaml") -> Dict[str, Any]:
    """
    Load symbols configuration from YAML file.
    
    Args:
        symbols_file: Path to symbols.yaml
    
    Returns:
        Dictionary with fx_pairs and metadata
    """
    path = Path(symbols_file)
    
    if not path.exists():
        error(f"Symbols file not found: {symbols_file}")
        return {}
    
    try:
        with open(path, 'r') as f:
            symbols = yaml.safe_load(f)
        return symbols
    except Exception as e:
        error(f"Failed to load symbols: {e}")
        return {}


def load_fx_pairs(symbols_file: str = "config/symbols.yaml") -> List[str]:
    """
    Load all FX pairs as a flat list.
    
    Args:
        symbols_file: Path to symbols.yaml
    
    Returns:
        Flat list of all FX pair symbols
        Example: ["EURUSD", "GBPUSD", "USDJPY", ...]
    """
    data = load_symbols(symbols_file)
    
    if not data:
        return []
    
    pairs = []
    fx_pairs = data.get("fx_pairs", {})
    
    for group in fx_pairs.values():
        if group is not None:
            pairs.extend(group)
    
    return pairs


def get_pairs_by_group(group: str, symbols_file: str = "config/symbols.yaml") -> List[str]:
    """
    Get FX pairs for a specific group.
    
    Args:
        group: Group name (e.g., "majors", "gold", "yen_crosses")
        symbols_file: Path to symbols.yaml
    
    Returns:
        List of FX pair symbols in the group
    """
    data = load_symbols(symbols_file)
    fx_pairs = data.get("fx_pairs", {})
    return fx_pairs.get(group, [])


def get_symbol_metadata(symbols_file: str = "config/symbols.yaml") -> Dict[str, str]:
    """
    Get metadata from symbols.yaml.
    
    Args:
        symbols_file: Path to symbols.yaml
    
    Returns:
        Metadata dictionary with default_exchange, sec_type, etc.
    """
    data = load_symbols(symbols_file)
    return data.get("metadata", {})

