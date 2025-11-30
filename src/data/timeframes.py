"""
Timeframe Mappings and Helpers
Maps timeframes to IBKR bar sizes and file suffixes
"""

TIMEFRAME_MAP = {
    "1 min":  {"ibkr": "1 min",  "suffix": "1m"},
    "5 mins": {"ibkr": "5 mins", "suffix": "5m"},
    "15 mins": {"ibkr": "15 mins", "suffix": "15m"},
    "1 hour": {"ibkr": "1 hour", "suffix": "1h"},
    "4 hours": {"ibkr": "4 hours", "suffix": "4h"},
    "1 day":  {"ibkr": "1 day",  "suffix": "1d"},
}

INTRINSIC_MAP = {
    0.0005: {"suffix": "dc005", "desc": "ultra_short"},
    0.0010: {"suffix": "dc010", "desc": "short"},
    0.0020: {"suffix": "dc020", "desc": "intraday"},
    0.0050: {"suffix": "dc050", "desc": "swing"},
    0.0100: {"suffix": "dc100", "desc": "macro_shift"},
}


def timeframe_to_suffix(tf: str) -> str:
    """
    Convert timeframe string to file suffix.
    
    Args:
        tf: Timeframe string (e.g., "1 hour", "1 min")
    
    Returns:
        Suffix string (e.g., "1h", "1m")
    """
    if tf in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[tf]["suffix"]
    
    # Fallback: try to normalize
    tf_lower = tf.lower().strip()
    for key, value in TIMEFRAME_MAP.items():
        if key.lower() == tf_lower:
            return value["suffix"]
    
    # Default: sanitize the timeframe
    return tf.replace(" ", "").lower()


def timeframe_to_bar_size(tf: str) -> str:
    """
    Convert timeframe string to IBKR bar size.
    
    Args:
        tf: Timeframe string (e.g., "1 hour", "1 min")
    
    Returns:
        IBKR bar size string (e.g., "1 hour", "1 min")
    """
    if tf in TIMEFRAME_MAP:
        return TIMEFRAME_MAP[tf]["ibkr"]
    
    # Fallback: try to normalize
    tf_lower = tf.lower().strip()
    for key, value in TIMEFRAME_MAP.items():
        if key.lower() == tf_lower:
            return value["ibkr"]
    
    # Default: return as-is (assume it's already in IBKR format)
    return tf

