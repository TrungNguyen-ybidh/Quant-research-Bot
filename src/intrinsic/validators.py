"""
Intrinsic Validators - Consistency checks for intrinsic-time data
Validates event DataFrames before saving
"""

import pandas as pd
from src.utils.logger import warning, success, error


def validate_intrinsic_events(df: pd.DataFrame) -> bool:
    """
    Validate intrinsic-time event DataFrame.
    
    Checks for:
    - Required columns matching exact schema
    - Monotonic event_id
    - Monotonic timestamps
    - Direction values (+1 or -1)
    - Return calculations
    - Duration calculations
    
    Args:
        df: DataFrame with intrinsic events
    
    Returns:
        True if valid, False otherwise
    """
    if df.empty:
        error("Intrinsic events DataFrame is empty")
        return False
    
    # Required columns (exact schema)
    required_cols = [
        'event_id', 'direction', 'dc_timestamp', 'dc_price',
        'previous_extreme_price', 'dc_return', 'overshoot_price',
        'overshoot_return', 'clock_start_ts', 'clock_end_ts',
        'duration_seconds', 'symbol', 'delta'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        error(f"Missing required columns: {', '.join(missing_cols)}")
        return False
    
    # Check event_id is monotonic
    if 'event_id' in df.columns:
        if not df['event_id'].is_monotonic_increasing:
            error("event_id is not monotonic")
            return False
        
        # Check event_id starts at 1 and is sequential
        expected_ids = list(range(1, len(df) + 1))
        if not df['event_id'].tolist() == expected_ids:
            warning("event_id is not sequential starting from 1")
    
    # Check direction values
    if 'direction' in df.columns:
        valid_directions = [+1, -1]
        invalid_directions = df[~df['direction'].isin(valid_directions)]['direction'].unique()
        if len(invalid_directions) > 0:
            error(f"Invalid directions found: {invalid_directions}. Must be +1 or -1")
            return False
    
    # Check timestamp ordering
    if 'dc_timestamp' in df.columns:
        if not df['dc_timestamp'].is_monotonic_increasing:
            warning("dc_timestamp is not sorted. Consider sorting before saving.")
        
        # Check for duplicate timestamps
        duplicates = df['dc_timestamp'].duplicated().sum()
        if duplicates > 0:
            warning(f"Found {duplicates} duplicate dc_timestamps")
    
    # Check clock timestamps
    if 'clock_start_ts' in df.columns and 'clock_end_ts' in df.columns:
        invalid_durations = (df['clock_end_ts'] < df['clock_start_ts']).sum()
        if invalid_durations > 0:
            error(f"Found {invalid_durations} events where clock_end_ts < clock_start_ts")
            return False
    
    # Check duration_seconds
    if 'duration_seconds' in df.columns:
        negative_durations = (df['duration_seconds'] < 0).sum()
        if negative_durations > 0:
            error(f"Found {negative_durations} negative duration_seconds")
            return False
    
    # Check price consistency
    if 'dc_price' in df.columns and 'previous_extreme_price' in df.columns:
        # Check that prices are positive
        if (df['dc_price'] <= 0).any() or (df['previous_extreme_price'] <= 0).any():
            error("Found non-positive prices")
            return False
    
    # Check overshoot_price consistency
    if 'overshoot_price' in df.columns and 'dc_price' in df.columns and 'direction' in df.columns:
        for idx, row in df.iterrows():
            if row['direction'] == +1:  # UP: overshoot should be >= dc_price
                if row['overshoot_price'] < row['dc_price']:
                    warning(f"Event {row['event_id']}: UP direction but overshoot_price < dc_price")
            elif row['direction'] == -1:  # DOWN: overshoot should be <= dc_price
                if row['overshoot_price'] > row['dc_price']:
                    warning(f"Event {row['event_id']}: DOWN direction but overshoot_price > dc_price")
    
    success("Intrinsic events validation complete")
    return True
