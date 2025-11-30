"""
Event Builder - Overshoot logic and event construction
Builds intrinsic-time events (DC + overshoot) from clock-time data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.intrinsic.dc_engine import DCEngine
from src.intrinsic.loader import calculate_mid_price
from src.utils.logger import info, warning


def build_intrinsic_events(df: pd.DataFrame, delta: float) -> pd.DataFrame:
    """
    Build intrinsic-time events from clock-time DataFrame.
    
    Processes 1-minute bars and generates DC + overshoot events.
    Each row in the output represents one EVENT (not one candle).
    
    Event Schema:
    - event_id: Sequential event counter
    - direction: +1 (UP) or -1 (DOWN)
    - dc_timestamp: When DC occurred
    - dc_price: Price at DC point
    - previous_extreme_price: Previous extreme before this DC
    - dc_return: Return from previous extreme to DC
    - overshoot_price: Maximum overshoot price after DC
    - overshoot_return: Return from DC to overshoot
    - clock_start_ts: First clock bar timestamp in this event
    - clock_end_ts: Last clock bar timestamp in this event
    - duration_seconds: Duration of event in seconds
    - symbol: Trading symbol
    - delta: Threshold used
    
    Args:
        df: DataFrame with 1-minute clock-time bars
        delta: Directional-change threshold (e.g., 0.0010)
    
    Returns:
        DataFrame with intrinsic-time events matching exact schema
    """
    if df.empty:
        return pd.DataFrame()
    
    # Calculate mid-price
    try:
        mid_prices = calculate_mid_price(df)
    except ValueError as e:
        warning(f"Error calculating mid-price: {e}")
        return pd.DataFrame()
    
    # Initialize DC engine
    dc_engine = DCEngine(delta)
    
    events = []
    
    # Track current event state
    current_event_id = None
    current_dc_timestamp = None
    current_dc_price = None
    current_direction = None
    current_previous_extreme = None
    current_clock_start_ts = None
    current_clock_end_ts = None
    
    # Track overshoot
    overshoot_price = None
    overshoot_timestamp = None
    
    # Track extreme timestamp (when extreme price was reached)
    extreme_timestamp = None
    
    # Get symbol from DataFrame if available
    symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else None
    
    total_rows = len(df)
    info(f"Processing {total_rows} clock-time bars for {symbol} with delta {delta}")
    
    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        price = mid_prices.iloc[idx]
        
        # Detect DC
        is_dc, direction, dc_price, previous_extreme = dc_engine.detect_dc(price)
        
        if is_dc:
            # Finalize previous event (if exists) with overshoot
            if current_event_id is not None:
                # Calculate overshoot return
                if overshoot_price is not None and current_dc_price is not None:
                    overshoot_return = (overshoot_price - current_dc_price) / current_dc_price
                    if current_direction == -1:  # DOWN direction
                        overshoot_return = -overshoot_return  # Negative for down moves
                else:
                    overshoot_return = 0.0
                
                # Calculate duration
                if current_clock_start_ts is not None and current_clock_end_ts is not None:
                    duration_seconds = (current_clock_end_ts - current_clock_start_ts).total_seconds()
                else:
                    duration_seconds = 0.0
                
                # Calculate DC return
                if current_previous_extreme is not None and current_dc_price is not None:
                    dc_return = (current_dc_price - current_previous_extreme) / current_previous_extreme
                    if current_direction == -1:  # DOWN direction
                        dc_return = -dc_return  # Negative for down moves
                else:
                    dc_return = 0.0
                
                events.append({
                    'event_id': current_event_id,
                    'direction': current_direction,
                    'dc_timestamp': current_dc_timestamp,
                    'dc_price': current_dc_price,
                    'previous_extreme_price': current_previous_extreme,
                    'dc_return': dc_return,
                    'overshoot_price': overshoot_price if overshoot_price is not None else current_dc_price,
                    'overshoot_return': overshoot_return,
                    'clock_start_ts': current_clock_start_ts,
                    'clock_end_ts': current_clock_end_ts,
                    'duration_seconds': duration_seconds,
                    'symbol': symbol,
                    'delta': delta
                })
            
            # Start new event
            # Use extreme timestamp if available, otherwise use current timestamp
            dc_ts = extreme_timestamp if extreme_timestamp is not None else timestamp
            
            current_event_id = dc_engine.event_id
            current_dc_timestamp = dc_ts  # Use timestamp when extreme occurred
            current_dc_price = dc_price
            current_direction = direction
            current_previous_extreme = previous_extreme
            current_clock_start_ts = dc_ts
            current_clock_end_ts = timestamp  # End at current timestamp (when DC detected)
            overshoot_price = price  # Initialize overshoot with current price
            overshoot_timestamp = timestamp
            extreme_timestamp = timestamp  # Reset for new event
        
        else:
            # Update current event
            if current_event_id is not None:
                current_clock_end_ts = timestamp
                
                # Track overshoot and its timestamp
                if current_direction == +1:  # UP mode: track highest price
                    if overshoot_price is None or price > overshoot_price:
                        overshoot_price = price
                        overshoot_timestamp = timestamp
                elif current_direction == -1:  # DOWN mode: track lowest price
                    if overshoot_price is None or price < overshoot_price:
                        overshoot_price = price
                        overshoot_timestamp = timestamp
    
    # Finalize last event if we have one
    if current_event_id is not None:
        # Calculate overshoot return
        if overshoot_price is not None and current_dc_price is not None:
            overshoot_return = (overshoot_price - current_dc_price) / current_dc_price
            if current_direction == -1:  # DOWN direction
                overshoot_return = -overshoot_return
        else:
            overshoot_return = 0.0
        
        # Calculate duration
        if current_clock_start_ts is not None and current_clock_end_ts is not None:
            duration_seconds = (current_clock_end_ts - current_clock_start_ts).total_seconds()
        else:
            duration_seconds = 0.0
        
        # Calculate DC return
        if current_previous_extreme is not None and current_dc_price is not None:
            dc_return = (current_dc_price - current_previous_extreme) / current_previous_extreme
            if current_direction == -1:  # DOWN direction
                dc_return = -dc_return
        else:
            dc_return = 0.0
        
        events.append({
            'event_id': current_event_id,
            'direction': current_direction,
            'dc_timestamp': current_dc_timestamp,
            'dc_price': current_dc_price,
            'previous_extreme_price': current_previous_extreme,
            'dc_return': dc_return,
            'overshoot_price': overshoot_price if overshoot_price is not None else current_dc_price,
            'overshoot_return': overshoot_return,
            'clock_start_ts': current_clock_start_ts,
            'clock_end_ts': current_clock_end_ts,
            'duration_seconds': duration_seconds,
            'symbol': symbol,
            'delta': delta
        })
    
    if not events:
        warning(f"No events generated for delta {delta}")
        return pd.DataFrame()
    
    # Create DataFrame with exact schema
    events_df = pd.DataFrame(events)
    
    # Ensure timestamps are datetime
    timestamp_cols = ['dc_timestamp', 'clock_start_ts', 'clock_end_ts']
    for col in timestamp_cols:
        if col in events_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(events_df[col]):
                events_df[col] = pd.to_datetime(events_df[col])
    
    # Sort by event_id to ensure monotonic
    events_df = events_df.sort_values('event_id').reset_index(drop=True)
    
    # Ensure event_id is monotonic (1, 2, 3, ...)
    events_df['event_id'] = range(1, len(events_df) + 1)
    
    info(f"Generated {len(events_df)} intrinsic events for delta {delta}")
    
    return events_df
