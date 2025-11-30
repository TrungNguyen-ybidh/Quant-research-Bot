"""
Directional-Change Engine - Core DC detection logic
Detects directional changes based on price movements exceeding threshold
"""

from typing import Optional, Tuple
from src.utils.logger import info, warning


class DCEngine:
    """
    Directional-Change detection engine.
    
    Maintains:
    - last_extreme_price: The extreme price from the previous DC
    - current_mode: "up" or "down"
    - event_id: Counter for events
    
    Events trigger when:
    - price moves +δ from last low → UP DC event (+1)
    - price moves -δ from last high → DOWN DC event (-1)
    """
    
    def __init__(self, delta: float):
        """
        Initialize DC engine with threshold.
        
        Args:
            delta: Directional-change threshold (e.g., 0.0010 = 0.1%)
        """
        self.delta = delta
        self.reset()
    
    def reset(self):
        """Reset engine state for new data series."""
        self.last_extreme_price: Optional[float] = None
        self.current_mode: Optional[str] = None  # "up" or "down"
        self.event_id: int = 0
        self.current_extreme: Optional[float] = None  # Extreme in current mode
        self.current_extreme_timestamp = None  # Timestamp when extreme occurred
    
    def detect_dc(self, price: float) -> Tuple[bool, Optional[int], Optional[float], Optional[float]]:
        """
        Detect if a directional change has occurred.
        
        Args:
            price: Current price
        
        Returns:
            Tuple of (is_dc, direction, dc_price, previous_extreme_price)
            - is_dc: True if DC detected
            - direction: +1 for UP, -1 for DOWN, None if no DC
            - dc_price: Price at which DC occurred (current extreme)
            - previous_extreme_price: The last extreme price before this DC
        """
        # Initialize on first price
        if self.last_extreme_price is None:
            self.last_extreme_price = price
            self.current_extreme = price
            return False, None, None, None
        
        # Update current extreme based on mode
        if self.current_mode == "up":
            # In up mode: track highest price
            self.current_extreme = max(self.current_extreme, price)
        elif self.current_mode == "down":
            # In down mode: track lowest price
            self.current_extreme = min(self.current_extreme, price)
        else:
            # No mode yet: determine initial direction
            # Check if price moved +δ from last low (UP DC)
            change_up = (price - self.last_extreme_price) / self.last_extreme_price
            if change_up >= self.delta:
                # UP DC detected: price moved +δ from last low
                previous_extreme = self.last_extreme_price
                self.last_extreme_price = price
                self.current_mode = "up"
                self.current_extreme = price
                self.event_id += 1
                return True, +1, price, previous_extreme
            
            # Check if price moved -δ from last high (DOWN DC)
            change_down = (self.last_extreme_price - price) / self.last_extreme_price
            if change_down >= self.delta:
                # DOWN DC detected: price moved -δ from last high
                previous_extreme = self.last_extreme_price
                self.last_extreme_price = price
                self.current_mode = "down"
                self.current_extreme = price
                self.event_id += 1
                return True, -1, price, previous_extreme
            
            # No DC yet, update current extreme
            if price > self.current_extreme:
                self.current_extreme = price
            elif price < self.current_extreme:
                self.current_extreme = price
            
            return False, None, None, None
        
        # Check for DC in opposite direction from current extreme
        if self.current_mode == "up":
            # In up mode: check for DOWN DC (price moves -δ from high)
            if price < self.current_extreme:
                change = (self.current_extreme - price) / self.current_extreme
                if change >= self.delta:
                    # DOWN DC detected
                    previous_extreme = self.current_extreme
                    dc_price = self.current_extreme
                    self.last_extreme_price = self.current_extreme
                    self.current_mode = "down"
                    self.current_extreme = price
                    self.event_id += 1
                    return True, -1, dc_price, previous_extreme
        
        elif self.current_mode == "down":
            # In down mode: check for UP DC (price moves +δ from low)
            if price > self.current_extreme:
                change = (price - self.current_extreme) / self.current_extreme
                if change >= self.delta:
                    # UP DC detected
                    previous_extreme = self.current_extreme
                    dc_price = self.current_extreme
                    self.last_extreme_price = self.current_extreme
                    self.current_mode = "up"
                    self.current_extreme = price
                    self.event_id += 1
                    return True, +1, dc_price, previous_extreme
        
        return False, None, None, None
    
    def get_state(self) -> dict:
        """
        Get current engine state.
        
        Returns:
            Dictionary with last_extreme_price, current_mode, event_id, current_extreme
        """
        return {
            "last_extreme_price": self.last_extreme_price,
            "current_mode": self.current_mode,
            "event_id": self.event_id,
            "current_extreme": self.current_extreme
        }
