"""
IBKR Client - Interactive Brokers API Connector
Clean, production-ready connector for historical data and real-time streaming
"""

import threading
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

from src.utils.logger import info, success, error, warning
from src.api.contract_builder import build_fx_contract


class IBKRClient(EWrapper, EClient):
    """
    IBKR API Client
    Handles connection, data requests, and callbacks
    """
    
    def __init__(self, host="127.0.0.1", port=7497, client_id=1, timeout=15):
        """
        Initialize IBKR client
        
        Args:
            host: TWS/Gateway host (default: "127.0.0.1")
            port: TWS/Gateway port (7497=PAPER, 7496=LIVE, default: 7497)
            client_id: Unique client identifier (default: 1)
            timeout: Connection timeout in seconds (default: 15)
        """
        EClient.__init__(self, self)
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        
        self.historical_data = []
        self.connected = False
        self.error_messages = []
        self.data_received = False
        self.data_complete = False
    
    # ---------------------
    # CONNECTION EVENTS
    # ---------------------
    def nextValidId(self, orderId):
        """Called when connection is established"""
        self.connected = True
        success("Connected to IBKR")
    
    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):
        """
        Error callback - IBKR API signature with all parameters
        
        Args:
            reqId: Request ID
            errorTime: Error timestamp (string)
            errorCode: Error code
            errorString: Error message
            advancedOrderRejectJson: Advanced order reject JSON (optional)
        """
        # Always store error messages for checking later
        self.error_messages.append(f"Error {errorCode}: {errorString}")
        
        # Filter out informational messages
        # Codes 2104, 2106, 2158 are connection status messages, not errors
        informational_codes = {2104, 2106, 2158}
        
        # Log all errors >= 1000 except informational ones
        if errorCode >= 1000 and errorCode not in informational_codes:
            error(f"IBKR Error {errorCode}: {errorString}")
        
        # Special handling for critical errors
        if errorCode == 162:
            error(f"CRITICAL: {errorString}")
            error("TWS session is connected from a different IP address. Please close other TWS sessions or configure TWS to allow multiple connections.")
        elif errorCode == 10285:
            error(f"CRITICAL: {errorString}")
            error("API version issue. The setConnectionOptions may not have taken effect.")
    
    # ---------------------
    # HISTORICAL DATA EVENTS
    # ---------------------
    def historicalData(self, reqId, bar):
        """
        Callback for historical data bars
        
        Args:
            reqId: Request ID
            bar: Historical data bar object
        """
        self.historical_data.append(bar)
        self.data_received = True
        # Log first few bars for debugging
        if len(self.historical_data) <= 3:
            info(f"Received bar {len(self.historical_data)}: {bar.date} O={bar.open:.5f} C={bar.close:.5f}")
    
    def historicalDataEnd(self, reqId, start, end):
        """
        Called when historical data request is complete
        
        Args:
            reqId: Request ID
            start: Start date string
            end: End date string
        """
        self.data_complete = True
        success(f"Historical data received ({len(self.historical_data)} bars)")
    
    # ---------------------
    # START + STOP
    # ---------------------
    def start(self):
        """Start the socket thread safely"""
        thread = threading.Thread(target=self.run)
        thread.daemon = True
        thread.start()
        time.sleep(1)  # Give connection time to establish
    
    def stop(self):
        """Stop the connection"""
        self.disconnect()
        self.connected = False


# ---------------------------------------------------
# PUBLIC FUNCTION FOR USER: FETCH HISTORICAL FX DATA
# ---------------------------------------------------
def fetch_fx_history(
    pair: str,
    bar_size: str = "1 hour",
    duration: str = "365 D",
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
    what_to_show: str = "MIDPOINT",
    end_date_time: str = ""
):
    """
    Fetch historical data for any FX pair.
    
    Args:
        pair: FX pair symbol (e.g., "EURUSD")
        bar_size: Bar size (e.g., "1 hour", "1 day")
        duration: Duration string (e.g., "365 D", "1 Y")
        host: TWS/Gateway host (default: "127.0.0.1")
        port: TWS/Gateway port (default: 7497 for paper)
        client_id: Client identifier (default: 1)
        what_to_show: Data type - "MIDPOINT" for FX (default: "MIDPOINT")
        end_date_time: End date/time string (empty = now, format: "YYYYMMDD HH:MM:SS")
    
    Returns:
        List of historical data bars
    """
    client = IBKRClient(host=host, port=port, client_id=client_id)
    
    # Set API version to 180 (supports fractional size rules, minimum 163 required)
    # This must be called before connect()
    # Note: setConnectionOptions may not be available in all ibapi versions
    if hasattr(client, 'setConnectionOptions'):
        try:
            client.setConnectionOptions("v180")
        except Exception as e:
            warning(f"Could not set connection options: {e}")
    else:
        warning("setConnectionOptions not available in this ibapi version")
    
    # Connect
    client.connect(client.host, client.port, client.client_id)
    client.start()
    
    # Wait a bit more for connection to fully establish
    time.sleep(2)
    
    # Check for connection errors
    if not client.connected:
        error("Failed to establish connection to IBKR")
        client.stop()
        return []
    
    # Build contract
    contract = build_fx_contract(pair)
    
    # Request data
    info(f"Requesting {pair} ({bar_size}, {duration})")
    
    client.reqHistoricalData(
        1,  # reqId
        contract,
        end_date_time,  # endDateTime (empty = now)
        duration,  # durationStr
        bar_size,  # barSizeSetting
        what_to_show,  # whatToShow (best for FX: MIDPOINT)
        0,  # useRTH (0 = all hours)
        1,  # formatDate
        False,  # keepUpToDate
        []  # chartOptions
    )
    
    # Wait for data with timeout (longer for smaller timeframes with more data)
    # 1-minute bars can have 43,200 bars for 30 days, so need more time
    if "1 min" in bar_size or "5 mins" in bar_size:
        timeout = 90  # 90 seconds for minute-level data
    else:
        timeout = 60  # 60 seconds for larger timeframes
    start_time = time.time()
    while not client.data_complete and (time.time() - start_time) < timeout:
        time.sleep(0.1)
    
    # Check for critical errors that would prevent data
    critical_errors = [msg for msg in client.error_messages if "162" in msg or "10285" in msg or "200" in msg]
    if critical_errors:
        error(f"Critical errors detected: {critical_errors}")
    
    if not client.data_complete:
        warning(f"Historical data request timed out after {timeout} seconds")
        # Filter out informational codes from error messages
        informational_codes = {2104, 2106, 2158}
        real_errors = [
            msg for msg in client.error_messages 
            if not any(f"Error {code}:" in msg for code in informational_codes)
        ]
        if real_errors:
            error(f"Errors during request: {real_errors[-5:]}")  # Show last 5 real errors
        if not client.data_received:
            error("No data bars were received at all")
        else:
            info(f"Received {len(client.historical_data)} bars before timeout")
    
    # Give a small buffer for any remaining callbacks
    time.sleep(0.5)
    
    # Stop connection
    client.stop()
    
    return client.historical_data

