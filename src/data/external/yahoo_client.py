"""
Yahoo Finance Client
Fetches market data for cross-asset analysis (DXY, SPX, VIX, commodities)

Fixed version with robust error handling for Yahoo API changes.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
from src.utils.logger import info, success, error


class YahooClient:
    """
    Client for fetching market data from Yahoo Finance.
    
    Requires yfinance package: pip install yfinance
    No API key required.
    """
    
    # Common tickers for cross-asset analysis
    TICKERS = {
        # Dollar Index - try multiple alternatives
        'DXY': 'DX=F',           # Dollar Index Futures (more reliable)
        
        # Equity Indices
        'SPX': 'SPY',            # S&P 500 ETF (more reliable than ^GSPC)
        'NDX': 'QQQ',            # Nasdaq 100 ETF
        'VIX': 'VIXY',           # VIX ETF (more reliable than ^VIX)
        'DJI': 'DIA',            # Dow Jones ETF
        
        # Bonds (ETF proxies)
        'TLT': 'TLT',            # 20+ Year Treasury ETF
        'IEF': 'IEF',            # 7-10 Year Treasury ETF
        'SHY': 'SHY',            # 1-3 Year Treasury ETF
        
        # Commodities
        'GOLD': 'GLD',           # Gold ETF (more reliable than GC=F)
        'OIL': 'USO',            # Oil ETF (more reliable than CL=F)
        'SILVER': 'SLV',         # Silver ETF
        
        # FX (for cross-validation)
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'USDJPY': 'USDJPY=X',
        'USDCHF': 'USDCHF=X',
        'AUDUSD': 'AUDUSD=X',
        'USDCAD': 'USDCAD=X',
    }
    
    # Fallback tickers if primary fails
    FALLBACK_TICKERS = {
        'DXY': ['DX=F', 'UUP', 'USDU'],      # Dollar alternatives
        'SPX': ['SPY', '^GSPC', 'IVV'],       # S&P alternatives
        'VIX': ['VIXY', '^VIX', 'VXX'],       # VIX alternatives
        'GOLD': ['GLD', 'GC=F', 'IAU'],       # Gold alternatives
        'OIL': ['USO', 'CL=F', 'BNO'],        # Oil alternatives
    }
    
    # Timeframe mapping
    INTERVAL_MAP = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '60m',
        '1d': '1d',
        '1wk': '1wk',
        '1mo': '1mo',
    }
    
    def __init__(self):
        """Initialize Yahoo Finance client."""
        self._yf = None
        self._connected = False
        self._connect()
    
    def _connect(self) -> bool:
        """Import yfinance library."""
        try:
            import yfinance as yf
            self._yf = yf
            self._connected = True
            success("Yahoo Finance client initialized")
            return True
        except ImportError:
            error("yfinance not installed. Run: pip install yfinance")
            return False
    
    @property
    def is_connected(self) -> bool:
        """Check if client is available."""
        return self._connected and self._yf is not None
    
    def _resolve_ticker(self, symbol: str) -> str:
        """
        Resolve symbol alias to Yahoo ticker.
        
        Args:
            symbol: Symbol name or alias (e.g., 'DXY', 'SPX', 'GOLD')
        
        Returns:
            Yahoo Finance ticker string
        """
        symbol = symbol.upper()
        return self.TICKERS.get(symbol, symbol)
    
    def _get_fallback_tickers(self, symbol: str) -> List[str]:
        """Get list of fallback tickers for a symbol."""
        symbol = symbol.upper()
        primary = self._resolve_ticker(symbol)
        fallbacks = self.FALLBACK_TICKERS.get(symbol, [])
        
        # Return primary first, then fallbacks
        all_tickers = [primary] + [t for t in fallbacks if t != primary]
        return all_tickers
    
    def _fetch_with_ticker_object(self, ticker: str, start_date: str, 
                                   end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Fetch using Ticker object method (more robust).
        """
        try:
            ticker_obj = self._yf.Ticker(ticker)
            data = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True
            )
            return data if len(data) > 0 else None
        except Exception as e:
            return None
    
    def _fetch_with_download(self, ticker: str, start_date: str,
                              end_date: str, interval: str) -> Optional[pd.DataFrame]:
        """
        Fetch using download method.
        """
        try:
            data = self._yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            return data if len(data) > 0 else None
        except Exception as e:
            return None
    
    def fetch_ohlcv(self,
                    symbol: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a symbol with fallback support.
        
        Args:
            symbol: Symbol or alias (e.g., 'DXY', 'SPX', '^GSPC')
            start_date: Start date 'YYYY-MM-DD' (default: 5 years ago)
            end_date: End date 'YYYY-MM-DD' (default: today)
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
        
        Returns:
            DataFrame with OHLCV columns, or None if failed
        """
        if not self.is_connected:
            error("Yahoo Finance client not available")
            return None
        
        # Default date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        # Map interval
        yf_interval = self.INTERVAL_MAP.get(interval, interval)
        
        # Get list of tickers to try
        tickers_to_try = self._get_fallback_tickers(symbol)
        
        for ticker in tickers_to_try:
            info(f"Trying {symbol} with ticker: {ticker}")
            
            # Try Ticker object method first (more robust)
            data = self._fetch_with_ticker_object(ticker, start_date, end_date, yf_interval)
            
            # If that fails, try download method
            if data is None or len(data) == 0:
                data = self._fetch_with_download(ticker, start_date, end_date, yf_interval)
            
            if data is not None and len(data) > 0:
                # Standardize column names
                data.columns = [c.lower() for c in data.columns]
                
                # Ensure we have expected columns
                expected = ['open', 'high', 'low', 'close', 'volume']
                for col in expected:
                    if col not in data.columns:
                        data[col] = 0.0
                
                # Add symbol column
                data['symbol'] = symbol
                data['ticker_used'] = ticker
                
                success(f"Fetched {len(data)} bars for {symbol} using {ticker}")
                return data
        
        error(f"All tickers failed for {symbol}")
        return None
    
    def fetch_close(self,
                    symbol: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    interval: str = '1d') -> Optional[pd.Series]:
        """
        Fetch only close prices for a symbol.
        
        Args:
            symbol: Symbol or alias
            start_date: Start date
            end_date: End date
            interval: Data interval
        
        Returns:
            Series with close prices
        """
        df = self.fetch_ohlcv(symbol, start_date, end_date, interval)
        
        if df is not None and 'close' in df.columns:
            close = df['close']
            close.name = symbol
            return close
        return None
    
    def fetch_multiple(self,
                       symbols: List[str],
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       interval: str = '1d',
                       column: str = 'close') -> pd.DataFrame:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of symbols or aliases
            start_date: Start date
            end_date: End date
            interval: Data interval
            column: Which column to extract ('close', 'open', etc.)
        
        Returns:
            DataFrame with symbols as columns
        """
        data = {}
        
        for symbol in symbols:
            df = self.fetch_ohlcv(symbol, start_date, end_date, interval)
            if df is not None and column in df.columns:
                data[symbol] = df[column]
        
        if not data:
            error("No data fetched for any symbol")
            return pd.DataFrame()
        
        result = pd.DataFrame(data)
        result = result.sort_index().ffill()
        
        return result
    
    def fetch_cross_asset_data(self,
                               start_date: Optional[str] = None,
                               end_date: Optional[str] = None,
                               interval: str = '1d') -> pd.DataFrame:
        """
        Fetch common cross-asset data for correlation analysis.
        
        Fetches: DXY, SPX, VIX, GOLD, OIL
        
        Args:
            start_date: Start date
            end_date: End date
            interval: Data interval
        
        Returns:
            DataFrame with cross-asset close prices
        """
        core_assets = ['DXY', 'SPX', 'VIX', 'GOLD', 'OIL']
        return self.fetch_multiple(core_assets, start_date, end_date, interval)
    
    def compute_returns(self,
                        symbol: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        interval: str = '1d',
                        method: str = 'simple') -> Optional[pd.Series]:
        """
        Fetch symbol and compute returns.
        
        Args:
            symbol: Symbol or alias
            start_date: Start date
            end_date: End date
            interval: Data interval
            method: 'simple' or 'log'
        
        Returns:
            Series with returns
        """
        close = self.fetch_close(symbol, start_date, end_date, interval)
        
        if close is None:
            return None
        
        if method == 'log':
            returns = np.log(close / close.shift(1))
        else:
            returns = close.pct_change()
        
        returns.name = f"{symbol}_returns"
        return returns
    
    def test_connection(self) -> bool:
        """
        Test if Yahoo Finance is working.
        
        Returns:
            True if at least one test ticker works
        """
        test_tickers = ['SPY', 'AAPL', 'GLD']
        
        for ticker in test_tickers:
            try:
                info(f"Testing ticker: {ticker}")
                ticker_obj = self._yf.Ticker(ticker)
                data = ticker_obj.history(period='5d')
                if len(data) > 0:
                    success(f"✓ {ticker} works - {len(data)} rows")
                    return True
            except Exception as e:
                error(f"✗ {ticker} failed: {e}")
        
        return False


def get_yahoo_client() -> YahooClient:
    """Factory function to get Yahoo Finance client."""
    return YahooClient()


# Convenience functions
def fetch_dxy(start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              interval: str = '1d') -> Optional[pd.Series]:
    """Fetch US Dollar Index (or proxy)."""
    client = get_yahoo_client()
    return client.fetch_close('DXY', start_date, end_date, interval)


def fetch_spx(start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              interval: str = '1d') -> Optional[pd.Series]:
    """Fetch S&P 500 (SPY ETF)."""
    client = get_yahoo_client()
    return client.fetch_close('SPX', start_date, end_date, interval)


def fetch_vix(start_date: Optional[str] = None,
              end_date: Optional[str] = None,
              interval: str = '1d') -> Optional[pd.Series]:
    """Fetch VIX (VIXY ETF)."""
    client = get_yahoo_client()
    return client.fetch_close('VIX', start_date, end_date, interval)


def fetch_gold(start_date: Optional[str] = None,
               end_date: Optional[str] = None,
               interval: str = '1d') -> Optional[pd.Series]:
    """Fetch Gold (GLD ETF)."""
    client = get_yahoo_client()
    return client.fetch_close('GOLD', start_date, end_date, interval)