# data_sources.py
# Abstract data source interface and implementations for FinMind and yfinance

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any

# Import data sources
try:
    from FinMind.data import FinMindApi
    FINMIND_AVAILABLE = True
except ImportError:
    FINMIND_AVAILABLE = False
    print("[DATA_SOURCES] FinMind not available. Install with: pip install FinMind")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("[DATA_SOURCES] yfinance not available. Install with: pip install yfinance")


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the display name of this data source."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a description of this data source."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this data source is available."""
        pass
    
    @abstractmethod
    def fetch_data(self, symbol: str, period: str = "6mo") -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a symbol.
        
        Args:
            symbol: The stock symbol
            period: Time period ("1mo", "6mo", "1y", "max")
            
        Returns:
            Pandas DataFrame with columns [Open, High, Low, Close, Volume] or None if failed
        """
        pass
    
    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get basic information about a symbol.
        
        Returns:
            Dictionary with symbol metadata or None if not found
        """
        pass
    
    @abstractmethod
    def supports_market(self, symbol: str) -> bool:
        """Check if this data source supports the given symbol/market."""
        pass


class FinMindDataSource(DataSource):
    """FinMind data source for Taiwan stocks."""
    
    def __init__(self):
        self.api = None
        self._initialize_api()
    
    def _initialize_api(self):
        """Initialize the FinMind API."""
        if not FINMIND_AVAILABLE:
            return
            
        try:
            self.api = FinMindApi()
            finmind_token = os.environ.get('FINMIND_TOKEN')
            if finmind_token:
                self.api.login_by_token(api_token=finmind_token)
                print("[FINMIND] Logged in successfully.")
            else:
                print("[FINMIND_WARN] FINMIND_TOKEN not found. Usage may be limited.")
        except Exception as e:
            print(f"[FINMIND_ERROR] Failed to initialize API: {e}")
            self.api = None
    
    def get_name(self) -> str:
        return "FinMind (Taiwan)"
    
    def get_description(self) -> str:
        return "Taiwan stock market data via FinMind API"
    
    def is_available(self) -> bool:
        return FINMIND_AVAILABLE and self.api is not None
    
    def supports_market(self, symbol: str) -> bool:
        """FinMind supports Taiwan stocks (.TW, .TWO) and some international stocks."""
        return symbol.endswith(('.TW', '.TWO'))
    
    def fetch_data(self, symbol: str, period: str = "6mo") -> Optional[Tuple[np.ndarray, pd.DatetimeIndex]]:
        """Fetch data from FinMind."""
        if not self.is_available():
            return None
            
        try:
            # Handle Taiwan stock symbols
            if symbol.endswith('.TW'):
                stock_id = symbol.replace('.TW', '')
                dataset = "TaiwanStockPrice"
            elif symbol.endswith('.TWO'):
                stock_id = symbol.replace('.TWO', '')
                dataset = "TaiwanStockPrice"
            else:
                # For other symbols, try to use as-is
                stock_id = symbol
                dataset = "TaiwanStockPrice"
            
            # Calculate date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            days_map = {"1y": 365, "5y": 365*5, "6mo": 180, "1mo": 30, "max": 365 * 20}
            days = days_map.get(period, 180)
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Fetch data
            df = self.api.get_data(
                dataset=dataset,
                data_id=stock_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty or 'date' not in df.columns:
                print(f"  [FINMIND_WARN] No/incomplete data for '{symbol}'.")
                return None
            
            # Standardize column names
            df.rename(columns={
                "open": "Open", "max": "High", "min": "Low", 
                "close": "Close", "Trading_Volume": "Volume"
            }, inplace=True)
            
            required_cols = {"Open", "High", "Low", "Close", "Volume"}
            if not required_cols.issubset(df.columns):
                print(f"  [FINMIND_WARN] Incomplete data for '{symbol}'. Missing required columns.")
                return None
            
            # Process data
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            df['Volume'] = df['Volume'] / 1000  # Convert to thousands
            
            return df
            
        except Exception as e:
            print(f"  [FINMIND_ERROR] Could not fetch data for '{symbol}'. Reason: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information from FinMind."""
        if not self.is_available():
            return None
            
        try:
            stock_id = symbol.replace('.TW', '').replace('.TWO', '')
            df_info = self.api.get_data(
                dataset="TaiwanStockInfo",
                data_id=stock_id
            )
            
            if df_info.empty:
                return None
                
            info = df_info.iloc[0]
            stock_name = info.get('stock_name', symbol)
            industry = info.get('industry_category', 'N/A')
            
            return {
                "symbol": symbol,
                "name": f"{stock_name} ({industry})",
                "theme": "FinMind",
                "focus": "Taiwan",
                "link": ""
            }
            
        except Exception as e:
            print(f"[FINMIND_ERROR] Failed to get info for {symbol}: {e}")
            return None


class YFinanceDataSource(DataSource):
    """yfinance data source for US and international stocks."""
    
    def __init__(self):
        self._cache = {}
    
    def get_name(self) -> str:
        return "Yahoo Finance (US/Global)"
    
    def get_description(self) -> str:
        return "US and global stock market data via Yahoo Finance"
    
    def is_available(self) -> bool:
        return YFINANCE_AVAILABLE
    
    def supports_market(self, symbol: str) -> bool:
        """yfinance supports most global markets, especially US stocks."""
        # yfinance works best with US stocks and major international symbols
        # Avoid Taiwan-specific symbols that are better handled by FinMind
        return not symbol.endswith(('.TW', '.TWO'))
    
    def fetch_data(self, symbol: str, period: str = "6mo") -> Optional[Tuple[np.ndarray, pd.DatetimeIndex]]:
        """Fetch data from Yahoo Finance."""
        if not self.is_available():
            return None
            
        try:
            # Map periods to yfinance format
            period_map = {
                "1mo": "1mo",
                "6mo": "6mo", 
                "1y": "1y",
                "max": "max"
            }
            yf_period = period_map.get(period, "6mo")
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=yf_period)
            
            if df.empty:
                print(f"  [YFINANCE_WARN] No data for '{symbol}'.")
                return None
            
            # Standardize column names
            df.rename(columns={
                "Open": "Open",
                "High": "High", 
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume"
            }, inplace=True)
            
            # Process data
            df.sort_index(inplace=True)
            df['Volume'] = df['Volume'] / 1000  # Convert to thousands
            
            return df
            
        except Exception as e:
            print(f"  [YFINANCE_ERROR] Could not fetch data for '{symbol}'. Reason: {e}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information from Yahoo Finance."""
        if not self.is_available():
            return None
            
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info:
                return None
            
            company_name = info.get('longName', info.get('shortName', symbol))
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            
            return {
                "symbol": symbol,
                "name": f"{company_name} ({sector}/{industry})",
                "theme": "YFinance",
                "focus": "US/Global",
                "link": ""
            }
            
        except Exception as e:
            print(f"[YFINANCE_ERROR] Failed to get info for {symbol}: {e}")
            return None


class DataSourceManager:
    """Manages multiple data sources and provides a unified interface."""
    
    def __init__(self):
        self.sources = {}
        self.default_source = None
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize available data sources."""
        # Add FinMind source
        finmind_source = FinMindDataSource()
        if finmind_source.is_available():
            self.sources["finmind"] = finmind_source
            self.default_source = "finmind"
        
        # Add yfinance source
        yfinance_source = YFinanceDataSource()
        if yfinance_source.is_available():
            self.sources["yfinance"] = yfinance_source
            if not self.default_source:
                self.default_source = "yfinance"
        
        print(f"[DATA_SOURCES] Initialized {len(self.sources)} data sources: {list(self.sources.keys())}")
        if self.default_source:
            print(f"[DATA_SOURCES] Default source: {self.default_source}")
    
    def get_available_sources(self) -> Dict[str, DataSource]:
        """Get all available data sources."""
        return self.sources.copy()
    
    def get_source(self, source_name: str) -> Optional[DataSource]:
        """Get a specific data source by name."""
        return self.sources.get(source_name)
    
    def get_default_source(self) -> Optional[DataSource]:
        """Get the default data source."""
        if self.default_source:
            return self.sources[self.default_source]
        return None
    
    def set_default_source(self, source_name: str) -> bool:
        """Set the default data source."""
        if source_name in self.sources:
            self.default_source = source_name
            return True
        return False
    
    def get_best_source_for_symbol(self, symbol: str) -> Optional[DataSource]:
        """Get the best data source for a given symbol."""
        # Check if any source explicitly supports this symbol
        for source in self.sources.values():
            if source.supports_market(symbol):
                return source
        
        # Fall back to default source
        return self.get_default_source()
    
    def fetch_data(self, symbol: str, period: str = "6mo", source_name: str = None) -> Optional[pd.DataFrame]:
        """Fetch data using the best available source or specified source."""
        if source_name:
            source = self.get_source(source_name)
            if source:
                return source.fetch_data(symbol, period)
            return None
        
        # Use best source for symbol
        source = self.get_best_source_for_symbol(symbol)
        if source:
            return source.fetch_data(symbol, period)
        
        return None
    
    def get_symbol_info(self, symbol: str, source_name: str = None) -> Optional[Dict[str, Any]]:
        """Get symbol info using the best available source or specified source."""
        if source_name:
            source = self.get_source(source_name)
            if source:
                return source.get_symbol_info(symbol)
            return None
        
        # Use best source for symbol
        source = self.get_best_source_for_symbol(symbol)
        if source:
            return source.get_symbol_info(symbol)
        
        return None


# Global instance
data_source_manager = DataSourceManager()