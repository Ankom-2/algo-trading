"""
Enhanced Data Source Manager
Handles any valid ticker symbol with multiple data sources and validation
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any, Tuple
import requests
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time

class EnhancedDataSource:
    """
    Advanced data source manager that can handle any valid ticker symbol
    with multiple data providers and intelligent fallback mechanisms
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Cache for validated symbols and prices
        self.symbol_cache = {}
        self.price_cache = {}
        self.cache_duration = self.config.get('cache_duration', 300)  # 5 minutes
        
        # Data providers configuration
        self.primary_provider = 'yfinance'
        self.fallback_providers = ['yfinance']  # Can be extended
        
        # Rate limiting
        self.last_request_time = {}
        self.request_delay = 0.1  # 100ms between requests
        
        # Supported market suffixes
        self.market_suffixes = {
            # Indian Markets
            'NSE': '.NS',  # National Stock Exchange
            'BSE': '.BO',  # Bombay Stock Exchange
            
            # US Markets  
            'NASDAQ': '',
            'NYSE': '',
            
            # Other Major Markets
            'LSE': '.L',   # London Stock Exchange
            'TSE': '.T',   # Tokyo Stock Exchange
            'HKEX': '.HK', # Hong Kong Exchange
            'ASX': '.AX',  # Australian Securities Exchange
            'TSX': '.TO',  # Toronto Stock Exchange
            'FRA': '.F',   # Frankfurt Stock Exchange
            
            # Crypto
            'CRYPTO': '-USD'  # Crypto pairs
        }
        
        self.logger.info("Enhanced Data Source Manager initialized")
    
    def validate_and_normalize_symbol(self, symbol: str, market_hint: str = None) -> Dict[str, Any]:
        """
        Validate and normalize a ticker symbol
        
        Args:
            symbol: Raw ticker symbol input
            market_hint: Optional hint about which market (NSE, NYSE, etc.)
            
        Returns:
            Dictionary with validation results and normalized symbol
        """
        try:
            symbol = symbol.strip().upper()
            
            # Check cache first
            cache_key = f"{symbol}_{market_hint}"
            if cache_key in self.symbol_cache:
                cached_result, cache_time = self.symbol_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < self.cache_duration:
                    return cached_result
            
            # Try different symbol variations based on market hint or common patterns
            symbol_variations = self._generate_symbol_variations(symbol, market_hint)
            
            # Test each variation
            for variation in symbol_variations:
                validation_result = self._test_symbol_validity(variation)
                if validation_result['valid']:
                    # Cache the successful result
                    self.symbol_cache[cache_key] = (validation_result, datetime.now())
                    return validation_result
            
            # If no variation worked
            return {
                'valid': False,
                'original_symbol': symbol,
                'error': 'Symbol not found in any supported market',
                'tried_variations': symbol_variations
            }
            
        except Exception as e:
            self.logger.error(f"Error validating symbol {symbol}: {str(e)}")
            return {
                'valid': False,
                'original_symbol': symbol,
                'error': str(e)
            }
    
    def _generate_symbol_variations(self, symbol: str, market_hint: str = None) -> List[str]:
        """Generate possible variations of a symbol based on market patterns"""
        variations = [symbol]  # Always try the original first
        
        # If market hint is provided, try that suffix first
        if market_hint and market_hint.upper() in self.market_suffixes:
            suffix = self.market_suffixes[market_hint.upper()]
            if suffix and not symbol.endswith(suffix):
                variations.insert(0, symbol + suffix)
        
        # Common patterns for Indian stocks
        if not any(symbol.endswith(suffix) for suffix in ['.NS', '.BO']):
            variations.extend([symbol + '.NS', symbol + '.BO'])
        
        # Crypto patterns
        if not symbol.endswith('-USD') and len(symbol) <= 5:
            variations.append(symbol + '-USD')
        
        # Common US market patterns (no suffix needed, but try anyway)
        # This handles cases where user might add unnecessary suffixes
        if symbol.endswith('.US'):
            variations.append(symbol[:-3])
        
        # European markets
        if not any(symbol.endswith(suffix) for suffix in ['.L', '.F', '.PA']):
            variations.extend([symbol + '.L', symbol + '.F'])
        
        return variations
    
    def _test_symbol_validity(self, symbol: str) -> Dict[str, Any]:
        """Test if a symbol is valid using yfinance"""
        try:
            # Rate limiting
            provider_key = f"yfinance_{symbol}"
            if provider_key in self.last_request_time:
                time_since_last = time.time() - self.last_request_time[provider_key]
                if time_since_last < self.request_delay:
                    time.sleep(self.request_delay - time_since_last)
            
            self.last_request_time[provider_key] = time.time()
            
            ticker = yf.Ticker(symbol)
            
            # Try to get basic info and recent data
            info = ticker.info
            hist = ticker.history(period='5d', timeout=10)
            
            # Check if we got valid data
            if hist.empty and not info:
                return {'valid': False, 'error': 'No data available'}
            
            if hist.empty and info.get('regularMarketPrice') is None:
                return {'valid': False, 'error': 'No price data available'}
            
            # Extract available information
            current_price = None
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
            elif info.get('regularMarketPrice'):
                current_price = info['regularMarketPrice']
            elif info.get('previousClose'):
                current_price = info['previousClose']
            
            result = {
                'valid': True,
                'normalized_symbol': symbol,
                'name': info.get('longName', info.get('shortName', symbol)),
                'price': float(current_price) if current_price else None,
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown'),
                'market': self._identify_market(symbol, info),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'data_source': 'yfinance',
                'last_updated': datetime.now()
            }
            
            return result
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _identify_market(self, symbol: str, info: Dict) -> str:
        """Identify which market/region a symbol belongs to"""
        exchange = info.get('exchange', '').upper()
        
        # Indian markets
        if symbol.endswith('.NS') or 'NSI' in exchange or 'NSE' in exchange:
            return 'India (NSE)'
        if symbol.endswith('.BO') or 'BSE' in exchange:
            return 'India (BSE)'
        
        # US markets
        if exchange in ['NASDAQ', 'NYSE', 'AMEX'] or exchange.startswith('NMS'):
            return f'US ({exchange})'
        
        # Crypto
        if symbol.endswith('-USD') or 'crypto' in exchange.lower():
            return 'Cryptocurrency'
        
        # European markets
        if symbol.endswith('.L') or 'LSE' in exchange:
            return 'UK (LSE)'
        if symbol.endswith('.F') or 'FRA' in exchange:
            return 'Germany (Frankfurt)'
        
        # Other markets
        if symbol.endswith('.T'):
            return 'Japan (TSE)'
        if symbol.endswith('.HK'):
            return 'Hong Kong'
        if symbol.endswith('.AX'):
            return 'Australia (ASX)'
        if symbol.endswith('.TO'):
            return 'Canada (TSX)'
        
        return info.get('exchange', 'Unknown')
    
    def get_real_time_data(self, symbol: str, period: str = '1d') -> Dict[str, Any]:
        """
        Get real-time or latest available data for a symbol
        
        Args:
            symbol: Validated and normalized symbol
            period: Data period (1d, 5d, 1mo, etc.)
            
        Returns:
            Dictionary with price data and metadata
        """
        try:
            # Check cache first
            cache_key = f"price_{symbol}_{period}"
            if cache_key in self.price_cache:
                cached_data, cache_time = self.price_cache[cache_key]
                # Use shorter cache for real-time data (1 minute)
                if (datetime.now() - cache_time).total_seconds() < 60:
                    return cached_data
            
            ticker = yf.Ticker(symbol)
            
            # Get historical data for the period
            hist = ticker.history(period=period)
            info = ticker.info
            
            if hist.empty:
                return {'error': 'No price data available'}
            
            # Calculate metrics
            latest_data = hist.iloc[-1]
            prev_data = hist.iloc[-2] if len(hist) > 1 else latest_data
            
            price_change = latest_data['Close'] - prev_data['Close']
            price_change_pct = (price_change / prev_data['Close'] * 100) if prev_data['Close'] != 0 else 0
            
            # Volume analysis
            avg_volume = hist['Volume'].mean() if len(hist) > 1 else latest_data['Volume']
            volume_ratio = latest_data['Volume'] / avg_volume if avg_volume > 0 else 1
            
            # Volatility (if enough data)
            volatility = None
            if len(hist) >= 20:
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            result = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price': round(float(latest_data['Close']), 4),
                'open': round(float(latest_data['Open']), 4),
                'high': round(float(latest_data['High']), 4),
                'low': round(float(latest_data['Low']), 4),
                'volume': int(latest_data['Volume']),
                'change': round(price_change, 4),
                'change_percent': round(price_change_pct, 2),
                'volume_ratio': round(volume_ratio, 2),
                'volatility': round(volatility, 2) if volatility else None,
                'period_high': round(hist['High'].max(), 4),
                'period_low': round(hist['Low'].min(), 4),
                'currency': info.get('currency', 'USD'),
                'market_state': info.get('marketState', 'Unknown'),
                'data_source': 'yfinance'
            }
            
            # Cache the result
            self.price_cache[cache_key] = (result, datetime.now())
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting real-time data for {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def get_historical_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
        """
        Get historical data for backtesting and analysis
        
        Args:
            symbol: Validated symbol
            period: Period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                self.logger.warning(f"No historical data available for {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            hist.columns = [col.lower() for col in hist.columns]
            hist['symbol'] = symbol
            
            # Add basic technical indicators
            if len(hist) >= 20:
                hist = self._add_technical_indicators(hist)
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to the dataframe"""
        try:
            # Simple Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Exponential Moving Averages  
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            return df
    
    def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for symbols matching a query
        This is a basic implementation - in production, you'd use a proper symbol search API
        """
        results = []
        
        # Try exact match first
        validation = self.validate_and_normalize_symbol(query)
        if validation['valid']:
            results.append(validation)
        
        # Try common variations and popular symbols
        popular_symbols = {
            # US Tech
            'AAPL': 'Apple Inc.',
            'GOOGL': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc.',
            
            # Indian Stocks
            'RELIANCE.NS': 'Reliance Industries Limited',
            'TCS.NS': 'Tata Consultancy Services Limited',
            'HDFCBANK.NS': 'HDFC Bank Limited',
            'INFY.NS': 'Infosys Limited',
            'HINDUNILVR.NS': 'Hindustan Unilever Limited',
            'ICICIBANK.NS': 'ICICI Bank Limited',
            'ITC.NS': 'ITC Limited',
            
            # Crypto
            'BTC-USD': 'Bitcoin USD',
            'ETH-USD': 'Ethereum USD',
            'ADA-USD': 'Cardano USD',
            'DOT-USD': 'Polkadot USD',
        }
        
        # Search for partial matches
        query_upper = query.upper()
        for symbol, name in popular_symbols.items():
            if query_upper in symbol or query_upper in name.upper():
                if len(results) < limit:
                    validation = self.validate_and_normalize_symbol(symbol)
                    if validation['valid'] and validation not in results:
                        results.append(validation)
        
        return results[:limit]

# Example usage and testing functions
def test_enhanced_data_source():
    """Test the enhanced data source with various symbols"""
    print("🧪 Testing Enhanced Data Source Manager")
    print("=" * 50)
    
    data_source = EnhancedDataSource()
    
    # Test cases
    test_symbols = [
        'AAPL',           # US stock
        'RELIANCE',       # Indian stock (should auto-detect .NS)
        'TCS.NS',         # Indian stock with suffix
        'BTC-USD',        # Cryptocurrency
        'GOOGL',          # US tech stock
        'TSLA',           # US auto stock
        'INVALID123',     # Invalid symbol
        'HDFCBANK.NS',    # Indian bank
    ]
    
    for symbol in test_symbols:
        print(f"\n📊 Testing symbol: {symbol}")
        
        # Test validation
        validation = data_source.validate_and_normalize_symbol(symbol)
        print(f"  ✅ Valid: {validation['valid']}")
        
        if validation['valid']:
            print(f"  🏷️  Name: {validation['name']}")
            print(f"  💰 Price: ${validation['price']}")
            print(f"  🏢 Exchange: {validation['exchange']}")
            print(f"  🌍 Market: {validation['market']}")
            
            # Test real-time data
            rt_data = data_source.get_real_time_data(validation['normalized_symbol'])
            if 'error' not in rt_data:
                print(f"  📈 Change: {rt_data['change_percent']:+.2f}%")
                print(f"  📊 Volume: {rt_data['volume']:,}")
        else:
            print(f"  ❌ Error: {validation['error']}")


if __name__ == '__main__':
    test_enhanced_data_source()
