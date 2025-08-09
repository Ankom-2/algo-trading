"""
Advanced Data Loader for High-Performance Algorithmic Trading
Handles multiple data sources with caching, validation, and preprocessing
"""
import pandas as pd
import numpy as np
import yfinance as yf
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import os
import pickle
import warnings
warnings.filterwarnings('ignore')


class AdvancedDataLoader:
    """
    Sophisticated data loader with multiple sources and advanced features
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data paths
        self.raw_data_path = config.get('raw_path', 'data/raw/')
        self.processed_data_path = config.get('processed_path', 'data/processed/')
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Data source configuration
        self.symbols = config.get('symbols', ['AAPL', 'GOOGL', 'MSFT'])
        self.timeframes = config.get('timeframes', ['1d'])
        self.lookback_days = config.get('lookback_days', 252)
        
        # Caching
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_duration = config.get('cache_duration', 3600)  # 1 hour
        self.data_cache = {}
        
        # Data quality parameters
        self.min_data_points = config.get('min_data_points', 100)
        self.max_missing_pct = config.get('max_missing_pct', 0.05)  # 5%
        
        self.logger.info("Advanced Data Loader initialized")
    
    async def load_historical_data(self, symbols: Optional[List[str]] = None, 
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  timeframe: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Load historical data for multiple symbols asynchronously
        
        Args:
            symbols: List of symbols to load
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            timeframe: Data timeframe ('1d', '1h', etc.)
        
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        
        if symbols is None:
            symbols = self.symbols
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"Loading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Load data concurrently
        tasks = []
        for symbol in symbols:
            task = self._load_symbol_data(symbol, start_date, end_date, timeframe)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        data_dict = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                self.logger.error(f"Failed to load data for {symbol}: {str(result)}")
                continue
            
            if result is not None and len(result) > 0:
                data_dict[symbol] = result
            else:
                self.logger.warning(f"No data returned for {symbol}")
        
        self.logger.info(f"Successfully loaded data for {len(data_dict)} symbols")
        return data_dict
    
    async def _load_symbol_data(self, symbol: str, start_date: str, 
                               end_date: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data for a single symbol"""
        
        # Check cache first
        cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        if self.cache_enabled and cache_key in self.data_cache:
            cached_data, cache_time = self.data_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                return cached_data
        
        try:
            # Try loading from file first
            file_data = self._load_from_file(symbol, start_date, end_date, timeframe)
            if file_data is not None:
                return file_data
            
            # Download from API
            data = await self._download_from_api(symbol, start_date, end_date, timeframe)
            
            if data is None or len(data) == 0:
                return None
            
            # Validate and clean data
            cleaned_data = self._validate_and_clean_data(data, symbol)
            
            if cleaned_data is None:
                return None
            
            # Add technical indicators
            enhanced_data = self._add_basic_indicators(cleaned_data)
            
            # Save to file
            self._save_to_file(enhanced_data, symbol, timeframe)
            
            # Cache the data
            if self.cache_enabled:
                self.data_cache[cache_key] = (enhanced_data, datetime.now())
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {str(e)}")
            return None
    
    def _load_from_file(self, symbol: str, start_date: str, end_date: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load data from saved file"""
        filename = f"{symbol}_{timeframe}.pkl"
        filepath = os.path.join(self.processed_data_path, filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Check if data covers the requested date range
            data_start = data.index[0].strftime('%Y-%m-%d')
            data_end = data.index[-1].strftime('%Y-%m-%d')
            
            if data_start <= start_date and data_end >= end_date:
                # Filter to requested date range
                mask = (data.index >= start_date) & (data.index <= end_date)
                return data.loc[mask]
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached data for {symbol}: {str(e)}")
        
        return None
    
    async def _download_from_api(self, symbol: str, start_date: str, 
                                end_date: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Download data from financial API"""
        
        try:
            # Use yfinance as primary source
            ticker = yf.Ticker(symbol)
            
            # Map timeframe
            interval_map = {
                '1d': '1d',
                '1h': '1h',
                '5m': '5m',
                '15m': '15m',
                '1m': '1m'
            }
            
            interval = interval_map.get(timeframe, '1d')
            
            # Download data
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                return None
            
            # Standardize column names
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add symbol column
            data['symbol'] = symbol
            
            return data
            
        except Exception as e:
            self.logger.error(f"API download failed for {symbol}: {str(e)}")
            return None
    
    def _validate_and_clean_data(self, data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """Validate and clean the loaded data"""
        
        if data is None or len(data) == 0:
            self.logger.warning(f"No data to validate for {symbol}")
            return None
        
        # Check minimum data points
        if len(data) < self.min_data_points:
            self.logger.warning(f"Insufficient data for {symbol}: {len(data)} < {self.min_data_points}")
            return None
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing columns for {symbol}: {missing_columns}")
            return None
        
        # Check for missing values
        missing_pct = data[required_columns].isnull().sum().sum() / (len(data) * len(required_columns))
        if missing_pct > self.max_missing_pct:
            self.logger.warning(f"Too many missing values for {symbol}: {missing_pct:.2%}")
        
        # Clean the data
        df = data.copy()
        
        # Remove rows with missing OHLC data
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Fill missing volume with 0
        df['volume'] = df['volume'].fillna(0)
        
        # Remove invalid price data
        price_cols = ['open', 'high', 'low', 'close']
        df = df[(df[price_cols] > 0).all(axis=1)]
        
        # Check high >= low, close between high and low
        df = df[df['high'] >= df['low']]
        df = df[(df['close'] >= df['low']) & (df['close'] <= df['high'])]
        df = df[(df['open'] >= df['low']) & (df['open'] <= df['high'])]
        
        # Remove extreme price movements (likely data errors)
        for col in price_cols:
            returns = df[col].pct_change()
            extreme_threshold = 0.5  # 50% single-day movement
            df = df[np.abs(returns) < extreme_threshold]
        
        # Sort by date
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        if len(df) < self.min_data_points:
            self.logger.warning(f"Insufficient data after cleaning for {symbol}: {len(df)}")
            return None
        
        self.logger.debug(f"Cleaned data for {symbol}: {len(df)} records")
        return df
    
    def _add_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators to the data"""
        
        df = data.copy()
        
        try:
            # Returns
            df['returns'] = df['close'].pct_change()
            
            # Moving averages
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=min(50, len(df))).mean()
            
            # Exponential moving averages
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = 20
            df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
            bb_std = df['close'].rolling(window=bb_period).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price volatility
            df['volatility'] = df['returns'].rolling(window=20).std()
            
            # Average True Range
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['atr'] = true_range.rolling(window=14).mean()
            
        except Exception as e:
            self.logger.warning(f"Failed to add some indicators: {str(e)}")
        
        return df
    
    def _save_to_file(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """Save data to file for future use"""
        filename = f"{symbol}_{timeframe}.pkl"
        filepath = os.path.join(self.processed_data_path, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save data for {symbol}: {str(e)}")
    
    async def load_real_time_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Load real-time data for symbols"""
        
        real_time_data = {}
        
        try:
            # Using yfinance for real-time data
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Get latest price data
                hist = ticker.history(period='1d', interval='1m')
                if not hist.empty:
                    latest = hist.iloc[-1]
                    
                    real_time_data[symbol] = {
                        'price': latest['Close'],
                        'volume': latest['Volume'],
                        'timestamp': hist.index[-1],
                        'bid': info.get('bid', latest['Close']),
                        'ask': info.get('ask', latest['Close']),
                        'bid_size': info.get('bidSize', 0),
                        'ask_size': info.get('askSize', 0),
                        'day_high': latest['High'],
                        'day_low': latest['Low'],
                        'prev_close': info.get('previousClose', latest['Close'])
                    }
        
        except Exception as e:
            self.logger.error(f"Failed to load real-time data: {str(e)}")
        
        return real_time_data
    
    def get_data_quality_report(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report for a symbol"""
        
        if data is None or len(data) == 0:
            return {'status': 'no_data'}
        
        # Basic statistics
        total_records = len(data)
        date_range = (data.index[0], data.index[-1])
        
        # Missing data analysis
        missing_data = data.isnull().sum()
        missing_pct = (missing_data / total_records) * 100
        
        # Price data validation
        price_cols = ['open', 'high', 'low', 'close']
        valid_ohlc = ((data['high'] >= data['low']) & 
                     (data['close'] >= data['low']) & 
                     (data['close'] <= data['high']) & 
                     (data['open'] >= data['low']) & 
                     (data['open'] <= data['high'])).sum()
        
        # Volume analysis
        zero_volume_days = (data['volume'] == 0).sum()
        
        # Price movement analysis
        returns = data['close'].pct_change().dropna()
        extreme_moves = (np.abs(returns) > 0.2).sum()  # >20% moves
        
        return {
            'symbol': symbol,
            'total_records': total_records,
            'date_range': date_range,
            'missing_data': missing_data.to_dict(),
            'missing_percentages': missing_pct.to_dict(),
            'valid_ohlc_records': valid_ohlc,
            'zero_volume_days': zero_volume_days,
            'extreme_price_moves': extreme_moves,
            'data_quality_score': self._calculate_quality_score(
                total_records, missing_pct.mean(), valid_ohlc/total_records, extreme_moves/total_records
            )
        }
    
    def _calculate_quality_score(self, total_records: int, avg_missing_pct: float, 
                                valid_ohlc_ratio: float, extreme_move_ratio: float) -> float:
        """Calculate overall data quality score (0-100)"""
        
        score = 100.0
        
        # Penalize for insufficient data
        if total_records < self.min_data_points:
            score -= 30
        
        # Penalize for missing data
        score -= avg_missing_pct * 2
        
        # Penalize for invalid OHLC
        score -= (1 - valid_ohlc_ratio) * 50
        
        # Penalize for too many extreme moves
        if extreme_move_ratio > 0.05:  # >5% extreme moves
            score -= 20
        
        return max(0, min(100, score))
    
    def clear_cache(self):
        """Clear data cache"""
        self.data_cache.clear()
        self.logger.info("Data cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cached_items': len(self.data_cache),
            'cache_enabled': self.cache_enabled,
            'cache_duration': self.cache_duration
        }
