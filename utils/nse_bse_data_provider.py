"""
Enhanced NSE/BSE Data Provider
Provides comprehensive data for Indian stock markets
"""

import yfinance as yf
import pandas as pd
import requests
from typing import Dict, List, Optional
import json
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class NSEBSESymbols:
    """Comprehensive NSE/BSE symbol database"""
    
    @staticmethod
    def get_nse_top_stocks() -> List[Dict]:
        """Get top NSE stocks with proper information"""
        return [
            # Nifty 50 stocks
            {"symbol": "RELIANCE.NS", "name": "Reliance Industries Ltd", "sector": "Oil & Gas"},
            {"symbol": "TCS.NS", "name": "Tata Consultancy Services Ltd", "sector": "IT Services"},
            {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Banking"},
            {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "sector": "Banking"},
            {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever Ltd", "sector": "FMCG"},
            {"symbol": "INFY.NS", "name": "Infosys Ltd", "sector": "IT Services"},
            {"symbol": "ITC.NS", "name": "ITC Ltd", "sector": "FMCG"},
            {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Banking"},
            {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel Ltd", "sector": "Telecom"},
            {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank Ltd", "sector": "Banking"},
            {"symbol": "LT.NS", "name": "Larsen & Toubro Ltd", "sector": "Infrastructure"},
            {"symbol": "HCLTECH.NS", "name": "HCL Technologies Ltd", "sector": "IT Services"},
            {"symbol": "ASIANPAINT.NS", "name": "Asian Paints Ltd", "sector": "Paints"},
            {"symbol": "MARUTI.NS", "name": "Maruti Suzuki India Ltd", "sector": "Automobile"},
            {"symbol": "AXISBANK.NS", "name": "Axis Bank Ltd", "sector": "Banking"},
            {"symbol": "TITAN.NS", "name": "Titan Company Ltd", "sector": "Consumer Goods"},
            {"symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical Industries Ltd", "sector": "Pharma"},
            {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement Ltd", "sector": "Cement"},
            {"symbol": "WIPRO.NS", "name": "Wipro Ltd", "sector": "IT Services"},
            {"symbol": "NESTLEIND.NS", "name": "Nestle India Ltd", "sector": "FMCG"},
            {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance Ltd", "sector": "NBFC"},
            {"symbol": "POWERGRID.NS", "name": "Power Grid Corporation of India Ltd", "sector": "Power"},
            {"symbol": "NTPC.NS", "name": "NTPC Ltd", "sector": "Power"},
            {"symbol": "TECHM.NS", "name": "Tech Mahindra Ltd", "sector": "IT Services"},
            {"symbol": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Automobile"},
            {"symbol": "ADANIPORTS.NS", "name": "Adani Ports and Special Economic Zone Ltd", "sector": "Infrastructure"},
            {"symbol": "ONGC.NS", "name": "Oil and Natural Gas Corporation Ltd", "sector": "Oil & Gas"},
            {"symbol": "COALINDIA.NS", "name": "Coal India Ltd", "sector": "Mining"},
            {"symbol": "TATASTEEL.NS", "name": "Tata Steel Ltd", "sector": "Steel"},
            {"symbol": "GRASIM.NS", "name": "Grasim Industries Ltd", "sector": "Cement"},
            {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv Ltd", "sector": "Financial Services"},
            {"symbol": "M&M.NS", "name": "Mahindra & Mahindra Ltd", "sector": "Automobile"},
            {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories Ltd", "sector": "Pharma"},
            {"symbol": "INDUSINDBK.NS", "name": "IndusInd Bank Ltd", "sector": "Banking"},
            {"symbol": "CIPLA.NS", "name": "Cipla Ltd", "sector": "Pharma"},
            {"symbol": "EICHERMOT.NS", "name": "Eicher Motors Ltd", "sector": "Automobile"},
            {"symbol": "BRITANNIA.NS", "name": "Britannia Industries Ltd", "sector": "FMCG"},
            {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories Ltd", "sector": "Pharma"},
            {"symbol": "HEROMOTOCO.NS", "name": "Hero MotoCorp Ltd", "sector": "Automobile"},
            {"symbol": "JSWSTEEL.NS", "name": "JSW Steel Ltd", "sector": "Steel"},
            
            # Additional popular stocks
            {"symbol": "ADANIENT.NS", "name": "Adani Enterprises Ltd", "sector": "Infrastructure"},
            {"symbol": "HINDALCO.NS", "name": "Hindalco Industries Ltd", "sector": "Metals"},
            {"symbol": "BAJAJ-AUTO.NS", "name": "Bajaj Auto Ltd", "sector": "Automobile"},
            {"symbol": "SBILIFE.NS", "name": "SBI Life Insurance Company Ltd", "sector": "Insurance"},
            {"symbol": "SHREECEM.NS", "name": "Shree Cement Ltd", "sector": "Cement"},
            {"symbol": "HDFCLIFE.NS", "name": "HDFC Life Insurance Company Ltd", "sector": "Insurance"},
            {"symbol": "BPCL.NS", "name": "Bharat Petroleum Corporation Ltd", "sector": "Oil & Gas"},
            {"symbol": "TATACONSUMER.NS", "name": "Tata Consumer Products Ltd", "sector": "FMCG"},
            {"symbol": "APOLLOHOSP.NS", "name": "Apollo Hospitals Enterprise Ltd", "sector": "Healthcare"},
            {"symbol": "TRENT.NS", "name": "Trent Ltd", "sector": "Retail"}
        ]
    
    @staticmethod
    def get_bse_top_stocks() -> List[Dict]:
        """Get top BSE stocks"""
        return [
            {"symbol": "RELIANCE.BO", "name": "Reliance Industries Ltd", "sector": "Oil & Gas"},
            {"symbol": "TCS.BO", "name": "Tata Consultancy Services Ltd", "sector": "IT Services"},
            {"symbol": "HDFCBANK.BO", "name": "HDFC Bank Ltd", "sector": "Banking"},
            {"symbol": "ICICIBANK.BO", "name": "ICICI Bank Ltd", "sector": "Banking"},
            {"symbol": "HINDUNILVR.BO", "name": "Hindustan Unilever Ltd", "sector": "FMCG"},
            {"symbol": "INFY.BO", "name": "Infosys Ltd", "sector": "IT Services"},
            {"symbol": "ITC.BO", "name": "ITC Ltd", "sector": "FMCG"},
            {"symbol": "SBIN.BO", "name": "State Bank of India", "sector": "Banking"},
            {"symbol": "BHARTIARTL.BO", "name": "Bharti Airtel Ltd", "sector": "Telecom"},
            {"symbol": "KOTAKBANK.BO", "name": "Kotak Mahindra Bank Ltd", "sector": "Banking"}
        ]
    
    @classmethod
    def search_symbol(cls, query: str, exchange: str = 'NSE') -> List[Dict]:
        """Search for symbols by name or symbol"""
        stocks = cls.get_nse_top_stocks() if exchange == 'NSE' else cls.get_bse_top_stocks()
        query = query.upper()
        
        results = []
        for stock in stocks:
            symbol_match = query in stock['symbol'].upper()
            name_match = query in stock['name'].upper()
            
            if symbol_match or name_match:
                results.append(stock)
        
        return results[:10]

class EnhancedNSEBSEData:
    """Enhanced data provider for NSE/BSE with better error handling and caching"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
    def get_stock_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Get historical stock data with caching"""
        cache_key = f"{symbol}_{period}"
        
        # Check cache
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                return data
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            # Cache the data
            self.cache[cache_key] = (datetime.now(), data)
            
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote data"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current data
            hist = ticker.history(period='5d')
            if hist.empty:
                return {}
            
            info = ticker.info
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            # Calculate change
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
            
            return {
                'symbol': symbol,
                'current_price': round(float(current_price), 2),
                'previous_close': round(float(prev_close), 2),
                'change': round(float(change), 2),
                'change_percent': round(float(change_percent), 2),
                'volume': int(hist['Volume'].iloc[-1]),
                'high': round(float(hist['High'].iloc[-1]), 2),
                'low': round(float(hist['Low'].iloc[-1]), 2),
                'open': round(float(hist['Open'].iloc[-1]), 2)
            }
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return {}
    
    def get_comprehensive_data(self, symbol: str) -> Dict:
        """Get comprehensive stock data including fundamentals"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='2y')
            
            if hist.empty:
                return {}
            
            # Current price data
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            # 52-week data
            year_data = hist.tail(252)  # ~1 year of trading days
            week_52_high = year_data['High'].max()
            week_52_low = year_data['Low'].min()
            
            # Today's data
            today_high = hist['High'].iloc[-1]
            today_low = hist['Low'].iloc[-1]
            today_volume = hist['Volume'].iloc[-1]
            
            # Calculate technical indicators
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
            
            # Price changes
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', symbol.replace('.NS', '').replace('.BO', '')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'current_price': round(float(current_price), 2),
                'previous_close': round(float(prev_close), 2),
                'price_change': round(float(change), 2),
                'price_change_percent': round(float(change_percent), 2),
                'today_high': round(float(today_high), 2),
                'today_low': round(float(today_low), 2),
                'week_52_high': round(float(week_52_high), 2),
                'week_52_low': round(float(week_52_low), 2),
                'volume': int(today_volume),
                'avg_volume': int(hist['Volume'].tail(30).mean()),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 'N/A',
                'book_value': round(info.get('bookValue', 0), 2) if info.get('bookValue') else 'N/A',
                'dividend_yield': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 'N/A',
                'sma_20': round(float(sma_20), 2) if not pd.isna(sma_20) else 'N/A',
                'sma_50': round(float(sma_50), 2) if not pd.isna(sma_50) else 'N/A',
                'currency': info.get('currency', 'INR'),
                'exchange': info.get('exchange', 'NSI' if '.NS' in symbol else 'BSE'),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error fetching comprehensive data for {symbol}: {e}")
            return {}
    
    def get_trending_analysis(self, symbols: List[str]) -> List[Dict]:
        """Get trending analysis for multiple symbols"""
        trending_data = []
        
        for symbol in symbols:
            try:
                data = self.get_real_time_quote(symbol)
                if data and abs(data.get('change_percent', 0)) > 0:
                    # Add trending score based on volume and price movement
                    data['trending_score'] = abs(data['change_percent']) * (data.get('volume', 0) / 1000000)
                    trending_data.append(data)
            except:
                continue
        
        # Sort by trending score
        trending_data.sort(key=lambda x: x.get('trending_score', 0), reverse=True)
        return trending_data[:20]  # Top 20 trending stocks
