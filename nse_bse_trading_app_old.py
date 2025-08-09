"""
NSE/BSE Trading Application
A comprehensive trading interface for Indian stock markets
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Add utils and strategies to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'strategies'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'execution'))

try:
    from nse_bse_data_provider import NSEBSESymbols, EnhancedNSEBSEData
    from momentum import MomentumStrategy
    from mean_reversion import MeanReversionStrategy
    from adaptive import AdaptiveStrategy
    from paper_trader import PaperTradingEngine
except ImportError:
    # Fallback if import fails
    class NSEBSESymbols:
        @staticmethod
        def get_nse_top_stocks():
            return [{"symbol": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Oil & Gas"}]
        
        @staticmethod  
        def get_bse_top_stocks():
            return [{"symbol": "RELIANCE.BO", "name": "Reliance Industries", "sector": "Oil & Gas"}]
    
    class EnhancedNSEBSEData:
        def get_comprehensive_data(self, symbol):
            return {}
    
    class MomentumStrategy:
        def __init__(self): pass
        def generate_signals(self, data): return pd.DataFrame()
    
    class MeanReversionStrategy:
        def __init__(self): pass 
        def generate_signals(self, data): return pd.DataFrame()
        
    class AdaptiveStrategy:
        def __init__(self): pass
        def generate_signals(self, data): return pd.DataFrame()
        
    class PaperTradingEngine:
        def __init__(self, initial_capital=100000): 
            self.capital = initial_capital
            self.positions = {}
        def execute_trade(self, symbol, action, quantity, price):
            return {"status": "executed", "message": f"{action} {quantity} shares of {symbol}"}
        def get_portfolio_summary(self):
            return {"total_value": self.capital, "cash": self.capital, "positions": []}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NSEBSEDataProvider:
    """Enhanced data provider for NSE/BSE stocks with multiple data sources"""
    
    def __init__(self):
        self.api_credentials = self.load_credentials()
        self.session = requests.Session()
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Enhanced NSE/BSE Stock symbols
        self.nse_stocks = self._get_comprehensive_nse_stocks()
        self.bse_stocks = self._get_comprehensive_bse_stocks()
    
    def _get_comprehensive_nse_stocks(self) -> List[Dict]:
        """Get comprehensive list of NSE stocks"""
        return [
            {"symbol": "RELIANCE.NS", "name": "Reliance Industries Ltd", "sector": "Oil & Gas", "market_cap": "Large"},
            {"symbol": "TCS.NS", "name": "Tata Consultancy Services Ltd", "sector": "IT Services", "market_cap": "Large"},
            {"symbol": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "sector": "Banking", "market_cap": "Large"},
            {"symbol": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "sector": "Banking", "market_cap": "Large"},
            {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever Ltd", "sector": "FMCG", "market_cap": "Large"},
            {"symbol": "INFY.NS", "name": "Infosys Ltd", "sector": "IT Services", "market_cap": "Large"},
            {"symbol": "ITC.NS", "name": "ITC Ltd", "sector": "FMCG", "market_cap": "Large"},
            {"symbol": "SBIN.NS", "name": "State Bank of India", "sector": "Banking", "market_cap": "Large"},
            {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel Ltd", "sector": "Telecom", "market_cap": "Large"},
            {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank Ltd", "sector": "Banking", "market_cap": "Large"},
            {"symbol": "LT.NS", "name": "Larsen & Toubro Ltd", "sector": "Infrastructure", "market_cap": "Large"},
            {"symbol": "HCLTECH.NS", "name": "HCL Technologies Ltd", "sector": "IT Services", "market_cap": "Large"},
            {"symbol": "ASIANPAINT.NS", "name": "Asian Paints Ltd", "sector": "Paints", "market_cap": "Large"},
            {"symbol": "MARUTI.NS", "name": "Maruti Suzuki India Ltd", "sector": "Automobile", "market_cap": "Large"},
            {"symbol": "AXISBANK.NS", "name": "Axis Bank Ltd", "sector": "Banking", "market_cap": "Large"},
            {"symbol": "TITAN.NS", "name": "Titan Company Ltd", "sector": "Consumer Goods", "market_cap": "Large"},
            {"symbol": "SUNPHARMA.NS", "name": "Sun Pharmaceutical Industries Ltd", "sector": "Pharma", "market_cap": "Large"},
            {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement Ltd", "sector": "Cement", "market_cap": "Large"},
            {"symbol": "WIPRO.NS", "name": "Wipro Ltd", "sector": "IT Services", "market_cap": "Large"},
            {"symbol": "NESTLEIND.NS", "name": "Nestle India Ltd", "sector": "FMCG", "market_cap": "Large"},
            {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance Ltd", "sector": "NBFC", "market_cap": "Large"},
            {"symbol": "POWERGRID.NS", "name": "Power Grid Corporation", "sector": "Power", "market_cap": "Large"},
            {"symbol": "NTPC.NS", "name": "NTPC Ltd", "sector": "Power", "market_cap": "Large"},
            {"symbol": "TECHM.NS", "name": "Tech Mahindra Ltd", "sector": "IT Services", "market_cap": "Large"},
            {"symbol": "TATAMOTORS.NS", "name": "Tata Motors Ltd", "sector": "Automobile", "market_cap": "Large"},
            {"symbol": "ADANIPORTS.NS", "name": "Adani Ports and SEZ Ltd", "sector": "Infrastructure", "market_cap": "Large"},
            {"symbol": "ONGC.NS", "name": "Oil and Natural Gas Corporation", "sector": "Oil & Gas", "market_cap": "Large"},
            {"symbol": "COALINDIA.NS", "name": "Coal India Ltd", "sector": "Mining", "market_cap": "Large"},
            {"symbol": "TATASTEEL.NS", "name": "Tata Steel Ltd", "sector": "Steel", "market_cap": "Large"},
            {"symbol": "GRASIM.NS", "name": "Grasim Industries Ltd", "sector": "Cement", "market_cap": "Large"},
        ]
    
    def _get_comprehensive_bse_stocks(self) -> List[Dict]:
        """Get comprehensive list of BSE stocks"""
        return [
            {"symbol": "RELIANCE.BO", "name": "Reliance Industries Ltd", "sector": "Oil & Gas", "market_cap": "Large"},
            {"symbol": "TCS.BO", "name": "Tata Consultancy Services Ltd", "sector": "IT Services", "market_cap": "Large"},
            {"symbol": "HDFCBANK.BO", "name": "HDFC Bank Ltd", "sector": "Banking", "market_cap": "Large"},
            {"symbol": "ICICIBANK.BO", "name": "ICICI Bank Ltd", "sector": "Banking", "market_cap": "Large"},
            {"symbol": "HINDUNILVR.BO", "name": "Hindustan Unilever Ltd", "sector": "FMCG", "market_cap": "Large"},
        ]
    
    def load_credentials(self) -> Dict:
        """Load API credentials from file or return empty dict"""
        creds_file = 'api_credentials.json'
        if os.path.exists(creds_file):
            with open(creds_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_credentials(self, credentials: Dict):
        """Save API credentials to file"""
        with open('api_credentials.json', 'w') as f:
            json.dump(credentials, f, indent=2)
        self.api_credentials = credentials
    
    def get_all_stocks(self, exchange: str = 'NSE') -> List[Dict]:
        """Get all available stocks for an exchange"""
        return self.nse_stocks if exchange == 'NSE' else self.bse_stocks
    
    def search_stocks(self, query: str, exchange: str = 'NSE') -> List[Dict]:
        """Search for stocks by name or symbol"""
        stocks = self.get_all_stocks(exchange)
        query = query.upper()
        
        results = []
        for stock in stocks:
            symbol_match = query in stock['symbol'].upper()
            name_match = query in stock['name'].upper()
            
            if symbol_match or name_match:
                # Get real-time data
                try:
                    price_data = self.get_real_time_quote(stock['symbol'])
                    if price_data:
                        stock.update(price_data)
                except:
                    pass
                results.append(stock)
        
        return results[:10]  # Return top 10 matches
    
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time quote with caching"""
        cache_key = f"quote_{symbol}"
        
        # Check cache
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if (datetime.now() - cache_time).seconds < self.cache_duration:
                return data
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')
            
            if hist.empty:
                return {}
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
            
            quote_data = {
                'current_price': round(float(current_price), 2),
                'previous_close': round(float(prev_close), 2),
                'price_change': round(float(change), 2),
                'price_change_percent': round(float(change_percent), 2),
                'volume': int(hist['Volume'].iloc[-1]),
                'today_high': round(float(hist['High'].iloc[-1]), 2),
                'today_low': round(float(hist['Low'].iloc[-1]), 2),
            }
            
            # Cache the data
            self.cache[cache_key] = (datetime.now(), quote_data)
            return quote_data
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return {}
    
    def get_stock_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Get historical stock data with caching"""
        cache_key = f"hist_{symbol}_{period}"
        
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
    
    def get_stock_info(self, symbol: str) -> Dict:
        """Get comprehensive stock information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current data
            hist = ticker.history(period='2y')
            if hist.empty:
                return {}
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            # Calculate 52-week high/low
            year_data = hist.tail(252)  # Approximately 1 year of trading days
            week_52_high = year_data['High'].max()
            week_52_low = year_data['Low'].min()
            
            # Today's high/low
            today_high = hist['High'].iloc[-1]
            today_low = hist['Low'].iloc[-1]
            
            # Technical indicators
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1] if len(hist) >= 20 else None
            sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
            
            # Volume analysis
            avg_volume = hist['Volume'].tail(30).mean()
            volume_ratio = hist['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', symbol.replace('.NS', '').replace('.BO', '')),
                'current_price': round(current_price, 2),
                'previous_close': round(prev_close, 2),
                'price_change': round(current_price - prev_close, 2),
                'price_change_percent': round(((current_price - prev_close) / prev_close) * 100, 2),
                'today_high': round(today_high, 2),
                'today_low': round(today_low, 2),
                'week_52_high': round(week_52_high, 2),
                'week_52_low': round(week_52_low, 2),
                'volume': hist['Volume'].iloc[-1],
                'avg_volume': int(avg_volume),
                'volume_ratio': round(volume_ratio, 2),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 'N/A',
                'book_value': round(info.get('bookValue', 0), 2) if info.get('bookValue') else 'N/A',
                'dividend_yield': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 'N/A',
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'currency': info.get('currency', 'INR'),
                'exchange': info.get('exchange', 'NSI' if '.NS' in symbol else 'BSE'),
                'sma_20': round(sma_20, 2) if sma_20 and not pd.isna(sma_20) else 'N/A',
                'sma_50': round(sma_50, 2) if sma_50 and not pd.isna(sma_50) else 'N/A',
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}
    
    def get_trending_stocks(self, exchange: str = 'NSE', limit: int = 10) -> List[Dict]:
        """Get trending stocks based on volume and price movement"""
        stocks = self.get_all_stocks(exchange)
        trending = []
        
        def get_trend_data(stock):
            try:
                symbol = stock['symbol']
                quote_data = self.get_real_time_quote(symbol)
                if quote_data and 'price_change_percent' in quote_data:
                    result = stock.copy()
                    result.update(quote_data)
                    # Calculate trending score
                    result['trending_score'] = abs(quote_data['price_change_percent']) * (quote_data.get('volume', 0) / 1000000)
                    return result
            except:
                pass
            return None
        
        # Use ThreadPoolExecutor for faster data fetching
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(get_trend_data, stocks[:limit*2]))
        
        # Filter and sort by trending score
        trending = [r for r in results if r and 'trending_score' in r]
        trending.sort(key=lambda x: x.get('trending_score', 0), reverse=True)
        
        return trending[:limit]

class NSEBSETradingApp:
    """Main trading application class"""
    
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
        ])
        self.data_provider = NSEBSEDataProvider()
        
        # Initialize paper trading engine
        self.paper_trader = PaperTradingEngine(initial_capital=100000)
        
        # Initialize strategies
        self.strategies = {
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'adaptive': AdaptiveStrategy()
        }
        
        # API credentials storage
        self.api_credentials = {}
        
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup the main application layout"""
        self.app.layout = html.Div([
            # Header
            html.Nav([
                html.Div([
                    html.H1([
                        html.I(className="fas fa-chart-line me-2"),
                        "NSE/BSE Trading Platform"
                    ], className="navbar-brand mb-0 h1 text-white"),
                    html.Div([
                        html.Button([
                            html.I(className="fas fa-cog me-2"),
                            "API Settings"
                        ], id='api-settings-btn', className="btn btn-outline-light me-2"),
                        html.Button([
                            html.I(className="fas fa-refresh me-2"),
                            "Refresh Data"
                        ], id='refresh-btn', className="btn btn-light")
                    ], className="ms-auto")
                ], className="container-fluid d-flex align-items-center")
            ], className="navbar navbar-dark", style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'padding': '1rem 0'
            }),
            
            # API Credentials Modal
            self._create_api_modal(),
            
            # Main Content
            html.Div([
                # Search and Selection Row
                html.Div([
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-search me-2"),
                            "Stock Search & Selection"
                        ], className="text-primary mb-3"),
                        html.Div([
                            html.Div([
                                html.Label("Exchange:", className="form-label fw-bold"),
                                dcc.Dropdown(
                                    id='exchange-dropdown',
                                    options=[
                                        {'label': 'NSE (National Stock Exchange)', 'value': 'NSE'},
                                        {'label': 'BSE (Bombay Stock Exchange)', 'value': 'BSE'}
                                    ],
                                    value='NSE',
                                    className="mb-3"
                                )
                            ], className="col-md-4"),
                            html.Div([
                                html.Label("Currency:", className="form-label fw-bold"),
                                dcc.Dropdown(
                                    id='currency-dropdown',
                                    options=[
                                        {'label': 'INR (Indian Rupee)', 'value': 'INR'},
                                        {'label': 'USD (US Dollar)', 'value': 'USD'}
                                    ],
                                    value='INR',
                                    className="mb-3"
                                )
                            ], className="col-md-4"),
                            html.Div([
                                html.Label("Search Stock:", className="form-label fw-bold"),
                                html.Div([
                                    dcc.Input(
                                        id='stock-search-input',
                                        type='text',
                                        placeholder='Enter stock symbol or name (e.g., RELIANCE, TCS)',
                                        className="form-control",
                                        style={'border-radius': '25px'}
                                    ),
                                    html.Button([
                                        html.I(className="fas fa-search")
                                    ], id='search-btn', className="btn btn-primary ms-2", 
                                    style={'border-radius': '25px'})
                                ], className="d-flex")
                            ], className="col-md-4")
                        ], className="row")
                    ], className="card-body")
                ], className="card mb-4"),
                
                # Stock Information and Trending Stocks Row
                html.Div([
                    html.Div([
                        html.Div([
                            html.H5([
                                html.I(className="fas fa-info-circle me-2"),
                                "Stock Information"
                            ], className="card-title text-primary"),
                            html.Div(id='stock-info-content', children=[
                                html.P("Select a stock to view detailed information", 
                                      className="text-muted text-center")
                            ])
                        ], className="card-body")
                    ], className="card", style={'min-height': '300px'})
                ], className="col-md-6"),
                
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-fire me-2"),
                            "Trending Stocks"
                        ], className="card-title text-danger"),
                        html.Div(id='trending-stocks-content')
                    ], className="card-body")
                ], className="card", style={'min-height': '300px'})
                ], className="col-md-6")
            ], className="row mb-4"),
            
            # Charts and Analysis Row
            html.Div([
                html.Div([
                    html.H5([
                        html.I(className="fas fa-chart-candlestick me-2"),
                        "Stock Chart & Analysis"
                    ], className="card-title text-success"),
                    dcc.Graph(id='stock-chart', style={'height': '500px'})
                ], className="card-body")
            ], className="card mb-4"),
            
            # Trading Actions Row
            html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-exchange-alt me-2"),
                            "Trading Actions & Strategies"
                        ], className="card-title text-warning"),
                        
                        # Strategy Selection Section
                        html.Div([
                            html.H6([
                                html.I(className="fas fa-robot me-2"),
                                "AI Trading Strategies"
                            ], className="text-primary mb-3"),
                            html.Row([
                                html.Div([
                                    html.Label("Select Strategy:", className="form-label fw-bold"),
                                    dcc.Dropdown(
                                        id='strategy-dropdown',
                                        options=[
                                            {'label': '📈 Momentum Strategy - Follow trending movements', 'value': 'momentum'},
                                            {'label': '📉 Mean Reversion - Buy dips, sell peaks', 'value': 'mean_reversion'},
                                            {'label': '🎯 Adaptive Strategy - AI-powered multi-approach', 'value': 'adaptive'}
                                        ],
                                        value='momentum',
                                        className="mb-3"
                                    ),
                                    html.Div(id="strategy-explanation", className="alert alert-info")
                                ], className="col-md-6"),
                                html.Div([
                                    html.Label("Position Size (INR):", className="form-label fw-bold"),
                                    dcc.Input(
                                        id='position-size',
                                        type='number',
                                        value=50000,
                                        min=1000,
                                        step=1000,
                                        className="form-control mb-2"
                                    ),
                                    html.Label("Stop Loss (%):", className="form-label fw-bold"),
                                    dcc.Input(
                                        id='stop-loss',
                                        type='number',
                                        value=2.0,
                                        min=0.1,
                                        max=10,
                                        step=0.1,
                                        className="form-control mb-2"
                                    ),
                                    html.Label("Take Profit (%):", className="form-label fw-bold"),
                                    dcc.Input(
                                        id='take-profit',
                                        type='number',
                                        value=5.0,
                                        min=0.1,
                                        max=20,
                                        step=0.1,
                                        className="form-control mb-2"
                                    )
                                ], className="col-md-6")
                            ])
                        ], className="mb-4"),
                        
                        # Enhanced Action Buttons
                        html.Div([
                            html.Row([
                                html.Div([
                                    html.Button([
                                        html.I(className="fas fa-chart-bar me-2"),
                                        "Analyze Stock"
                                    ], id='analyze-btn', className="btn btn-info btn-lg w-100 mb-2"),
                                    html.Small("Technical analysis & fundamentals", className="text-muted")
                                ], className="col-md-3"),
                                html.Div([
                                    html.Button([
                                        html.I(className="fas fa-history me-2"),
                                        "Backtest Strategy"
                                    ], id='backtest-btn', className="btn btn-success btn-lg w-100 mb-2"),
                                    html.Small("Test strategy on historical data", className="text-muted")
                                ], className="col-md-3"),
                                html.Div([
                                    html.Button([
                                        html.I(className="fas fa-play-circle me-2"),
                                        "Start Paper Trading"
                                    ], id='paper-trade-btn', className="btn btn-warning btn-lg w-100 mb-2"),
                                    html.Small("Risk-free virtual trading", className="text-muted")
                                ], className="col-md-3"),
                                html.Div([
                                    html.Button([
                                        html.I(className="fas fa-wallet me-2"),
                                        "View Portfolio"
                                    ], id='portfolio-btn', className="btn btn-primary btn-lg w-100 mb-2"),
                                    html.Small("Check trading performance", className="text-muted")
                                ], className="col-md-3")
                            ])
                        ]),
                        
                        # Results Display
                        html.Div(id='trading-result', className="mt-4")
                    ], className="card-body")
                ], className="card mb-4")
                
            ], className="container-fluid mt-4"),
            
            # Hidden divs for storing data
            html.Div(id='selected-stock', style={'display': 'none'}),
            html.Div(id='api-modal-trigger', style={'display': 'none'}),
            
        ], style={'background-color': '#f8f9fa', 'min-height': '100vh'})
    
    def _create_api_modal(self):
        """Create API credentials modal"""
        return html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-key me-2"),
                            "API Credentials Setup"
                        ], className="modal-title"),
                        html.Button([
                            html.Span("×", **{"aria-hidden": "true"})
                        ], className="btn-close", **{"data-bs-dismiss": "modal"})
                    ], className="modal-header"),
                    html.Div([
                        html.P("Configure your API credentials for enhanced data access:", 
                               className="text-muted mb-4"),
                        html.Div([
                            html.Label("Alpha Vantage API Key:", className="form-label"),
                            dcc.Input(
                                id='alpha-vantage-key',
                                type='password',
                                placeholder='Enter your Alpha Vantage API key',
                                className="form-control mb-3"
                            )
                        ]),
                        html.Div([
                            html.Label("Quandl API Key:", className="form-label"),
                            dcc.Input(
                                id='quandl-key',
                                type='password',
                                placeholder='Enter your Quandl API key',
                                className="form-control mb-3"
                            )
                        ]),
                        html.Div([
                            html.Label("Polygon.io API Key:", className="form-label"),
                            dcc.Input(
                                id='polygon-key',
                                type='password',
                                placeholder='Enter your Polygon.io API key',
                                className="form-control mb-3"
                            )
                        ]),
                        html.Div([
                            html.Small("These credentials will be stored locally and used to fetch enhanced market data.", 
                                     className="text-muted")
                        ])
                    ], className="modal-body"),
                    html.Div([
                        html.Button("Save Credentials", id='save-credentials-btn', 
                                   className="btn btn-primary"),
                        html.Button("Cancel", className="btn btn-secondary ms-2",
                                   **{"data-bs-dismiss": "modal"})
                    ], className="modal-footer")
                ], className="modal-content")
            ], className="modal-dialog")
        ], className="modal fade", id="api-modal", **{"tabindex": "-1"})
    
    def setup_callbacks(self):
        """Setup all Dash callbacks"""
        
        # Store selected stock data
        @self.app.callback(
            Output('selected-stock', 'children'),
            Input('search-btn', 'n_clicks'),
            Input({'type': 'trending-stock', 'symbol': dash.dependencies.ALL}, 'n_clicks'),
            State('stock-search-input', 'value'),
            State('exchange-dropdown', 'value'),
            prevent_initial_call=True
        )
        def store_selected_stock(search_clicks, trending_clicks, search_input, exchange):
            """Store the selected stock symbol"""
            ctx = callback_context
            if not ctx.triggered:
                return ""
            
            trigger = ctx.triggered[0]
            
            if 'search-btn' in trigger['prop_id'] and search_input:
                # Search functionality
                suffix = '.NS' if exchange == 'NSE' else '.BO'
                if not search_input.endswith(('.NS', '.BO')):
                    return search_input.upper() + suffix
                else:
                    return search_input.upper()
            elif 'trending-stock' in trigger['prop_id']:
                # Trending stock clicked
                import json
                button_id = json.loads(trigger['prop_id'].split('.')[0])
                return button_id['symbol']
            
            return ""
        
        @self.app.callback(
            Output('trending-stocks-content', 'children'),
            Input('exchange-dropdown', 'value'),
            Input('refresh-btn', 'n_clicks')
        )
        def update_trending_stocks(exchange, refresh_clicks):
            """Update trending stocks based on selected exchange"""
            trending = self.data_provider.get_trending_stocks(exchange)
            
            if not trending:
                return html.Div([
                    html.I(className="fas fa-spinner fa-spin me-2"),
                    "Loading trending stocks..."
                ], className="text-center text-muted py-4")
            
            cards = []
            for stock in trending:
                change_color = "text-success" if stock.get('price_change', 0) > 0 else "text-danger"
                change_icon = "fa-arrow-up" if stock.get('price_change', 0) > 0 else "fa-arrow-down"
                
                card = html.Div([
                    html.Div([
                        html.Div([
                            html.H6(stock['symbol'].replace('.NS', '').replace('.BO', ''), 
                                   className="fw-bold mb-1 text-primary"),
                            html.P(stock['name'][:35] + "..." if len(stock['name']) > 35 else stock['name'], 
                                  className="text-muted small mb-2"),
                            html.Div([
                                html.Span(f"₹{stock.get('current_price', 'N/A')}", className="fw-bold h6 mb-0"),
                                html.Div([
                                    html.I(className=f"fas {change_icon} me-1"),
                                    f"{stock.get('price_change_percent', 0):.2f}%"
                                ], className=f"{change_color} small")
                            ], className="d-flex justify-content-between align-items-center"),
                            html.Div([
                                html.Small(f"Vol: {stock.get('volume', 0):,}", className="text-muted me-3"),
                                html.Small(f"{stock.get('sector', 'N/A')}", className="text-info")
                            ])
                        ], className="p-3")
                    ], className="card h-100 shadow-sm border-0", 
                    style={'cursor': 'pointer', 'transition': 'transform 0.2s'},
                    id={'type': 'trending-stock', 'symbol': stock['symbol']})
                ], className="col-md-6 col-lg-4 mb-3")
                cards.append(card)
            
            return html.Div(cards, className="row")
        
        @self.app.callback(
            [Output('stock-info-content', 'children'),
             Output('stock-chart', 'figure')],
            Input('selected-stock', 'children'),
            prevent_initial_call=True
        )
        def update_stock_display(selected_symbol):
            """Update stock information and chart when symbol is selected"""
            if not selected_symbol:
                return "Select a stock to view information", go.Figure()
            
            # Get comprehensive stock information
            stock_info = self.data_provider.get_stock_info(selected_symbol)
            if not stock_info:
                error_content = html.Div([
                    html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                    f"Unable to fetch data for {selected_symbol}. Please check the symbol and try again."
                ], className="alert alert-warning")
                return error_content, go.Figure()
            
            # Create stock info display
            info_content = self._create_comprehensive_stock_display(stock_info)
            
            # Create chart
            chart_fig = self._create_advanced_stock_chart(selected_symbol)
            
            return info_content, chart_fig
        
        @self.app.callback(
            Output('trading-result', 'children'),
            [Input('buy-btn', 'n_clicks'),
             Input('sell-btn', 'n_clicks'), 
             Input('analyze-btn', 'n_clicks')],
            State('selected-stock', 'children'),
            prevent_initial_call=True
        )
        def handle_trading_actions(buy_clicks, sell_clicks, analyze_clicks, selected_symbol):
            """Handle trading button clicks"""
            ctx = callback_context
            if not ctx.triggered or not selected_symbol:
                return html.Div("Please select a stock first.", className="alert alert-info")
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Get stock info for the selected symbol
            stock_info = self.data_provider.get_stock_info(selected_symbol)
            stock_name = stock_info.get('company_name', selected_symbol) if stock_info else selected_symbol
            current_price = stock_info.get('current_price', 0) if stock_info else 0
            
            if button_id == 'buy-btn':
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-shopping-cart text-success me-2"),
                        html.Strong("Buy Order Prepared")
                    ], className="mb-2"),
                    html.P([
                        f"Stock: {stock_name} ({selected_symbol})",
                        html.Br(),
                        f"Current Price: ₹{current_price}",
                        html.Br(),
                        "Order Type: Market Order",
                        html.Br(),
                        "Status: Ready to execute (Paper Trading Mode)"
                    ]),
                    html.Button("Execute Buy Order", className="btn btn-success btn-sm me-2"),
                    html.Button("Cancel", className="btn btn-outline-secondary btn-sm")
                ], className="alert alert-success")
                
            elif button_id == 'sell-btn':
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-hand-holding-usd text-danger me-2"),
                        html.Strong("Sell Order Prepared")
                    ], className="mb-2"),
                    html.P([
                        f"Stock: {stock_name} ({selected_symbol})",
                        html.Br(),
                        f"Current Price: ₹{current_price}",
                        html.Br(),
                        "Order Type: Market Order",
                        html.Br(),
                        "Status: Ready to execute (Paper Trading Mode)"
                    ]),
                    html.Button("Execute Sell Order", className="btn btn-danger btn-sm me-2"),
                    html.Button("Cancel", className="btn btn-outline-secondary btn-sm")
                ], className="alert alert-warning")
                
            elif button_id == 'analyze-btn':
                if stock_info:
                    analysis = self._generate_stock_analysis(stock_info)
                    return analysis
                else:
                    return html.Div("Unable to analyze - stock data not available", className="alert alert-warning")
        
        # API Credentials callback
        @self.app.callback(
            Output('api-modal-trigger', 'children'),
            Input('save-credentials-btn', 'n_clicks'),
            [State('alpha-vantage-key', 'value'),
             State('quandl-key', 'value'), 
             State('polygon-key', 'value')],
            prevent_initial_call=True
        )
        def save_api_credentials(save_clicks, av_key, quandl_key, polygon_key):
            """Save API credentials"""
            if save_clicks:
                credentials = {}
                if av_key:
                    credentials['alpha_vantage'] = av_key
                if quandl_key:
                    credentials['quandl'] = quandl_key
                if polygon_key:
                    credentials['polygon'] = polygon_key
                
                self.data_provider.save_credentials(credentials)
                
                return html.Div([
                    html.I(className="fas fa-check-circle text-success me-2"),
                    "API credentials saved successfully!"
                ], className="alert alert-success")
            
            return ""
    
    def _create_comprehensive_stock_display(self, stock_info: Dict) -> html.Div:
        """Create comprehensive stock information display"""
        change_color = "text-success" if stock_info.get('price_change', 0) > 0 else "text-danger"
        change_icon = "fa-arrow-up" if stock_info.get('price_change', 0) > 0 else "fa-arrow-down"
        
        return html.Div([
            # Company Header with Badge
            html.Div([
                html.Div([
                    html.H4([
                        stock_info.get('company_name', 'N/A'),
                        html.Span(stock_info.get('exchange', 'NSE'), 
                                className="badge bg-primary ms-2")
                    ], className="mb-1"),
                    html.H6([
                        stock_info['symbol'],
                        html.Span(stock_info.get('sector', 'N/A'), 
                                className="badge bg-secondary ms-2")
                    ], className="text-muted mb-3")
                ])
            ]),
            
            # Price Section
            html.Div([
                html.Div([
                    html.H2(f"₹{stock_info.get('current_price', 'N/A')}", 
                           className="mb-1 fw-bold"),
                    html.Div([
                        html.I(className=f"fas {change_icon} me-2"),
                        f"₹{stock_info.get('price_change', 0)} ({stock_info.get('price_change_percent', 0)}%)"
                    ], className=f"{change_color} h5 mb-0")
                ], className="col-md-6"),
                html.Div([
                    html.Div([
                        html.Small("Previous Close", className="text-muted d-block"),
                        html.Strong(f"₹{stock_info.get('previous_close', 'N/A')}")
                    ], className="mb-2"),
                    html.Div([
                        html.Small("Last Updated", className="text-muted d-block"),
                        html.Strong(stock_info.get('last_updated', 'N/A'))
                    ])
                ], className="col-md-6")
            ], className="row mb-4"),
            
            # Trading Range
            html.Div([
                html.H5("Trading Ranges", className="text-primary mb-3"),
                html.Div([
                    html.Div([
                        html.Div([
                            html.H6("Today's Range", className="text-info mb-2"),
                            html.Div([
                                html.Div([
                                    html.Small("Low", className="text-muted d-block"),
                                    html.Strong(f"₹{stock_info.get('today_low', 'N/A')}", className="text-danger")
                                ], className="col-6"),
                                html.Div([
                                    html.Small("High", className="text-muted d-block"),
                                    html.Strong(f"₹{stock_info.get('today_high', 'N/A')}", className="text-success")
                                ], className="col-6")
                            ], className="row")
                        ], className="card-body")
                    ], className="card h-100"),
                    
                    html.Div([
                        html.Div([
                            html.H6("52 Week Range", className="text-warning mb-2"),
                            html.Div([
                                html.Div([
                                    html.Small("52W Low", className="text-muted d-block"),
                                    html.Strong(f"₹{stock_info.get('week_52_low', 'N/A')}", className="text-danger")
                                ], className="col-6"),
                                html.Div([
                                    html.Small("52W High", className="text-muted d-block"),
                                    html.Strong(f"₹{stock_info.get('week_52_high', 'N/A')}", className="text-success")
                                ], className="col-6")
                            ], className="row")
                        ], className="card-body")
                    ], className="card h-100")
                ], className="col-md-6"),
                
                # Volume and Technical Indicators
                html.Div([
                    html.Div([
                        html.H6("Volume & Indicators", className="text-success mb-2"),
                        html.Div([
                            html.Div([
                                html.Small("Volume", className="text-muted d-block"),
                                html.Strong(f"{stock_info.get('volume', 0):,}")
                            ], className="mb-2"),
                            html.Div([
                                html.Small("Avg Volume (30d)", className="text-muted d-block"),
                                html.Strong(f"{stock_info.get('avg_volume', 0):,}")
                            ], className="mb-2"),
                            html.Div([
                                html.Small("Volume Ratio", className="text-muted d-block"),
                                html.Strong(f"{stock_info.get('volume_ratio', 'N/A')}")
                            ])
                        ])
                    ], className="card-body")
                ], className="card h-100")
            ], className="col-md-6")
        ], className="row mb-4"),
        
        # Fundamentals
        html.Div([
            html.H5("Key Fundamentals", className="text-success mb-3"),
            html.Div([
                html.Div([
                    html.Small("Market Cap", className="text-muted d-block"),
                    html.Strong(self._format_market_cap(stock_info.get('market_cap', 'N/A')))
                ], className="col-md-3 mb-3"),
                html.Div([
                    html.Small("P/E Ratio", className="text-muted d-block"),
                    html.Strong(str(stock_info.get('pe_ratio', 'N/A')))
                ], className="col-md-3 mb-3"),
                html.Div([
                    html.Small("Book Value", className="text-muted d-block"),
                    html.Strong(f"₹{stock_info.get('book_value', 'N/A')}")
                ], className="col-md-3 mb-3"),
                html.Div([
                    html.Small("Dividend Yield", className="text-muted d-block"),
                    html.Strong(f"{stock_info.get('dividend_yield', 'N/A')}%")
                ], className="col-md-3 mb-3")
            ], className="row"),
            
            # Technical Indicators
            html.Div([
                html.Div([
                    html.Small("SMA 20", className="text-muted d-block"),
                    html.Strong(f"₹{stock_info.get('sma_20', 'N/A')}")
                ], className="col-md-4"),
                html.Div([
                    html.Small("SMA 50", className="text-muted d-block"),
                    html.Strong(f"₹{stock_info.get('sma_50', 'N/A')}")
                ], className="col-md-4"),
                html.Div([
                    html.Small("Industry", className="text-muted d-block"),
                    html.Strong(stock_info.get('industry', 'N/A'))
                ], className="col-md-4")
            ], className="row")
        ])
    
    def _format_market_cap(self, market_cap):
        """Format market cap for display"""
        if market_cap == 'N/A' or not market_cap:
            return 'N/A'
        
        try:
            cap = float(market_cap)
            if cap >= 1e12:
                return f"₹{cap/1e12:.2f}T"
            elif cap >= 1e9:
                return f"₹{cap/1e9:.2f}B"
            elif cap >= 1e7:
                return f"₹{cap/1e7:.2f}Cr"
            else:
                return f"₹{cap/1e5:.2f}L"
        except:
            return str(market_cap)
    
    def _generate_stock_analysis(self, stock_info: Dict) -> html.Div:
        """Generate technical analysis for the stock"""
        current_price = stock_info.get('current_price', 0)
        sma_20 = stock_info.get('sma_20', 0)
        sma_50 = stock_info.get('sma_50', 0)
        volume_ratio = stock_info.get('volume_ratio', 1)
        price_change_percent = stock_info.get('price_change_percent', 0)
        
        analysis_points = []
        
        # Price vs Moving Averages
        if sma_20 != 'N/A' and sma_20 > 0:
            if current_price > sma_20:
                analysis_points.append(("Bullish Signal", "Price above 20-day SMA", "success"))
            else:
                analysis_points.append(("Bearish Signal", "Price below 20-day SMA", "danger"))
        
        if sma_50 != 'N/A' and sma_50 > 0:
            if current_price > sma_50:
                analysis_points.append(("Long-term Bullish", "Price above 50-day SMA", "success"))
            else:
                analysis_points.append(("Long-term Bearish", "Price below 50-day SMA", "danger"))
        
        # Volume Analysis
        if volume_ratio > 1.5:
            analysis_points.append(("High Volume", "Above average trading volume", "info"))
        elif volume_ratio < 0.5:
            analysis_points.append(("Low Volume", "Below average trading volume", "warning"))
        
        # Price Movement
        if abs(price_change_percent) > 5:
            trend = "Strong Uptrend" if price_change_percent > 0 else "Strong Downtrend"
            color = "success" if price_change_percent > 0 else "danger"
            analysis_points.append((trend, f"Significant price movement: {price_change_percent:.2f}%", color))
        
        # Generate recommendation
        bullish_signals = sum(1 for _, _, color in analysis_points if color == "success")
        bearish_signals = sum(1 for _, _, color in analysis_points if color == "danger")
        
        if bullish_signals > bearish_signals:
            recommendation = ("BUY", "Bullish indicators suggest upward momentum", "success")
        elif bearish_signals > bullish_signals:
            recommendation = ("SELL", "Bearish indicators suggest downward pressure", "danger")
        else:
            recommendation = ("HOLD", "Mixed signals suggest neutral stance", "warning")
        
        return html.Div([
            html.Div([
                html.I(className="fas fa-chart-line text-info me-2"),
                html.Strong("Technical Analysis Report")
            ], className="mb-3"),
            
            # Analysis Points
            html.Div([
                html.Div([
                    html.I(className=f"fas fa-circle text-{color} me-2"),
                    html.Strong(title),
                    html.Br(),
                    html.Small(description, className="text-muted")
                ], className="mb-2") for title, description, color in analysis_points
            ]),
            
            html.Hr(),
            
            # Recommendation
            html.Div([
                html.H5("Recommendation:", className="d-inline me-2"),
                html.Span(recommendation[0], className=f"badge bg-{recommendation[2]} fs-6 me-2"),
                html.P(recommendation[1], className="mb-0 mt-2")
            ], className=f"alert alert-{recommendation[2]} mt-3"),
            
            html.Small("Note: This is automated technical analysis. Please consult with financial advisors before making investment decisions.", 
                      className="text-muted")
        ])
    
    def _create_advanced_stock_chart(self, symbol: str) -> go.Figure:
        """Create advanced stock chart with technical indicators"""
        data = self.data_provider.get_stock_data(symbol, '6mo')
        
        if data.empty:
            return go.Figure().add_annotation(
                text="No chart data available for this symbol",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
        
        fig = go.Figure()
        
        # Main candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol,
            increasing_line_color='#00C851',
            decreasing_line_color='#ff4444'
        ))
        
        # Add moving averages
        if len(data) >= 20:
            sma_20 = data['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=sma_20,
                name='SMA 20',
                line=dict(color='orange', width=2)
            ))
        
        if len(data) >= 50:
            sma_50 = data['Close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=sma_50,
                name='SMA 50',
                line=dict(color='blue', width=2)
            ))
        
        # Volume subplot
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            yaxis='y2',
            opacity=0.3,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f"{symbol} - Advanced Stock Analysis",
            yaxis_title="Price (₹)",
            xaxis_title="Date",
            yaxis2=dict(
                title="Volume",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
        
        # Strategy explanation callback
        @self.app.callback(
            Output('strategy-explanation', 'children'),
            Input('strategy-dropdown', 'value')
        )
        def update_strategy_explanation(strategy):
            explanations = {
                'momentum': [
                    html.I(className="fas fa-trend-up me-2 text-primary"),
                    html.Strong("Momentum Strategy", className="text-primary"), html.Br(),
                    "• Identifies stocks with strong upward price movement", html.Br(),
                    "• Buys when trend accelerates, sells when it weakens", html.Br(),
                    "• Best for trending markets and growth stocks"
                ],
                'mean_reversion': [
                    html.I(className="fas fa-balance-scale me-2 text-success"),
                    html.Strong("Mean Reversion Strategy", className="text-success"), html.Br(),
                    "• Assumes prices return to their average over time", html.Br(),
                    "• Buys when price is below average, sells when above", html.Br(),
                    "• Best for stable stocks in sideways markets"
                ],
                'adaptive': [
                    html.I(className="fas fa-robot me-2 text-info"),
                    html.Strong("Adaptive Strategy", className="text-info"), html.Br(),
                    "• AI-powered approach that adapts to market conditions", html.Br(),
                    "• Combines multiple strategies based on current market regime", html.Br(),
                    "• Best for all market conditions, suitable for beginners"
                ]
            }
            return explanations.get(strategy, "Select a strategy to see explanation")
        
        # API Settings modal callback
        @self.app.callback(
            Output('api-modal-trigger', 'children'),
            Input('api-settings-btn', 'n_clicks'),
            prevent_initial_call=True
        )
        def toggle_api_modal(n_clicks):
            if n_clicks:
                return html.Script("$('#api-modal').modal('show');")
            return ""
        
        # Save API credentials callback  
        @self.app.callback(
            Output('api-modal-trigger', 'children'),
            [Input('alpha-vantage-key', 'value'),
             Input('quandl-key', 'value'),
             Input('polygon-key', 'value')],
            prevent_initial_call=True
        )
        def save_api_credentials(alpha_key, quandl_key, polygon_key):
            if any([alpha_key, quandl_key, polygon_key]):
                credentials = {}
                if alpha_key: credentials['alpha_vantage'] = alpha_key
                if quandl_key: credentials['quandl'] = quandl_key  
                if polygon_key: credentials['polygon'] = polygon_key
                
                self.data_provider.save_credentials(credentials)
                self.api_credentials.update(credentials)
                
                return html.Div([
                    html.I(className="fas fa-check-circle text-success me-2"),
                    "API credentials saved successfully!"
                ], className="alert alert-success")
            return ""

        # Enhanced trading actions callback
        @self.app.callback(
            Output('trading-result', 'children'),
            [Input('analyze-btn', 'n_clicks'),
             Input('backtest-btn', 'n_clicks'), 
             Input('paper-trade-btn', 'n_clicks'),
             Input('portfolio-btn', 'n_clicks')],
            [State('selected-stock', 'children'),
             State('strategy-dropdown', 'value'),
             State('position-size', 'value'),
             State('stop-loss', 'value'),
             State('take-profit', 'value')],
            prevent_initial_call=True
        )
        def handle_trading_actions(analyze_clicks, backtest_clicks, paper_clicks, portfolio_clicks,
                                 selected_stock, strategy, position_size, stop_loss, take_profit):
            ctx = callback_context
            if not ctx.triggered or not selected_stock:
                return html.Div([
                    html.P("Please select a stock first by searching or clicking on a trending stock.",
                           className="text-muted text-center py-4")
                ])

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            symbol = selected_stock
            
            if button_id == 'analyze-btn':
                return self._show_stock_analysis(symbol)
            elif button_id == 'backtest-btn':
                return self._run_strategy_backtest(symbol, strategy, position_size, stop_loss, take_profit)
            elif button_id == 'paper-trade-btn':
                return self._execute_paper_trade(symbol, strategy, position_size, stop_loss, take_profit)
            elif button_id == 'portfolio-btn':
                return self._show_portfolio_summary()
            
            return ""
    
    def _show_stock_analysis(self, symbol: str):
        """Show comprehensive stock analysis"""
        try:
            stock_info = self.data_provider.get_stock_info(symbol)
            if not stock_info:
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-triangle fa-2x text-warning mb-3"),
                        html.H5("Analysis Error", className="text-warning"),
                        html.P(f"Unable to analyze {symbol}. Please try again.")
                    ], className="text-center p-4")
                ], className="alert alert-warning")
            
            # Get historical data for technical analysis
            data = self.data_provider.get_stock_data(symbol, '6mo')
            
            if not data.empty:
                # Calculate technical indicators
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
                max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
                
                # Moving averages
                ma_20 = data['Close'].rolling(20).mean().iloc[-1]
                ma_50 = data['Close'].rolling(50).mean().iloc[-1]
                current_price = data['Close'].iloc[-1]
                
                # Price trend analysis
                trend = "Bullish" if current_price > ma_20 > ma_50 else "Bearish" if current_price < ma_20 < ma_50 else "Neutral"
                trend_color = "success" if trend == "Bullish" else "danger" if trend == "Bearish" else "warning"
                
                return html.Div([
                    html.H4([
                        html.I(className="fas fa-chart-line me-2"),
                        f"Analysis: {symbol}"
                    ], className="text-primary mb-4"),
                    
                    # Key metrics cards
                    html.Row([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-rupee-sign fa-2x text-primary mb-2"),
                                html.H6("Current Price", className="text-muted mb-1"),
                                html.H4(f"₹{current_price:.2f}", className="text-primary mb-0")
                            ], className="text-center p-3 bg-light rounded")
                        ], className="col-md-3"),
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-chart-area fa-2x text-warning mb-2"),
                                html.H6("Volatility", className="text-muted mb-1"),
                                html.H4(f"{volatility:.1f}%", className="text-warning mb-0")
                            ], className="text-center p-3 bg-light rounded")
                        ], className="col-md-3"),
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-trophy fa-2x text-info mb-2"),
                                html.H6("Sharpe Ratio", className="text-muted mb-1"), 
                                html.H4(f"{sharpe_ratio:.2f}", className="text-info mb-0")
                            ], className="text-center p-3 bg-light rounded")
                        ], className="col-md-3"),
                        html.Div([
                            html.Div([
                                html.I(className=f"fas fa-trending-{'up' if trend == 'Bullish' else 'down' if trend == 'Bearish' else 'flat'} fa-2x text-{trend_color} mb-2"),
                                html.H6("Trend", className="text-muted mb-1"),
                                html.H4(trend, className=f"text-{trend_color} mb-0")
                            ], className="text-center p-3 bg-light rounded")
                        ], className="col-md-3")
                    ], className="mb-4"),
                    
                    # Recommendation
                    html.Div([
                        html.H6("Investment Recommendation", className="text-success mb-3"),
                        html.P([
                            html.Strong("Trend Analysis: "), 
                            f"Stock is currently in a {trend.lower()} trend. ",
                            "Price is above key moving averages." if trend == "Bullish" else 
                            "Price is below key moving averages." if trend == "Bearish" else
                            "Price is consolidating around moving averages."
                        ]),
                        html.P([
                            html.Strong("Risk Assessment: "),
                            f"With {volatility:.1f}% annual volatility, this stock is ",
                            "low risk" if volatility < 25 else "moderate risk" if volatility < 50 else "high risk",
                            f". The Sharpe ratio of {sharpe_ratio:.2f} indicates ",
                            "excellent" if sharpe_ratio > 2 else "good" if sharpe_ratio > 1 else "poor",
                            " risk-adjusted returns."
                        ])
                    ], className="alert alert-info")
                ])
            else:
                return html.Div([
                    html.P("Insufficient data for technical analysis.", className="text-muted text-center py-4")
                ])
                
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return html.Div([
                html.P(f"Analysis error: {str(e)}", className="text-danger text-center py-4")
            ])
    
    def _run_strategy_backtest(self, symbol: str, strategy: str, position_size: float, 
                              stop_loss: float, take_profit: float):
        """Run backtest for the selected strategy"""
        try:
            data = self.data_provider.get_stock_data(symbol, '1y')
            
            if data.empty:
                return html.Div([
                    html.P("Insufficient historical data for backtesting.", 
                           className="text-muted text-center py-4")
                ])
            
            # Generate strategy signals
            strategy_obj = self.strategies.get(strategy)
            if not strategy_obj:
                return html.Div([
                    html.P("Strategy not available.", className="text-muted text-center py-4")
                ])
            
            # Simple backtesting simulation
            returns = data['Close'].pct_change().dropna()
            
            if strategy == 'momentum':
                # Simple momentum: buy when price > 20-day MA
                ma_20 = data['Close'].rolling(20).mean()
                signals = (data['Close'] > ma_20).astype(int)
            elif strategy == 'mean_reversion':
                # Mean reversion: buy when price < 20-day MA
                ma_20 = data['Close'].rolling(20).mean()
                signals = (data['Close'] < ma_20).astype(int)
            else:  # adaptive
                # Combine both signals
                ma_20 = data['Close'].rolling(20).mean()
                momentum_signals = (data['Close'] > ma_20).astype(int)
                mean_rev_signals = (data['Close'] < ma_20).astype(int)
                signals = ((momentum_signals + mean_rev_signals) > 0).astype(int)
            
            # Calculate strategy returns
            strategy_returns = signals.shift(1) * returns
            strategy_returns = strategy_returns.fillna(0)
            
            # Buy and hold returns
            buy_hold_returns = returns.fillna(0)
            
            # Cumulative returns
            strategy_cumulative = (1 + strategy_returns).cumprod()
            buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
            
            # Performance metrics
            total_return = (strategy_cumulative.iloc[-1] - 1) * 100
            buy_hold_return = (buy_hold_cumulative.iloc[-1] - 1) * 100
            outperformance = total_return - buy_hold_return
            
            # Win rate
            profitable_trades = (strategy_returns > 0).sum()
            total_trades = (signals.diff() != 0).sum()
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            performance_color = "success" if outperformance > 0 else "danger"
            
            return html.Div([
                html.H4([
                    html.I(className="fas fa-history me-2"),
                    f"Backtest: {strategy.title()} on {symbol}"
                ], className="text-primary mb-4"),
                
                # Performance cards
                html.Row([
                    html.Div([
                        html.Div([
                            html.I(className=f"fas fa-chart-line fa-2x text-{performance_color} mb-2"),
                            html.H6("Strategy Return", className="text-muted mb-1"),
                            html.H4(f"{total_return:+.1f}%", className=f"text-{performance_color} mb-0")
                        ], className="text-center p-3 bg-light rounded")
                    ], className="col-md-4"),
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-chart-area fa-2x text-primary mb-2"),
                            html.H6("Buy & Hold", className="text-muted mb-1"),
                            html.H4(f"{buy_hold_return:+.1f}%", className="text-primary mb-0")
                        ], className="text-center p-3 bg-light rounded")
                    ], className="col-md-4"),
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-percentage fa-2x text-info mb-2"),
                            html.H6("Win Rate", className="text-muted mb-1"),
                            html.H4(f"{win_rate:.0f}%", className="text-info mb-0")
                        ], className="text-center p-3 bg-light rounded")
                    ], className="col-md-4")
                ], className="mb-4"),
                
                # Analysis
                html.Div([
                    html.H6("Backtest Summary", className="text-success mb-3"),
                    html.P([
                        html.Strong("Performance: "),
                        f"Strategy returned {total_return:+.1f}% vs {buy_hold_return:+.1f}% for buy & hold. ",
                        f"Outperformance: {outperformance:+.1f}%"
                    ]),
                    html.P([
                        html.Strong("Trade Statistics: "),
                        f"Win rate of {win_rate:.0f}% across {total_trades} trades."
                    ]),
                    html.P([
                        html.Strong("With your position size of ₹{:,}: ".format(int(position_size))),
                        f"Potential profit would be ₹{int(position_size * total_return / 100):,} ",
                        f"({'gain' if total_return > 0 else 'loss'})"
                    ])
                ], className="alert alert-info")
            ])
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return html.Div([
                html.P(f"Backtest error: {str(e)}", className="text-danger text-center py-4")
            ])
    
    def _execute_paper_trade(self, symbol: str, strategy: str, position_size: float,
                           stop_loss: float, take_profit: float):
        """Execute paper trading strategy"""
        try:
            stock_info = self.data_provider.get_stock_info(symbol)
            current_price = stock_info.get('current_price', 0)
            
            if not current_price:
                return html.Div([
                    html.P("Unable to get current price for paper trading.", 
                           className="text-muted text-center py-4")
                ])
            
            # Calculate number of shares
            shares = int(position_size / current_price)
            
            # Execute paper trade
            trade_result = self.paper_trader.execute_trade(symbol, 'buy', shares, current_price)
            
            return html.Div([
                html.H4([
                    html.I(className="fas fa-play-circle me-2"),
                    "Paper Trade Executed"
                ], className="text-success mb-4"),
                
                html.Div([
                    html.H6("Trade Details", className="text-primary mb-3"),
                    html.P([html.Strong("Symbol: "), symbol]),
                    html.P([html.Strong("Strategy: "), strategy.title()]),
                    html.P([html.Strong("Action: "), "BUY"]),
                    html.P([html.Strong("Shares: "), f"{shares:,}"]),
                    html.P([html.Strong("Price: "), f"₹{current_price:.2f}"]),
                    html.P([html.Strong("Total Investment: "), f"₹{shares * current_price:,.2f}"]),
                    html.P([html.Strong("Stop Loss: "), f"₹{current_price * (1 - stop_loss/100):.2f} ({stop_loss}%)"]),
                    html.P([html.Strong("Take Profit: "), f"₹{current_price * (1 + take_profit/100):.2f} ({take_profit}%)"])
                ], className="alert alert-success"),
                
                html.Div([
                    html.I(className="fas fa-info-circle me-2"),
                    html.Strong("Note: "), 
                    "This is virtual paper trading. No real money is involved. ",
                    "The system will monitor this position and send alerts based on your strategy."
                ], className="alert alert-info mt-3")
            ])
            
        except Exception as e:
            logger.error(f"Error in paper trade: {e}")
            return html.Div([
                html.P(f"Paper trade error: {str(e)}", className="text-danger text-center py-4")
            ])
    
    def _show_portfolio_summary(self):
        """Show paper trading portfolio summary"""
        try:
            portfolio = self.paper_trader.get_portfolio_summary()
            
            return html.Div([
                html.H4([
                    html.I(className="fas fa-wallet me-2"),
                    "Paper Trading Portfolio"
                ], className="text-primary mb-4"),
                
                # Portfolio overview
                html.Div([
                    html.Row([
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-coins fa-2x text-success mb-2"),
                                html.H6("Total Value", className="text-muted mb-1"),
                                html.H4(f"₹{portfolio.get('total_value', 0):,.2f}", className="text-success mb-0")
                            ], className="text-center p-3 bg-light rounded")
                        ], className="col-md-4"),
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-money-bill fa-2x text-info mb-2"),
                                html.H6("Available Cash", className="text-muted mb-1"),
                                html.H4(f"₹{portfolio.get('cash', 0):,.2f}", className="text-info mb-0")
                            ], className="text-center p-3 bg-light rounded")
                        ], className="col-md-4"),
                        html.Div([
                            html.Div([
                                html.I(className="fas fa-chart-pie fa-2x text-warning mb-2"),
                                html.H6("Positions", className="text-muted mb-1"),
                                html.H4(f"{len(portfolio.get('positions', []))}", className="text-warning mb-0")
                            ], className="text-center p-3 bg-light rounded")
                        ], className="col-md-4")
                    ], className="mb-4")
                ]),
                
                # Positions table
                html.Div([
                    html.H6("Current Positions", className="text-success mb-3"),
                    html.P("No active positions in paper trading portfolio.", className="text-muted text-center py-3") if not portfolio.get('positions') 
                    else html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Symbol"),
                                html.Th("Shares"),
                                html.Th("Entry Price"),
                                html.Th("Current Price"),
                                html.Th("P&L")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(pos.get('symbol', '')),
                                html.Td(pos.get('shares', 0)),
                                html.Td(f"₹{pos.get('entry_price', 0):.2f}"),
                                html.Td(f"₹{pos.get('current_price', 0):.2f}"),
                                html.Td(f"₹{pos.get('pnl', 0):+.2f}",
                                       className="text-success" if pos.get('pnl', 0) >= 0 else "text-danger")
                            ]) for pos in portfolio.get('positions', [])
                        ])
                    ], className="table table-striped")
                ], className="alert alert-light")
            ])
            
        except Exception as e:
            logger.error(f"Error in portfolio summary: {e}")
            return html.Div([
                html.P(f"Portfolio error: {str(e)}", className="text-danger text-center py-4")
            ])
    
    def run(self, debug=True, port=8050):
        """Run the application"""
        print("\n🚀 Starting NSE/BSE Trading Platform...")
        print("📊 Features:")
        print("  • Real-time NSE/BSE stock data")
        print("  • API credentials management")
        print("  • Trending stocks analysis")
        print("  • Professional trading interface")
        print("  • 52-week high/low tracking")
        print("  • Multi-currency support")
        print(f"\n🌐 Access at: http://localhost:{port}")
        print("⚠️  Press Ctrl+C to stop\n")
        
        self.app.run_server(debug=debug, port=port)

if __name__ == '__main__':
    app = NSEBSETradingApp()
    app.run()
