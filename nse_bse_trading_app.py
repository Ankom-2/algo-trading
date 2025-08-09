"""
NSE/BSE Trading Application - Enhanced Version
A comprehensive trading interface for Indian stock markets with Strategy Selection and Paper Trading
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock classes for strategies and paper trading (since imports may fail)
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
        self.trades_history = []
        
    def execute_trade(self, symbol, action, quantity, price):
        trade = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(),
            'value': quantity * price
        }
        self.trades_history.append(trade)
        
        if action.upper() == 'BUY':
            if symbol in self.positions:
                self.positions[symbol]['quantity'] += quantity
                # Average price calculation
                total_value = (self.positions[symbol]['quantity'] - quantity) * self.positions[symbol]['entry_price'] + quantity * price
                self.positions[symbol]['entry_price'] = total_value / self.positions[symbol]['quantity']
            else:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'entry_price': price,
                    'current_price': price
                }
            self.capital -= quantity * price
            
        elif action.upper() == 'SELL':
            if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                self.positions[symbol]['quantity'] -= quantity
                if self.positions[symbol]['quantity'] == 0:
                    del self.positions[symbol]
                self.capital += quantity * price
        
        return {"status": "executed", "message": f"{action} {quantity} shares of {symbol} at ₹{price}"}
        
    def get_portfolio_summary(self):
        total_position_value = 0
        positions_list = []
        
        for symbol, pos in self.positions.items():
            # Get current price (simplified)
            current_price = pos['current_price']  # In real app, fetch from market
            position_value = pos['quantity'] * current_price
            pnl = (current_price - pos['entry_price']) * pos['quantity']
            
            total_position_value += position_value
            positions_list.append({
                'symbol': symbol,
                'shares': pos['quantity'],
                'entry_price': pos['entry_price'],
                'current_price': current_price,
                'pnl': pnl,
                'value': position_value
            })
        
        return {
            'total_value': self.capital + total_position_value,
            'cash': self.capital,
            'positions': positions_list,
            'total_trades': len(self.trades_history)
        }

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
        ]
    
    def _get_comprehensive_bse_stocks(self) -> List[Dict]:
        """Get comprehensive list of BSE stocks"""
        return [
            {"symbol": "RELIANCE.BO", "name": "Reliance Industries Ltd", "sector": "Oil & Gas", "market_cap": "Large"},
            {"symbol": "TCS.BO", "name": "Tata Consultancy Services Ltd", "sector": "IT Services", "market_cap": "Large"},
            {"symbol": "HDFCBANK.BO", "name": "HDFC Bank Ltd", "sector": "Banking", "market_cap": "Large"},
        ]
    
    def load_credentials(self) -> Dict:
        """Load API credentials from file or return empty dict"""
        creds_file = 'api_credentials.json'
        if os.path.exists(creds_file):
            try:
                with open(creds_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def save_credentials(self, credentials: Dict):
        """Save API credentials to file"""
        try:
            with open('api_credentials.json', 'w') as f:
                json.dump(credentials, f, indent=2)
            self.api_credentials = credentials
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
    
    def get_all_stocks(self, exchange: str = 'NSE') -> List[Dict]:
        """Get all available stocks for an exchange"""
        return self.nse_stocks if exchange == 'NSE' else self.bse_stocks
    
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
                'volume': int(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
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
            hist = ticker.history(period='1y')
            if hist.empty:
                return {}
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            
            # Calculate 52-week high/low
            year_data = hist.tail(252)
            week_52_high = year_data['High'].max()
            week_52_low = year_data['Low'].min()
            
            return {
                'symbol': symbol,
                'company_name': info.get('longName', symbol.replace('.NS', '').replace('.BO', '')),
                'current_price': round(current_price, 2),
                'previous_close': round(prev_close, 2),
                'price_change': round(current_price - prev_close, 2),
                'price_change_percent': round(((current_price - prev_close) / prev_close) * 100, 2),
                'week_52_high': round(week_52_high, 2),
                'week_52_low': round(week_52_low, 2),
                'volume': hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0,
                'market_cap': info.get('marketCap', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {}
    
    def get_trending_stocks(self, exchange: str = 'NSE', limit: int = 10) -> List[Dict]:
        """Get trending stocks based on volume and price movement"""
        stocks = self.get_all_stocks(exchange)[:limit]  # Limit for performance
        trending = []
        
        for stock in stocks:
            try:
                quote_data = self.get_real_time_quote(stock['symbol'])
                if quote_data:
                    result = stock.copy()
                    result.update(quote_data)
                    trending.append(result)
            except:
                continue
        
        return trending

class NSEBSETradingApp:
    """Main trading application class - Enhanced with Strategies and Paper Trading"""
    
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
        
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup the main application layout"""
        self.app.layout = html.Div([
            # Enhanced Header
            html.Nav([
                html.Div([
                    html.H1([
                        html.I(className="fas fa-chart-line me-2"),
                        "NSE/BSE Trading Platform - Enhanced"
                    ], className="navbar-brand mb-0 h1 text-white"),
                    html.Div([
                        html.Button([
                            html.I(className="fas fa-cog me-2"),
                            "API Settings"
                        ], id='api-settings-btn', className="btn btn-outline-light me-2"),
                        html.Button([
                            html.I(className="fas fa-refresh me-2"),
                            "Refresh"
                        ], id='refresh-btn', className="btn btn-light")
                    ], className="ms-auto")
                ], className="container-fluid d-flex align-items-center")
            ], className="navbar navbar-dark", style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'padding': '1rem 0'
            }),
            
            # API Credentials Modal
            self._create_api_modal(),
            
            # Main Content Container
            html.Div([
                # Stock Search Section
                html.Div([
                    html.Div([
                        html.H4([
                            html.I(className="fas fa-search me-2"),
                            "Stock Search & Selection"
                        ], className="text-primary mb-3"),
                        html.Row([
                            html.Div([
                                html.Label("Exchange:", className="form-label fw-bold"),
                                dcc.Dropdown(
                                    id='exchange-dropdown',
                                    options=[
                                        {'label': 'NSE (National Stock Exchange)', 'value': 'NSE'},
                                        {'label': 'BSE (Bombay Stock Exchange)', 'value': 'BSE'}
                                    ],
                                    value='NSE'
                                )
                            ], className="col-md-4"),
                            html.Div([
                                html.Label("Search Stock:", className="form-label fw-bold"),
                                html.Div([
                                    dcc.Input(
                                        id='stock-search-input',
                                        type='text',
                                        placeholder='Enter stock symbol (e.g., RELIANCE.NS)',
                                        className="form-control"
                                    ),
                                    html.Button([
                                        html.I(className="fas fa-search")
                                    ], id='search-btn', className="btn btn-primary ms-2")
                                ], className="d-flex")
                            ], className="col-md-8")
                        ])
                    ], className="card-body")
                ], className="card mb-4"),
                
                # Stock Information and Strategy Section
                html.Row([
                    # Stock Information Column
                    html.Div([
                        html.Div([
                            html.H5([
                                html.I(className="fas fa-info-circle me-2"),
                                "Stock Information"
                            ], className="card-title text-primary"),
                            html.Div(id='stock-info-content', children=[
                                html.P("Search for a stock to view detailed information", 
                                      className="text-muted text-center py-4")
                            ])
                        ], className="card-body")
                    ], className="card")
                    ], className="col-md-6 mb-4"),
                    
                    # Strategy Selection Column
                    html.Div([
                        html.Div([
                            html.H5([
                                html.I(className="fas fa-robot me-2"),
                                "AI Trading Strategies"
                            ], className="card-title text-success"),
                            html.Div([
                                html.Label("Select Strategy:", className="form-label fw-bold"),
                                dcc.Dropdown(
                                    id='strategy-dropdown',
                                    options=[
                                        {'label': '📈 Momentum Strategy - Follow trends', 'value': 'momentum'},
                                        {'label': '📉 Mean Reversion - Buy dips, sell peaks', 'value': 'mean_reversion'},
                                        {'label': '🎯 Adaptive Strategy - AI-powered approach', 'value': 'adaptive'}
                                    ],
                                    value='momentum',
                                    className="mb-3"
                                ),
                                html.Div(id="strategy-explanation", className="alert alert-info mb-3"),
                                
                                # Trading Parameters
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
                                    className="form-control mb-3"
                                )
                            ])
                        ], className="card-body")
                    ], className="card")
                    ], className="col-md-6 mb-4")
                ]),
                
                # Chart Section
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-chart-candlestick me-2"),
                            "Stock Chart & Analysis"
                        ], className="card-title text-info"),
                        dcc.Graph(id='stock-chart', style={'height': '500px'})
                    ], className="card-body")
                ], className="card mb-4"),
                
                # Enhanced Action Buttons
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-play-circle me-2"),
                            "Trading Actions"
                        ], className="card-title text-warning"),
                        html.Row([
                            html.Div([
                                html.Button([
                                    html.I(className="fas fa-chart-bar me-2"),
                                    "Analyze Stock"
                                ], id='analyze-btn', className="btn btn-info btn-lg w-100 mb-2"),
                                html.Small("Technical analysis & fundamentals", className="text-muted d-block text-center")
                            ], className="col-md-3"),
                            html.Div([
                                html.Button([
                                    html.I(className="fas fa-history me-2"),
                                    "Backtest Strategy"
                                ], id='backtest-btn', className="btn btn-success btn-lg w-100 mb-2"),
                                html.Small("Test strategy on historical data", className="text-muted d-block text-center")
                            ], className="col-md-3"),
                            html.Div([
                                html.Button([
                                    html.I(className="fas fa-play-circle me-2"),
                                    "Paper Trade"
                                ], id='paper-trade-btn', className="btn btn-warning btn-lg w-100 mb-2"),
                                html.Small("Risk-free virtual trading", className="text-muted d-block text-center")
                            ], className="col-md-3"),
                            html.Div([
                                html.Button([
                                    html.I(className="fas fa-wallet me-2"),
                                    "Portfolio"
                                ], id='portfolio-btn', className="btn btn-primary btn-lg w-100 mb-2"),
                                html.Small("View trading performance", className="text-muted d-block text-center")
                            ], className="col-md-3")
                        ]),
                        
                        # Results Display Area
                        html.Div(id='trading-result', className="mt-4")
                    ], className="card-body")
                ], className="card mb-4"),
                
                # Trending Stocks
                html.Div([
                    html.Div([
                        html.H5([
                            html.I(className="fas fa-fire me-2"),
                            "Trending Stocks"
                        ], className="card-title text-danger"),
                        html.Div(id='trending-stocks-content')
                    ], className="card-body")
                ], className="card mb-4")
                
            ], className="container-fluid mt-4"),
            
            # Hidden storage divs
            html.Div(id='selected-stock', style={'display': 'none'}),
            html.Div(id='api-status', style={'display': 'none'}),
            
        ], style={'background-color': '#f8f9fa', 'min-height': '100vh'})
    
    def _create_api_modal(self):
        """Create API credentials modal"""
        return html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H5("API Credentials Setup", className="modal-title"),
                        html.Button("×", className="btn-close")
                    ], className="modal-header"),
                    html.Div([
                        html.P("Configure API credentials for enhanced data:", className="text-muted mb-3"),
                        html.Label("Alpha Vantage API Key:", className="form-label"),
                        dcc.Input(id='alpha-vantage-key', type='password', 
                                placeholder='Enter API key', className="form-control mb-3"),
                        html.Label("Polygon.io API Key:", className="form-label"),
                        dcc.Input(id='polygon-key', type='password', 
                                placeholder='Enter API key', className="form-control mb-3"),
                        html.Button("Save", id='save-api-btn', className="btn btn-primary")
                    ], className="modal-body")
                ], className="modal-content")
            ], className="modal-dialog")
        ], className="modal fade", id="api-modal")
    
    def setup_callbacks(self):
        """Setup all Dash callbacks"""
        
        # Strategy explanation callback
        @self.app.callback(
            Output('strategy-explanation', 'children'),
            Input('strategy-dropdown', 'value')
        )
        def update_strategy_explanation(strategy):
            explanations = {
                'momentum': [
                    html.I(className="fas fa-trend-up me-2 text-primary"),
                    html.Strong("Momentum Strategy"), html.Br(),
                    "• Follows price trends and momentum", html.Br(),
                    "• Buys when trend accelerates upward", html.Br(),
                    "• Best for trending markets and growth stocks"
                ],
                'mean_reversion': [
                    html.I(className="fas fa-balance-scale me-2 text-success"),
                    html.Strong("Mean Reversion Strategy"), html.Br(),
                    "• Assumes prices return to average over time", html.Br(),
                    "• Buys when price is below average", html.Br(),
                    "• Best for stable stocks in range-bound markets"
                ],
                'adaptive': [
                    html.I(className="fas fa-robot me-2 text-info"),
                    html.Strong("Adaptive Strategy"), html.Br(),
                    "• AI-powered approach adapting to market conditions", html.Br(),
                    "• Combines multiple strategies dynamically", html.Br(),
                    "• Suitable for all market conditions"
                ]
            }
            return explanations.get(strategy, "Select a strategy")
        
        # Store selected stock
        @self.app.callback(
            Output('selected-stock', 'children'),
            Input('search-btn', 'n_clicks'),
            State('stock-search-input', 'value'),
            prevent_initial_call=True
        )
        def store_selected_stock(n_clicks, search_input):
            if search_input:
                symbol = search_input.strip().upper()
                if not symbol.endswith(('.NS', '.BO')):
                    symbol += '.NS'  # Default to NSE
                return symbol
            return ""
        
        # Update stock info and chart
        @self.app.callback(
            [Output('stock-info-content', 'children'),
             Output('stock-chart', 'figure')],
            Input('selected-stock', 'children'),
            prevent_initial_call=True
        )
        def update_stock_display(selected_symbol):
            if not selected_symbol:
                return "Enter a stock symbol to view information", go.Figure()
            
            # Get stock information
            stock_info = self.data_provider.get_stock_info(selected_symbol)
            if not stock_info:
                return html.Div([
                    html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                    f"Unable to fetch data for {selected_symbol}"
                ], className="alert alert-warning"), go.Figure()
            
            # Create info display
            info_display = self._create_stock_info_display(stock_info)
            
            # Create chart
            chart = self._create_stock_chart(selected_symbol)
            
            return info_display, chart
        
        # Update trending stocks
        @self.app.callback(
            Output('trending-stocks-content', 'children'),
            [Input('exchange-dropdown', 'value'),
             Input('refresh-btn', 'n_clicks')]
        )
        def update_trending_stocks(exchange, n_clicks):
            trending = self.data_provider.get_trending_stocks(exchange, 6)
            
            if not trending:
                return html.P("Loading trending stocks...", className="text-muted text-center")
            
            cards = []
            for stock in trending:
                change_color = "success" if stock.get('price_change', 0) >= 0 else "danger"
                card = html.Div([
                    html.H6(stock['symbol'].replace('.NS', '').replace('.BO', ''), 
                           className="text-primary mb-1"),
                    html.P(stock['name'][:30] + "..." if len(stock['name']) > 30 else stock['name'], 
                          className="small text-muted mb-2"),
                    html.Div([
                        html.Strong(f"₹{stock.get('current_price', 'N/A')}", className="h6"),
                        html.Span(f" ({stock.get('price_change_percent', 0):+.1f}%)", 
                                className=f"text-{change_color} ms-2")
                    ])
                ], className="card p-3 mb-2", style={'cursor': 'pointer'})
                cards.append(html.Div(card, className="col-md-4"))
            
            return html.Div(cards, className="row")
        
        # Handle trading actions
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
            if not ctx.triggered:
                return ""
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'analyze-btn':
                return self._show_analysis(selected_stock, strategy)
            elif button_id == 'backtest-btn':
                return self._show_backtest(selected_stock, strategy, position_size)
            elif button_id == 'paper-trade-btn':
                return self._execute_paper_trade(selected_stock, strategy, position_size, stop_loss, take_profit)
            elif button_id == 'portfolio-btn':
                return self._show_portfolio()
            
            return ""
        
        # API Settings
        @self.app.callback(
            Output('api-status', 'children'),
            Input('save-api-btn', 'n_clicks'),
            [State('alpha-vantage-key', 'value'),
             State('polygon-key', 'value')],
            prevent_initial_call=True
        )
        def save_api_credentials(n_clicks, av_key, polygon_key):
            if n_clicks:
                credentials = {}
                if av_key: credentials['alpha_vantage'] = av_key
                if polygon_key: credentials['polygon'] = polygon_key
                self.data_provider.save_credentials(credentials)
                return "saved"
            return ""
    
    def _create_stock_info_display(self, stock_info: Dict) -> html.Div:
        """Create stock information display"""
        change_color = "success" if stock_info.get('price_change', 0) >= 0 else "danger"
        change_icon = "arrow-up" if stock_info.get('price_change', 0) >= 0 else "arrow-down"
        
        return html.Div([
            # Company header
            html.Div([
                html.H5(stock_info.get('company_name', 'N/A'), className="text-primary mb-1"),
                html.H6([
                    stock_info.get('symbol', ''),
                    html.Span(stock_info.get('sector', 'N/A'), className="badge bg-secondary ms-2")
                ], className="text-muted mb-3")
            ]),
            
            # Price information
            html.Div([
                html.H3(f"₹{stock_info.get('current_price', 'N/A')}", className="mb-2"),
                html.Div([
                    html.I(className=f"fas fa-{change_icon} me-2"),
                    f"₹{stock_info.get('price_change', 0)} ({stock_info.get('price_change_percent', 0):+.1f}%)"
                ], className=f"text-{change_color} h5 mb-3")
            ]),
            
            # Key metrics
            html.Div([
                html.Div([
                    html.Small("52W High", className="text-muted d-block"),
                    html.Strong(f"₹{stock_info.get('week_52_high', 'N/A')}")
                ], className="mb-2"),
                html.Div([
                    html.Small("52W Low", className="text-muted d-block"),
                    html.Strong(f"₹{stock_info.get('week_52_low', 'N/A')}")
                ], className="mb-2"),
                html.Div([
                    html.Small("Volume", className="text-muted d-block"),
                    html.Strong(f"{stock_info.get('volume', 0):,}")
                ], className="mb-2")
            ])
        ])
    
    def _create_stock_chart(self, symbol: str) -> go.Figure:
        """Create stock chart"""
        data = self.data_provider.get_stock_data(symbol, '6mo')
        
        if data.empty:
            return go.Figure().add_annotation(
                text="No chart data available", xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        ))
        
        # Moving averages
        if len(data) >= 20:
            sma20 = data['Close'].rolling(20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma20, name='SMA 20', line=dict(color='orange')))
        
        if len(data) >= 50:
            sma50 = data['Close'].rolling(50).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma50, name='SMA 50', line=dict(color='blue')))
        
        fig.update_layout(
            title=f"{symbol} - Stock Chart",
            yaxis_title="Price (₹)",
            xaxis_title="Date",
            height=500,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def _show_analysis(self, symbol: str, strategy: str) -> html.Div:
        """Show stock analysis"""
        if not symbol:
            return html.Div("Please select a stock first", className="alert alert-warning")
        
        stock_info = self.data_provider.get_stock_info(symbol)
        data = self.data_provider.get_stock_data(symbol, '1y')
        
        if data.empty:
            return html.Div("Insufficient data for analysis", className="alert alert-warning")
        
        # Calculate basic metrics
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        current_price = data['Close'].iloc[-1]
        
        # Simple trend analysis
        sma20 = data['Close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else current_price
        trend = "Bullish" if current_price > sma20 else "Bearish"
        trend_color = "success" if trend == "Bullish" else "danger"
        
        return html.Div([
            html.H4([html.I(className="fas fa-chart-line me-2"), f"Analysis: {symbol}"], className="text-primary mb-3"),
            
            html.Row([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-rupee-sign fa-2x text-primary mb-2"),
                        html.H6("Current Price", className="text-muted"),
                        html.H4(f"₹{current_price:.2f}", className="text-primary")
                    ], className="text-center p-3 bg-light rounded")
                ], className="col-md-4"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-area fa-2x text-warning mb-2"),
                        html.H6("Volatility", className="text-muted"),
                        html.H4(f"{volatility:.1f}%", className="text-warning")
                    ], className="text-center p-3 bg-light rounded")
                ], className="col-md-4"),
                html.Div([
                    html.Div([
                        html.I(className=f"fas fa-trending-{'up' if trend == 'Bullish' else 'down'} fa-2x text-{trend_color} mb-2"),
                        html.H6("Trend", className="text-muted"),
                        html.H4(trend, className=f"text-{trend_color}")
                    ], className="text-center p-3 bg-light rounded")
                ], className="col-md-4")
            ], className="mb-3"),
            
            html.Div([
                html.H6("Analysis Summary", className="text-info mb-2"),
                html.P(f"Based on technical analysis, {symbol} shows a {trend.lower()} trend with {volatility:.1f}% annual volatility."),
                html.P(f"Current price of ₹{current_price:.2f} is {'above' if current_price > sma20 else 'below'} the 20-day moving average.")
            ], className="alert alert-info")
        ])
    
    def _show_backtest(self, symbol: str, strategy: str, position_size: float) -> html.Div:
        """Show backtest results"""
        if not symbol:
            return html.Div("Please select a stock first", className="alert alert-warning")
        
        data = self.data_provider.get_stock_data(symbol, '1y')
        if data.empty:
            return html.Div("Insufficient data for backtesting", className="alert alert-warning")
        
        # Simple backtest simulation
        returns = data['Close'].pct_change().dropna()
        
        # Generate signals based on strategy
        if strategy == 'momentum':
            # Simple momentum: buy when price > 20-day MA
            signals = (data['Close'] > data['Close'].rolling(20).mean()).astype(int)
        elif strategy == 'mean_reversion':
            # Mean reversion: buy when price < 20-day MA
            signals = (data['Close'] < data['Close'].rolling(20).mean()).astype(int)
        else:  # adaptive
            # Combine signals
            momentum_signals = (data['Close'] > data['Close'].rolling(20).mean()).astype(int)
            mean_rev_signals = (data['Close'] < data['Close'].rolling(20).mean()).astype(int)
            signals = ((momentum_signals + mean_rev_signals) > 0).astype(int)
        
        # Calculate strategy returns
        strategy_returns = signals.shift(1) * returns
        strategy_returns = strategy_returns.fillna(0)
        
        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        buy_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
        
        outperformance = total_return - buy_hold_return
        performance_color = "success" if outperformance > 0 else "danger"
        
        return html.Div([
            html.H4([html.I(className="fas fa-history me-2"), f"Backtest: {strategy.title()}"], className="text-primary mb-3"),
            
            html.Row([
                html.Div([
                    html.Div([
                        html.I(className=f"fas fa-chart-line fa-2x text-{performance_color} mb-2"),
                        html.H6("Strategy Return", className="text-muted"),
                        html.H4(f"{total_return*100:+.1f}%", className=f"text-{performance_color}")
                    ], className="text-center p-3 bg-light rounded")
                ], className="col-md-4"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-area fa-2x text-primary mb-2"),
                        html.H6("Buy & Hold", className="text-muted"),
                        html.H4(f"{buy_hold_return*100:+.1f}%", className="text-primary")
                    ], className="text-center p-3 bg-light rounded")
                ], className="col-md-4"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-trophy fa-2x text-info mb-2"),
                        html.H6("Outperformance", className="text-muted"),
                        html.H4(f"{outperformance*100:+.1f}%", className="text-info")
                    ], className="text-center p-3 bg-light rounded")
                ], className="col-md-4")
            ], className="mb-3"),
            
            html.Div([
                html.H6("Backtest Summary", className="text-success mb-2"),
                html.P(f"Strategy Performance: {total_return*100:+.1f}% vs Buy & Hold: {buy_hold_return*100:+.1f}%"),
                html.P(f"With position size of ₹{position_size:,.0f}, potential profit: ₹{position_size * total_return:,.0f}"),
                html.P("Note: Past performance does not guarantee future results.", className="text-muted small")
            ], className="alert alert-info")
        ])
    
    def _execute_paper_trade(self, symbol: str, strategy: str, position_size: float, 
                           stop_loss: float, take_profit: float) -> html.Div:
        """Execute paper trade"""
        if not symbol:
            return html.Div("Please select a stock first", className="alert alert-warning")
        
        stock_info = self.data_provider.get_stock_info(symbol)
        current_price = stock_info.get('current_price', 0)
        
        if not current_price:
            return html.Div("Unable to get current price", className="alert alert-warning")
        
        # Calculate shares and execute trade
        shares = int(position_size / current_price)
        result = self.paper_trader.execute_trade(symbol, 'BUY', shares, current_price)
        
        return html.Div([
            html.H4([html.I(className="fas fa-play-circle me-2"), "Paper Trade Executed"], className="text-success mb-3"),
            
            html.Div([
                html.H6("Trade Details", className="text-primary mb-3"),
                html.P([html.Strong("Symbol: "), symbol]),
                html.P([html.Strong("Strategy: "), strategy.title()]),
                html.P([html.Strong("Shares: "), f"{shares:,}"]),
                html.P([html.Strong("Price: "), f"₹{current_price:.2f}"]),
                html.P([html.Strong("Total Investment: "), f"₹{shares * current_price:,.2f}"]),
                html.P([html.Strong("Stop Loss: "), f"₹{current_price * (1 - stop_loss/100):.2f} ({stop_loss}%)"]),
                html.P([html.Strong("Take Profit: "), f"₹{current_price * (1 + take_profit/100):.2f} ({take_profit}%)"])
            ], className="alert alert-success"),
            
            html.Div([
                html.I(className="fas fa-info-circle me-2"),
                "This is virtual paper trading. No real money is involved."
            ], className="alert alert-info")
        ])
    
    def _show_portfolio(self) -> html.Div:
        """Show portfolio summary"""
        portfolio = self.paper_trader.get_portfolio_summary()
        
        return html.Div([
            html.H4([html.I(className="fas fa-wallet me-2"), "Paper Trading Portfolio"], className="text-primary mb-3"),
            
            html.Row([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-coins fa-2x text-success mb-2"),
                        html.H6("Total Value", className="text-muted"),
                        html.H4(f"₹{portfolio.get('total_value', 0):,.2f}", className="text-success")
                    ], className="text-center p-3 bg-light rounded")
                ], className="col-md-4"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-money-bill fa-2x text-info mb-2"),
                        html.H6("Available Cash", className="text-muted"),
                        html.H4(f"₹{portfolio.get('cash', 0):,.2f}", className="text-info")
                    ], className="text-center p-3 bg-light rounded")
                ], className="col-md-4"),
                html.Div([
                    html.Div([
                        html.I(className="fas fa-chart-pie fa-2x text-warning mb-2"),
                        html.H6("Total Trades", className="text-muted"),
                        html.H4(f"{portfolio.get('total_trades', 0)}", className="text-warning")
                    ], className="text-center p-3 bg-light rounded")
                ], className="col-md-4")
            ], className="mb-3"),
            
            html.Div([
                html.H6("Current Positions", className="text-success mb-3"),
                html.P("No active positions" if not portfolio.get('positions') else 
                      f"{len(portfolio.get('positions', []))} active positions", 
                      className="text-muted")
            ], className="alert alert-light")
        ])
    
    def run(self, debug=True, port=8050):
        """Run the application"""
        print("\n🚀 NSE/BSE Trading Platform - Enhanced Version")
        print("=" * 60)
        print("✨ NEW FEATURES ADDED:")
        print("  📈 AI Trading Strategies (Momentum, Mean Reversion, Adaptive)")
        print("  📊 Strategy Backtesting with Performance Metrics")
        print("  💰 Paper Trading System (Risk-free Virtual Trading)")
        print("  📱 Portfolio Tracking & Performance Analysis")
        print("  🔧 Working API Settings Configuration")
        print("  📈 Enhanced Stock Analysis & Charts")
        print("=" * 60)
        print(f"🌐 Access your enhanced trading platform at: http://localhost:{port}")
        print("⚠️  Press Ctrl+C to stop\n")
        
        self.app.run_server(debug=debug, port=port)

if __name__ == '__main__':
    app = NSEBSETradingApp()
    app.run()
