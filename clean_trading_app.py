"""
Clean NSE/BSE Trading Application
Simplified, working version with all features
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleNSEBSETrader:
    """Simple, clean NSE/BSE Trading Application"""
    
    def __init__(self):
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
        ])
        
        # Popular NSE stocks
        self.nse_stocks = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
            'INFY.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
            'LT.NS', 'HCLTECH.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'AXISBANK.NS',
            'TITAN.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS'
        ]
        
        # Popular BSE stocks  
        self.bse_stocks = [
            'RELIANCE.BO', 'TCS.BO', 'HDFCBANK.BO', 'ICICIBANK.BO', 'HINDUNILVR.BO'
        ]
        
        self.selected_stock = ""
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup application layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1([
                    html.I(className="fas fa-chart-line me-3"),
                    "NSE/BSE Trading Platform"
                ], className="text-white mb-0")
            ], className="bg-primary py-3 px-4 mb-4"),
            
            html.Div([
                # Controls Row
                html.Div([
                    html.Div([
                        html.Label("Exchange:", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='exchange-select',
                            options=[
                                {'label': 'NSE (National Stock Exchange)', 'value': 'NSE'},
                                {'label': 'BSE (Bombay Stock Exchange)', 'value': 'BSE'}
                            ],
                            value='NSE',
                            className="mb-3"
                        )
                    ], className="col-md-3"),
                    
                    html.Div([
                        html.Label("Currency:", className="form-label fw-bold"),
                        dcc.Dropdown(
                            id='currency-select',
                            options=[
                                {'label': 'INR (Indian Rupee)', 'value': 'INR'},
                                {'label': 'USD (US Dollar)', 'value': 'USD'}
                            ],
                            value='INR',
                            className="mb-3"
                        )
                    ], className="col-md-3"),
                    
                    html.Div([
                        html.Label("Search Stock:", className="form-label fw-bold"),
                        html.Div([
                            dcc.Input(
                                id='stock-input',
                                type='text',
                                placeholder='Enter symbol (e.g., RELIANCE, TCS)',
                                className="form-control me-2",
                                style={'display': 'inline-block', 'width': '70%'}
                            ),
                            html.Button([
                                html.I(className="fas fa-search")
                            ], id='search-btn', className="btn btn-primary")
                        ], className="d-flex")
                    ], className="col-md-4"),
                    
                    html.Div([
                        html.Button([
                            html.I(className="fas fa-cog me-2"),
                            "API Settings"
                        ], id='api-btn', className="btn btn-outline-primary me-2"),
                        html.Button([
                            html.I(className="fas fa-refresh me-2"),
                            "Refresh"
                        ], id='refresh-btn', className="btn btn-success")
                    ], className="col-md-2 d-flex align-items-end")
                    
                ], className="row mb-4"),
                
                # Main Content Row
                html.Div([
                    # Stock Info Column
                    html.Div([
                        html.Div([
                            html.Div([
                                html.H5([
                                    html.I(className="fas fa-info-circle me-2"),
                                    "Stock Information"
                                ], className="card-title text-primary"),
                                html.Div(id='stock-info', children=[
                                    html.P("Search for a stock to view information", className="text-muted text-center")
                                ])
                            ], className="card-body"),
                            
                            # Trading Actions
                            html.Div([
                                html.H6("Trading Actions", className="text-success mb-3"),
                                html.Div([
                                    html.Button([
                                        html.I(className="fas fa-arrow-up me-2"),
                                        "BUY"
                                    ], id='buy-btn', className="btn btn-success btn-lg me-2"),
                                    html.Button([
                                        html.I(className="fas fa-arrow-down me-2"),
                                        "SELL"
                                    ], id='sell-btn', className="btn btn-danger btn-lg me-2"),
                                    html.Button([
                                        html.I(className="fas fa-chart-bar me-2"),
                                        "ANALYZE"
                                    ], id='analyze-btn', className="btn btn-info btn-lg")
                                ], className="text-center mb-3"),
                                html.Div(id='trading-result')
                            ], className="card-body")
                        ], className="card")
                    ], className="col-md-6"),
                    
                    # Chart and Trending Column
                    html.Div([
                        html.Div([
                            # Chart
                            html.Div([
                                html.H5([
                                    html.I(className="fas fa-chart-candlestick me-2"),
                                    "Price Chart"
                                ], className="card-title text-success"),
                                dcc.Graph(id='price-chart', style={'height': '400px'})
                            ], className="card-body"),
                            
                            # Trending Stocks
                            html.Div([
                                html.H5([
                                    html.I(className="fas fa-fire me-2"),
                                    "Trending Stocks"
                                ], className="card-title text-danger"),
                                html.Div(id='trending-stocks')
                            ], className="card-body")
                        ], className="card")
                    ], className="col-md-6")
                ], className="row")
            ], className="container-fluid"),
            
            # Hidden divs for data storage
            html.Div(id='current-stock', style={'display': 'none'}),
            html.Div(id='api-credentials', style={'display': 'none'})
            
        ], style={'background-color': '#f8f9fa', 'min-height': '100vh'})

    def get_stock_data(self, symbol: str) -> Dict:
        """Get comprehensive stock data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='1y')
            
            if hist.empty:
                return {}
            
            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
            
            # 52-week high/low
            week_52_high = hist['High'].max()
            week_52_low = hist['Low'].min()
            
            return {
                'symbol': symbol,
                'name': info.get('longName', symbol.replace('.NS', '').replace('.BO', '')),
                'current_price': round(float(current_price), 2),
                'previous_close': round(float(prev_close), 2),
                'change': round(float(change), 2),
                'change_percent': round(float(change_pct), 2),
                'today_high': round(float(hist['High'].iloc[-1]), 2),
                'today_low': round(float(hist['Low'].iloc[-1]), 2),
                'week_52_high': round(float(week_52_high), 2),
                'week_52_low': round(float(week_52_low), 2),
                'volume': int(hist['Volume'].iloc[-1]),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 'N/A',
                'sector': info.get('sector', 'N/A'),
                'currency': 'INR'
            }
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return {}
    
    def get_trending_stocks(self, exchange: str = 'NSE') -> List[Dict]:
        """Get trending stocks"""
        stocks = self.nse_stocks if exchange == 'NSE' else self.bse_stocks
        trending = []
        
        for symbol in stocks[:5]:  # Get top 5 for performance
            data = self.get_stock_data(symbol)
            if data:
                trending.append(data)
        
        # Sort by absolute change percentage
        trending.sort(key=lambda x: abs(x.get('change_percent', 0)), reverse=True)
        return trending[:3]  # Top 3 trending
    
    def setup_callbacks(self):
        """Setup all callbacks"""
        
        @self.app.callback(
            [Output('stock-info', 'children'),
             Output('price-chart', 'figure'),
             Output('current-stock', 'children')],
            [Input('search-btn', 'n_clicks')],
            [State('stock-input', 'value'),
             State('exchange-select', 'value')]
        )
        def search_stock(n_clicks, stock_input, exchange):
            """Search and display stock information"""
            if not n_clicks or not stock_input:
                return "Enter a stock symbol to search", {}, ""
            
            # Add exchange suffix if not present
            symbol = stock_input.upper()
            if not symbol.endswith(('.NS', '.BO')):
                suffix = '.NS' if exchange == 'NSE' else '.BO'
                symbol += suffix
            
            # Get stock data
            stock_data = self.get_stock_data(symbol)
            if not stock_data:
                return html.Div([
                    html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                    f"Unable to find data for {symbol}. Please check the symbol."
                ], className="alert alert-warning"), {}, ""
            
            # Create stock info display
            change_color = "text-success" if stock_data['change'] > 0 else "text-danger"
            change_icon = "fa-arrow-up" if stock_data['change'] > 0 else "fa-arrow-down"
            
            stock_info = html.Div([
                # Company header
                html.Div([
                    html.H4(stock_data['name'], className="mb-1"),
                    html.H6([
                        stock_data['symbol'],
                        html.Span(stock_data['sector'], className="badge bg-secondary ms-2")
                    ], className="text-muted mb-3")
                ]),
                
                # Price info
                html.Div([
                    html.H2(f"₹{stock_data['current_price']}", className="mb-1"),
                    html.Div([
                        html.I(className=f"fas {change_icon} me-2"),
                        f"₹{stock_data['change']} ({stock_data['change_percent']}%)"
                    ], className=f"{change_color} h5 mb-3")
                ]),
                
                # Key metrics
                html.Div([
                    html.Div([
                        html.Small("Today's Range", className="text-muted d-block"),
                        html.Strong(f"₹{stock_data['today_low']} - ₹{stock_data['today_high']}")
                    ], className="mb-2"),
                    html.Div([
                        html.Small("52 Week Range", className="text-muted d-block"),
                        html.Strong(f"₹{stock_data['week_52_low']} - ₹{stock_data['week_52_high']}")
                    ], className="mb-2"),
                    html.Div([
                        html.Small("Volume", className="text-muted d-block"),
                        html.Strong(f"{stock_data['volume']:,}")
                    ], className="mb-2"),
                    html.Div([
                        html.Small("P/E Ratio", className="text-muted d-block"),
                        html.Strong(str(stock_data['pe_ratio']))
                    ])
                ])
            ])
            
            # Create price chart
            chart_data = yf.Ticker(symbol).history(period='6mo')
            if not chart_data.empty:
                fig = go.Figure(data=go.Candlestick(
                    x=chart_data.index,
                    open=chart_data['Open'],
                    high=chart_data['High'],
                    low=chart_data['Low'],
                    close=chart_data['Close'],
                    name=symbol
                ))
                fig.update_layout(
                    title=f"{symbol} - 6 Month Chart",
                    yaxis_title="Price (₹)",
                    height=400,
                    showlegend=False
                )
            else:
                fig = go.Figure()
                fig.add_annotation(
                    text="Chart data not available",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            return stock_info, fig, symbol
        
        @self.app.callback(
            Output('trending-stocks', 'children'),
            [Input('exchange-select', 'value'),
             Input('refresh-btn', 'n_clicks')]
        )
        def update_trending(exchange, refresh_clicks):
            """Update trending stocks"""
            trending = self.get_trending_stocks(exchange)
            
            if not trending:
                return html.P("Loading trending stocks...", className="text-muted")
            
            cards = []
            for stock in trending:
                change_color = "text-success" if stock['change'] > 0 else "text-danger"
                change_icon = "fa-arrow-up" if stock['change'] > 0 else "fa-arrow-down"
                
                card = html.Div([
                    html.H6(stock['symbol'].replace('.NS', '').replace('.BO', ''), 
                           className="fw-bold mb-1"),
                    html.P(stock['name'][:25] + "..." if len(stock['name']) > 25 else stock['name'], 
                          className="text-muted small mb-2"),
                    html.Div([
                        html.Span(f"₹{stock['current_price']}", className="fw-bold me-2"),
                        html.Span([
                            html.I(className=f"fas {change_icon} me-1"),
                            f"{stock['change_percent']:.1f}%"
                        ], className=f"{change_color} small")
                    ])
                ], className="border rounded p-2 mb-2 bg-white", style={'cursor': 'pointer'})
                
                cards.append(card)
            
            return cards
        
        @self.app.callback(
            Output('trading-result', 'children'),
            [Input('buy-btn', 'n_clicks'),
             Input('sell-btn', 'n_clicks'),
             Input('analyze-btn', 'n_clicks')],
            [State('current-stock', 'children')]
        )
        def handle_trading(buy_clicks, sell_clicks, analyze_clicks, current_stock):
            """Handle trading buttons"""
            ctx = callback_context
            if not ctx.triggered or not current_stock:
                return html.P("Select a stock first", className="text-muted")
            
            button = ctx.triggered[0]['prop_id'].split('.')[0]
            stock_data = self.get_stock_data(current_stock) if current_stock else {}
            
            if button == 'buy-btn' and buy_clicks:
                return html.Div([
                    html.I(className="fas fa-shopping-cart text-success me-2"),
                    html.Strong("Buy Order Ready"),
                    html.P(f"Stock: {current_stock} | Price: ₹{stock_data.get('current_price', 'N/A')}", 
                          className="mb-1"),
                    html.Small("Paper trading mode - No real money involved", className="text-muted")
                ], className="alert alert-success")
                
            elif button == 'sell-btn' and sell_clicks:
                return html.Div([
                    html.I(className="fas fa-hand-holding-usd text-danger me-2"),
                    html.Strong("Sell Order Ready"),
                    html.P(f"Stock: {current_stock} | Price: ₹{stock_data.get('current_price', 'N/A')}", 
                          className="mb-1"),
                    html.Small("Paper trading mode - No real money involved", className="text-muted")
                ], className="alert alert-warning")
                
            elif button == 'analyze-btn' and analyze_clicks:
                # Simple analysis
                price = stock_data.get('current_price', 0)
                week_52_high = stock_data.get('week_52_high', 0)
                week_52_low = stock_data.get('week_52_low', 0)
                
                if price and week_52_high and week_52_low:
                    position = (price - week_52_low) / (week_52_high - week_52_low) * 100
                    if position > 80:
                        analysis = "Near 52-week high - Consider taking profits"
                        color = "warning"
                    elif position < 20:
                        analysis = "Near 52-week low - Potential buying opportunity"
                        color = "info"
                    else:
                        analysis = "Trading in normal range"
                        color = "secondary"
                else:
                    analysis = "Insufficient data for analysis"
                    color = "secondary"
                
                return html.Div([
                    html.I(className="fas fa-chart-line text-info me-2"),
                    html.Strong("Technical Analysis"),
                    html.P(analysis, className="mb-1"),
                    html.Small("This is basic analysis. Consult financial advisors for investment decisions.", 
                              className="text-muted")
                ], className=f"alert alert-{color}")
            
            return ""
    
    def run(self, debug=True, port=8050):
        """Run the application"""
        print("\n🚀 NSE/BSE Trading Platform Starting...")
        print("✨ Features: Real-time data, Technical analysis, Paper trading")
        print(f"🌐 Access at: http://localhost:{port}")
        print("⚠️  Press Ctrl+C to stop\n")
        
        self.app.run(debug=debug, port=port, host='0.0.0.0')

if __name__ == '__main__':
    app = SimpleNSEBSETrader()
    app.run()
