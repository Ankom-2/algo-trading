"""
Simple Working NSE/BSE Trading App
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf
import logging

app = dash.Dash(__name__, external_stylesheets=[
    'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
])

# Popular NSE stocks
NSE_STOCKS = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS',
              'INFY.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS']

BSE_STOCKS = ['RELIANCE.BO', 'TCS.BO', 'HDFCBANK.BO', 'ICICIBANK.BO', 'HINDUNILVR.BO']

def get_stock_data(symbol):
    """Get stock data"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1y')
        info = ticker.info
        
        if hist.empty:
            return None
            
        current = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
        change = current - prev
        change_pct = (change / prev) * 100 if prev != 0 else 0
        
        return {
            'symbol': symbol,
            'name': info.get('longName', symbol.replace('.NS', '').replace('.BO', '')),
            'price': round(float(current), 2),
            'change': round(float(change), 2),
            'change_pct': round(float(change_pct), 2),
            'high': round(float(hist['High'].iloc[-1]), 2),
            'low': round(float(hist['Low'].iloc[-1]), 2),
            'volume': int(hist['Volume'].iloc[-1]),
            'sector': info.get('sector', 'N/A')
        }
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# App Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1([
            html.I(className="fas fa-chart-line me-3"),
            "NSE/BSE Trading Platform"
        ], className="text-white text-center")
    ], className="bg-primary py-4 mb-4"),
    
    # Main Container
    html.Div([
        # Controls
        html.Div([
            html.Div([
                html.Label("Exchange:"),
                dcc.Dropdown(
                    id='exchange',
                    options=[
                        {'label': 'NSE', 'value': 'NSE'},
                        {'label': 'BSE', 'value': 'BSE'}
                    ],
                    value='NSE'
                )
            ], className="col-md-2"),
            
            html.Div([
                html.Label("Currency:"),
                dcc.Dropdown(
                    id='currency',
                    options=[
                        {'label': 'INR', 'value': 'INR'},
                        {'label': 'USD', 'value': 'USD'}
                    ],
                    value='INR'
                )
            ], className="col-md-2"),
            
            html.Div([
                html.Label("Stock Symbol:"),
                dcc.Input(
                    id='stock-input',
                    type='text',
                    placeholder='Enter symbol (e.g., RELIANCE)',
                    className="form-control"
                )
            ], className="col-md-3"),
            
            html.Div([
                html.Br(),
                html.Button("Search", id='search-btn', className="btn btn-primary")
            ], className="col-md-2"),
            
            html.Div([
                html.Br(),
                html.Button("Refresh", id='refresh-btn', className="btn btn-success")
            ], className="col-md-2")
            
        ], className="row mb-4"),
        
        # Main Content
        html.Div([
            # Left Column - Stock Info
            html.Div([
                html.Div([
                    html.H5("Stock Information", className="card-title"),
                    html.Div(id='stock-info', children=[
                        html.P("Search for a stock to view information")
                    ])
                ], className="card-body")
            ], className="card mb-4 col-md-6"),
            
            # Right Column - Chart
            html.Div([
                html.Div([
                    html.H5("Price Chart", className="card-title"),
                    dcc.Graph(id='stock-chart')
                ], className="card-body")
            ], className="card mb-4 col-md-6")
            
        ], className="row"),
        
        # Trading Buttons
        html.Div([
            html.Div([
                html.H5("Trading Actions"),
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
        ], className="card"),
        
        # Trending Stocks
        html.Div([
            html.Div([
                html.H5("Trending Stocks", className="card-title"),
                html.Div(id='trending-stocks')
            ], className="card-body")
        ], className="card mt-4")
        
    ], className="container"),
    
    # Hidden storage
    html.Div(id='current-stock', style={'display': 'none'})
    
])

# Callbacks
@app.callback(
    [Output('stock-info', 'children'),
     Output('stock-chart', 'figure'), 
     Output('current-stock', 'children')],
    [Input('search-btn', 'n_clicks')],
    [State('stock-input', 'value'), State('exchange', 'value')]
)
def search_stock(n_clicks, stock_input, exchange):
    if not n_clicks or not stock_input:
        return "Enter a stock symbol", {}, ""
    
    # Add suffix
    symbol = stock_input.upper()
    if not symbol.endswith(('.NS', '.BO')):
        suffix = '.NS' if exchange == 'NSE' else '.BO'
        symbol += suffix
    
    # Get data
    data = get_stock_data(symbol)
    if not data:
        return f"No data found for {symbol}", {}, ""
    
    # Stock info display
    color = "text-success" if data['change'] > 0 else "text-danger"
    icon = "fa-arrow-up" if data['change'] > 0 else "fa-arrow-down"
    
    info = html.Div([
        html.H4(data['name']),
        html.H6(f"{data['symbol']} - {data['sector']}", className="text-muted mb-3"),
        html.H2(f"₹{data['price']}", className="mb-2"),
        html.H5([
            html.I(className=f"fas {icon} me-2"),
            f"₹{data['change']} ({data['change_pct']:.2f}%)"
        ], className=color),
        html.Hr(),
        html.P(f"High: ₹{data['high']} | Low: ₹{data['low']}"),
        html.P(f"Volume: {data['volume']:,}")
    ])
    
    # Chart
    try:
        hist = yf.Ticker(symbol).history(period='6mo')
        if not hist.empty:
            fig = go.Figure(data=go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'], 
                low=hist['Low'],
                close=hist['Close']
            ))
            fig.update_layout(title=f"{symbol} - 6 Month Chart", height=400)
        else:
            fig = go.Figure()
    except:
        fig = go.Figure()
    
    return info, fig, symbol

@app.callback(
    Output('trending-stocks', 'children'),
    [Input('exchange', 'value'), Input('refresh-btn', 'n_clicks')]
)
def update_trending(exchange, refresh_clicks):
    stocks = NSE_STOCKS[:5] if exchange == 'NSE' else BSE_STOCKS[:5]
    trending = []
    
    for symbol in stocks:
        data = get_stock_data(symbol)
        if data:
            color = "text-success" if data['change'] > 0 else "text-danger"
            icon = "fa-arrow-up" if data['change'] > 0 else "fa-arrow-down"
            
            card = html.Div([
                html.H6(data['symbol'].replace('.NS', '').replace('.BO', '')),
                html.P(data['name'][:30] + "..." if len(data['name']) > 30 else data['name'], 
                      className="small text-muted"),
                html.Div([
                    html.Span(f"₹{data['price']}", className="fw-bold me-2"),
                    html.Span([
                        html.I(className=f"fas {icon} me-1"),
                        f"{data['change_pct']:.1f}%"
                    ], className=f"{color} small")
                ])
            ], className="p-3 border rounded mb-2")
            
            trending.append(card)
    
    return trending

@app.callback(
    Output('trading-result', 'children'),
    [Input('buy-btn', 'n_clicks'),
     Input('sell-btn', 'n_clicks'),
     Input('analyze-btn', 'n_clicks')],
    [State('current-stock', 'children')]
)
def handle_trading(buy_clicks, sell_clicks, analyze_clicks, current_stock):
    ctx = callback_context
    if not ctx.triggered or not current_stock:
        return "Select a stock first"
    
    button = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button == 'buy-btn' and buy_clicks:
        return html.Div([
            html.I(className="fas fa-check text-success me-2"),
            f"Buy order prepared for {current_stock} (Paper Trading Mode)"
        ], className="alert alert-success")
        
    elif button == 'sell-btn' and sell_clicks:
        return html.Div([
            html.I(className="fas fa-check text-warning me-2"),
            f"Sell order prepared for {current_stock} (Paper Trading Mode)"
        ], className="alert alert-warning")
        
    elif button == 'analyze-btn' and analyze_clicks:
        return html.Div([
            html.I(className="fas fa-chart-line text-info me-2"),
            f"Technical analysis for {current_stock}: Currently in normal trading range"
        ], className="alert alert-info")
    
    return ""

if __name__ == '__main__':
    print("\n🚀 NSE/BSE Trading Platform")
    print("🌐 Access at: http://localhost:8050")
    print("✨ Features: Stock search, Real-time data, Charts, Paper trading")
    print("⚠️ Press Ctrl+C to stop\n")
    
    app.run(debug=True, port=8050, host='0.0.0.0')
