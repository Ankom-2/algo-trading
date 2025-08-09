"""
Advanced Configuration for World-Class Algorithmic Trading System
Optimized for maximum profitability with minimal risk
"""
import os
from typing import Dict, Any

# Trading configuration optimized for consistent profitability
TRADING_CONFIG = {
    # Data sources and paths
    'data': {
        'raw_path': 'data/raw/',
        'processed_path': 'data/processed/',
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMZN', 'META', 'SPY', 'QQQ'],
        'timeframes': ['1m', '5m', '15m', '1h', '1d'],
        'lookback_days': 252,  # 1 year of trading data
    },
    
    # Advanced risk management for consistent profits
    'risk_management': {
        'max_portfolio_risk': 0.02,  # 2% max portfolio risk per trade
        'max_position_size': 0.10,   # 10% max position size
        'max_drawdown': 0.05,        # 5% max drawdown before halt
        'stop_loss_multiplier': 1.5, # Dynamic stop loss
        'take_profit_ratio': 3.0,    # 3:1 reward to risk ratio
        'risk_free_rate': 0.045,     # Current risk-free rate
        'volatility_lookback': 20,   # Days for volatility calculation
        'correlation_threshold': 0.7, # Max correlation between positions
    },
    
    # Strategy parameters optimized through extensive backtesting
    'strategies': {
        'momentum': {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bollinger_period': 20,
            'bollinger_std': 2,
        },
        'mean_reversion': {
            'lookback_period': 20,
            'entry_threshold': 2.0,  # Standard deviations
            'exit_threshold': 0.5,
            'min_volume_ratio': 1.5,
        },
        'adaptive': {
            'regime_detection_period': 50,
            'volatility_threshold': 0.02,
            'trend_strength_period': 14,
            'adaptation_speed': 0.1,
        }
    },
    
    # Execution parameters for optimal fills
    'execution': {
        'paper_trading': True,
        'slippage_model': 'linear',
        'slippage_rate': 0.0005,     # 5 bps
        'commission_rate': 0.0001,   # 1 bp
        'min_trade_amount': 1000,    # $1000 minimum
        'max_orders_per_second': 10,
        'order_timeout': 30,         # seconds
    },
    
    # Backtesting configuration
    'backtest': {
        'start_date': '2020-01-01',
        'end_date': '2024-12-31',
        'initial_capital': 1000000,  # $1M starting capital
        'benchmark': 'SPY',
        'rebalance_frequency': 'daily',
        'transaction_costs': True,
    },
    
    # Performance targets
    'targets': {
        'annual_return': 0.25,       # 25% target return
        'max_drawdown': 0.05,        # 5% max drawdown
        'sharpe_ratio': 2.0,         # Target Sharpe ratio
        'win_rate': 0.65,            # 65% win rate target
        'profit_factor': 2.5,        # Profit factor target
    },
    
    # Logging and monitoring
    'logging': {
        'level': 'INFO',
        'file': 'logs/trading.log',
        'max_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
    },
    
    # API configurations (for real trading)
    'api': {
        'provider': 'alpaca',  # or 'interactive_brokers', 'td_ameritrade', 'zerodha', 'iifl'
        'paper_trading': True,
        'rate_limit': 200,  # requests per minute
        
        # Indian Market Brokers
        'zerodha': {
            'api_key': '',  # To be set via UI
            'api_secret': '',
            'request_token': '',
            'access_token': '',
            'base_url': 'https://api.kite.trade',
            'instruments_url': 'https://api.kite.trade/instruments',
            'rate_limit': 3,  # requests per second
        },
        'iifl': {
            'api_key': '',  # To be set via UI
            'api_secret': '',
            'client_id': '',
            'password': '',
            'base_url': 'https://ttblaze.iifl.com/interactive',
            'rate_limit': 10,  # requests per second
        }
    },
    
    # Indian Market Configuration
    'indian_markets': {
        'exchanges': ['NSE', 'BSE'],
        'indices': ['NIFTY 50', 'NIFTY BANK', 'SENSEX', 'NIFTY IT'],
        'common_symbols': [
            'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR',
            'ICICIBANK', 'ITC', 'KOTAKBANK', 'LT', 'SBIN',
            'BHARTIARTL', 'ASIANPAINT', 'AXISBANK', 'MARUTI', 'HCLTECH'
        ],
        'option_symbols': [
            'NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'HDFCBANK'
        ],
        'trading_hours': {
            'start': '09:15',
            'end': '15:30',
            'timezone': 'Asia/Kolkata'
        }
    },
    
    # Options Trading Configuration
    'options': {
        'strategies': {
            'long_straddle': {
                'description': 'Buy ATM call and put options',
                'max_loss': 'Limited to premium paid',
                'max_profit': 'Unlimited',
                'breakeven': 'Strike ± premium paid',
                'volatility_requirement': 'high',
                'time_decay_impact': 'negative'
            },
            'long_strangle': {
                'description': 'Buy OTM call and put options',
                'max_loss': 'Limited to premium paid',
                'max_profit': 'Unlimited',
                'breakeven': 'Call strike + premium, Put strike - premium',
                'volatility_requirement': 'high',
                'time_decay_impact': 'negative'
            },
            'short_straddle': {
                'description': 'Sell ATM call and put options',
                'max_loss': 'Unlimited',
                'max_profit': 'Limited to premium received',
                'volatility_requirement': 'low',
                'time_decay_impact': 'positive'
            },
            'iron_condor': {
                'description': 'Sell call spread and put spread',
                'max_loss': 'Limited',
                'max_profit': 'Limited to net premium',
                'volatility_requirement': 'low',
                'time_decay_impact': 'positive'
            }
        },
        'default_params': {
            'dte_range': [7, 45],  # Days to expiration range
            'delta_range': [0.15, 0.85],  # Delta range for option selection
            'min_open_interest': 100,
            'min_volume': 10,
            'max_bid_ask_spread': 0.1
        }
    },
    
    # Strategy Optimization Parameters
    'strategy_optimization': {
        'enabled': True,
        'auto_select_strategy': True,
        'evaluation_period': 30,  # Days
        'min_trades_for_evaluation': 10,
        'performance_weights': {
            'sharpe_ratio': 0.3,
            'total_return': 0.25,
            'max_drawdown': 0.2,
            'win_rate': 0.15,
            'profit_factor': 0.1
        },
        'strategy_switching_cooldown': 5  # Days before switching strategies
    }
}
