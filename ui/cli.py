"""
Command Line Interface for Algorithmic Trading System
Simple CLI for testing strategies and running backtests
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import asyncio

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import AdvancedDataLoader
from utils.optimizer import PortfolioOptimizer
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.adaptive import AdaptiveStrategy
from backtest.engine import BacktestEngine
from execution.paper_trader import PaperTrader
import config

class AlgoTradingCLI:
    """Command Line Interface for Algo Trading System"""
    
    def __init__(self):
        self.data_loader = AdvancedDataLoader(config.CONFIG)
        self.optimizer = PortfolioOptimizer(config.CONFIG)
        self.backtest_engine = BacktestEngine(config.CONFIG)
        
        # Available strategies
        self.strategies = {
            'momentum': MomentumStrategy,
            'mean_reversion': MeanReversionStrategy,
            'adaptive': AdaptiveStrategy
        }
        
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']
    
    def print_banner(self):
        """Print application banner"""
        print("=" * 60)
        print("🚀 ALGORITHMIC TRADING SYSTEM")
        print("Advanced Portfolio Optimization & Strategy Testing")
        print("=" * 60)
        print()
    
    def print_menu(self):
        """Print main menu"""
        print("📋 MAIN MENU")
        print("-" * 30)
        print("1. 📊 Run Strategy Backtest")
        print("2. 🎯 Optimize Portfolio")
        print("3. 📈 Show Market Data")
        print("4. 💼 View Current Positions")
        print("5. 📊 Performance Report")
        print("6. 🔄 Paper Trading Demo")
        print("7. ⚙️  Configuration")
        print("0. 🚪 Exit")
        print()
    
    async def run_backtest_menu(self):
        """Interactive backtest menu"""
        print("\n📊 STRATEGY BACKTEST")
        print("-" * 30)
        
        # Strategy selection
        print("Available strategies:")
        for i, (key, _) in enumerate(self.strategies.items(), 1):
            print(f"  {i}. {key.replace('_', ' ').title()}")
        
        try:
            strategy_choice = int(input("\nSelect strategy (1-3): ")) - 1
            strategy_names = list(self.strategies.keys())
            selected_strategy = strategy_names[strategy_choice]
        except (ValueError, IndexError):
            print("❌ Invalid strategy selection!")
            return
        
        # Symbol selection
        print(f"\nAvailable symbols: {', '.join(self.symbols)}")
        symbols_input = input("Enter symbols (comma-separated) or press Enter for default: ").strip()
        if symbols_input:
            symbols = [s.strip().upper() for s in symbols_input.split(',')]
        else:
            symbols = self.symbols[:3]  # Use first 3 as default
        
        # Date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        date_input = input(f"Days to backtest (default 365): ").strip()
        if date_input.isdigit():
            days = int(date_input)
            start_date = end_date - timedelta(days=days)
        
        print(f"\n🔄 Running backtest for {selected_strategy} strategy...")
        print(f"   Symbols: {', '.join(symbols)}")
        print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # Load data
            data = await self.data_loader.load_multiple_symbols(symbols, start_date, end_date)
            
            # Initialize strategy
            strategy_class = self.strategies[selected_strategy]
            strategy = strategy_class(config.CONFIG)
            
            # Run backtest
            results = await self.backtest_engine.run_backtest(strategy, data)
            
            self.display_backtest_results(results)
            
        except Exception as e:
            print(f"❌ Backtest failed: {str(e)}")
    
    async def optimize_portfolio_menu(self):
        """Interactive portfolio optimization menu"""
        print("\n🎯 PORTFOLIO OPTIMIZATION")
        print("-" * 30)
        
        # Method selection
        methods = ['bayesian', 'genetic', 'mean_variance']
        print("Optimization methods:")
        for i, method in enumerate(methods, 1):
            print(f"  {i}. {method.replace('_', ' ').title()}")
        
        try:
            method_choice = int(input("\nSelect method (1-3): ")) - 1
            selected_method = methods[method_choice]
        except (ValueError, IndexError):
            print("❌ Invalid method selection!")
            return
        
        print(f"\n🔄 Running {selected_method} optimization...")
        
        try:
            # Load data for optimization
            symbols = self.symbols[:5]  # Use first 5 symbols
            end_date = datetime.now()
            start_date = end_date - timedelta(days=252)  # 1 year
            
            data = await self.data_loader.load_multiple_symbols(symbols, start_date, end_date)
            returns_data = data.pct_change().dropna()
            
            # Run optimization
            results = await self.optimizer.optimize_portfolio(returns_data, selected_method)
            
            self.display_optimization_results(results)
            
        except Exception as e:
            print(f"❌ Optimization failed: {str(e)}")
    
    async def show_market_data_menu(self):
        """Show current market data"""
        print("\n📈 MARKET DATA")
        print("-" * 30)
        
        try:
            # Load recent data
            symbols = self.symbols[:5]
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            data = await self.data_loader.load_multiple_symbols(symbols, start_date, end_date)
            
            print("Recent closing prices:")
            print(data.tail().round(2))
            
            print("\nDaily returns (%):")
            returns = data.pct_change() * 100
            print(returns.tail().round(2))
            
        except Exception as e:
            print(f"❌ Failed to load market data: {str(e)}")
    
    def display_backtest_results(self, results: Dict[str, Any]):
        """Display backtest results in a formatted table"""
        print("\n📊 BACKTEST RESULTS")
        print("=" * 50)
        
        if not results:
            print("❌ No results to display")
            return
        
        # Performance metrics
        metrics = [
            ("Total Return", f"{results.get('total_return', 0):.2%}"),
            ("Annualized Return", f"{results.get('annualized_return', 0):.2%}"),
            ("Volatility", f"{results.get('volatility', 0):.2%}"),
            ("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}"),
            ("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}"),
            ("Sortino Ratio", f"{results.get('sortino_ratio', 0):.2f}"),
            ("Calmar Ratio", f"{results.get('calmar_ratio', 0):.2f}"),
            ("Number of Trades", f"{results.get('num_trades', 0)}"),
            ("Win Rate", f"{results.get('win_rate', 0):.2%}")
        ]
        
        for metric, value in metrics:
            print(f"{metric:.<20} {value:>15}")
        
        print("\n" + "=" * 50)
    
    def display_optimization_results(self, results: Dict[str, Any]):
        """Display optimization results"""
        print("\n🎯 OPTIMIZATION RESULTS")
        print("=" * 50)
        
        if not results:
            print("❌ No results to display")
            return
        
        # Portfolio weights
        if 'weights' in results:
            print("\nOptimal Portfolio Weights:")
            print("-" * 30)
            for symbol, weight in results['weights'].items():
                print(f"{symbol:.<10} {weight:.2%}")
        
        # Portfolio metrics
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"\nExpected Return: {metrics.get('expected_return', 0):.2%}")
            print(f"Expected Volatility: {metrics.get('volatility', 0):.2%}")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"VaR (95%): {metrics.get('var_95', 0):.2%}")
        
        print("\n" + "=" * 50)
    
    async def paper_trading_demo(self):
        """Run a paper trading demonstration"""
        print("\n🔄 PAPER TRADING DEMO")
        print("-" * 30)
        
        try:
            # Initialize paper trader
            paper_trader = PaperTrader(config.CONFIG)
            
            # Initialize with some cash
            initial_capital = 100000
            paper_trader.cash = initial_capital
            
            print(f"Starting with ${initial_capital:,.2f} cash")
            
            # Load recent data
            symbols = ['AAPL', 'GOOGL', 'MSFT']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=10)
            
            data = await self.data_loader.load_multiple_symbols(symbols, start_date, end_date)
            current_prices = data.iloc[-1]
            
            # Execute some demo trades
            demo_trades = [
                {'symbol': 'AAPL', 'action': 'buy', 'quantity': 50},
                {'symbol': 'GOOGL', 'action': 'buy', 'quantity': 20},
                {'symbol': 'MSFT', 'action': 'buy', 'quantity': 30}
            ]
            
            print("\nExecuting demo trades:")
            for trade in demo_trades:
                symbol = trade['symbol']
                quantity = trade['quantity']
                price = current_prices[symbol]
                
                result = await paper_trader.execute_trade(
                    symbol, trade['action'], quantity, price
                )
                
                if result['success']:
                    print(f"✅ {trade['action'].upper()} {quantity} {symbol} @ ${price:.2f}")
                else:
                    print(f"❌ Failed to {trade['action']} {symbol}: {result['message']}")
            
            # Show portfolio status
            portfolio_value = await paper_trader.get_portfolio_value(current_prices)
            print(f"\nPortfolio Summary:")
            print(f"Cash: ${paper_trader.cash:,.2f}")
            print(f"Positions Value: ${portfolio_value - paper_trader.cash:,.2f}")
            print(f"Total Value: ${portfolio_value:,.2f}")
            print(f"P&L: ${portfolio_value - initial_capital:,.2f}")
            
        except Exception as e:
            print(f"❌ Paper trading demo failed: {str(e)}")
    
    def show_configuration(self):
        """Show current configuration"""
        print("\n⚙️  CONFIGURATION")
        print("-" * 30)
        
        important_configs = [
            'optimization_method',
            'max_position_size',
            'risk_free_rate',
            'lookback_period',
            'rebalance_threshold'
        ]
        
        for key in important_configs:
            value = config.CONFIG.get(key, 'Not set')
            print(f"{key.replace('_', ' ').title():.<25} {value}")
    
    async def run(self):
        """Main CLI loop"""
        self.print_banner()
        
        while True:
            self.print_menu()
            
            try:
                choice = input("Select option (0-7): ").strip()
                
                if choice == '0':
                    print("\n👋 Goodbye! Happy trading!")
                    break
                elif choice == '1':
                    await self.run_backtest_menu()
                elif choice == '2':
                    await self.optimize_portfolio_menu()
                elif choice == '3':
                    await self.show_market_data_menu()
                elif choice == '4':
                    print("\n💼 Current positions: Empty (Demo mode)")
                elif choice == '5':
                    print("\n📊 Performance report: Run a backtest first!")
                elif choice == '6':
                    await self.paper_trading_demo()
                elif choice == '7':
                    self.show_configuration()
                else:
                    print("❌ Invalid option! Please try again.")
                
                input("\nPress Enter to continue...")
                print("\n" + "="*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Happy trading!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {str(e)}")
                input("\nPress Enter to continue...")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Algorithmic Trading CLI')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'GOOGL', 'MSFT'],
                       help='Symbols to analyze')
    parser.add_argument('--strategy', choices=['momentum', 'mean_reversion', 'adaptive'],
                       default='momentum', help='Strategy to use')
    parser.add_argument('--days', type=int, default=365,
                       help='Number of days for backtesting')
    
    args = parser.parse_args()
    
    # If command line args provided, run non-interactive mode
    if len(sys.argv) > 1:
        print("Running in non-interactive mode...")
        # Add non-interactive mode implementation here
    else:
        # Interactive mode
        cli = AlgoTradingCLI()
        asyncio.run(cli.run())

if __name__ == "__main__":
    main()
