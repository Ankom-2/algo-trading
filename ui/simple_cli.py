"""
Simple Command Line Interface for Algorithmic Trading System
Lightweight CLI for quick testing and demonstrations
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SimpleAlgoTradingCLI:
    """Simple Command Line Interface for Algo Trading System"""
    
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']
        self.strategies = ['momentum', 'mean_reversion', 'adaptive']
    
    def print_banner(self):
        """Print application banner"""
        print("=" * 60)
        print("🚀 ALGORITHMIC TRADING SYSTEM - SIMPLE CLI")
        print("Lightweight interface for strategy testing")
        print("=" * 60)
        print()
    
    def print_menu(self):
        """Print main menu"""
        print("📋 MAIN MENU")
        print("-" * 30)
        print("1. 📊 Run Strategy Demo")
        print("2. 🎯 Portfolio Optimization Demo")
        print("3. 📈 Show Sample Market Data")
        print("4. 💼 Sample Portfolio Analysis")
        print("5. 📊 Performance Metrics Demo")
        print("0. 🚪 Exit")
        print()
    
    def create_sample_data(self, symbols=None, days=365):
        """Create sample market data"""
        if symbols is None:
            symbols = self.symbols[:4]
        
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        np.random.seed(42)
        
        # Base prices for different stocks
        base_prices = {
            'AAPL': 150,
            'GOOGL': 2800,
            'MSFT': 300,
            'AMZN': 3200,
            'TSLA': 800,
            'NVDA': 400
        }
        
        data = {}
        for symbol in symbols:
            base_price = base_prices.get(symbol, 100)
            # Generate realistic price movements
            returns = np.random.normal(0.0005, 0.02, len(dates))
            prices = base_price * np.cumprod(1 + returns)
            data[symbol] = prices
        
        return pd.DataFrame(data, index=dates)
    
    def run_strategy_demo(self):
        """Run a strategy demonstration"""
        print("\n📊 STRATEGY DEMO")
        print("-" * 30)
        
        # Strategy selection
        print("Available strategies:")
        for i, strategy in enumerate(self.strategies, 1):
            print(f"  {i}. {strategy.replace('_', ' ').title()}")
        
        try:
            choice = int(input("\nSelect strategy (1-3): ")) - 1
            selected_strategy = self.strategies[choice]
        except (ValueError, IndexError):
            print("❌ Invalid selection!")
            return
        
        print(f"\n🔄 Running {selected_strategy.replace('_', ' ').title()} strategy demo...")
        
        # Generate sample data
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        data = self.create_sample_data(symbols, 252)
        returns = data.pct_change().dropna()
        
        # Simulate strategy results
        np.random.seed(hash(selected_strategy) % 1000)
        results = {
            'strategy': selected_strategy,
            'total_return': np.random.uniform(0.15, 0.35),
            'sharpe_ratio': np.random.uniform(1.2, 2.5),
            'max_drawdown': np.random.uniform(-0.15, -0.05),
            'volatility': np.random.uniform(0.12, 0.25),
            'win_rate': np.random.uniform(0.55, 0.75)
        }
        
        # Display results
        print("\n📊 STRATEGY RESULTS")
        print("=" * 40)
        print(f"Strategy: {selected_strategy.replace('_', ' ').title()}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print("=" * 40)
    
    def portfolio_optimization_demo(self):
        """Run portfolio optimization demonstration"""
        print("\n🎯 PORTFOLIO OPTIMIZATION DEMO")
        print("-" * 30)
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        data = self.create_sample_data(symbols, 252)
        returns = data.pct_change().dropna()
        
        # Calculate basic portfolio metrics
        print("Asset Statistics:")
        print("-" * 20)
        annual_returns = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        for symbol in symbols:
            print(f"{symbol}: Return {annual_returns[symbol]:.2%}, "
                  f"Volatility {annual_volatility[symbol]:.2%}")
        
        # Simple optimization (equal weights vs market cap weighted)
        equal_weights = np.array([0.25, 0.25, 0.25, 0.25])
        market_cap_weights = np.array([0.35, 0.25, 0.20, 0.20])  # Simulated market cap weights
        
        portfolios = {
            'Equal Weight': equal_weights,
            'Market Cap Weighted': market_cap_weights
        }
        
        print(f"\nPortfolio Comparison:")
        print("-" * 25)
        
        for name, weights in portfolios.items():
            portfolio_return = np.sum(annual_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            sharpe_ratio = (portfolio_return - 0.02) / portfolio_volatility  # Assuming 2% risk-free rate
            
            print(f"\n{name}:")
            print(f"  Expected Return: {portfolio_return:.2%}")
            print(f"  Volatility: {portfolio_volatility:.2%}")
            print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"  Weights: {dict(zip(symbols, weights))}")
    
    def show_sample_market_data(self):
        """Show sample market data"""
        print("\n📈 SAMPLE MARKET DATA")
        print("-" * 30)
        
        symbols = self.symbols[:5]
        data = self.create_sample_data(symbols, 10)
        
        print("Recent Prices (Last 5 days):")
        print(data.tail().round(2))
        
        print("\nDaily Returns (%):")
        returns = data.pct_change() * 100
        print(returns.tail().round(2))
    
    def sample_portfolio_analysis(self):
        """Show sample portfolio analysis"""
        print("\n💼 SAMPLE PORTFOLIO ANALYSIS")
        print("-" * 30)
        
        # Sample portfolio
        portfolio = {
            'AAPL': {'shares': 100, 'avg_cost': 150.00},
            'GOOGL': {'shares': 10, 'avg_cost': 2800.00},
            'MSFT': {'shares': 80, 'avg_cost': 300.00},
            'Cash': {'value': 15000.00}
        }
        
        # Current prices (simulated)
        current_prices = {
            'AAPL': 165.50,
            'GOOGL': 2950.00,
            'MSFT': 325.75
        }
        
        total_value = 0
        print("Portfolio Holdings:")
        print("-" * 20)
        
        for symbol, holding in portfolio.items():
            if symbol == 'Cash':
                print(f"Cash: ${holding['value']:,.2f}")
                total_value += holding['value']
            else:
                shares = holding['shares']
                avg_cost = holding['avg_cost']
                current_price = current_prices[symbol]
                market_value = shares * current_price
                cost_basis = shares * avg_cost
                pnl = market_value - cost_basis
                pnl_pct = (pnl / cost_basis) * 100
                
                print(f"{symbol}: {shares} shares @ ${current_price:.2f}")
                print(f"  Market Value: ${market_value:,.2f}")
                print(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
                
                total_value += market_value
        
        print(f"\nTotal Portfolio Value: ${total_value:,.2f}")
    
    def performance_metrics_demo(self):
        """Show performance metrics demonstration"""
        print("\n📊 PERFORMANCE METRICS DEMO")
        print("-" * 30)
        
        # Generate sample performance data
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        daily_returns = np.random.normal(0.0008, 0.02, len(dates))
        cumulative_returns = np.cumprod(1 + daily_returns)
        
        # Calculate metrics
        total_return = cumulative_returns[-1] - 1
        annualized_return = (1 + daily_returns.mean()) ** 252 - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility
        
        # Max drawdown calculation
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        # Display metrics
        print("Performance Metrics:")
        print("-" * 20)
        print(f"Total Return: {total_return:.2%}")
        print(f"Annualized Return: {annualized_return:.2%}")
        print(f"Volatility: {volatility:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        
        # Show best/worst days
        best_day = daily_returns.max()
        worst_day = daily_returns.min()
        print(f"\nBest Day: {best_day:.2%}")
        print(f"Worst Day: {worst_day:.2%}")
    
    def run(self):
        """Main CLI loop"""
        self.print_banner()
        
        while True:
            self.print_menu()
            
            try:
                choice = input("Select option (0-5): ").strip()
                
                if choice == '0':
                    print("\n👋 Goodbye! Happy trading!")
                    break
                elif choice == '1':
                    self.run_strategy_demo()
                elif choice == '2':
                    self.portfolio_optimization_demo()
                elif choice == '3':
                    self.show_sample_market_data()
                elif choice == '4':
                    self.sample_portfolio_analysis()
                elif choice == '5':
                    self.performance_metrics_demo()
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
    cli = SimpleAlgoTradingCLI()
    cli.run()

if __name__ == "__main__":
    main()
