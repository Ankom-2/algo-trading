"""
UI Configuration and Launch Script
Easily configure and launch different interfaces for the trading system
"""

import os
import sys
import subprocess
import json
from typing import Dict, Any

def print_banner():
    """Print application banner"""
    print("=" * 70)
    print("🚀 ALGORITHMIC TRADING SYSTEM - UI LAUNCHER")
    print("Choose your preferred interface")
    print("=" * 70)

def print_interfaces():
    """Print available interfaces"""
    print("\n📱 AVAILABLE INTERFACES")
    print("-" * 40)
    print("1. 🌐 Web Dashboard (Dash) - Full-featured web interface")
    print("2. 💻 Command Line Interface - Quick terminal-based access")
    print("3. 📊 Jupyter Notebook - Interactive analysis environment")
    print("4. ⚙️  Configuration Manager - Update system settings")
    print("0. 🚪 Exit")
    print()

def launch_web_dashboard():
    """Launch the web dashboard"""
    print("\n🌐 Launching Web Dashboard...")
    print("Opening at http://localhost:8050")
    print("Press Ctrl+C to stop the dashboard")
    print("-" * 40)
    
    try:
        # Change to the UI directory and run dashboard
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        dashboard_path = os.path.join(ui_dir, 'dashboard.py')
        subprocess.run([sys.executable, dashboard_path])
    except KeyboardInterrupt:
        print("\n✅ Dashboard stopped.")
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {str(e)}")

def launch_cli():
    """Launch the command line interface"""
    print("\n💻 Launching Command Line Interface...")
    print("-" * 40)
    
    try:
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        cli_path = os.path.join(ui_dir, 'cli.py')
        subprocess.run([sys.executable, cli_path])
    except Exception as e:
        print(f"❌ Failed to launch CLI: {str(e)}")

def create_jupyter_notebook():
    """Create and launch a Jupyter notebook for analysis"""
    print("\n📊 Creating Jupyter Notebook...")
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Algorithmic Trading Analysis\n",
                    "\n",
                    "This notebook provides interactive analysis capabilities for the algorithmic trading system.\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Import necessary libraries\n",
                    "import sys\n",
                    "import os\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from datetime import datetime, timedelta\n",
                    "\n",
                    "# Add parent directory to path\n",
                    "sys.path.append('..')\n",
                    "\n",
                    "# Import trading system modules\n",
                    "from utils.data_loader import DataLoader\n",
                    "from utils.optimizer import PortfolioOptimizer\n",
                    "from strategies.momentum import MomentumStrategy\n",
                    "from backtest.engine import BacktestEngine\n",
                    "import config\n",
                    "\n",
                    "print(\"✅ Trading system modules loaded successfully!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Market Data\n",
                    "\n",
                    "Start by loading some market data for analysis:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Initialize data loader\n",
                    "data_loader = DataLoader(config.CONFIG)\n",
                    "\n",
                    "# Define symbols and date range\n",
                    "symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']\n",
                    "end_date = datetime.now()\n",
                    "start_date = end_date - timedelta(days=365)\n",
                    "\n",
                    "print(f\"Loading data for {symbols} from {start_date.date()} to {end_date.date()}...\")\n",
                    "\n",
                    "# Note: In Jupyter, use await with async functions like this:\n",
                    "# data = await data_loader.load_multiple_symbols(symbols, start_date, end_date)\n",
                    "# For now, we'll create sample data\n",
                    "\n",
                    "# Create sample data for demonstration\n",
                    "dates = pd.date_range(start=start_date, end=end_date, freq='D')\n",
                    "np.random.seed(42)\n",
                    "sample_data = pd.DataFrame({\n",
                    "    'AAPL': 150 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))),\n",
                    "    'GOOGL': 2800 * np.cumprod(1 + np.random.normal(0.0008, 0.025, len(dates))),\n",
                    "    'MSFT': 300 * np.cumprod(1 + np.random.normal(0.0009, 0.022, len(dates))),\n",
                    "    'AMZN': 3200 * np.cumprod(1 + np.random.normal(0.0007, 0.028, len(dates)))\n",
                    "}, index=dates)\n",
                    "\n",
                    "print(f\"✅ Sample data created with shape: {sample_data.shape}\")\n",
                    "sample_data.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Visualize Price Data"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Plot price charts\n",
                    "plt.figure(figsize=(12, 8))\n",
                    "\n",
                    "# Normalize prices to start at 1 for comparison\n",
                    "normalized_data = sample_data / sample_data.iloc[0]\n",
                    "\n",
                    "for symbol in symbols:\n",
                    "    plt.plot(normalized_data.index, normalized_data[symbol], label=symbol, linewidth=2)\n",
                    "\n",
                    "plt.title('Stock Price Performance (Normalized)', fontsize=16, fontweight='bold')\n",
                    "plt.xlabel('Date', fontsize=12)\n",
                    "plt.ylabel('Normalized Price', fontsize=12)\n",
                    "plt.legend()\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Calculate Returns and Risk Metrics"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Calculate daily returns\n",
                    "returns = sample_data.pct_change().dropna()\n",
                    "\n",
                    "# Calculate basic statistics\n",
                    "print(\"📊 RETURN STATISTICS\")\n",
                    "print(\"=\"*50)\n",
                    "print(\"\\nAnnualized Returns:\")\n",
                    "print((returns.mean() * 252).round(4))\n",
                    "\n",
                    "print(\"\\nAnnualized Volatility:\")\n",
                    "print((returns.std() * np.sqrt(252)).round(4))\n",
                    "\n",
                    "print(\"\\nSharpe Ratios (assuming 2% risk-free rate):\")\n",
                    "risk_free_rate = 0.02\n",
                    "sharpe_ratios = (returns.mean() * 252 - risk_free_rate) / (returns.std() * np.sqrt(252))\n",
                    "print(sharpe_ratios.round(4))\n",
                    "\n",
                    "# Correlation matrix\n",
                    "print(\"\\n📈 CORRELATION MATRIX\")\n",
                    "print(\"=\"*30)\n",
                    "correlation_matrix = returns.corr()\n",
                    "print(correlation_matrix.round(3))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Visualize Risk-Return Profile"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create risk-return scatter plot\n",
                    "plt.figure(figsize=(10, 6))\n",
                    "\n",
                    "annual_returns = returns.mean() * 252\n",
                    "annual_volatility = returns.std() * np.sqrt(252)\n",
                    "\n",
                    "plt.scatter(annual_volatility, annual_returns, s=100, alpha=0.7)\n",
                    "\n",
                    "for i, symbol in enumerate(symbols):\n",
                    "    plt.annotate(symbol, (annual_volatility[i], annual_returns[i]), \n",
                    "                xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')\n",
                    "\n",
                    "plt.xlabel('Annual Volatility', fontsize=12)\n",
                    "plt.ylabel('Annual Return', fontsize=12)\n",
                    "plt.title('Risk-Return Profile', fontsize=16, fontweight='bold')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Portfolio Optimization Example"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Simple equal-weight portfolio example\n",
                    "equal_weights = np.array([0.25, 0.25, 0.25, 0.25])\n",
                    "portfolio_returns = (returns * equal_weights).sum(axis=1)\n",
                    "\n",
                    "# Calculate portfolio metrics\n",
                    "portfolio_annual_return = portfolio_returns.mean() * 252\n",
                    "portfolio_annual_volatility = portfolio_returns.std() * np.sqrt(252)\n",
                    "portfolio_sharpe = (portfolio_annual_return - 0.02) / portfolio_annual_volatility\n",
                    "\n",
                    "print(\"🎯 EQUAL-WEIGHT PORTFOLIO METRICS\")\n",
                    "print(\"=\"*40)\n",
                    "print(f\"Annual Return: {portfolio_annual_return:.2%}\")\n",
                    "print(f\"Annual Volatility: {portfolio_annual_volatility:.2%}\")\n",
                    "print(f\"Sharpe Ratio: {portfolio_sharpe:.2f}\")\n",
                    "\n",
                    "# Plot portfolio vs individual assets\n",
                    "plt.figure(figsize=(12, 6))\n",
                    "cumulative_returns = (1 + returns).cumprod()\n",
                    "portfolio_cumulative = (1 + portfolio_returns).cumprod()\n",
                    "\n",
                    "for symbol in symbols:\n",
                    "    plt.plot(cumulative_returns.index, cumulative_returns[symbol], \n",
                    "             label=symbol, alpha=0.7, linewidth=1)\n",
                    "\n",
                    "plt.plot(portfolio_cumulative.index, portfolio_cumulative, \n",
                    "         label='Equal-Weight Portfolio', color='red', linewidth=3)\n",
                    "\n",
                    "plt.title('Cumulative Returns: Individual Assets vs Portfolio', fontsize=16, fontweight='bold')\n",
                    "plt.xlabel('Date', fontsize=12)\n",
                    "plt.ylabel('Cumulative Return', fontsize=12)\n",
                    "plt.legend()\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Next Steps\n",
                    "\n",
                    "This notebook provides a starting point for analysis. You can:\n",
                    "\n",
                    "1. **Load real data**: Replace sample data with actual market data using the DataLoader\n",
                    "2. **Test strategies**: Initialize and backtest different trading strategies\n",
                    "3. **Optimize portfolios**: Use the PortfolioOptimizer for advanced optimization\n",
                    "4. **Risk analysis**: Implement VaR, CVaR, and other risk metrics\n",
                    "5. **Strategy comparison**: Compare multiple strategies side by side\n",
                    "\n",
                    "Happy analyzing! 🚀"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    try:
        ui_dir = os.path.dirname(os.path.abspath(__file__))
        notebook_path = os.path.join(ui_dir, 'trading_analysis.ipynb')
        
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        print(f"✅ Jupyter notebook created: {notebook_path}")
        print("📊 Launching Jupyter...")
        
        # Launch Jupyter
        subprocess.run(['jupyter', 'notebook', notebook_path])
        
    except Exception as e:
        print(f"❌ Failed to create/launch Jupyter notebook: {str(e)}")

def configuration_manager():
    """Manage system configuration"""
    print("\n⚙️  CONFIGURATION MANAGER")
    print("-" * 40)
    
    # Import config
    sys.path.append('..')
    import config
    
    config_items = [
        ('optimization_method', 'Optimization Method'),
        ('max_position_size', 'Maximum Position Size'),
        ('risk_free_rate', 'Risk Free Rate'),
        ('lookback_period', 'Lookback Period'),
        ('rebalance_threshold', 'Rebalance Threshold'),
        ('max_iterations', 'Max Optimization Iterations')
    ]
    
    print("Current Configuration:")
    print("-" * 25)
    for key, description in config_items:
        value = config.CONFIG.get(key, 'Not set')
        print(f"{description}: {value}")
    
    print("\nTo modify configuration, edit the config.py file directly.")

def main():
    """Main launcher function"""
    print_banner()
    
    while True:
        print_interfaces()
        
        try:
            choice = input("Select interface (0-4): ").strip()
            
            if choice == '0':
                print("\n👋 Goodbye!")
                break
            elif choice == '1':
                launch_web_dashboard()
            elif choice == '2':
                launch_cli()
            elif choice == '3':
                create_jupyter_notebook()
            elif choice == '4':
                configuration_manager()
            else:
                print("❌ Invalid option! Please try again.")
            
            input("\nPress Enter to return to main menu...")
            print("\n" + "="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
