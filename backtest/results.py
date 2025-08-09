"""
Comprehensive Backtest Results Analysis and Visualization
Advanced analytics for strategy performance evaluation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class BacktestResults:
    """
    Comprehensive backtest results analysis and visualization
    """
    
    def __init__(self, results: Dict[str, Any], config: Dict[str, Any] = None):
        self.results = results
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Extract key components
        self.trades = results.get('trades', [])
        self.equity_curve = results.get('equity_curve', [])
        self.performance_stats = {k: v for k, v in results.items() 
                                if k not in ['trades', 'equity_curve']}
        
        self.logger.info("Backtest results initialized")
    
    def generate_report(self, output_dir: str = 'backtest_results') -> Dict[str, str]:
        """
        Generate comprehensive backtest report
        
        Args:
            output_dir: Directory to save report files
            
        Returns:
            Dictionary with file paths of generated reports
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        report_files = {}
        
        try:
            # Generate text summary
            summary_file = output_path / 'performance_summary.txt'
            self._generate_text_summary(summary_file)
            report_files['summary'] = str(summary_file)
            
            # Generate detailed CSV reports
            trades_file = output_path / 'trades_detail.csv'
            self._export_trades_csv(trades_file)
            report_files['trades'] = str(trades_file)
            
            equity_file = output_path / 'equity_curve.csv'
            self._export_equity_csv(equity_file)
            report_files['equity'] = str(equity_file)
            
            # Generate visualizations
            charts_file = output_path / 'performance_charts.html'
            self._generate_interactive_charts(charts_file)
            report_files['charts'] = str(charts_file)
            
            # Generate static plots
            plots_dir = output_path / 'plots'
            plots_dir.mkdir(exist_ok=True)
            plot_files = self._generate_static_plots(plots_dir)
            report_files.update(plot_files)
            
            # Generate JSON export
            json_file = output_path / 'results.json'
            self._export_json(json_file)
            report_files['json'] = str(json_file)
            
            self.logger.info(f"Report generated in {output_dir}")
            return report_files
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise
    
    def _generate_text_summary(self, output_file: Path):
        """Generate text performance summary"""
        
        with open(output_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ALGORITHMIC TRADING BACKTEST RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic information
            f.write("BACKTEST CONFIGURATION\n")
            f.write("-" * 30 + "\n")
            f.write(f"Start Date: {self.results.get('start_date', 'N/A')}\n")
            f.write(f"End Date: {self.results.get('end_date', 'N/A')}\n")
            f.write(f"Initial Capital: ${self.results.get('final_capital', 0) - self.results.get('total_return', 0) * 100000:,.2f}\n")
            f.write(f"Final Capital: ${self.results.get('final_capital', 0):,.2f}\n\n")
            
            # Performance metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Return: {self.results.get('total_return', 0):.2%}\n")
            f.write(f"Annualized Return: {self.results.get('annualized_return', 0):.2%}\n")
            f.write(f"Volatility: {self.results.get('volatility', 0):.2%}\n")
            f.write(f"Sharpe Ratio: {self.results.get('sharpe_ratio', 0):.3f}\n")
            f.write(f"Sortino Ratio: {self.results.get('sortino_ratio', 0):.3f}\n")
            f.write(f"Calmar Ratio: {self.results.get('calmar_ratio', 0):.3f}\n")
            f.write(f"Maximum Drawdown: {self.results.get('max_drawdown', 0):.2%}\n\n")
            
            # Trading statistics
            f.write("TRADING STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Trades: {self.results.get('total_trades', 0)}\n")
            f.write(f"Winning Trades: {self.results.get('winning_trades', 0)}\n")
            f.write(f"Losing Trades: {self.results.get('losing_trades', 0)}\n")
            f.write(f"Win Rate: {self.results.get('win_rate', 0):.2%}\n")
            f.write(f"Average Win: ${self.results.get('avg_win', 0):.2f}\n")
            f.write(f"Average Loss: ${self.results.get('avg_loss', 0):.2f}\n")
            f.write(f"Profit Factor: {self.results.get('profit_factor', 0):.3f}\n\n")
            
            # Risk analysis
            if 'risk_analysis' in self.results:
                risk = self.results['risk_analysis']
                f.write("RISK ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(f"VaR (95%): {risk.get('var_95', 0):.2%}\n")
                f.write(f"VaR (99%): {risk.get('var_99', 0):.2%}\n")
                f.write(f"CVaR (95%): {risk.get('cvar_95', 0):.2%}\n")
                f.write(f"Skewness: {risk.get('skewness', 0):.3f}\n")
                f.write(f"Kurtosis: {risk.get('kurtosis', 0):.3f}\n\n")
            
            # Strategy breakdown
            if 'strategy_performance' in self.results:
                f.write("STRATEGY PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                for strategy, stats in self.results['strategy_performance'].items():
                    f.write(f"\n{strategy.upper()}:\n")
                    f.write(f"  Total Trades: {stats.get('total_trades', 0)}\n")
                    f.write(f"  Win Rate: {stats.get('win_rate', 0):.2%}\n")
                    f.write(f"  Total P&L: ${stats.get('total_pnl', 0):.2f}\n")
                    f.write(f"  Avg P&L per Trade: ${stats.get('avg_pnl', 0):.2f}\n")
                f.write("\n")
            
            # Monthly performance
            if 'monthly_returns' in self.results:
                monthly = self.results['monthly_returns']
                f.write("MONTHLY PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average Monthly Return: {monthly.get('mean_monthly_return', 0):.2%}\n")
                f.write(f"Monthly Volatility: {monthly.get('std_monthly_return', 0):.2%}\n")
                f.write(f"Best Month: {monthly.get('best_month', 0):.2%}\n")
                f.write(f"Worst Month: {monthly.get('worst_month', 0):.2%}\n")
                f.write(f"Positive Months: {monthly.get('positive_months', 0)}\n")
                f.write(f"Negative Months: {monthly.get('negative_months', 0)}\n\n")
    
    def _export_trades_csv(self, output_file: Path):
        """Export trades to CSV"""
        
        if not self.trades:
            return
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'strategy': trade.strategy,
                'confidence': trade.confidence,
                'exit_reason': trade.exit_reason
            })
        
        df = pd.DataFrame(trades_data)
        df.to_csv(output_file, index=False)
        self.logger.info(f"Trades exported to {output_file}")
    
    def _export_equity_csv(self, output_file: Path):
        """Export equity curve to CSV"""
        
        if not self.equity_curve:
            return
        
        df = pd.DataFrame(self.equity_curve)
        df.to_csv(output_file, index=False)
        self.logger.info(f"Equity curve exported to {output_file}")
    
    def print_summary(self):
        """Print a quick summary to console"""
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"Total Return: {self.results.get('total_return', 0):.2%}")
        print(f"Annualized Return: {self.results.get('annualized_return', 0):.2%}")
        print(f"Sharpe Ratio: {self.results.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {self.results.get('max_drawdown', 0):.2%}")
        print(f"Win Rate: {self.results.get('win_rate', 0):.2%}")
        print(f"Total Trades: {self.results.get('total_trades', 0)}")
        print(f"Profit Factor: {self.results.get('profit_factor', 0):.3f}")
        
        print("="*60)
