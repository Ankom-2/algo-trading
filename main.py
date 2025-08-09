"""
Main Entry Point for World-Class Algorithmic Trading System
Designed for consistent profitability with minimal risk
"""
import asyncio
import sys
import os
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TRADING_CONFIG
from utils.logger import TradingLogger
from utils.data_loader import AdvancedDataLoader
from utils.optimizer import PortfolioOptimizer
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.adaptive import AdaptiveStrategy
from risk_management.position_sizing import KellyPositionSizing
from risk_management.stop_loss import DynamicStopLoss
from risk_management.drawdown_control import DrawdownController
from backtest.engine import AdvancedBacktestEngine
from execution.paper_trader import SmartPaperTrader
from backtest.results import BacktestResults


class WorldClassTradingSystem:
    """
    Advanced algorithmic trading system designed for consistent profitability
    """
    
    def __init__(self):
        self.config = TRADING_CONFIG
        self.logger = TradingLogger(self.config['logging'])
        self.data_loader = AdvancedDataLoader(self.config['data'])
        self.optimizer = PortfolioOptimizer(self.config)
        
        # Initialize strategies
        self.strategies = self._initialize_strategies()
        
        # Initialize risk management
        self.position_sizer = KellyPositionSizing(self.config['risk_management'])
        self.stop_loss = DynamicStopLoss(self.config['risk_management'])
        self.drawdown_controller = DrawdownController(self.config['risk_management'])
        
        # Initialize paper trading
        from execution.paper_trader import SmartPaperTrader
        self.paper_trader = SmartPaperTrader(
            self.strategies, self.position_sizer, self.stop_loss,
            self.drawdown_controller, self.config, self.logger
        )
        
        self.logger.info("World-Class Trading System Initialized")
    
    def _initialize_strategies(self) -> List:
        """Initialize and optimize trading strategies"""
        strategies = [
            MomentumStrategy(self.config['strategies']['momentum']),
            MeanReversionStrategy(self.config['strategies']['mean_reversion']),
            AdaptiveStrategy(self.config['strategies']['adaptive'])
        ]
        
        self.logger.info(f"Initialized {len(strategies)} trading strategies")
        return strategies
    
    async def run_backtest(self) -> Dict[str, Any]:
        """Run comprehensive backtest with optimization"""
        self.logger.info("Starting comprehensive backtesting...")
        
        # Load historical data
        data = await self.data_loader.load_historical_data(
            symbols=self.config['data']['symbols'],
            start_date=self.config['backtest']['start_date'],
            end_date=self.config['backtest']['end_date']
        )
        
        # Initialize backtest engine
        backtest_engine = AdvancedBacktestEngine(
            strategies=self.strategies,
            position_sizer=self.position_sizer,
            stop_loss=self.stop_loss,
            drawdown_controller=self.drawdown_controller,
            config=self.config
        )
        
        # Run backtest
        results = await backtest_engine.run(data)
        
        # Analyze performance
        analysis = self.performance_analyzer.analyze(results)
        
        self.logger.info(f"Backtest completed. Sharpe Ratio: {analysis['sharpe_ratio']:.2f}")
        self.logger.info(f"Annual Return: {analysis['annual_return']:.2%}")
        self.logger.info(f"Max Drawdown: {analysis['max_drawdown']:.2%}")
        self.logger.info(f"Win Rate: {analysis['win_rate']:.2%}")
        
        return analysis
    
    async def optimize_strategies(self) -> Dict[str, Any]:
        """Optimize strategy parameters for maximum performance"""
        self.logger.info("Starting strategy optimization...")
        
        # Load data for optimization
        data = await self.data_loader.load_historical_data(
            symbols=self.config['data']['symbols'][:3],  # Use subset for optimization
            start_date='2022-01-01',
            end_date='2024-12-31'
        )
        
        # Run optimization
        optimized_params = await self.optimizer.optimize_parameters(
            strategies=self.strategies,
            data=data,
            objective='sharpe_ratio'
        )
        
        self.logger.info("Strategy optimization completed")
        return optimized_params
    
    async def run_paper_trading(self):
        """Run live paper trading with optimized strategies"""
        self.logger.info("Starting paper trading...")
        
        # Initialize paper trader
        paper_trader = SmartPaperTrader(
            strategies=self.strategies,
            position_sizer=self.position_sizer,
            stop_loss=self.stop_loss,
            drawdown_controller=self.drawdown_controller,
            config=self.config,
            logger=self.logger
        )
        
        # Start live trading
        await paper_trader.start()
    
    async def run_full_system(self):
        """Run the complete trading system pipeline"""
        try:
            self.logger.info("=== Starting World-Class Trading System ===")
            
            # Step 1: Run backtest
            backtest_results = await self.run_backtest()
            
            # Step 2: Optimize strategies if performance is below target
            if backtest_results['sharpe_ratio'] < self.config['targets']['sharpe_ratio']:
                self.logger.info("Performance below target, optimizing strategies...")
                optimized_params = await self.optimize_strategies()
                
                # Update strategy parameters
                for strategy, params in optimized_params.items():
                    strategy.update_parameters(params)
            
            # Step 3: Start paper trading if backtest is satisfactory
            if (backtest_results['sharpe_ratio'] >= 1.5 and 
                backtest_results['max_drawdown'] <= self.config['targets']['max_drawdown']):
                
                self.logger.info("Backtest results satisfactory, starting paper trading...")
                await self.run_paper_trading()
            else:
                self.logger.warning("Backtest results below threshold, paper trading not started")
                
        except Exception as e:
            self.logger.error(f"System error: {str(e)}")
            raise


async def main():
    """Main entry point"""
    print("🚀 World-Class Algorithmic Trading System")
    print("=" * 50)
    
    # Initialize trading system
    trading_system = WorldClassTradingSystem()
    
    # Run the complete system
    await trading_system.run_full_system()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
