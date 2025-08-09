"""
Advanced Algorithmic Trading System Launcher
Integrates Indian brokers, options strategies, and automatic optimization
"""
import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import TRADING_CONFIG
from utils.strategy_optimizer import StrategyOptimizer
from execution.indian_brokers import IndianBrokerFactory, ZerodhaBroker, IIFLBroker
from strategies.options import OptionsStrategyManager
from utils.logger import setup_logger
from utils.data_loader import DataLoader


class AdvancedTradingSystem:
    """Advanced trading system with Indian market support and auto-optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or TRADING_CONFIG
        self.logger = setup_logger('AdvancedTradingSystem')
        
        # Initialize components
        self.strategy_optimizer = StrategyOptimizer(self.config['strategy_optimization'])
        self.options_manager = OptionsStrategyManager()
        self.data_loader = DataLoader(self.config['data'])
        
        # Broker connections
        self.brokers = {}
        self.active_broker = None
        
        # Trading state
        self.is_trading = False
        self.positions = {}
        self.orders = []
        
        # Performance tracking
        self.performance_history = []
        self.last_optimization = None
        
        self.logger.info("Advanced Trading System initialized")
    
    async def initialize_brokers(self, broker_configs: Dict[str, Dict[str, str]]):
        """Initialize broker connections"""
        self.logger.info("Initializing broker connections...")
        
        for broker_name, config in broker_configs.items():
            try:
                if not config.get('api_key') or not config.get('api_secret'):
                    self.logger.warning(f"Incomplete configuration for {broker_name}, skipping")
                    continue
                
                broker = IndianBrokerFactory.create_broker(broker_name, config)
                
                # Test connection
                if broker_name == 'zerodha' and config.get('access_token'):
                    profile = broker.get_profile()
                    self.logger.info(f"Connected to Zerodha - User: {profile.get('user_name', 'Unknown')}")
                elif broker_name == 'iifl':
                    # IIFL requires login
                    if config.get('password'):
                        token = broker.login(config['password'])
                        self.logger.info(f"Connected to IIFL - Token received")
                
                self.brokers[broker_name] = broker
                if not self.active_broker:
                    self.active_broker = broker
                    self.logger.info(f"Set {broker_name} as active broker")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {broker_name}: {e}")
    
    def load_market_data(self, symbols: List[str], timeframe: str = '1d', 
                        days_back: int = 365) -> pd.DataFrame:
        """Load market data for symbols"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            data = self.data_loader.load_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            
            self.logger.info(f"Loaded data for {len(symbols)} symbols: {', '.join(symbols)}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            return pd.DataFrame()
    
    def optimize_strategy_selection(self, data: pd.DataFrame, symbols: List[str]) -> str:
        """Optimize strategy selection based on current market conditions"""
        try:
            self.logger.info("Running strategy optimization...")
            
            # Evaluate all strategies
            strategy_performances = self.strategy_optimizer.evaluate_strategies(data, symbols)
            
            # Analyze market regime
            market_regime = self.strategy_optimizer.analyze_market_regime(data)
            
            # Select optimal strategy
            optimal_strategy = self.strategy_optimizer.select_optimal_strategy(
                strategy_performances, market_regime
            )
            
            self.logger.info(f"Optimal strategy selected: {optimal_strategy}")
            self.logger.info(f"Market regime: {market_regime.market_phase}, "
                           f"Volatility: {market_regime.volatility_regime}, "
                           f"Trend strength: {market_regime.trend_strength:.2f}")
            
            self.last_optimization = datetime.now()
            
            # Save performance history
            self.strategy_optimizer.save_performance_history('data/strategy_performance.pkl')
            
            return optimal_strategy
            
        except Exception as e:
            self.logger.error(f"Error in strategy optimization: {e}")
            return 'adaptive'  # Fallback to adaptive strategy
    
    async def generate_trading_signals(self, data: pd.DataFrame, symbols: List[str], 
                                     strategy_name: str = None) -> List:
        """Generate trading signals using optimal strategy"""
        try:
            if not strategy_name:
                strategy_name = self.optimize_strategy_selection(data, symbols)
            
            # Get the optimal strategy
            strategy = self.strategy_optimizer.get_current_strategy()
            
            all_signals = []
            
            for symbol in symbols:
                if symbol not in data.columns:
                    continue
                
                # Prepare symbol-specific data
                symbol_data = pd.DataFrame({
                    'close': data[symbol],
                    'open': data[symbol] * (1 + np.random.normal(0, 0.001, len(data))),
                    'high': data[symbol] * (1 + abs(np.random.normal(0, 0.01, len(data)))),
                    'low': data[symbol] * (1 - abs(np.random.normal(0, 0.01, len(data)))),
                    'volume': np.random.randint(10000, 1000000, len(data))
                }).dropna()
                
                # Generate signals
                signals = strategy.calculate_signals(symbol_data, symbol)
                all_signals.extend(signals)
                
                self.logger.info(f"Generated {len(signals)} signals for {symbol}")
            
            return all_signals
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            return []
    
    async def execute_signals(self, signals: List) -> List[str]:
        """Execute trading signals through active broker"""
        executed_orders = []
        
        if not self.active_broker:
            self.logger.warning("No active broker available for trade execution")
            return executed_orders
        
        for signal in signals:
            try:
                # Create broker order from signal
                from execution.broker import BrokerOrder, OrderType
                
                order = BrokerOrder(
                    symbol=signal.symbol,
                    side='buy' if signal.signal.value > 0 else 'sell',
                    quantity=100,  # Default quantity, should be calculated based on position sizing
                    order_type=OrderType.MARKET,
                    price=signal.price
                )
                
                # Execute order
                order_id = self.active_broker.place_order(order)
                executed_orders.append(order_id)
                self.orders.append(order)
                
                self.logger.info(f"Executed {order.side} order for {order.symbol}, ID: {order_id}")
                
            except Exception as e:
                self.logger.error(f"Error executing signal for {signal.symbol}: {e}")
        
        return executed_orders
    
    async def monitor_positions(self):
        """Monitor current positions and manage risk"""
        if not self.active_broker:
            return
        
        try:
            positions = self.active_broker.get_positions()
            self.positions = {pos.symbol: pos for pos in positions}
            
            total_value = sum(pos.market_value for pos in positions)
            total_pnl = sum(pos.unrealized_pnl for pos in positions)
            
            self.logger.info(f"Portfolio value: ₹{total_value:,.2f}, "
                           f"Unrealized P&L: ₹{total_pnl:,.2f}")
            
            # Risk management checks
            max_drawdown = self.config['risk_management']['max_drawdown']
            if total_pnl / total_value < -max_drawdown:
                self.logger.warning("Maximum drawdown exceeded, halting trading")
                self.is_trading = False
            
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")
    
    def start_trading(self, symbols: List[str], auto_optimize: bool = True):
        """Start the trading system"""
        self.logger.info("🚀 Starting Advanced Algorithmic Trading System")
        self.logger.info(f"Symbols: {', '.join(symbols)}")
        self.logger.info(f"Auto-optimization: {'Enabled' if auto_optimize else 'Disabled'}")
        
        self.is_trading = True
        
        # Start main trading loop
        asyncio.run(self._trading_loop(symbols, auto_optimize))
    
    async def _trading_loop(self, symbols: List[str], auto_optimize: bool):
        """Main trading loop"""
        optimization_interval = timedelta(hours=1)  # Optimize every hour
        signal_interval = timedelta(minutes=5)  # Generate signals every 5 minutes
        
        last_signal_time = datetime.now()
        
        while self.is_trading:
            try:
                current_time = datetime.now()
                
                # Check if market is open (Indian market hours)
                if not self._is_market_open():
                    await asyncio.sleep(60)  # Check again in 1 minute
                    continue
                
                # Load fresh market data
                data = self.load_market_data(symbols, timeframe='1m', days_back=30)
                
                if data.empty:
                    self.logger.warning("No market data available")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue
                
                # Strategy optimization (if enabled and due)
                if (auto_optimize and 
                    (not self.last_optimization or 
                     current_time - self.last_optimization > optimization_interval)):
                    
                    optimal_strategy = self.optimize_strategy_selection(data, symbols)
                    self.logger.info(f"Strategy optimized: {optimal_strategy}")
                
                # Generate and execute signals
                if current_time - last_signal_time > signal_interval:
                    signals = await self.generate_trading_signals(data, symbols)
                    
                    if signals:
                        executed_orders = await self.execute_signals(signals)
                        self.logger.info(f"Executed {len(executed_orders)} orders")
                    
                    last_signal_time = current_time
                
                # Monitor positions
                await self.monitor_positions()
                
                # Wait before next iteration
                await asyncio.sleep(30)  # 30 second intervals
                
            except KeyboardInterrupt:
                self.logger.info("Trading interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
        
        self.logger.info("Trading system stopped")
    
    def _is_market_open(self) -> bool:
        """Check if Indian market is currently open"""
        now = datetime.now()
        
        # Simple check - market open Monday to Friday, 9:15 AM to 3:30 PM IST
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.positions:
            return {}
        
        total_invested = sum(pos.quantity * pos.avg_price for pos in self.positions.values())
        total_current_value = sum(pos.market_value for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'total_invested': total_invested,
            'current_value': total_current_value,
            'unrealized_pnl': total_pnl,
            'return_percentage': (total_pnl / total_invested * 100) if total_invested > 0 else 0,
            'num_positions': len(self.positions),
            'last_updated': datetime.now()
        }
    
    def stop_trading(self):
        """Stop the trading system"""
        self.logger.info("Stopping trading system...")
        self.is_trading = False


def main():
    """Main entry point"""
    print("🇮🇳 Advanced Indian Market Algorithmic Trading System")
    print("=" * 60)
    print("Features:")
    print("• 🔗 Zerodha Kite & IIFL API Integration")
    print("• 🎭 Options Strategies (Straddle, Strangle)")
    print("• 🤖 Automatic Strategy Optimization")
    print("• ⚡ Real-time Market Analysis")
    print("• 📊 Advanced Risk Management")
    print("=" * 60)
    
    # Initialize trading system
    trading_system = AdvancedTradingSystem()
    
    # Example usage
    try:
        # Indian market symbols
        symbols = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'NIFTY']
        
        print(f"\n📈 Starting with symbols: {', '.join(symbols)}")
        print("\n⚠️  Currently running in DEMO mode with simulated data")
        print("💡 Configure your broker APIs in the dashboard for live trading")
        
        # Start the web dashboard in a separate thread
        import threading
        from ui.enhanced_dashboard import app
        
        def run_dashboard():
            app.run(debug=False, host='0.0.0.0', port=8050)
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        print(f"\n🚀 Dashboard starting at: http://localhost:8050")
        print("📊 Open the dashboard to configure APIs and monitor trading")
        print("\n⏹️  Press Ctrl+C to stop the system")
        
        # Keep the main thread alive
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Trading system shutting down...")
            trading_system.stop_trading()
            
    except Exception as e:
        print(f"❌ Error starting trading system: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
