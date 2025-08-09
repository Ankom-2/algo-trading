"""
Smart Paper Trading Engine for Risk-Free Strategy Validation
Simulates real trading conditions with advanced order management
"""
import pandas as pd
import numpy as np
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import uuid


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class PaperOrder:
    """Paper trading order representation"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = None
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    strategy: Optional[str] = None
    confidence: float = 0.0


@dataclass
class Position:
    """Trading position representation"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    created_at: datetime = None
    last_updated: datetime = None


class SmartPaperTrader:
    """
    Advanced paper trading engine with realistic simulation features
    """
    
    def __init__(self, strategies: List, position_sizer, stop_loss, 
                 drawdown_controller, config: Dict[str, Any], logger):
        
        self.strategies = strategies
        self.position_sizer = position_sizer
        self.stop_loss = stop_loss
        self.drawdown_controller = drawdown_controller
        self.config = config
        self.logger = logger
        
        # Trading state
        self.initial_capital = config.get('initial_capital', 100000)
        self.current_capital = self.initial_capital
        self.positions = {}  # symbol -> Position
        self.orders = {}     # order_id -> PaperOrder
        self.trade_history = []
        
        # Execution parameters
        self.slippage_rate = config.get('slippage_rate', 0.0005)  # 5 bps
        self.commission_rate = config.get('commission_rate', 0.0001)  # 1 bp
        self.min_trade_amount = config.get('min_trade_amount', 1000)
        self.max_orders_per_second = config.get('max_orders_per_second', 10)
        
        # Market simulation
        self.market_impact = config.get('market_impact', 0.0002)
        self.order_fill_probability = config.get('order_fill_probability', 0.95)
        self.partial_fill_probability = config.get('partial_fill_probability', 0.1)
        
        # Performance tracking
        self.performance_metrics = {}
        self.daily_pnl = []
        self.equity_curve = []
        
        # Rate limiting
        self.last_order_time = datetime.now()
        self.orders_this_second = 0
        
        self.logger.info("Smart Paper Trader initialized")
    
    async def start(self):
        """Start paper trading loop"""
        self.logger.info("Starting paper trading simulation")
        
        try:
            # Main trading loop
            while True:
                await self._trading_iteration()
                await asyncio.sleep(1)  # 1-second intervals
                
        except KeyboardInterrupt:
            self.logger.info("Paper trading stopped by user")
        except Exception as e:
            self.logger.error(f"Paper trading error: {str(e)}")
        finally:
            await self._shutdown()
    
    async def _trading_iteration(self):
        """Single iteration of the trading loop"""
        
        try:
            # Update market data (simulated)
            market_data = await self._get_simulated_market_data()
            
            # Update positions and portfolio value
            self._update_positions(market_data)
            self._update_portfolio_value()
            
            # Update drawdown controller
            self.drawdown_controller.update_portfolio_value(self.current_capital)
            
            # Check if trading is allowed
            if not self._can_trade():
                return
            
            # Generate signals from strategies
            all_signals = []
            for strategy in self.strategies:
                try:
                    # Get recent data for strategy (simulated)
                    strategy_data = self._prepare_strategy_data(market_data)
                    signals = strategy.generate_signals(strategy_data)
                    all_signals.extend(signals)
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy.name} signal generation error: {str(e)}")
            
            # Process signals and create orders
            if all_signals:
                await self._process_signals(all_signals, market_data)
            
            # Process pending orders
            await self._process_pending_orders(market_data)
            
            # Update stop losses
            self._update_stop_losses(market_data)
            
            # Log performance periodically
            if len(self.equity_curve) % 100 == 0:  # Every 100 iterations
                self._log_performance()
            
        except Exception as e:
            self.logger.error(f"Trading iteration error: {str(e)}")
    
    async def _get_simulated_market_data(self) -> Dict[str, Any]:
        """Generate simulated market data"""
        
        # In a real implementation, this would fetch actual market data
        # For simulation, we generate realistic price movements
        
        market_data = {}
        base_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        for symbol in base_symbols:
            # Generate realistic price data
            base_price = 150.0  # Starting price
            volatility = 0.02   # 2% daily volatility
            
            # Random walk with drift
            price_change = np.random.normal(0.0001, volatility / np.sqrt(24 * 60))  # Per minute
            current_price = base_price * (1 + price_change)
            
            market_data[symbol] = {
                'price': current_price,
                'bid': current_price * 0.9995,
                'ask': current_price * 1.0005,
                'volume': np.random.randint(1000, 10000),
                'timestamp': datetime.now()
            }
        
        return market_data
    
    def _prepare_strategy_data(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare data for strategy signal generation"""
        
        # Create a simple DataFrame for strategy consumption
        # In production, this would be more sophisticated
        
        data_points = []
        for symbol, data in market_data.items():
            data_points.append({
                'symbol': symbol,
                'close': data['price'],
                'high': data['price'] * 1.001,
                'low': data['price'] * 0.999,
                'open': data['price'] * 0.9995,
                'volume': data['volume']
            })
        
        df = pd.DataFrame(data_points)
        df.index = pd.date_range(start=datetime.now() - timedelta(minutes=len(df)), 
                                periods=len(df), freq='1min')
        
        return df
    
    async def _process_signals(self, signals: List, market_data: Dict[str, Any]):
        """Process trading signals and create orders"""
        
        for signal in signals:
            try:
                # Check if we can open new positions
                position_size = self._calculate_position_size(signal)
                risk_amount = position_size * self.current_capital
                
                can_trade, reason = self.drawdown_controller.can_open_new_position(
                    position_size, risk_amount
                )
                
                if not can_trade:
                    self.logger.info(f"Signal rejected for {signal.symbol}: {reason}")
                    continue
                
                # Create order
                order = await self._create_order_from_signal(signal, market_data)
                if order:
                    self.orders[order.id] = order
                    self.logger.info(f"Order created: {order.id} for {order.symbol}")
                
            except Exception as e:
                self.logger.error(f"Signal processing error: {str(e)}")
    
    def _calculate_position_size(self, signal) -> float:
        """Calculate position size for a signal"""
        
        # Get current volatility estimate
        volatility = 0.02  # Simplified, would use actual volatility
        
        # Use position sizer
        position_size = self.position_sizer.calculate_position_size(
            signal.confidence,
            volatility,
            self.current_capital
        )
        
        # Apply drawdown controller scaling
        scaling_factor = self.drawdown_controller.get_position_scaling_factor()
        position_size *= scaling_factor
        
        return position_size
    
    async def _create_order_from_signal(self, signal, market_data: Dict[str, Any]) -> Optional[PaperOrder]:
        """Create paper order from trading signal"""
        
        if signal.symbol not in market_data:
            return None
        
        # Rate limiting check
        if not self._check_rate_limit():
            return None
        
        # Calculate order parameters
        position_size = self._calculate_position_size(signal)
        market_price = market_data[signal.symbol]['price']
        
        # Convert position size to shares
        dollar_amount = position_size * self.current_capital
        if dollar_amount < self.min_trade_amount:
            return None
        
        shares = dollar_amount / market_price
        
        # Determine order side
        side = 'buy' if signal.signal.value > 0 else 'sell'
        
        # Create order
        order = PaperOrder(
            id=str(uuid.uuid4()),
            symbol=signal.symbol,
            side=side,
            quantity=shares,
            order_type=OrderType.MARKET,
            price=market_price,
            created_at=datetime.now(),
            strategy=getattr(signal, 'strategy', 'unknown'),
            confidence=signal.confidence
        )
        
        return order
    
    async def _process_pending_orders(self, market_data: Dict[str, Any]):
        """Process all pending orders"""
        
        pending_orders = [order for order in self.orders.values() 
                         if order.status == OrderStatus.PENDING]
        
        for order in pending_orders:
            await self._try_fill_order(order, market_data)
    
    async def _try_fill_order(self, order: PaperOrder, market_data: Dict[str, Any]):
        """Attempt to fill a pending order"""
        
        if order.symbol not in market_data:
            return
        
        market_info = market_data[order.symbol]
        current_price = market_info['price']
        
        # Simulate order fill probability
        if np.random.random() > self.order_fill_probability:
            return
        
        # Calculate fill price with slippage and market impact
        fill_price = self._calculate_fill_price(order, current_price)
        
        # Check for partial fill
        fill_quantity = order.quantity
        if np.random.random() < self.partial_fill_probability:
            fill_quantity *= np.random.uniform(0.5, 0.9)
        
        # Execute the fill
        self._execute_fill(order, fill_price, fill_quantity)
    
    def _calculate_fill_price(self, order: PaperOrder, market_price: float) -> float:
        """Calculate realistic fill price including slippage and market impact"""
        
        # Base slippage
        slippage = self.slippage_rate * market_price
        
        # Market impact based on order size
        impact = self.market_impact * market_price * (order.quantity / 1000)
        
        # Apply slippage and impact
        if order.side == 'buy':
            fill_price = market_price + slippage + impact
        else:
            fill_price = market_price - slippage - impact
        
        return max(0.01, fill_price)  # Ensure positive price
    
    def _execute_fill(self, order: PaperOrder, fill_price: float, fill_quantity: float):
        """Execute order fill and update positions"""
        
        # Calculate commission
        commission = fill_quantity * fill_price * self.commission_rate
        
        # Update order
        order.filled_price = fill_price
        order.filled_quantity += fill_quantity
        order.filled_at = datetime.now()
        
        if order.filled_quantity >= order.quantity * 0.99:  # 99% filled
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Update position
        self._update_position(order.symbol, order.side, fill_quantity, fill_price)
        
        # Update capital
        if order.side == 'buy':
            self.current_capital -= (fill_quantity * fill_price + commission)
        else:
            self.current_capital += (fill_quantity * fill_price - commission)
        
        # Record trade
        trade_record = {
            'timestamp': order.filled_at,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': fill_quantity,
            'price': fill_price,
            'commission': commission,
            'strategy': order.strategy,
            'confidence': order.confidence
        }
        self.trade_history.append(trade_record)
        
        # Log trade
        self.logger.info(f"Order filled: {order.symbol} {order.side} {fill_quantity:.2f} @ {fill_price:.4f}")
    
    def _update_position(self, symbol: str, side: str, quantity: float, price: float):
        """Update position after trade execution"""
        
        if symbol not in self.positions:
            # New position
            if side == 'buy':
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price,
                    unrealized_pnl=0.0,
                    created_at=datetime.now()
                )
            # Don't create position for sells without existing position
        else:
            # Update existing position
            position = self.positions[symbol]
            
            if side == 'buy':
                # Add to position
                total_quantity = position.quantity + quantity
                total_cost = (position.quantity * position.avg_price + quantity * price)
                position.avg_price = total_cost / total_quantity if total_quantity > 0 else 0
                position.quantity = total_quantity
            else:
                # Reduce position
                if position.quantity >= quantity:
                    # Calculate realized P&L
                    realized_pnl = (price - position.avg_price) * quantity
                    position.realized_pnl += realized_pnl
                    position.quantity -= quantity
                    
                    # Remove position if completely closed
                    if position.quantity <= 0.01:  # Close to zero
                        del self.positions[symbol]
            
            if symbol in self.positions:
                self.positions[symbol].last_updated = datetime.now()
    
    def _update_positions(self, market_data: Dict[str, Any]):
        """Update all positions with current market prices"""
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['price']
                position.current_price = current_price
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity
    
    def _update_portfolio_value(self):
        """Update current portfolio value"""
        
        # Cash + unrealized P&L
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        self.current_capital = self.initial_capital + unrealized_pnl
        
        # Add to equity curve
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'value': self.current_capital,
            'unrealized_pnl': unrealized_pnl
        })
    
    def _update_stop_losses(self, market_data: Dict[str, Any]):
        """Update stop losses for active positions"""
        
        for symbol in list(self.positions.keys()):
            if symbol in market_data:
                current_price = market_data[symbol]['price']
                
                # Check if stop loss is triggered
                if self.stop_loss.is_stop_triggered(symbol, current_price):
                    self.logger.warning(f"Stop loss triggered for {symbol} at {current_price}")
                    
                    # Create market order to close position
                    position = self.positions[symbol]
                    close_order = PaperOrder(
                        id=str(uuid.uuid4()),
                        symbol=symbol,
                        side='sell' if position.quantity > 0 else 'buy',
                        quantity=abs(position.quantity),
                        order_type=OrderType.MARKET,
                        price=current_price,
                        created_at=datetime.now(),
                        strategy='stop_loss'
                    )
                    
                    # Immediately fill stop loss order
                    self._execute_fill(close_order, current_price, close_order.quantity)
                    self.stop_loss.remove_stop(symbol)
                
                # Update trailing stops
                else:
                    self.stop_loss.update_trailing_stop(symbol, current_price)
    
    def _can_trade(self) -> bool:
        """Check if trading is allowed"""
        
        # Check drawdown state
        if self.drawdown_controller.current_state.value in ['lockdown']:
            return False
        
        # Check if market is open (simplified)
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
        
        # Check trading hours (9:30 AM - 4:00 PM ET, simplified)
        if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
            return False
        
        return True
    
    def _check_rate_limit(self) -> bool:
        """Check if we can place another order (rate limiting)"""
        
        now = datetime.now()
        
        # Reset counter if new second
        if (now - self.last_order_time).total_seconds() >= 1.0:
            self.orders_this_second = 0
            self.last_order_time = now
        
        # Check limit
        if self.orders_this_second >= self.max_orders_per_second:
            return False
        
        self.orders_this_second += 1
        return True
    
    def _log_performance(self):
        """Log current performance metrics"""
        
        if len(self.equity_curve) < 2:
            return
        
        # Calculate metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Get equity values
        equity_values = [point['value'] for point in self.equity_curve]
        returns = np.diff(equity_values) / equity_values[:-1]
        
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            max_drawdown = self._calculate_max_drawdown(equity_values)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        performance = {
            'timestamp': datetime.now(),
            'total_return': total_return,
            'portfolio_value': self.current_capital,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trade_history),
            'open_positions': len(self.positions)
        }
        
        self.performance_metrics = performance
        self.logger.info(f"Performance: Return {total_return:.2%}, Sharpe {sharpe_ratio:.2f}, DD {max_drawdown:.2%}")
    
    def _calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = equity_values[0]
        max_dd = 0.0
        
        for value in equity_values:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    async def _shutdown(self):
        """Shutdown paper trader"""
        
        self.logger.info("Shutting down paper trader")
        
        # Close all positions
        for symbol in list(self.positions.keys()):
            self.logger.info(f"Closing position in {symbol}")
            # Implementation would close positions
        
        # Final performance report
        self._log_performance()
        
        # Save trade history and performance data
        self._save_results()
    
    def _save_results(self):
        """Save trading results for analysis"""
        
        try:
            # Save trade history
            if self.trade_history:
                trade_df = pd.DataFrame(self.trade_history)
                trade_df.to_csv('results/paper_trades.csv', index=False)
            
            # Save equity curve
            if self.equity_curve:
                equity_df = pd.DataFrame(self.equity_curve)
                equity_df.to_csv('results/paper_equity_curve.csv', index=False)
            
            # Save performance metrics
            if self.performance_metrics:
                import json
                with open('results/paper_performance.json', 'w') as f:
                    json.dump(self.performance_metrics, f, default=str, indent=2)
            
            self.logger.info("Results saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        
        return {
            'portfolio_value': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'open_positions': len(self.positions),
            'pending_orders': len([o for o in self.orders.values() if o.status == OrderStatus.PENDING]),
            'total_trades': len(self.trade_history),
            'drawdown_state': self.drawdown_controller.current_state.value,
            'is_trading': self._can_trade()
        }
