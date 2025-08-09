"""
Advanced Backtesting Engine for Strategy Validation
High-performance backtesting with realistic simulation and comprehensive analytics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestTrade:
    """Individual trade record for backtesting"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: Optional[float] = None
    strategy: Optional[str] = None
    confidence: float = 0.0
    exit_reason: str = 'unknown'


class AdvancedBacktestEngine:
    """
    Sophisticated backtesting engine with advanced features
    """
    
    def __init__(self, strategies: List, position_sizer, stop_loss, 
                 drawdown_controller, config: Dict[str, Any]):
        
        self.strategies = strategies
        self.position_sizer = position_sizer
        self.stop_loss = stop_loss
        self.drawdown_controller = drawdown_controller
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Backtest parameters
        self.initial_capital = config.get('initial_capital', 100000)
        self.start_date = config.get('start_date', '2020-01-01')
        self.end_date = config.get('end_date', '2024-12-31')
        
        # Execution simulation
        self.commission_rate = config.get('commission_rate', 0.0001)
        self.slippage_rate = config.get('slippage_rate', 0.0005)
        self.include_transaction_costs = config.get('transaction_costs', True)
        
        # State tracking
        self.current_capital = self.initial_capital
        self.positions = {}  # symbol -> position info
        self.trades = []
        self.equity_curve = []
        self.performance_stats = {}
        
        self.logger.info("Advanced Backtest Engine initialized")
    
    async def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run comprehensive backtesting
        
        Args:
            data: Dictionary with symbol as key and historical data as value
            
        Returns:
            Comprehensive backtest results
        """
        
        self.logger.info("Starting advanced backtesting")
        
        try:
            # Prepare data
            aligned_data = self._align_data(data)
            
            # Initialize tracking
            self._initialize_backtest()
            
            # Run main backtest loop
            results = await self._run_backtest_loop(aligned_data)
            
            # Calculate comprehensive performance metrics
            performance = self._calculate_performance_metrics()
            
            # Generate detailed analysis
            analysis = self._generate_analysis()
            
            # Combine results
            final_results = {
                **performance,
                **analysis,
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'config': self.config
            }
            
            self.logger.info("Backtesting completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Backtesting failed: {str(e)}")
            raise
    
    def _align_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align data from multiple symbols into a single DataFrame"""
        
        aligned_data = {}
        
        for symbol, df in data.items():
            if df is None or len(df) == 0:
                continue
            
            # Filter date range
            mask = (df.index >= self.start_date) & (df.index <= self.end_date)
            filtered_df = df.loc[mask].copy()
            
            if len(filtered_df) == 0:
                continue
            
            # Add symbol prefix to columns
            for col in filtered_df.columns:
                if col != 'symbol':
                    new_col = f"{symbol}_{col}"
                    aligned_data[new_col] = filtered_df[col]
        
        if not aligned_data:
            raise ValueError("No valid data after alignment")
        
        # Create combined DataFrame
        combined_df = pd.DataFrame(aligned_data)
        combined_df = combined_df.dropna(how='all')
        
        self.logger.info(f"Data aligned: {len(combined_df)} rows, {len(aligned_data)} columns")
        return combined_df
    
    def _initialize_backtest(self):
        """Initialize backtest state"""
        self.current_capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        # Initialize drawdown controller
        self.drawdown_controller.reset_lockdown()
    
    async def _run_backtest_loop(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Main backtesting loop"""
        
        total_rows = len(data)
        self.logger.info(f"Processing {total_rows} time periods")
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            
            # Progress logging
            if i % 1000 == 0:
                progress = (i / total_rows) * 100
                self.logger.info(f"Backtest progress: {progress:.1f}%")
            
            # Update positions with current prices
            self._update_positions(row, timestamp)
            
            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(row)
            
            # Update drawdown controller
            self.drawdown_controller.update_portfolio_value(portfolio_value, timestamp)
            
            # Record equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'value': portfolio_value,
                'positions': len(self.positions)
            })
            
            # Check stop losses
            self._check_stop_losses(row, timestamp)
            
            # Generate new signals if allowed to trade
            if self._can_trade(timestamp):
                await self._process_signals(row, timestamp)
        
        return {'status': 'completed'}
    
    def _update_positions(self, row: pd.Series, timestamp: pd.Timestamp):
        """Update position values with current market prices"""
        
        for symbol in list(self.positions.keys()):
            price_col = f"{symbol}_close"
            
            if price_col in row and not pd.isna(row[price_col]):
                current_price = row[price_col]
                position = self.positions[symbol]
                
                # Update unrealized P&L
                if position['side'] == 'long':
                    position['unrealized_pnl'] = (current_price - position['avg_price']) * position['quantity']
                else:  # short
                    position['unrealized_pnl'] = (position['avg_price'] - current_price) * position['quantity']
                
                position['current_price'] = current_price
                position['last_updated'] = timestamp
    
    def _calculate_portfolio_value(self, row: pd.Series) -> float:
        """Calculate current portfolio value"""
        
        cash = self.current_capital
        unrealized_pnl = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        
        return cash + unrealized_pnl
    
    def _check_stop_losses(self, row: pd.Series, timestamp: pd.Timestamp):
        """Check and execute stop losses"""
        
        for symbol in list(self.positions.keys()):
            price_col = f"{symbol}_close"
            
            if price_col in row and not pd.isna(row[price_col]):
                current_price = row[price_col]
                
                if self.stop_loss.is_stop_triggered(symbol, current_price):
                    self.logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                    
                    # Close position
                    self._close_position(symbol, current_price, timestamp, 'stop_loss')
    
    async def _process_signals(self, row: pd.Series, timestamp: pd.Timestamp):
        """Process signals from strategies"""
        
        # Extract data for each symbol
        symbols = self._extract_symbols_from_row(row)
        
        for symbol in symbols:
            # Create symbol-specific data
            symbol_data = self._create_symbol_dataframe(symbol, row, timestamp)
            
            # Generate signals from all strategies
            for strategy in self.strategies:
                try:
                    signals = strategy.generate_signals(symbol_data)
                    
                    for signal in signals:
                        await self._process_signal(signal, row, timestamp)
                        
                except Exception as e:
                    self.logger.warning(f"Strategy {strategy.name} error: {str(e)}")
    
    def _extract_symbols_from_row(self, row: pd.Series) -> List[str]:
        """Extract available symbols from data row"""
        symbols = set()
        for col in row.index:
            if '_close' in col:
                symbol = col.replace('_close', '')
                symbols.add(symbol)
        return list(symbols)
    
    def _create_symbol_dataframe(self, symbol: str, row: pd.Series, timestamp: pd.Timestamp) -> pd.DataFrame:
        """Create DataFrame for a specific symbol"""
        
        # Get symbol-specific columns
        symbol_cols = [col for col in row.index if col.startswith(f"{symbol}_")]
        
        if not symbol_cols:
            return pd.DataFrame()
        
        # Create basic OHLCV data
        data_dict = {
            'symbol': symbol,
            'close': row.get(f"{symbol}_close", np.nan),
            'open': row.get(f"{symbol}_open", np.nan),
            'high': row.get(f"{symbol}_high", np.nan),
            'low': row.get(f"{symbol}_low", np.nan),
            'volume': row.get(f"{symbol}_volume", 0),
        }
        
        # Add any additional indicators
        for col in symbol_cols:
            indicator_name = col.replace(f"{symbol}_", "")
            if indicator_name not in data_dict:
                data_dict[indicator_name] = row[col]
        
        # Create DataFrame
        df = pd.DataFrame([data_dict], index=[timestamp])
        
        return df
    
    async def _process_signal(self, signal, row: pd.Series, timestamp: pd.Timestamp):
        """Process individual trading signal"""
        
        symbol = signal.symbol
        
        # Check if we can trade this signal
        position_size = self._calculate_position_size(signal)
        risk_amount = position_size * self._calculate_portfolio_value(row)
        
        can_trade, reason = self.drawdown_controller.can_open_new_position(
            position_size, risk_amount
        )
        
        if not can_trade:
            return
        
        # Check if we already have a position
        if symbol in self.positions:
            # Skip if same direction
            existing_side = self.positions[symbol]['side']
            new_side = 'long' if signal.signal.value > 0 else 'short'
            
            if existing_side == new_side:
                return
            else:
                # Close existing position first
                current_price = row[f"{symbol}_close"]
                self._close_position(symbol, current_price, timestamp, 'signal_reversal')
        
        # Open new position
        self._open_position(signal, row, timestamp)
    
    def _calculate_position_size(self, signal) -> float:
        """Calculate position size for signal"""
        
        # Use the position sizer
        volatility = 0.02  # Simplified, would use actual volatility
        position_size = self.position_sizer.calculate_position_size(
            signal.confidence,
            volatility,
            self.current_capital
        )
        
        # Apply drawdown scaling
        scaling_factor = self.drawdown_controller.get_position_scaling_factor()
        position_size *= scaling_factor
        
        return position_size
    
    def _open_position(self, signal, row: pd.Series, timestamp: pd.Timestamp):
        """Open new trading position"""
        
        symbol = signal.symbol
        price_col = f"{symbol}_close"
        
        if price_col not in row or pd.isna(row[price_col]):
            return
        
        entry_price = row[price_col]
        
        # Apply slippage
        if self.include_transaction_costs:
            if signal.signal.value > 0:  # Buy
                entry_price *= (1 + self.slippage_rate)
            else:  # Sell
                entry_price *= (1 - self.slippage_rate)
        
        # Calculate position size
        position_size = self._calculate_position_size(signal)
        dollar_amount = position_size * self.current_capital
        shares = dollar_amount / entry_price
        
        # Calculate commission
        commission = 0
        if self.include_transaction_costs:
            commission = shares * entry_price * self.commission_rate
        
        # Create position
        side = 'long' if signal.signal.value > 0 else 'short'
        
        self.positions[symbol] = {
            'side': side,
            'quantity': shares,
            'avg_price': entry_price,
            'current_price': entry_price,
            'unrealized_pnl': 0.0,
            'entry_time': timestamp,
            'strategy': getattr(signal, 'strategy', 'unknown'),
            'confidence': signal.confidence,
            'commission_paid': commission
        }
        
        # Update capital
        if side == 'long':
            self.current_capital -= (dollar_amount + commission)
        else:
            self.current_capital += (dollar_amount - commission)
        
        # Set stop loss
        self.stop_loss.calculate_stop_loss(
            symbol, entry_price, side, 
            self._create_symbol_dataframe(symbol, row, timestamp)
        )
        
        self.logger.debug(f"Opened {side} position in {symbol}: {shares:.2f} @ {entry_price:.4f}")
    
    def _close_position(self, symbol: str, exit_price: float, 
                       timestamp: pd.Timestamp, reason: str = 'unknown'):
        """Close existing position"""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Apply slippage on exit
        if self.include_transaction_costs:
            if position['side'] == 'long':
                exit_price *= (1 - self.slippage_rate)
            else:
                exit_price *= (1 + self.slippage_rate)
        
        # Calculate P&L
        if position['side'] == 'long':
            pnl = (exit_price - position['avg_price']) * position['quantity']
        else:  # short
            pnl = (position['avg_price'] - exit_price) * position['quantity']
        
        # Calculate commission
        commission = 0
        if self.include_transaction_costs:
            commission = position['quantity'] * exit_price * self.commission_rate
        
        net_pnl = pnl - commission - position['commission_paid']
        
        # Update capital
        if position['side'] == 'long':
            self.current_capital += (position['quantity'] * exit_price - commission)
        else:
            self.current_capital -= (position['quantity'] * exit_price + commission)
        
        # Record trade
        trade = BacktestTrade(
            entry_time=position['entry_time'],
            exit_time=timestamp,
            symbol=symbol,
            side=position['side'],
            entry_price=position['avg_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            pnl=net_pnl,
            strategy=position['strategy'],
            confidence=position['confidence'],
            exit_reason=reason
        )
        
        self.trades.append(trade)
        
        # Remove position and stop loss
        del self.positions[symbol]
        self.stop_loss.remove_stop(symbol)
        
        self.logger.debug(f"Closed {position['side']} position in {symbol}: PnL {net_pnl:.2f}")
    
    def _can_trade(self, timestamp: pd.Timestamp) -> bool:
        """Check if trading is allowed at this time"""
        
        # Check drawdown state
        if self.drawdown_controller.current_state.value == 'lockdown':
            return False
        
        # Check market hours (simplified)
        if timestamp.weekday() >= 5:  # Weekend
            return False
        
        return True
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        if not self.equity_curve or not self.trades:
            return {'error': 'insufficient_data'}
        
        # Equity curve analysis
        equity_values = [point['value'] for point in self.equity_curve]
        equity_series = pd.Series(equity_values)
        returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Drawdown analysis
        max_drawdown = self._calculate_max_drawdown(equity_values)
        calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown < 0 else 0
        
        # Trade analysis
        winning_trades = [t.pnl for t in self.trades if t.pnl > 0]
        losing_trades = [t.pnl for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        profit_factor = sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0.01
        sortino_ratio = (returns.mean() * np.sqrt(252) / downside_std) if downside_std > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(equity_values)) - 1,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'final_capital': equity_values[-1],
            'start_date': self.start_date,
            'end_date': self.end_date
        }
    
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
        
        return -max_dd  # Return as negative value
    
    def _generate_analysis(self) -> Dict[str, Any]:
        """Generate detailed analysis"""
        
        analysis = {}
        
        # Monthly returns analysis
        if len(self.equity_curve) > 30:
            analysis['monthly_returns'] = self._calculate_monthly_returns()
        
        # Strategy performance breakdown
        if self.trades:
            analysis['strategy_performance'] = self._analyze_strategy_performance()
        
        # Risk analysis
        analysis['risk_analysis'] = self._analyze_risk()
        
        return analysis
    
    def _calculate_monthly_returns(self) -> Dict[str, Any]:
        """Calculate monthly return statistics"""
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        
        # Resample to monthly
        monthly_values = equity_df['value'].resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()
        
        return {
            'mean_monthly_return': monthly_returns.mean(),
            'std_monthly_return': monthly_returns.std(),
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min(),
            'positive_months': (monthly_returns > 0).sum(),
            'negative_months': (monthly_returns < 0).sum()
        }
    
    def _analyze_strategy_performance(self) -> Dict[str, Any]:
        """Analyze performance by strategy"""
        
        strategy_stats = {}
        
        for trade in self.trades:
            strategy = trade.strategy or 'unknown'
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'trades': [],
                    'total_pnl': 0,
                    'wins': 0,
                    'losses': 0
                }
            
            strategy_stats[strategy]['trades'].append(trade.pnl)
            strategy_stats[strategy]['total_pnl'] += trade.pnl
            
            if trade.pnl > 0:
                strategy_stats[strategy]['wins'] += 1
            else:
                strategy_stats[strategy]['losses'] += 1
        
        # Calculate metrics for each strategy
        for strategy, stats in strategy_stats.items():
            total_trades = len(stats['trades'])
            if total_trades > 0:
                stats['win_rate'] = stats['wins'] / total_trades
                stats['avg_pnl'] = stats['total_pnl'] / total_trades
                stats['total_trades'] = total_trades
        
        return strategy_stats
    
    def _analyze_risk(self) -> Dict[str, Any]:
        """Perform risk analysis"""
        
        if not self.equity_curve:
            return {}
        
        equity_values = [point['value'] for point in self.equity_curve]
        returns = pd.Series(equity_values).pct_change().dropna()
        
        # Value at Risk (VaR)
        var_95 = returns.quantile(0.05) if len(returns) > 20 else 0
        var_99 = returns.quantile(0.01) if len(returns) > 100 else 0
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'skewness': returns.skew() if len(returns) > 3 else 0,
            'kurtosis': returns.kurtosis() if len(returns) > 3 else 0
        }
