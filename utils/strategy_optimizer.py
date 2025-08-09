"""
Strategy Optimizer - Automatic Optimal Strategy Selection
Continuously evaluates strategies and selects the best performing one
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import pickle
import os

from ..strategies.base import BaseStrategy, StrategyMetrics
from ..strategies.momentum import MomentumStrategy
from ..strategies.mean_reversion import MeanReversionStrategy
from ..strategies.adaptive import AdaptiveStrategy
from ..strategies.options import LongStraddleStrategy, LongStrangleStrategy
from ..config import TRADING_CONFIG


@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_duration: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    last_updated: datetime = None
    evaluation_period_days: int = 30
    confidence_score: float = 0.0


@dataclass
class MarketRegime:
    """Current market regime classification"""
    trend_strength: float  # -1 to 1 (bearish to bullish)
    volatility_regime: str  # 'low', 'medium', 'high'
    market_phase: str  # 'trending', 'ranging', 'volatile'
    momentum_strength: float  # 0 to 1
    mean_reversion_strength: float  # 0 to 1
    volatility_percentile: float  # 0 to 1
    regime_stability: float  # 0 to 1 (how stable the current regime is)


class StrategyOptimizer:
    """Automatic strategy selection and optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or TRADING_CONFIG['strategy_optimization']
        self.logger = logging.getLogger(__name__)
        
        # Initialize all available strategies
        self.strategies = {
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'adaptive': AdaptiveStrategy(),
            'long_straddle': LongStraddleStrategy(),
            'long_strangle': LongStrangleStrategy()
        }
        
        # Performance tracking
        self.performance_history: Dict[str, List[StrategyPerformance]] = defaultdict(list)
        self.current_strategy = 'adaptive'  # Default strategy
        self.last_strategy_switch = datetime.now()
        self.market_regime_history: List[MarketRegime] = []
        
        # Performance weights from config
        self.performance_weights = self.config.get('performance_weights', {
            'sharpe_ratio': 0.3,
            'total_return': 0.25,
            'max_drawdown': 0.2,
            'win_rate': 0.15,
            'profit_factor': 0.1
        })
    
    def evaluate_strategies(self, data: pd.DataFrame, symbols: List[str]) -> Dict[str, StrategyPerformance]:
        """Evaluate all strategies on recent data"""
        self.logger.info("Evaluating all strategies for optimal selection...")
        
        # Get evaluation period
        eval_days = self.config.get('evaluation_period', 30)
        eval_start = datetime.now() - timedelta(days=eval_days)
        
        # Filter data to evaluation period
        if isinstance(data.index, pd.DatetimeIndex):
            eval_data = data[data.index >= eval_start]
        else:
            eval_data = data.tail(eval_days)
        
        strategy_performances = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                performance = self._evaluate_single_strategy(
                    strategy, eval_data, symbols, strategy_name
                )
                strategy_performances[strategy_name] = performance
                
                # Update performance history
                self.performance_history[strategy_name].append(performance)
                
                # Keep only recent history (last 100 evaluations)
                if len(self.performance_history[strategy_name]) > 100:
                    self.performance_history[strategy_name] = \
                        self.performance_history[strategy_name][-100:]
                
            except Exception as e:
                self.logger.error(f"Error evaluating strategy {strategy_name}: {e}")
                continue
        
        return strategy_performances
    
    def _evaluate_single_strategy(self, strategy: BaseStrategy, data: pd.DataFrame, 
                                 symbols: List[str], strategy_name: str) -> StrategyPerformance:
        """Evaluate a single strategy"""
        all_returns = []
        all_signals = []
        trade_durations = []
        win_count = 0
        loss_count = 0
        gross_profit = 0.0
        gross_loss = 0.0
        
        # Simulate trading for each symbol
        for symbol in symbols:
            if symbol not in data.columns:
                continue
            
            symbol_data = pd.DataFrame({'close': data[symbol]}).dropna()
            if len(symbol_data) < 20:
                continue
            
            # Generate signals
            signals = strategy.calculate_signals(symbol_data, symbol)
            all_signals.extend(signals)
            
            # Simulate trades
            returns, trades_info = self._simulate_trades(symbol_data, signals)
            all_returns.extend(returns)
            
            # Analyze trades
            for trade in trades_info:
                trade_durations.append(trade['duration'])
                if trade['return'] > 0:
                    win_count += 1
                    gross_profit += trade['return']
                else:
                    loss_count += 1
                    gross_loss += abs(trade['return'])
        
        # Calculate performance metrics
        if not all_returns:
            return StrategyPerformance(strategy_name=strategy_name, last_updated=datetime.now())
        
        returns_series = pd.Series(all_returns)
        
        total_return = (1 + returns_series).prod() - 1
        volatility = returns_series.std() * np.sqrt(252) if len(returns_series) > 1 else 0
        sharpe_ratio = (returns_series.mean() * 252) / (volatility + 1e-6)
        
        # Calculate drawdown
        cumulative_returns = (1 + returns_series).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate other metrics
        win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
        profit_factor = gross_profit / (gross_loss + 1e-6) if gross_loss > 0 else 0
        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0
        
        # Calculate Calmar and Sortino ratios
        calmar_ratio = (returns_series.mean() * 252) / (abs(max_drawdown) + 1e-6)
        downside_returns = returns_series[returns_series < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-6
        sortino_ratio = (returns_series.mean() * 252) / downside_std
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            len(all_returns), win_rate, sharpe_ratio, len(symbols)
        )
        
        performance = StrategyPerformance(
            strategy_name=strategy_name,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trade_durations),
            avg_trade_duration=avg_trade_duration,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            last_updated=datetime.now(),
            confidence_score=confidence_score
        )
        
        return performance
    
    def _simulate_trades(self, data: pd.DataFrame, signals: List) -> Tuple[List[float], List[Dict]]:
        """Simulate trades based on signals"""
        returns = []
        trades = []
        position = None
        
        for signal in signals:
            if signal.signal.value > 0 and position is None:  # Buy signal
                position = {
                    'entry_price': signal.price,
                    'entry_time': signal.timestamp,
                    'type': 'long'
                }
            elif signal.signal.value < 0 and position is not None:  # Sell signal
                # Calculate return
                trade_return = (signal.price - position['entry_price']) / position['entry_price']
                returns.append(trade_return)
                
                # Calculate duration
                if hasattr(signal.timestamp, 'days'):
                    duration = (signal.timestamp - position['entry_time']).days
                else:
                    duration = 1  # Default duration
                
                trades.append({
                    'return': trade_return,
                    'duration': duration,
                    'entry_price': position['entry_price'],
                    'exit_price': signal.price
                })
                
                position = None
        
        return returns, trades
    
    def _calculate_confidence_score(self, num_trades: int, win_rate: float, 
                                  sharpe_ratio: float, num_symbols: int) -> float:
        """Calculate confidence score for strategy performance"""
        # More trades = higher confidence
        trade_confidence = min(num_trades / 50, 1.0)
        
        # Better metrics = higher confidence
        win_rate_confidence = min(win_rate * 2, 1.0)  # Scale 0.5-1.0 to 1.0-2.0
        sharpe_confidence = min(abs(sharpe_ratio) / 2, 1.0)
        
        # More symbols = higher confidence
        symbol_confidence = min(num_symbols / 10, 1.0)
        
        # Weighted average
        confidence = (
            trade_confidence * 0.4 +
            win_rate_confidence * 0.3 +
            sharpe_confidence * 0.2 +
            symbol_confidence * 0.1
        )
        
        return min(confidence, 1.0)
    
    def select_optimal_strategy(self, strategy_performances: Dict[str, StrategyPerformance], 
                               market_regime: MarketRegime = None) -> str:
        """Select the optimal strategy based on performance and market regime"""
        
        if not strategy_performances:
            return self.current_strategy
        
        # Calculate composite scores for each strategy
        strategy_scores = {}
        
        for strategy_name, performance in strategy_performances.items():
            # Skip strategies with insufficient data
            if performance.confidence_score < 0.3:
                continue
            
            score = self._calculate_composite_score(performance, market_regime)
            strategy_scores[strategy_name] = score
            
            self.logger.info(f"Strategy {strategy_name}: Score={score:.3f}, "
                           f"Return={performance.total_return:.2%}, "
                           f"Sharpe={performance.sharpe_ratio:.2f}")
        
        if not strategy_scores:
            self.logger.warning("No strategies meet confidence threshold, keeping current")
            return self.current_strategy
        
        # Select best strategy
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        # Check if we should switch strategies
        current_score = strategy_scores.get(self.current_strategy, 0)
        best_score = strategy_scores[best_strategy]
        
        # Only switch if the improvement is significant and cooldown period has passed
        min_improvement = 0.1  # 10% improvement required
        cooldown_days = self.config.get('strategy_switching_cooldown', 5)
        days_since_switch = (datetime.now() - self.last_strategy_switch).days
        
        should_switch = (
            best_score > current_score * (1 + min_improvement) and
            days_since_switch >= cooldown_days and
            best_strategy != self.current_strategy
        )
        
        if should_switch:
            self.logger.info(f"Switching strategy from {self.current_strategy} to {best_strategy}")
            self.logger.info(f"Score improvement: {current_score:.3f} -> {best_score:.3f}")
            self.current_strategy = best_strategy
            self.last_strategy_switch = datetime.now()
        
        return self.current_strategy
    
    def _calculate_composite_score(self, performance: StrategyPerformance, 
                                  market_regime: MarketRegime = None) -> float:
        """Calculate composite score for strategy ranking"""
        # Normalize metrics to 0-1 scale
        return_score = max(min(performance.total_return * 2, 1.0), 0.0)  # Normalize around 50% annual return
        sharpe_score = max(min(performance.sharpe_ratio / 3, 1.0), 0.0)  # Normalize around 3.0 Sharpe
        drawdown_score = max(1.0 + performance.max_drawdown * 5, 0.0)  # Invert drawdown (less is better)
        win_rate_score = performance.win_rate
        profit_factor_score = max(min(performance.profit_factor / 3, 1.0), 0.0)
        
        # Apply weights
        composite_score = (
            return_score * self.performance_weights['total_return'] +
            sharpe_score * self.performance_weights['sharpe_ratio'] +
            drawdown_score * self.performance_weights['max_drawdown'] +
            win_rate_score * self.performance_weights['win_rate'] +
            profit_factor_score * self.performance_weights['profit_factor']
        )
        
        # Apply confidence multiplier
        composite_score *= performance.confidence_score
        
        # Market regime adjustment
        if market_regime:
            regime_multiplier = self._get_regime_multiplier(performance.strategy_name, market_regime)
            composite_score *= regime_multiplier
        
        return composite_score
    
    def _get_regime_multiplier(self, strategy_name: str, market_regime: MarketRegime) -> float:
        """Get multiplier based on how well strategy fits current market regime"""
        multipliers = {
            'momentum': {
                'trending': 1.2,
                'ranging': 0.8,
                'volatile': 0.9
            },
            'mean_reversion': {
                'trending': 0.8,
                'ranging': 1.2,
                'volatile': 0.9
            },
            'long_straddle': {
                'trending': 0.7,
                'ranging': 0.9,
                'volatile': 1.3
            },
            'long_strangle': {
                'trending': 0.8,
                'ranging': 0.9,
                'volatile': 1.2
            },
            'adaptive': {
                'trending': 1.0,
                'ranging': 1.0,
                'volatile': 1.1
            }
        }
        
        return multipliers.get(strategy_name, {}).get(market_regime.market_phase, 1.0)
    
    def analyze_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Analyze current market regime"""
        if len(data) < 50:
            # Default regime for insufficient data
            return MarketRegime(
                trend_strength=0.0,
                volatility_regime='medium',
                market_phase='ranging',
                momentum_strength=0.5,
                mean_reversion_strength=0.5,
                volatility_percentile=0.5,
                regime_stability=0.5
            )
        
        # Calculate trend strength using multiple timeframes
        returns = data.pct_change().dropna()
        
        # Short-term trend (10 days)
        short_trend = returns.rolling(10).mean().iloc[-1]
        # Medium-term trend (20 days)  
        medium_trend = returns.rolling(20).mean().iloc[-1]
        # Long-term trend (50 days)
        long_trend = returns.rolling(50).mean().iloc[-1]
        
        trend_strength = (short_trend + medium_trend + long_trend) / 3
        
        # Calculate volatility
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        vol_percentile = (returns.rolling(20).std() <= volatility).mean()
        
        # Determine volatility regime
        if vol_percentile > 0.8:
            volatility_regime = 'high'
        elif vol_percentile < 0.3:
            volatility_regime = 'low'
        else:
            volatility_regime = 'medium'
        
        # Calculate momentum and mean reversion strengths
        momentum_strength = abs(trend_strength) * 10  # Scale and take absolute value
        momentum_strength = min(momentum_strength, 1.0)
        
        # Mean reversion strength (inverse of momentum when volatility is high)
        mean_reversion_strength = (1 - momentum_strength) * (vol_percentile * 2)
        mean_reversion_strength = min(mean_reversion_strength, 1.0)
        
        # Determine market phase
        if abs(trend_strength) > 0.01 and volatility < 0.25:
            market_phase = 'trending'
        elif volatility > 0.3:
            market_phase = 'volatile'
        else:
            market_phase = 'ranging'
        
        # Calculate regime stability (how consistent recent regime has been)
        recent_volatilities = returns.rolling(5).std().tail(10)
        regime_stability = 1 - (recent_volatilities.std() / recent_volatilities.mean())
        regime_stability = max(min(regime_stability, 1.0), 0.0)
        
        regime = MarketRegime(
            trend_strength=trend_strength * 10,  # Scale to reasonable range
            volatility_regime=volatility_regime,
            market_phase=market_phase,
            momentum_strength=momentum_strength,
            mean_reversion_strength=mean_reversion_strength,
            volatility_percentile=vol_percentile,
            regime_stability=regime_stability
        )
        
        self.market_regime_history.append(regime)
        
        # Keep only recent history
        if len(self.market_regime_history) > 100:
            self.market_regime_history = self.market_regime_history[-100:]
        
        return regime
    
    def get_current_strategy(self) -> BaseStrategy:
        """Get the currently selected optimal strategy"""
        return self.strategies[self.current_strategy]
    
    def save_performance_history(self, filepath: str):
        """Save performance history to file"""
        data = {
            'performance_history': dict(self.performance_history),
            'current_strategy': self.current_strategy,
            'last_strategy_switch': self.last_strategy_switch,
            'market_regime_history': self.market_regime_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Performance history saved to {filepath}")
    
    def load_performance_history(self, filepath: str):
        """Load performance history from file"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                self.performance_history = defaultdict(list, data['performance_history'])
                self.current_strategy = data.get('current_strategy', 'adaptive')
                self.last_strategy_switch = data.get('last_strategy_switch', datetime.now())
                self.market_regime_history = data.get('market_regime_history', [])
                
                self.logger.info(f"Performance history loaded from {filepath}")
                
            except Exception as e:
                self.logger.error(f"Error loading performance history: {e}")
        else:
            self.logger.info(f"Performance history file {filepath} not found, starting fresh")
