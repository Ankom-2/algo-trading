"""
Base Strategy Class for World-Class Algorithmic Trading System
Provides the foundation for all trading strategies with advanced features
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging


class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0
    STRONG_BUY = 2
    STRONG_SELL = -2


@dataclass
class TradingSignal:
    """Trading signal with comprehensive information"""
    symbol: str
    signal: SignalType
    confidence: float  # 0-1 confidence level
    price: float
    timestamp: pd.Timestamp
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade_duration: float = 0.0
    volatility: float = 0.0


class BaseStrategy(ABC):
    """
    Advanced base class for all trading strategies
    Implements sophisticated risk management and signal generation
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        self.name = name
        self.parameters = parameters
        self.logger = logging.getLogger(f"Strategy.{name}")
        
        # Strategy state
        self.is_active = True
        self.positions = {}
        self.signals_history = []
        self.performance_metrics = StrategyMetrics()
        
        # Risk management parameters
        self.max_position_size = parameters.get('max_position_size', 0.1)
        self.stop_loss_pct = parameters.get('stop_loss_pct', 0.02)
        self.take_profit_pct = parameters.get('take_profit_pct', 0.06)
        
        # Technical indicators cache
        self._indicators_cache = {}
        
        self.logger.info(f"Initialized {name} strategy")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate trading signals based on market data
        Must be implemented by each strategy
        """
        pass
    
    @abstractmethod
    def calculate_confidence(self, data: pd.DataFrame, signal: SignalType) -> float:
        """
        Calculate confidence level for a signal
        Must be implemented by each strategy
        """
        pass
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """Update strategy parameters dynamically"""
        self.parameters.update(new_parameters)
        self.logger.info(f"Updated parameters: {new_parameters}")
    
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float, 
                              volatility: float) -> float:
        """Calculate optimal position size using Kelly Criterion and volatility scaling"""
        if signal.confidence == 0:
            return 0.0
        
        # Kelly Criterion with confidence adjustment
        win_rate = max(0.5, signal.confidence)  # Minimum 50% assumed win rate
        avg_win = self.take_profit_pct
        avg_loss = self.stop_loss_pct
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        # Apply volatility scaling
        vol_scalar = min(1.0, 0.15 / volatility) if volatility > 0 else 1.0
        
        # Final position size with safety limits
        position_size = kelly_fraction * vol_scalar * signal.confidence
        position_size = max(0, min(position_size, self.max_position_size))
        
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, signal_type: SignalType, 
                           volatility: float) -> float:
        """Calculate dynamic stop loss based on volatility"""
        base_stop = self.stop_loss_pct
        vol_adjustment = min(0.05, volatility * 2)  # Cap at 5%
        
        dynamic_stop = base_stop + vol_adjustment
        
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return entry_price * (1 - dynamic_stop)
        else:
            return entry_price * (1 + dynamic_stop)
    
    def calculate_take_profit(self, entry_price: float, signal_type: SignalType) -> float:
        """Calculate take profit level"""
        if signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return entry_price * (1 + self.take_profit_pct)
        else:
            return entry_price * (1 - self.take_profit_pct)
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to the data"""
        df = data.copy()
        
        # Moving averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        df['BB_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std = df['close'].rolling(window=bb_period).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def validate_signal(self, signal: TradingSignal, market_conditions: Dict[str, Any]) -> bool:
        """Validate signal against market conditions and risk parameters"""
        
        # Minimum confidence threshold
        if signal.confidence < 0.6:
            return False
        
        # Market volatility check
        if market_conditions.get('volatility', 0) > 0.05:  # 5% daily volatility threshold
            return False
        
        # Volume confirmation
        if market_conditions.get('volume_ratio', 1) < 0.8:  # Minimum volume requirement
            return False
        
        return True
    
    def get_performance_metrics(self) -> StrategyMetrics:
        """Get current strategy performance metrics"""
        return self.performance_metrics
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.performance_metrics = StrategyMetrics()
    
    def log_signal(self, signal: TradingSignal):
        """Log trading signal for analysis"""
        self.signals_history.append(signal)
        self.logger.info(
            f"Generated {signal.signal.name} signal for {signal.symbol} "
            f"at {signal.price:.2f} with confidence {signal.confidence:.2f}"
        )
    
    def __str__(self) -> str:
        return f"{self.name} Strategy (Active: {self.is_active})"
