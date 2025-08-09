"""
Dynamic Stop Loss System for Optimal Risk Management
Implements multiple stop loss strategies with adaptive adjustment
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import logging


class StopLossType(Enum):
    FIXED_PERCENTAGE = "fixed_percentage"
    ATR_BASED = "atr_based"
    VOLATILITY_BASED = "volatility_based"
    TRAILING = "trailing"
    ADAPTIVE = "adaptive"
    SUPPORT_RESISTANCE = "support_resistance"


class DynamicStopLoss:
    """
    Advanced dynamic stop loss system that adapts to market conditions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Stop loss parameters
        self.base_stop_pct = config.get('stop_loss_pct', 0.02)  # 2%
        self.min_stop_pct = config.get('min_stop_pct', 0.005)   # 0.5%
        self.max_stop_pct = config.get('max_stop_pct', 0.05)    # 5%
        
        # ATR parameters
        self.atr_period = config.get('atr_period', 14)
        self.atr_multiplier = config.get('atr_multiplier', 2.0)
        
        # Trailing stop parameters
        self.trailing_stop_pct = config.get('trailing_stop_pct', 0.03)
        self.activation_threshold = config.get('activation_threshold', 0.02)
        
        # Adaptive parameters
        self.volatility_lookback = config.get('volatility_lookback', 20)
        self.regime_sensitivity = config.get('regime_sensitivity', 0.5)
        
        # Track active positions and stops
        self.active_stops = {}
        
    def calculate_stop_loss(self, symbol: str, entry_price: float, position_type: str,
                           data: pd.DataFrame, stop_type: StopLossType = StopLossType.ADAPTIVE,
                           **kwargs) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal stop loss level
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price for the position
            position_type: 'long' or 'short'
            data: Price data
            stop_type: Type of stop loss to calculate
            **kwargs: Additional parameters
        
        Returns:
            Tuple of (stop_price, metadata)
        """
        
        if len(data) < self.atr_period:
            # Fall back to fixed percentage for insufficient data
            return self._fixed_percentage_stop(entry_price, position_type), {}
        
        # Add technical indicators if not present
        if 'ATR' not in data.columns:
            data = self._add_technical_indicators(data)
        
        # Calculate stop based on type
        if stop_type == StopLossType.FIXED_PERCENTAGE:
            stop_price = self._fixed_percentage_stop(entry_price, position_type)
            metadata = {'type': 'fixed_percentage', 'percentage': self.base_stop_pct}
            
        elif stop_type == StopLossType.ATR_BASED:
            stop_price = self._atr_based_stop(entry_price, position_type, data)
            metadata = {'type': 'atr_based', 'atr_multiplier': self.atr_multiplier}
            
        elif stop_type == StopLossType.VOLATILITY_BASED:
            stop_price = self._volatility_based_stop(entry_price, position_type, data)
            metadata = {'type': 'volatility_based'}
            
        elif stop_type == StopLossType.SUPPORT_RESISTANCE:
            stop_price = self._support_resistance_stop(entry_price, position_type, data)
            metadata = {'type': 'support_resistance'}
            
        elif stop_type == StopLossType.ADAPTIVE:
            stop_price, metadata = self._adaptive_stop(entry_price, position_type, data)
            
        else:
            stop_price = self._fixed_percentage_stop(entry_price, position_type)
            metadata = {'type': 'default'}
        
        # Store active stop
        self.active_stops[symbol] = {
            'entry_price': entry_price,
            'stop_price': stop_price,
            'position_type': position_type,
            'stop_type': stop_type,
            'metadata': metadata
        }
        
        self.logger.info(f"Stop loss set for {symbol}: {stop_price:.4f} ({metadata.get('type', 'unknown')})")
        
        return stop_price, metadata
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add necessary technical indicators"""
        df = data.copy()
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(window=self.atr_period).mean()
        
        # Volatility
        returns = df['close'].pct_change()
        df['volatility'] = returns.rolling(window=self.volatility_lookback).std()
        
        # Support and Resistance levels
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_upper'] = sma_20 + (std_20 * 2)
        
        return df
    
    def _fixed_percentage_stop(self, entry_price: float, position_type: str) -> float:
        """Calculate fixed percentage stop loss"""
        if position_type.lower() == 'long':
            return entry_price * (1 - self.base_stop_pct)
        else:
            return entry_price * (1 + self.base_stop_pct)
    
    def _atr_based_stop(self, entry_price: float, position_type: str, data: pd.DataFrame) -> float:
        """Calculate ATR-based stop loss"""
        current_atr = data['ATR'].iloc[-1]
        
        if np.isnan(current_atr) or current_atr == 0:
            return self._fixed_percentage_stop(entry_price, position_type)
        
        if position_type.lower() == 'long':
            return entry_price - (current_atr * self.atr_multiplier)
        else:
            return entry_price + (current_atr * self.atr_multiplier)
    
    def _volatility_based_stop(self, entry_price: float, position_type: str, data: pd.DataFrame) -> float:
        """Calculate volatility-based stop loss"""
        current_vol = data['volatility'].iloc[-1]
        
        if np.isnan(current_vol) or current_vol == 0:
            return self._fixed_percentage_stop(entry_price, position_type)
        
        # Scale stop loss based on volatility
        vol_adjusted_stop = max(self.min_stop_pct, min(self.max_stop_pct, current_vol * 2))
        
        if position_type.lower() == 'long':
            return entry_price * (1 - vol_adjusted_stop)
        else:
            return entry_price * (1 + vol_adjusted_stop)
    
    def _support_resistance_stop(self, entry_price: float, position_type: str, data: pd.DataFrame) -> float:
        """Calculate support/resistance-based stop loss"""
        if position_type.lower() == 'long':
            # Place stop below recent support
            recent_support = data['support'].iloc[-5:].min()
            if not np.isnan(recent_support) and recent_support < entry_price:
                buffer = (entry_price - recent_support) * 0.1  # 10% buffer
                return recent_support - buffer
        else:
            # Place stop above recent resistance
            recent_resistance = data['resistance'].iloc[-5:].max()
            if not np.isnan(recent_resistance) and recent_resistance > entry_price:
                buffer = (recent_resistance - entry_price) * 0.1  # 10% buffer
                return recent_resistance + buffer
        
        # Fall back to ATR-based if support/resistance not viable
        return self._atr_based_stop(entry_price, position_type, data)
    
    def _adaptive_stop(self, entry_price: float, position_type: str, data: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Calculate adaptive stop loss using multiple methods"""
        
        # Calculate stops using different methods
        atr_stop = self._atr_based_stop(entry_price, position_type, data)
        vol_stop = self._volatility_based_stop(entry_price, position_type, data)
        sr_stop = self._support_resistance_stop(entry_price, position_type, data)
        fixed_stop = self._fixed_percentage_stop(entry_price, position_type)
        
        # Determine market regime
        regime = self._detect_market_regime(data)
        
        # Weight different stops based on regime
        if regime == 'trending':
            # In trending markets, use wider stops (ATR-based)
            final_stop = atr_stop
            weights = {'atr': 0.6, 'support_resistance': 0.3, 'volatility': 0.1}
        elif regime == 'range_bound':
            # In range-bound markets, use support/resistance
            final_stop = sr_stop
            weights = {'support_resistance': 0.5, 'volatility': 0.3, 'atr': 0.2}
        elif regime == 'high_volatility':
            # In high volatility, use volatility-based stops
            final_stop = vol_stop
            weights = {'volatility': 0.5, 'atr': 0.3, 'fixed': 0.2}
        else:
            # Default: weighted average of all methods
            stops = np.array([atr_stop, vol_stop, sr_stop, fixed_stop])
            weights_array = np.array([0.4, 0.3, 0.2, 0.1])
            
            # Remove invalid stops (NaN or extreme values)
            valid_mask = ~np.isnan(stops)
            if position_type.lower() == 'long':
                valid_mask &= (stops > 0) & (stops < entry_price)
            else:
                valid_mask &= (stops > entry_price)
            
            if np.any(valid_mask):
                final_stop = np.average(stops[valid_mask], weights=weights_array[valid_mask])
            else:
                final_stop = fixed_stop
            
            weights = {'adaptive': 1.0}
        
        # Apply risk limits
        final_stop = self._apply_risk_limits(entry_price, final_stop, position_type)
        
        metadata = {
            'type': 'adaptive',
            'regime': regime,
            'weights': weights,
            'component_stops': {
                'atr': atr_stop,
                'volatility': vol_stop,
                'support_resistance': sr_stop,
                'fixed': fixed_stop
            }
        }
        
        return final_stop, metadata
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime for stop loss adaptation"""
        
        if len(data) < 20:
            return 'normal'
        
        recent_data = data.tail(20)
        returns = recent_data['close'].pct_change().dropna()
        
        # Volatility regime
        volatility = returns.std()
        if volatility > 0.03:
            return 'high_volatility'
        elif volatility < 0.01:
            return 'low_volatility'
        
        # Trend regime
        price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1)
        if abs(price_change) > 0.1:
            return 'trending'
        elif abs(price_change) < 0.02:
            return 'range_bound'
        
        return 'normal'
    
    def _apply_risk_limits(self, entry_price: float, stop_price: float, position_type: str) -> float:
        """Apply risk limits to stop loss"""
        
        # Calculate stop loss percentage
        if position_type.lower() == 'long':
            stop_pct = (entry_price - stop_price) / entry_price
        else:
            stop_pct = (stop_price - entry_price) / entry_price
        
        # Apply limits
        stop_pct = max(self.min_stop_pct, min(self.max_stop_pct, stop_pct))
        
        # Recalculate stop price with limits
        if position_type.lower() == 'long':
            return entry_price * (1 - stop_pct)
        else:
            return entry_price * (1 + stop_pct)
    
    def update_trailing_stop(self, symbol: str, current_price: float) -> Optional[float]:
        """Update trailing stop for active position"""
        
        if symbol not in self.active_stops:
            return None
        
        position_info = self.active_stops[symbol]
        position_type = position_info['position_type']
        current_stop = position_info['stop_price']
        
        # Calculate new trailing stop
        if position_type.lower() == 'long':
            # For long positions, only move stop up
            new_stop = current_price * (1 - self.trailing_stop_pct)
            if new_stop > current_stop:
                self.active_stops[symbol]['stop_price'] = new_stop
                return new_stop
        else:
            # For short positions, only move stop down
            new_stop = current_price * (1 + self.trailing_stop_pct)
            if new_stop < current_stop:
                self.active_stops[symbol]['stop_price'] = new_stop
                return new_stop
        
        return current_stop
    
    def is_stop_triggered(self, symbol: str, current_price: float) -> bool:
        """Check if stop loss is triggered"""
        
        if symbol not in self.active_stops:
            return False
        
        position_info = self.active_stops[symbol]
        stop_price = position_info['stop_price']
        position_type = position_info['position_type']
        
        if position_type.lower() == 'long':
            return current_price <= stop_price
        else:
            return current_price >= stop_price
    
    def remove_stop(self, symbol: str):
        """Remove active stop loss for closed position"""
        if symbol in self.active_stops:
            del self.active_stops[symbol]
    
    def get_active_stops(self) -> Dict[str, Any]:
        """Get all active stop losses"""
        return self.active_stops.copy()
    
    def get_stop_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for stop loss effectiveness"""
        # This would track historical stop loss performance
        # Implementation depends on your trade tracking system
        return {
            'total_stops_triggered': 0,
            'average_loss_per_stop': 0.0,
            'stop_effectiveness_ratio': 0.0
        }
