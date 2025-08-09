"""
Advanced Position Sizing for Optimal Risk-Adjusted Returns
Implements Kelly Criterion, Risk Parity, and Volatility Scaling
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class PositionSizing:
    """Base class for position sizing algorithms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_position_size = config.get('max_position_size', 0.1)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        
    def calculate_position_size(self, signal_confidence: float, volatility: float,
                              portfolio_value: float, **kwargs) -> float:
        """Calculate position size - to be implemented by subclasses"""
        raise NotImplementedError


class KellyPositionSizing(PositionSizing):
    """
    Kelly Criterion position sizing with modifications for practical trading
    Maximizes log utility while controlling risk
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kelly_fraction = config.get('kelly_fraction', 0.25)  # Conservative Kelly
        self.min_win_rate = config.get('min_win_rate', 0.5)
        self.avg_win_loss_ratio = config.get('avg_win_loss_ratio', 2.0)
        
    def calculate_position_size(self, signal_confidence: float, volatility: float,
                              portfolio_value: float, **kwargs) -> float:
        """
        Calculate Kelly-based position size
        
        Args:
            signal_confidence: Confidence in the signal (0-1)
            volatility: Asset volatility
            portfolio_value: Current portfolio value
            **kwargs: Additional parameters (stop_loss_pct, take_profit_pct, etc.)
        """
        
        # Extract additional parameters
        stop_loss_pct = kwargs.get('stop_loss_pct', 0.02)
        take_profit_pct = kwargs.get('take_profit_pct', 0.06)
        
        # Estimate win rate based on signal confidence
        win_rate = max(self.min_win_rate, signal_confidence)
        
        # Estimate average win/loss based on stop loss and take profit
        avg_win = take_profit_pct
        avg_loss = stop_loss_pct
        
        # Kelly Criterion: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly_f = (b * win_rate - (1 - win_rate)) / b
        else:
            kelly_f = 0
        
        # Apply conservative scaling
        kelly_f *= self.kelly_fraction
        
        # Volatility adjustment
        vol_scalar = self._calculate_volatility_scalar(volatility)
        kelly_f *= vol_scalar
        
        # Confidence adjustment
        confidence_scalar = self._calculate_confidence_scalar(signal_confidence)
        kelly_f *= confidence_scalar
        
        # Apply position limits
        position_size = max(0, min(kelly_f, self.max_position_size))
        
        return position_size
    
    def _calculate_volatility_scalar(self, volatility: float) -> float:
        """Scale position size based on volatility"""
        target_vol = 0.02  # 2% daily volatility target
        if volatility > 0:
            scalar = min(1.0, target_vol / volatility)
        else:
            scalar = 1.0
        
        return max(0.1, scalar)  # Minimum 10% scaling
    
    def _calculate_confidence_scalar(self, confidence: float) -> float:
        """Scale position size based on signal confidence"""
        # Exponential scaling to reward high confidence
        return confidence ** 2
