"""
Advanced Drawdown Control System
Implements multiple drawdown protection mechanisms to preserve capital
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import logging
from datetime import datetime, timedelta


class DrawdownState(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    LOCKDOWN = "lockdown"


class DrawdownController:
    """
    Sophisticated drawdown control system with multiple protection layers
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Drawdown thresholds
        self.max_drawdown = config.get('max_drawdown', 0.1)  # 10%
        self.warning_drawdown = config.get('warning_drawdown', 0.05)  # 5%
        self.critical_drawdown = config.get('critical_drawdown', 0.08)  # 8%
        
        # Daily loss limits
        self.max_daily_loss = config.get('max_daily_loss', 0.03)  # 3%
        self.max_consecutive_losses = config.get('max_consecutive_losses', 5)
        
        # Recovery parameters
        self.recovery_threshold = config.get('recovery_threshold', 0.5)  # 50% drawdown recovery
        self.lockdown_duration = config.get('lockdown_duration', 24)  # hours
        
        # State tracking
        self.current_state = DrawdownState.NORMAL
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.daily_loss = 0.0
        self.consecutive_losses = 0
        self.last_update = None
        self.lockdown_until = None
        
        # Historical tracking
        self.equity_curve = []
        self.drawdown_history = []
        self.daily_returns = []
        
        self.logger.info("Drawdown Controller initialized")
    
    def update_portfolio_value(self, current_value: float, timestamp: Optional[datetime] = None) -> DrawdownState:
        """
        Update portfolio value and assess drawdown state
        
        Args:
            current_value: Current portfolio value
            timestamp: Timestamp of update
            
        Returns:
            Current drawdown state
        """
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Initialize peak if first update
        if self.peak_value == 0:
            self.peak_value = current_value
            self.last_update = timestamp
            return self.current_state
        
        # Update equity curve
        self.equity_curve.append({
            'timestamp': timestamp,
            'value': current_value,
            'peak': self.peak_value
        })
        
        # Calculate daily return if new day
        if self._is_new_day(timestamp):
            if len(self.equity_curve) > 1:
                prev_value = self.equity_curve[-2]['value']
                daily_return = (current_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
                self.daily_loss = min(0, daily_return)  # Reset daily loss for new day
            
            # Update consecutive losses
            if self.daily_loss < -0.01:  # More than 1% loss
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
        
        # Update peak value
        if current_value > self.peak_value:
            self.peak_value = current_value
            # Reset consecutive losses on new high
            self.consecutive_losses = 0
        
        # Calculate current drawdown
        self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        
        # Update drawdown history
        self.drawdown_history.append({
            'timestamp': timestamp,
            'drawdown': self.current_drawdown,
            'value': current_value
        })
        
        # Determine new state
        new_state = self._assess_drawdown_state()
        
        if new_state != self.current_state:
            self.logger.warning(f"Drawdown state changed: {self.current_state.value} -> {new_state.value}")
            self.current_state = new_state
        
        self.last_update = timestamp
        return self.current_state
    
    def _assess_drawdown_state(self) -> DrawdownState:
        """Assess current drawdown state based on multiple factors"""
        
        # Check if in lockdown
        if self.lockdown_until and datetime.now() < self.lockdown_until:
            return DrawdownState.LOCKDOWN
        
        # Check daily loss limits
        if abs(self.daily_loss) > self.max_daily_loss:
            self.logger.critical(f"Daily loss limit exceeded: {self.daily_loss:.2%}")
            self._trigger_lockdown()
            return DrawdownState.LOCKDOWN
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.logger.critical(f"Consecutive loss limit exceeded: {self.consecutive_losses}")
            self._trigger_lockdown()
            return DrawdownState.LOCKDOWN
        
        # Check drawdown levels
        if self.current_drawdown >= self.max_drawdown:
            self.logger.critical(f"Maximum drawdown reached: {self.current_drawdown:.2%}")
            self._trigger_lockdown()
            return DrawdownState.LOCKDOWN
        elif self.current_drawdown >= self.critical_drawdown:
            return DrawdownState.CRITICAL
        elif self.current_drawdown >= self.warning_drawdown:
            return DrawdownState.WARNING
        else:
            return DrawdownState.NORMAL
    
    def _trigger_lockdown(self):
        """Trigger emergency lockdown"""
        self.lockdown_until = datetime.now() + timedelta(hours=self.lockdown_duration)
        self.logger.critical(f"EMERGENCY LOCKDOWN TRIGGERED until {self.lockdown_until}")
    
    def _is_new_day(self, timestamp: datetime) -> bool:
        """Check if timestamp represents a new trading day"""
        if self.last_update is None:
            return True
        return timestamp.date() != self.last_update.date()
    
    def can_open_new_position(self, position_size: float, risk_amount: float) -> Tuple[bool, str]:
        """
        Check if new position can be opened given current drawdown state
        
        Args:
            position_size: Requested position size (as fraction of portfolio)
            risk_amount: Dollar amount at risk
            
        Returns:
            Tuple of (can_open, reason)
        """
        
        # Check lockdown
        if self.current_state == DrawdownState.LOCKDOWN:
            return False, "System in lockdown - no new positions allowed"
        
        # Check if position would exceed daily loss limit
        potential_daily_loss = abs(self.daily_loss) + (risk_amount / self.peak_value)
        if potential_daily_loss > self.max_daily_loss:
            return False, f"Position would exceed daily loss limit ({potential_daily_loss:.2%})"
        
        # Apply position size limits based on state
        max_position_size = self._get_max_position_size()
        if position_size > max_position_size:
            return False, f"Position size {position_size:.2%} exceeds limit {max_position_size:.2%}"
        
        return True, "Position approved"
    
    def _get_max_position_size(self) -> float:
        """Get maximum allowed position size based on current state"""
        
        if self.current_state == DrawdownState.NORMAL:
            return 0.10  # 10% max position
        elif self.current_state == DrawdownState.WARNING:
            return 0.06  # 6% max position
        elif self.current_state == DrawdownState.CRITICAL:
            return 0.03  # 3% max position
        else:  # LOCKDOWN
            return 0.0   # No new positions
    
    def get_position_scaling_factor(self) -> float:
        """Get position scaling factor based on drawdown state"""
        
        if self.current_state == DrawdownState.NORMAL:
            return 1.0
        elif self.current_state == DrawdownState.WARNING:
            return 0.7  # Reduce position sizes by 30%
        elif self.current_state == DrawdownState.CRITICAL:
            return 0.4  # Reduce position sizes by 60%
        else:  # LOCKDOWN
            return 0.0  # No new positions
    
    def should_close_positions(self) -> Tuple[bool, float]:
        """
        Determine if existing positions should be closed
        
        Returns:
            Tuple of (should_close, close_percentage)
        """
        
        if self.current_state == DrawdownState.LOCKDOWN:
            return True, 1.0  # Close all positions
        elif self.current_state == DrawdownState.CRITICAL:
            # Close half of positions
            return True, 0.5
        elif self.current_state == DrawdownState.WARNING:
            # Close losing positions only
            return False, 0.0
        else:
            return False, 0.0
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics"""
        
        if len(self.equity_curve) < 2:
            return {'status': 'insufficient_data'}
        
        # Calculate metrics
        values = [point['value'] for point in self.equity_curve]
        returns = np.diff(values) / values[:-1]
        
        # Drawdown metrics
        max_dd_period = self._calculate_max_drawdown_period()
        avg_drawdown = np.mean([dd['drawdown'] for dd in self.drawdown_history if dd['drawdown'] > 0])
        
        # Return metrics
        total_return = (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
        volatility = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = (np.mean(returns) / volatility) if volatility > 0 else 0
        
        # Risk metrics
        var_95 = np.percentile(returns, 5) if len(returns) > 10 else 0
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else 0
        
        return {
            'current_drawdown': self.current_drawdown,
            'max_drawdown': max(dd['drawdown'] for dd in self.drawdown_history) if self.drawdown_history else 0,
            'avg_drawdown': avg_drawdown if not np.isnan(avg_drawdown) else 0,
            'max_drawdown_duration': max_dd_period,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'consecutive_losses': self.consecutive_losses,
            'daily_loss': self.daily_loss,
            'current_state': self.current_state.value,
            'days_in_drawdown': self._days_in_current_drawdown()
        }
    
    def _calculate_max_drawdown_period(self) -> int:
        """Calculate maximum drawdown duration in days"""
        if not self.drawdown_history:
            return 0
        
        max_period = 0
        current_period = 0
        
        for dd in self.drawdown_history:
            if dd['drawdown'] > 0:
                current_period += 1
                max_period = max(max_period, current_period)
            else:
                current_period = 0
        
        return max_period
    
    def _days_in_current_drawdown(self) -> int:
        """Calculate days in current drawdown"""
        if self.current_drawdown <= 0:
            return 0
        
        days = 0
        for dd in reversed(self.drawdown_history):
            if dd['drawdown'] > 0:
                days += 1
            else:
                break
        
        return days
    
    def generate_recovery_plan(self) -> Dict[str, Any]:
        """Generate recovery plan based on current drawdown state"""
        
        if self.current_drawdown <= 0:
            return {'status': 'no_recovery_needed'}
        
        # Calculate required return to recover
        required_return = self.current_drawdown / (1 - self.current_drawdown)
        
        # Estimate recovery time based on historical performance
        if len(self.daily_returns) > 20:
            avg_daily_return = np.mean(self.daily_returns)
            if avg_daily_return > 0:
                estimated_days = int(required_return / avg_daily_return)
            else:
                estimated_days = 999  # Unknown
        else:
            estimated_days = 999  # Insufficient data
        
        # Recovery recommendations
        recommendations = []
        
        if self.current_state in [DrawdownState.CRITICAL, DrawdownState.LOCKDOWN]:
            recommendations.extend([
                "Reduce position sizes significantly",
                "Focus on high-probability setups only",
                "Consider reducing trading frequency",
                "Review and adjust risk management rules"
            ])
        elif self.current_state == DrawdownState.WARNING:
            recommendations.extend([
                "Reduce position sizes moderately",
                "Increase signal quality threshold",
                "Monitor correlation between positions"
            ])
        
        return {
            'current_drawdown': self.current_drawdown,
            'required_return_to_recover': required_return,
            'estimated_recovery_days': estimated_days,
            'recommendations': recommendations,
            'risk_reduction_factor': self.get_position_scaling_factor()
        }
    
    def reset_lockdown(self) -> bool:
        """Manually reset lockdown (use with caution)"""
        if self.lockdown_until:
            self.lockdown_until = None
            self.current_state = DrawdownState.NORMAL
            self.logger.warning("Lockdown manually reset")
            return True
        return False
    
    def export_drawdown_data(self) -> pd.DataFrame:
        """Export drawdown history as DataFrame for analysis"""
        if not self.drawdown_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.drawdown_history)
