"""
Options Trading Strategies
Implementation of Long Straddle, Long Strangle, and other options strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from .base import BaseStrategy, TradingSignal, SignalType, StrategyMetrics
from ..config import TRADING_CONFIG


@dataclass
class OptionContract:
    """Option contract details"""
    symbol: str
    strike: float
    expiry: datetime
    option_type: str  # 'CE' for Call, 'PE' for Put
    premium: float
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float
    open_interest: int
    volume: int
    bid: float
    ask: float


@dataclass
class OptionsPosition:
    """Options position with Greeks"""
    contracts: List[OptionContract]
    quantity: List[int]  # Positive for long, negative for short
    entry_price: List[float]
    total_premium_paid: float
    total_premium_received: float
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    breakeven_points: List[float] = None


class OptionsGreeksCalculator:
    """Calculate options Greeks using Black-Scholes model"""
    
    @staticmethod
    def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
        """Calculate call option price and Greeks"""
        from scipy.stats import norm
        import math
        
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        
        call_price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        
        # Greeks
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S*sigma*math.sqrt(T))
        theta = -(S*norm.pdf(d1)*sigma)/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*norm.cdf(d2)
        vega = S*norm.pdf(d1)*math.sqrt(T)
        
        return {
            'price': call_price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta / 365,  # Daily theta
            'vega': vega / 100     # Vega per 1% change in IV
        }
    
    @staticmethod
    def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
        """Calculate put option price and Greeks"""
        call_greeks = OptionsGreeksCalculator.black_scholes_call(S, K, T, r, sigma)
        
        put_price = call_greeks['price'] - S + K * np.exp(-r*T)
        delta = call_greeks['delta'] - 1
        
        return {
            'price': put_price,
            'delta': delta,
            'gamma': call_greeks['gamma'],
            'theta': call_greeks['theta'],
            'vega': call_greeks['vega']
        }


class LongStraddleStrategy(BaseStrategy):
    """Long Straddle Options Strategy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or TRADING_CONFIG['options']['strategies']['long_straddle']
        self.name = "Long Straddle"
        self.description = "Buy ATM call and put options expecting high volatility"
        self.logger = logging.getLogger(__name__)
        self.positions: Dict[str, OptionsPosition] = {}
        self.greeks_calculator = OptionsGreeksCalculator()
    
    def calculate_signals(self, data: pd.DataFrame, symbol: str = None) -> List[TradingSignal]:
        """Generate trading signals for long straddle"""
        signals = []
        
        if len(data) < 30:
            return signals
        
        # Calculate volatility indicators
        returns = data['close'].pct_change().dropna()
        current_iv = self._estimate_implied_volatility(returns)
        historical_vol = returns.rolling(20).std() * np.sqrt(252)
        
        # Check if volatility is expected to increase
        vol_percentile = self._calculate_volatility_percentile(historical_vol)
        
        # Entry conditions for long straddle
        if self._should_enter_straddle(data, current_iv, vol_percentile):
            # Find ATM options
            current_price = data['close'].iloc[-1]
            atm_strike = self._find_atm_strike(current_price)
            expiry = self._select_expiry()
            
            # Create call signal
            call_signal = TradingSignal(
                symbol=f"{symbol}_{atm_strike}CE_{expiry.strftime('%Y%m%d')}",
                signal=SignalType.BUY,
                confidence=0.8,
                price=current_price,
                timestamp=data.index[-1],
                reason=f"Long Straddle Entry - High volatility expected, IV: {current_iv:.2%}",
                metadata={
                    'strategy': 'long_straddle',
                    'leg': 'call',
                    'strike': atm_strike,
                    'expiry': expiry,
                    'option_type': 'CE'
                }
            )
            
            # Create put signal
            put_signal = TradingSignal(
                symbol=f"{symbol}_{atm_strike}PE_{expiry.strftime('%Y%m%d')}",
                signal=SignalType.BUY,
                confidence=0.8,
                price=current_price,
                timestamp=data.index[-1],
                reason=f"Long Straddle Entry - High volatility expected, IV: {current_iv:.2%}",
                metadata={
                    'strategy': 'long_straddle',
                    'leg': 'put',
                    'strike': atm_strike,
                    'expiry': expiry,
                    'option_type': 'PE'
                }
            )
            
            signals.extend([call_signal, put_signal])
        
        # Exit conditions
        exit_signals = self._check_exit_conditions(data, symbol)
        signals.extend(exit_signals)
        
        return signals
    
    def _should_enter_straddle(self, data: pd.DataFrame, current_iv: float, vol_percentile: float) -> bool:
        """Determine if conditions are right for straddle entry"""
        # Check for upcoming events or high volatility expectation
        recent_volatility = data['close'].pct_change().rolling(5).std().iloc[-1] * np.sqrt(252)
        
        conditions = [
            vol_percentile > 0.8,  # High volatility expected
            current_iv < recent_volatility * 1.2,  # IV not too high relative to realized vol
            len(data) > 20  # Sufficient data
        ]
        
        return all(conditions)
    
    def _find_atm_strike(self, current_price: float) -> float:
        """Find the ATM strike price"""
        # Round to nearest strike (assuming strikes are in multiples of 50/100)
        if current_price < 1000:
            strike_interval = 50
        else:
            strike_interval = 100
        
        return round(current_price / strike_interval) * strike_interval
    
    def _select_expiry(self) -> datetime:
        """Select optimal expiry date"""
        # Select expiry between 15-30 days
        today = datetime.now()
        target_dte = 25  # Days to expiration
        return today + timedelta(days=target_dte)
    
    def _estimate_implied_volatility(self, returns: pd.Series) -> float:
        """Estimate implied volatility"""
        return returns.rolling(20).std().iloc[-1] * np.sqrt(252)
    
    def _calculate_volatility_percentile(self, volatility: pd.Series) -> float:
        """Calculate volatility percentile"""
        current_vol = volatility.iloc[-1]
        return (volatility <= current_vol).mean()
    
    def _check_exit_conditions(self, data: pd.DataFrame, symbol: str) -> List[TradingSignal]:
        """Check for exit conditions"""
        signals = []
        
        if symbol not in self.positions:
            return signals
        
        position = self.positions[symbol]
        current_price = data['close'].iloc[-1]
        
        # Calculate current P&L
        current_pnl = self._calculate_position_pnl(position, current_price)
        
        # Exit conditions
        if current_pnl > position.total_premium_paid * 0.5:  # 50% profit
            signals.extend(self._create_exit_signals(symbol, "Profit target reached"))
        elif current_pnl < -position.total_premium_paid * 0.5:  # 50% loss
            signals.extend(self._create_exit_signals(symbol, "Stop loss triggered"))
        
        return signals
    
    def _calculate_position_pnl(self, position: OptionsPosition, current_price: float) -> float:
        """Calculate current position P&L"""
        # This would typically require real option pricing
        # For now, return simplified calculation
        return 0.0
    
    def _create_exit_signals(self, symbol: str, reason: str) -> List[TradingSignal]:
        """Create exit signals for straddle position"""
        return []  # Implementation would create sell signals for both legs


class LongStrangleStrategy(BaseStrategy):
    """Long Strangle Options Strategy"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or TRADING_CONFIG['options']['strategies']['long_strangle']
        self.name = "Long Strangle"
        self.description = "Buy OTM call and put options expecting high volatility"
        self.logger = logging.getLogger(__name__)
        self.positions: Dict[str, OptionsPosition] = {}
        self.greeks_calculator = OptionsGreeksCalculator()
    
    def calculate_signals(self, data: pd.DataFrame, symbol: str = None) -> List[TradingSignal]:
        """Generate trading signals for long strangle"""
        signals = []
        
        if len(data) < 30:
            return signals
        
        # Calculate volatility indicators
        returns = data['close'].pct_change().dropna()
        current_iv = self._estimate_implied_volatility(returns)
        historical_vol = returns.rolling(20).std() * np.sqrt(252)
        vol_percentile = self._calculate_volatility_percentile(historical_vol)
        
        # Entry conditions for long strangle
        if self._should_enter_strangle(data, current_iv, vol_percentile):
            current_price = data['close'].iloc[-1]
            call_strike = self._find_otm_call_strike(current_price)
            put_strike = self._find_otm_put_strike(current_price)
            expiry = self._select_expiry()
            
            # Create call signal (OTM)
            call_signal = TradingSignal(
                symbol=f"{symbol}_{call_strike}CE_{expiry.strftime('%Y%m%d')}",
                signal=SignalType.BUY,
                confidence=0.75,
                price=current_price,
                timestamp=data.index[-1],
                reason=f"Long Strangle Entry - High volatility expected, cheaper than straddle",
                metadata={
                    'strategy': 'long_strangle',
                    'leg': 'call',
                    'strike': call_strike,
                    'expiry': expiry,
                    'option_type': 'CE'
                }
            )
            
            # Create put signal (OTM)
            put_signal = TradingSignal(
                symbol=f"{symbol}_{put_strike}PE_{expiry.strftime('%Y%m%d')}",
                signal=SignalType.BUY,
                confidence=0.75,
                price=current_price,
                timestamp=data.index[-1],
                reason=f"Long Strangle Entry - High volatility expected, cheaper than straddle",
                metadata={
                    'strategy': 'long_strangle',
                    'leg': 'put',
                    'strike': put_strike,
                    'expiry': expiry,
                    'option_type': 'PE'
                }
            )
            
            signals.extend([call_signal, put_signal])
        
        return signals
    
    def _should_enter_strangle(self, data: pd.DataFrame, current_iv: float, vol_percentile: float) -> bool:
        """Determine if conditions are right for strangle entry"""
        recent_volatility = data['close'].pct_change().rolling(5).std().iloc[-1] * np.sqrt(252)
        
        conditions = [
            vol_percentile > 0.75,  # High volatility expected
            current_iv < recent_volatility * 1.3,  # IV not too expensive
            len(data) > 20
        ]
        
        return all(conditions)
    
    def _find_otm_call_strike(self, current_price: float) -> float:
        """Find OTM call strike (typically 5-10% OTM)"""
        otm_percentage = 0.07  # 7% OTM
        target_strike = current_price * (1 + otm_percentage)
        
        if current_price < 1000:
            strike_interval = 50
        else:
            strike_interval = 100
        
        return np.ceil(target_strike / strike_interval) * strike_interval
    
    def _find_otm_put_strike(self, current_price: float) -> float:
        """Find OTM put strike (typically 5-10% OTM)"""
        otm_percentage = 0.07  # 7% OTM
        target_strike = current_price * (1 - otm_percentage)
        
        if current_price < 1000:
            strike_interval = 50
        else:
            strike_interval = 100
        
        return np.floor(target_strike / strike_interval) * strike_interval
    
    def _select_expiry(self) -> datetime:
        """Select optimal expiry date"""
        today = datetime.now()
        target_dte = 30  # Slightly longer for strangle
        return today + timedelta(days=target_dte)
    
    def _estimate_implied_volatility(self, returns: pd.Series) -> float:
        """Estimate implied volatility"""
        return returns.rolling(20).std().iloc[-1] * np.sqrt(252)
    
    def _calculate_volatility_percentile(self, volatility: pd.Series) -> float:
        """Calculate volatility percentile"""
        current_vol = volatility.iloc[-1]
        return (volatility <= current_vol).mean()


class OptionsStrategyManager:
    """Manager for options strategies"""
    
    def __init__(self):
        self.strategies = {
            'long_straddle': LongStraddleStrategy(),
            'long_strangle': LongStrangleStrategy()
        }
        self.logger = logging.getLogger(__name__)
    
    def get_strategy(self, strategy_name: str) -> BaseStrategy:
        """Get strategy by name"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown options strategy: {strategy_name}")
        return self.strategies[strategy_name]
    
    def get_optimal_strategy(self, market_conditions: Dict[str, Any]) -> str:
        """Select optimal options strategy based on market conditions"""
        volatility = market_conditions.get('volatility', 0)
        trend_strength = market_conditions.get('trend_strength', 0)
        time_to_event = market_conditions.get('days_to_event', float('inf'))
        
        # Strategy selection logic
        if volatility > 0.3 and time_to_event < 7:
            return 'long_straddle'  # High vol event coming up
        elif volatility > 0.25:
            return 'long_strangle'  # High vol but cheaper entry
        elif volatility < 0.15 and abs(trend_strength) < 0.3:
            return 'short_straddle'  # Low vol, range-bound
        else:
            return 'long_strangle'  # Default for most conditions
    
    def calculate_position_greeks(self, position: OptionsPosition) -> Dict[str, float]:
        """Calculate net Greeks for a position"""
        net_delta = sum(contract.delta * qty for contract, qty in 
                       zip(position.contracts, position.quantity))
        net_gamma = sum(contract.gamma * qty for contract, qty in 
                       zip(position.contracts, position.quantity))
        net_theta = sum(contract.theta * qty for contract, qty in 
                       zip(position.contracts, position.quantity))
        net_vega = sum(contract.vega * qty for contract, qty in 
                      zip(position.contracts, position.quantity))
        
        return {
            'delta': net_delta,
            'gamma': net_gamma,
            'theta': net_theta,
            'vega': net_vega
        }
