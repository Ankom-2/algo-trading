"""
Advanced Mean Reversion Strategy for Optimal Profit Generation
Uses statistical analysis and machine learning for high-probability mean reversion trades
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from .base import BaseStrategy, TradingSignal, SignalType


class MeanReversionStrategy(BaseStrategy):
    """
    Sophisticated mean reversion strategy using statistical methods and regime detection
    Designed for high-probability reversion trades with optimal risk-reward
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__("AdvancedMeanReversion", parameters)
        
        # Mean reversion parameters
        self.lookback_period = parameters.get('lookback_period', 20)
        self.entry_threshold = parameters.get('entry_threshold', 2.0)  # Standard deviations
        self.exit_threshold = parameters.get('exit_threshold', 0.5)
        self.min_volume_ratio = parameters.get('min_volume_ratio', 1.5)
        
        # Statistical parameters
        self.z_score_lookback = 50
        self.half_life_period = 10
        self.adf_threshold = -2.5  # Augmented Dickey-Fuller test threshold
        
        # Regime detection
        self.regime_detector = None
        self.current_regime = 'unknown'
        
        # Performance tracking
        self.trade_history = []
        
        self.logger.info("Advanced Mean Reversion Strategy initialized")
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate mean reversion trading signals"""
        signals = []
        
        if len(data) < self.z_score_lookback:
            return signals
        
        # Add mean reversion indicators
        df = self._add_mean_reversion_indicators(data)
        
        # Detect market regime
        self._detect_market_regime(df)
        
        # Only trade in suitable regimes
        if self.current_regime not in ['mean_reverting', 'sideways']:
            return signals
        
        # Get latest data
        latest = df.iloc[-1]
        symbol = latest.get('symbol', 'UNKNOWN')
        
        # Calculate mean reversion score
        reversion_score = self._calculate_reversion_score(latest)
        
        # Determine signal
        signal_type = self._determine_reversion_signal(latest, reversion_score)
        
        if signal_type != SignalType.HOLD:
            confidence = self.calculate_confidence(df, signal_type)
            
            if confidence > 0.65:  # Higher threshold for mean reversion
                signal = TradingSignal(
                    symbol=symbol,
                    signal=signal_type,
                    confidence=confidence,
                    price=latest['close'],
                    timestamp=latest.name,
                    stop_loss=self.calculate_stop_loss(
                        latest['close'], signal_type, latest['volatility']
                    ),
                    take_profit=self.calculate_take_profit(
                        latest['close'], signal_type
                    ),
                    reason=f"Mean Reversion Score: {reversion_score:.3f}, Regime: {self.current_regime}",
                    metadata={
                        'z_score': latest['z_score'],
                        'bollinger_position': latest['bollinger_position'],
                        'reversion_score': reversion_score,
                        'regime': self.current_regime,
                        'mean_distance': latest['mean_distance']
                    }
                )
                
                signals.append(signal)
                self.log_signal(signal)
        
        return signals
    
    def _add_mean_reversion_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive mean reversion indicators"""
        df = self.add_technical_indicators(data)
        
        # Z-Score calculation
        rolling_mean = df['close'].rolling(window=self.lookback_period).mean()
        rolling_std = df['close'].rolling(window=self.lookback_period).std()
        df['z_score'] = (df['close'] - rolling_mean) / rolling_std
        
        # Bollinger Band position
        df['bollinger_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Distance from mean
        df['mean_distance'] = (df['close'] - rolling_mean) / rolling_mean
        
        # Percentage Price Oscillator (PPO)
        ema_fast = df['close'].ewm(span=12).mean()
        ema_slow = df['close'].ewm(span=26).mean()
        df['ppo'] = ((ema_fast - ema_slow) / ema_slow) * 100
        df['ppo_signal'] = df['ppo'].ewm(span=9).mean()
        df['ppo_histogram'] = df['ppo'] - df['ppo_signal']
        
        # Commodity Channel Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mean_deviation = typical_price.rolling(window=20).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        df['cci'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        # Detrended Price Oscillator
        df['dpo'] = df['close'] - df['close'].shift(int(self.lookback_period/2 + 1)).rolling(
            window=self.lookback_period
        ).mean()
        
        # Mean reversion probability
        df['reversion_prob'] = self._calculate_reversion_probability(df)
        
        # Half-life of mean reversion
        df['half_life'] = self._calculate_half_life(df)
        
        return df
    
    def _calculate_reversion_probability(self, data: pd.DataFrame) -> pd.Series:
        """Calculate probability of mean reversion"""
        z_scores = data['z_score'].dropna()
        if len(z_scores) < 30:
            return pd.Series(0.5, index=data.index)
        
        # Use historical z-score distribution to estimate reversion probability
        probs = []
        for i in range(len(z_scores)):
            if i < 29:
                probs.append(0.5)
            else:
                current_z = z_scores.iloc[i]
                historical_z = z_scores.iloc[i-29:i]
                
                # Calculate how often extreme z-scores reverted
                extreme_mask = np.abs(historical_z) > 1.5
                if extreme_mask.sum() > 0:
                    reverted = 0
                    for j in range(len(historical_z) - 5):
                        if np.abs(historical_z.iloc[j]) > 1.5:
                            future_z = historical_z.iloc[j+1:j+6]
                            if len(future_z) > 0 and np.abs(future_z.mean()) < np.abs(historical_z.iloc[j]):
                                reverted += 1
                    
                    prob = reverted / extreme_mask.sum() if extreme_mask.sum() > 0 else 0.5
                else:
                    prob = 0.5
                
                # Adjust probability based on current z-score magnitude
                prob = prob * (1 + min(2.0, np.abs(current_z)) / 3)
                probs.append(min(0.95, max(0.05, prob)))
        
        return pd.Series(probs, index=z_scores.index)
    
    def _calculate_half_life(self, data: pd.DataFrame) -> pd.Series:
        """Calculate half-life of mean reversion"""
        prices = data['close'].dropna()
        if len(prices) < 50:
            return pd.Series(self.half_life_period, index=data.index)
        
        half_lives = []
        for i in range(len(prices)):
            if i < 49:
                half_lives.append(self.half_life_period)
            else:
                recent_prices = prices.iloc[i-49:i+1]
                log_prices = np.log(recent_prices)
                lag_prices = log_prices.shift(1).dropna()
                
                if len(lag_prices) > 10:
                    # AR(1) regression: p(t) = a + b*p(t-1) + e(t)
                    X = lag_prices.values.reshape(-1, 1)
                    y = log_prices.iloc[1:].values
                    
                    try:
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression().fit(X, y)
                        b = model.coef_[0]
                        
                        if b < 1 and b > 0:
                            half_life = -np.log(2) / np.log(b)
                            half_life = max(1, min(50, half_life))
                        else:
                            half_life = self.half_life_period
                    except:
                        half_life = self.half_life_period
                else:
                    half_life = self.half_life_period
                
                half_lives.append(half_life)
        
        return pd.Series(half_lives, index=prices.index)
    
    def _detect_market_regime(self, data: pd.DataFrame):
        """Detect current market regime for mean reversion suitability"""
        if len(data) < 100:
            self.current_regime = 'unknown'
            return
        
        recent_data = data.tail(50)
        
        # Calculate regime indicators
        price_volatility = recent_data['close'].pct_change().std()
        trend_strength = abs(recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1)
        mean_reversion_freq = (np.abs(recent_data['z_score']) > 1.5).mean()
        
        # Classify regime
        if trend_strength > 0.15:  # Strong trend
            self.current_regime = 'trending'
        elif price_volatility > 0.03:  # High volatility
            self.current_regime = 'volatile'
        elif mean_reversion_freq > 0.3:  # Frequent mean reversions
            self.current_regime = 'mean_reverting'
        else:
            self.current_regime = 'sideways'
        
        self.logger.debug(f"Market regime detected: {self.current_regime}")
    
    def _calculate_reversion_score(self, latest_data: pd.Series) -> float:
        """Calculate comprehensive mean reversion score"""
        score = 0.0
        
        # Z-score component (40% weight)
        z_score = latest_data['z_score']
        if np.abs(z_score) > self.entry_threshold:
            z_component = -np.sign(z_score) * min(1.0, np.abs(z_score) / 3.0)
            score += z_component * 0.4
        
        # Bollinger position component (25% weight)
        bb_pos = latest_data['bollinger_position']
        if bb_pos < 0.1:  # Near lower band
            score += 0.25
        elif bb_pos > 0.9:  # Near upper band
            score -= 0.25
        
        # CCI component (20% weight)
        cci = latest_data['cci']
        if cci < -100:  # Oversold
            score += 0.2
        elif cci > 100:  # Overbought
            score -= 0.2
        
        # PPO histogram component (15% weight)
        ppo_hist = latest_data['ppo_histogram']
        if ppo_hist != 0:
            ppo_component = -np.sign(ppo_hist) * min(1.0, np.abs(ppo_hist) / 2.0)
            score += ppo_component * 0.15
        
        return score
    
    def _determine_reversion_signal(self, latest_data: pd.Series, reversion_score: float) -> SignalType:
        """Determine signal type based on mean reversion analysis"""
        
        # Strong reversion signals
        if (reversion_score > 0.6 and 
            latest_data['z_score'] < -self.entry_threshold and
            latest_data['volume_ratio'] > self.min_volume_ratio and
            latest_data['reversion_prob'] > 0.7):
            return SignalType.STRONG_BUY
        
        elif (reversion_score < -0.6 and 
              latest_data['z_score'] > self.entry_threshold and
              latest_data['volume_ratio'] > self.min_volume_ratio and
              latest_data['reversion_prob'] > 0.7):
            return SignalType.STRONG_SELL
        
        # Moderate reversion signals
        elif (reversion_score > 0.3 and 
              latest_data['z_score'] < -1.5 and
              latest_data['volume_ratio'] > 1.2):
            return SignalType.BUY
        
        elif (reversion_score < -0.3 and 
              latest_data['z_score'] > 1.5 and
              latest_data['volume_ratio'] > 1.2):
            return SignalType.SELL
        
        return SignalType.HOLD
    
    def calculate_confidence(self, data: pd.DataFrame, signal: SignalType) -> float:
        """Calculate confidence for mean reversion signals"""
        if len(data) < 30:
            return 0.5
        
        latest = data.iloc[-1]
        confidence = 0.5  # Base confidence
        
        # Statistical significance
        z_score_abs = abs(latest['z_score'])
        if z_score_abs > 2.0:
            confidence += 0.15
        elif z_score_abs > 1.5:
            confidence += 0.1
        
        # Reversion probability
        if latest['reversion_prob'] > 0.7:
            confidence += 0.15
        elif latest['reversion_prob'] > 0.6:
            confidence += 0.1
        
        # Volume confirmation
        if latest['volume_ratio'] > 1.5:
            confidence += 0.1
        
        # Half-life consideration
        if latest['half_life'] < 15:  # Quick mean reversion expected
            confidence += 0.05
        
        # Regime suitability
        if self.current_regime in ['mean_reverting', 'sideways']:
            confidence += 0.1
        elif self.current_regime == 'trending':
            confidence -= 0.15
        
        # Multiple indicator confirmation
        confirmations = 0
        if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            if latest['cci'] < -100:
                confirmations += 1
            if latest['bollinger_position'] < 0.2:
                confirmations += 1
            if latest['ppo_histogram'] < 0:
                confirmations += 1
        elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
            if latest['cci'] > 100:
                confirmations += 1
            if latest['bollinger_position'] > 0.8:
                confirmations += 1
            if latest['ppo_histogram'] > 0:
                confirmations += 1
        
        confidence += confirmations * 0.05
        
        return max(0.0, min(1.0, confidence))
