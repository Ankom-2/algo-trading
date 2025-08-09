"""
Advanced Momentum Strategy for Consistent Profitability
Combines multiple momentum indicators with machine learning for optimal signal generation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .base import BaseStrategy, TradingSignal, SignalType


class MomentumStrategy(BaseStrategy):
    """
    Sophisticated momentum strategy using multiple indicators and ML
    Designed for high probability momentum trades
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__("AdvancedMomentum", parameters)
        
        # Momentum-specific parameters
        self.rsi_period = parameters.get('rsi_period', 14)
        self.rsi_overbought = parameters.get('rsi_overbought', 70)
        self.rsi_oversold = parameters.get('rsi_oversold', 30)
        
        self.macd_fast = parameters.get('macd_fast', 12)
        self.macd_slow = parameters.get('macd_slow', 26)
        self.macd_signal = parameters.get('macd_signal', 9)
        
        self.bollinger_period = parameters.get('bollinger_period', 20)
        self.bollinger_std = parameters.get('bollinger_std', 2)
        
        # ML model for signal prediction
        self.ml_model = None
        self.scaler = StandardScaler()
        self.model_trained = False
        
        # Momentum thresholds
        self.momentum_threshold = 0.015  # 1.5% momentum threshold
        self.volume_multiplier = 1.5     # Volume confirmation multiplier
        
        self.logger.info("Advanced Momentum Strategy initialized")
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate momentum-based trading signals"""
        signals = []
        
        if len(data) < 50:  # Minimum data requirement
            return signals
        
        # Add technical indicators
        df = self.add_advanced_momentum_indicators(data)
        
        # Train ML model if not done
        if not self.model_trained and len(df) > 100:
            self._train_ml_model(df)
        
        # Get latest data point
        latest = df.iloc[-1]
        symbol = latest.get('symbol', 'UNKNOWN')
        
        # Calculate momentum score
        momentum_score = self._calculate_momentum_score(latest)
        
        # Determine signal type
        signal_type = self._determine_signal_type(latest, momentum_score)
        
        if signal_type != SignalType.HOLD:
            confidence = self.calculate_confidence(df, signal_type)
            
            if confidence > 0.6:  # Minimum confidence threshold
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
                    reason=f"Momentum Score: {momentum_score:.3f}",
                    metadata={
                        'rsi': latest['RSI'],
                        'macd_histogram': latest['MACD_histogram'],
                        'momentum_score': momentum_score,
                        'volume_ratio': latest['volume_ratio']
                    }
                )
                
                signals.append(signal)
                self.log_signal(signal)
        
        return signals
    
    def add_advanced_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced momentum indicators"""
        df = self.add_technical_indicators(data)
        
        # Rate of Change (ROC)
        df['ROC_10'] = df['close'].pct_change(10) * 100
        df['ROC_20'] = df['close'].pct_change(20) * 100
        
        # Momentum oscillator
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Price momentum
        df['price_momentum'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        
        # Stochastic oscillator
        low_min = df['low'].rolling(window=14).min()
        high_max = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Williams %R
        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
        
        # Commodity Channel Index (CCI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mean_deviation = typical_price.rolling(window=20).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        df['CCI'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        # Momentum strength
        df['momentum_strength'] = (
            np.abs(df['ROC_10']) + np.abs(df['ROC_20']) + 
            np.abs(df['price_momentum'] * 100)
        ) / 3
        
        return df
    
    def _calculate_momentum_score(self, latest_data: pd.Series) -> float:
        """Calculate comprehensive momentum score"""
        score = 0.0
        
        # RSI component (30% weight)
        if latest_data['RSI'] > 50:
            rsi_score = (latest_data['RSI'] - 50) / 50
        else:
            rsi_score = (latest_data['RSI'] - 50) / 50
        score += rsi_score * 0.3
        
        # MACD component (25% weight)
        if latest_data['MACD_histogram'] > 0:
            macd_score = min(1.0, latest_data['MACD_histogram'] / 0.01)
        else:
            macd_score = max(-1.0, latest_data['MACD_histogram'] / 0.01)
        score += macd_score * 0.25
        
        # Price momentum component (20% weight)
        price_mom_score = np.tanh(latest_data['price_momentum'] * 10)
        score += price_mom_score * 0.2
        
        # Stochastic component (15% weight)
        if latest_data['stoch_k'] > 50:
            stoch_score = (latest_data['stoch_k'] - 50) / 50
        else:
            stoch_score = (latest_data['stoch_k'] - 50) / 50
        score += stoch_score * 0.15
        
        # Volume confirmation (10% weight)
        volume_score = np.tanh((latest_data['volume_ratio'] - 1) * 2)
        score += volume_score * 0.1
        
        return score
    
    def _determine_signal_type(self, latest_data: pd.Series, momentum_score: float) -> SignalType:
        """Determine signal type based on momentum analysis"""
        
        # Strong momentum thresholds
        if momentum_score > 0.6 and latest_data['volume_ratio'] > self.volume_multiplier:
            # Additional confirmation for strong buy
            if (latest_data['RSI'] < 80 and latest_data['MACD_histogram'] > 0 and
                latest_data['close'] > latest_data['SMA_20']):
                return SignalType.STRONG_BUY
            elif momentum_score > 0.3:
                return SignalType.BUY
        
        elif momentum_score < -0.6 and latest_data['volume_ratio'] > self.volume_multiplier:
            # Additional confirmation for strong sell
            if (latest_data['RSI'] > 20 and latest_data['MACD_histogram'] < 0 and
                latest_data['close'] < latest_data['SMA_20']):
                return SignalType.STRONG_SELL
            elif momentum_score < -0.3:
                return SignalType.SELL
        
        # Moderate momentum signals
        elif momentum_score > 0.3 and latest_data['volume_ratio'] > 1.2:
            if self._confirm_uptrend(latest_data):
                return SignalType.BUY
        
        elif momentum_score < -0.3 and latest_data['volume_ratio'] > 1.2:
            if self._confirm_downtrend(latest_data):
                return SignalType.SELL
        
        return SignalType.HOLD
    
    def _confirm_uptrend(self, data: pd.Series) -> bool:
        """Confirm uptrend using multiple indicators"""
        confirmations = 0
        
        # Price above moving averages
        if data['close'] > data['SMA_20']:
            confirmations += 1
        if data['SMA_20'] > data['SMA_50']:
            confirmations += 1
        
        # MACD bullish
        if data['MACD'] > data['MACD_signal']:
            confirmations += 1
        
        # RSI in bullish zone but not overbought
        if 50 < data['RSI'] < 75:
            confirmations += 1
        
        return confirmations >= 3
    
    def _confirm_downtrend(self, data: pd.Series) -> bool:
        """Confirm downtrend using multiple indicators"""
        confirmations = 0
        
        # Price below moving averages
        if data['close'] < data['SMA_20']:
            confirmations += 1
        if data['SMA_20'] < data['SMA_50']:
            confirmations += 1
        
        # MACD bearish
        if data['MACD'] < data['MACD_signal']:
            confirmations += 1
        
        # RSI in bearish zone but not oversold
        if 25 < data['RSI'] < 50:
            confirmations += 1
        
        return confirmations >= 3
    
    def calculate_confidence(self, data: pd.DataFrame, signal: SignalType) -> float:
        """Calculate confidence level for momentum signals"""
        if len(data) < 20:
            return 0.5
        
        latest = data.iloc[-1]
        confidence = 0.5  # Base confidence
        
        # Volume confirmation
        if latest['volume_ratio'] > self.volume_multiplier:
            confidence += 0.1
        
        # Multiple timeframe confirmation
        if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            if (latest['momentum_10'] > 0 and latest['momentum_20'] > 0):
                confidence += 0.15
            if latest['price_momentum'] > self.momentum_threshold:
                confidence += 0.1
        elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
            if (latest['momentum_10'] < 0 and latest['momentum_20'] < 0):
                confidence += 0.15
            if latest['price_momentum'] < -self.momentum_threshold:
                confidence += 0.1
        
        # Volatility adjustment
        if latest['volatility'] < 0.02:  # Low volatility increases confidence
            confidence += 0.05
        elif latest['volatility'] > 0.04:  # High volatility decreases confidence
            confidence -= 0.1
        
        # ML model prediction (if available)
        if self.model_trained:
            ml_confidence = self._get_ml_confidence(latest)
            confidence = (confidence + ml_confidence) / 2
        
        return max(0.0, min(1.0, confidence))
    
    def _train_ml_model(self, data: pd.DataFrame):
        """Train ML model for signal prediction"""
        try:
            # Prepare features
            features = [
                'RSI', 'MACD_histogram', 'momentum_10', 'momentum_20',
                'stoch_k', 'williams_r', 'CCI', 'volume_ratio', 'volatility'
            ]
            
            # Create target variable (future returns)
            data_ml = data.dropna()
            data_ml['future_return'] = data_ml['close'].shift(-5) / data_ml['close'] - 1
            data_ml = data_ml.dropna()
            
            if len(data_ml) < 50:
                return
            
            X = data_ml[features].values
            y = data_ml['future_return'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.ml_model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            self.ml_model.fit(X_scaled, y)
            
            self.model_trained = True
            self.logger.info("ML model trained successfully")
            
        except Exception as e:
            self.logger.warning(f"ML model training failed: {str(e)}")
    
    def _get_ml_confidence(self, latest_data: pd.Series) -> float:
        """Get confidence from ML model prediction"""
        if not self.model_trained:
            return 0.5
        
        try:
            features = [
                'RSI', 'MACD_histogram', 'momentum_10', 'momentum_20',
                'stoch_k', 'williams_r', 'CCI', 'volume_ratio', 'volatility'
            ]
            
            X = np.array([latest_data[features].values])
            X_scaled = self.scaler.transform(X)
            
            prediction = self.ml_model.predict(X_scaled)[0]
            
            # Convert prediction to confidence
            confidence = 0.5 + np.tanh(prediction * 10) * 0.3
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.warning(f"ML prediction failed: {str(e)}")
            return 0.5
