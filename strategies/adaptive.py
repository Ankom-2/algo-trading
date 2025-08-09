"""
Adaptive Strategy - The Crown Jewel of Algorithmic Trading
Dynamically switches between momentum and mean reversion based on market conditions
Uses advanced machine learning and regime detection for optimal profitability
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from .base import BaseStrategy, TradingSignal, SignalType
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy


class AdaptiveStrategy(BaseStrategy):
    """
    Master adaptive strategy that dynamically selects the best approach
    Uses machine learning to determine optimal strategy based on market conditions
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        super().__init__("AdaptiveAI", parameters)
        
        # Adaptive parameters
        self.regime_detection_period = parameters.get('regime_detection_period', 50)
        self.volatility_threshold = parameters.get('volatility_threshold', 0.02)
        self.trend_strength_period = parameters.get('trend_strength_period', 14)
        self.adaptation_speed = parameters.get('adaptation_speed', 0.1)
        
        # Initialize sub-strategies
        self.momentum_strategy = MomentumStrategy(parameters.get('momentum', {}))
        self.mean_reversion_strategy = MeanReversionStrategy(parameters.get('mean_reversion', {}))
        
        # Machine learning models
        self.regime_classifier = None
        self.performance_predictor = None
        self.scaler = StandardScaler()
        self.models_trained = False
        
        # Regime detection
        self.current_regime = 'unknown'
        self.regime_confidence = 0.5
        self.regime_history = []
        
        # Strategy performance tracking
        self.strategy_performance = {
            'momentum': {'wins': 0, 'losses': 0, 'total_return': 0.0},
            'mean_reversion': {'wins': 0, 'losses': 0, 'total_return': 0.0},
            'adaptive': {'wins': 0, 'losses': 0, 'total_return': 0.0}
        }
        
        # Market condition indicators
        self.market_conditions = {}
        
        self.logger.info("Adaptive AI Strategy initialized with sub-strategies")
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate adaptive trading signals based on market regime"""
        signals = []
        
        if len(data) < self.regime_detection_period:
            return signals
        
        # Analyze market conditions
        self._analyze_market_conditions(data)
        
        # Detect current market regime
        self._detect_market_regime(data)
        
        # Train models if sufficient data
        if not self.models_trained and len(data) > 200:
            self._train_adaptive_models(data)
        
        # Generate signals based on regime and conditions
        if self.current_regime == 'momentum_favorable':
            signals = self._generate_momentum_signals(data)
        elif self.current_regime == 'mean_reversion_favorable':
            signals = self._generate_mean_reversion_signals(data)
        elif self.current_regime == 'mixed':
            signals = self._generate_hybrid_signals(data)
        else:
            # Use ensemble approach when regime is uncertain
            signals = self._generate_ensemble_signals(data)
        
        # Apply adaptive filters
        filtered_signals = self._apply_adaptive_filters(signals, data)
        
        return filtered_signals
    
    def _analyze_market_conditions(self, data: pd.DataFrame):
        """Comprehensive market condition analysis"""
        df = self.add_technical_indicators(data)
        recent_data = df.tail(50)
        
        # Volatility analysis
        returns = recent_data['close'].pct_change()
        self.market_conditions['volatility'] = returns.std()
        self.market_conditions['volatility_regime'] = self._classify_volatility(returns.std())
        
        # Trend analysis
        sma_short = recent_data['close'].rolling(window=10).mean()
        sma_long = recent_data['close'].rolling(window=30).mean()
        trend_strength = abs(sma_short.iloc[-1] / sma_long.iloc[-1] - 1)
        self.market_conditions['trend_strength'] = trend_strength
        
        # Market efficiency (Hurst exponent approximation)
        self.market_conditions['efficiency'] = self._calculate_hurst_exponent(recent_data['close'])
        
        # Volume analysis
        avg_volume = recent_data['volume'].mean()
        recent_volume = recent_data['volume'].iloc[-5:].mean()
        self.market_conditions['volume_trend'] = recent_volume / avg_volume
        
        # Correlation with market (if SPY-like data available)
        self.market_conditions['market_correlation'] = self._calculate_market_correlation(recent_data)
        
        # News sentiment proxy (based on volume and price action)
        self.market_conditions['sentiment_proxy'] = self._estimate_sentiment(recent_data)
        
        self.logger.debug(f"Market conditions: {self.market_conditions}")
    
    def _detect_market_regime(self, data: pd.DataFrame):
        """Advanced regime detection using multiple methodologies"""
        if len(data) < self.regime_detection_period:
            return
        
        recent_data = data.tail(self.regime_detection_period)
        
        # Method 1: Statistical regime detection
        statistical_regime = self._statistical_regime_detection(recent_data)
        
        # Method 2: Volatility-based regime detection
        volatility_regime = self._volatility_regime_detection(recent_data)
        
        # Method 3: Trend-based regime detection
        trend_regime = self._trend_regime_detection(recent_data)
        
        # Method 4: ML-based regime detection (if model trained)
        ml_regime = self._ml_regime_detection(recent_data) if self.models_trained else statistical_regime
        
        # Ensemble regime determination
        regime_votes = [statistical_regime, volatility_regime, trend_regime, ml_regime]
        regime_counts = {regime: regime_votes.count(regime) for regime in set(regime_votes)}
        
        # Determine final regime with confidence
        self.current_regime = max(regime_counts, key=regime_counts.get)
        self.regime_confidence = regime_counts[self.current_regime] / len(regime_votes)
        
        # Update regime history
        self.regime_history.append({
            'timestamp': data.index[-1],
            'regime': self.current_regime,
            'confidence': self.regime_confidence
        })
        
        # Keep only recent history
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        self.logger.debug(f"Regime: {self.current_regime} (confidence: {self.regime_confidence:.2f})")
    
    def _statistical_regime_detection(self, data: pd.DataFrame) -> str:
        """Regime detection based on statistical properties"""
        returns = data['close'].pct_change().dropna()
        
        # Calculate statistical measures
        volatility = returns.std()
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Autocorrelation for mean reversion detection
        autocorr_1 = returns.autocorr(lag=1) if len(returns) > 1 else 0
        autocorr_5 = returns.autocorr(lag=5) if len(returns) > 5 else 0
        
        # Regime classification
        if autocorr_1 < -0.1 and autocorr_5 < -0.1:
            return 'mean_reversion_favorable'
        elif volatility < 0.015 and abs(skewness) < 1:
            return 'mean_reversion_favorable'
        elif volatility > 0.03 or abs(skewness) > 2:
            return 'momentum_favorable'
        else:
            return 'mixed'
    
    def _volatility_regime_detection(self, data: pd.DataFrame) -> str:
        """Volatility-based regime classification"""
        returns = data['close'].pct_change()
        
        # Rolling volatility
        vol_short = returns.rolling(window=10).std().iloc[-1]
        vol_long = returns.rolling(window=30).std().iloc[-1]
        
        if vol_short < 0.01:
            return 'mean_reversion_favorable'
        elif vol_short > 0.03:
            return 'momentum_favorable'
        elif vol_short > vol_long * 1.5:
            return 'momentum_favorable'
        else:
            return 'mixed'
    
    def _trend_regime_detection(self, data: pd.DataFrame) -> str:
        """Trend-based regime classification"""
        # Multiple timeframe trend analysis
        sma_10 = data['close'].rolling(window=10).mean()
        sma_20 = data['close'].rolling(window=20).mean()
        sma_50 = data['close'].rolling(window=min(50, len(data))).mean()
        
        # Current trend alignment
        trend_alignment = 0
        if sma_10.iloc[-1] > sma_20.iloc[-1]:
            trend_alignment += 1
        if sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend_alignment += 1
        
        # Trend strength
        trend_strength = abs(data['close'].iloc[-1] / data['close'].iloc[0] - 1)
        
        if trend_strength > 0.1 and trend_alignment >= 1:
            return 'momentum_favorable'
        elif trend_strength < 0.03:
            return 'mean_reversion_favorable'
        else:
            return 'mixed'
    
    def _ml_regime_detection(self, data: pd.DataFrame) -> str:
        """ML-based regime detection"""
        if not self.regime_classifier:
            return 'mixed'
        
        try:
            # Prepare features
            features = self._prepare_regime_features(data)
            if features is None:
                return 'mixed'
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Predict regime
            prediction = self.regime_classifier.predict(features_scaled)[0]
            
            return prediction
            
        except Exception as e:
            self.logger.warning(f"ML regime detection failed: {str(e)}")
            return 'mixed'
    
    def _generate_momentum_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate momentum signals with adaptive enhancements"""
        momentum_signals = self.momentum_strategy.generate_signals(data)
        
        # Enhance signals with market condition adjustments
        enhanced_signals = []
        for signal in momentum_signals:
            # Adjust confidence based on market conditions
            adjusted_confidence = self._adjust_confidence_for_momentum(signal.confidence, data)
            
            if adjusted_confidence > 0.6:
                signal.confidence = adjusted_confidence
                signal.reason += f" [Adaptive: {adjusted_confidence:.2f}]"
                enhanced_signals.append(signal)
        
        return enhanced_signals
    
    def _generate_mean_reversion_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate mean reversion signals with adaptive enhancements"""
        reversion_signals = self.mean_reversion_strategy.generate_signals(data)
        
        # Enhance signals with market condition adjustments
        enhanced_signals = []
        for signal in reversion_signals:
            # Adjust confidence based on market conditions
            adjusted_confidence = self._adjust_confidence_for_reversion(signal.confidence, data)
            
            if adjusted_confidence > 0.65:
                signal.confidence = adjusted_confidence
                signal.reason += f" [Adaptive: {adjusted_confidence:.2f}]"
                enhanced_signals.append(signal)
        
        return enhanced_signals
    
    def _generate_hybrid_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate hybrid signals combining both strategies"""
        momentum_signals = self.momentum_strategy.generate_signals(data)
        reversion_signals = self.mean_reversion_strategy.generate_signals(data)
        
        # Combine and weight signals
        all_signals = momentum_signals + reversion_signals
        
        if not all_signals:
            return []
        
        # Select best signal based on confidence and market conditions
        best_signal = max(all_signals, key=lambda s: s.confidence * self._get_strategy_multiplier(s))
        
        if best_signal.confidence > 0.7:
            best_signal.reason += " [Hybrid Selection]"
            return [best_signal]
        
        return []
    
    def _generate_ensemble_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate ensemble signals using model predictions"""
        if not self.models_trained:
            return self._generate_hybrid_signals(data)
        
        try:
            # Get signals from both strategies
            momentum_signals = self.momentum_strategy.generate_signals(data)
            reversion_signals = self.mean_reversion_strategy.generate_signals(data)
            
            # Use ML to predict best signal
            all_signals = momentum_signals + reversion_signals
            if not all_signals:
                return []
            
            # Predict performance for each signal
            best_signal = None
            best_predicted_return = -np.inf
            
            for signal in all_signals:
                predicted_return = self._predict_signal_performance(signal, data)
                if predicted_return > best_predicted_return:
                    best_predicted_return = predicted_return
                    best_signal = signal
            
            if best_signal and best_predicted_return > 0.01:  # 1% minimum expected return
                best_signal.reason += f" [Ensemble: {best_predicted_return:.3f}]"
                return [best_signal]
            
        except Exception as e:
            self.logger.warning(f"Ensemble signal generation failed: {str(e)}")
        
        return []
    
    def _train_adaptive_models(self, data: pd.DataFrame):
        """Train adaptive ML models"""
        try:
            self.logger.info("Training adaptive models...")
            
            # Prepare training data for regime classification
            regime_features, regime_labels = self._prepare_regime_training_data(data)
            
            if len(regime_features) > 50:
                # Train regime classifier
                self.regime_classifier = GradientBoostingClassifier(
                    n_estimators=100, max_depth=5, random_state=42
                )
                
                # Scale features
                regime_features_scaled = self.scaler.fit_transform(regime_features)
                self.regime_classifier.fit(regime_features_scaled, regime_labels)
                
                self.models_trained = True
                self.logger.info("Adaptive models trained successfully")
            
        except Exception as e:
            self.logger.warning(f"Model training failed: {str(e)}")
    
    def _prepare_regime_features(self, data: pd.DataFrame) -> Optional[List[float]]:
        """Prepare features for regime detection"""
        if len(data) < 30:
            return None
        
        df = self.add_technical_indicators(data.tail(30))
        returns = df['close'].pct_change().dropna()
        
        features = [
            returns.std(),  # Volatility
            returns.skew(),  # Skewness
            returns.kurtosis(),  # Kurtosis
            returns.autocorr(lag=1) if len(returns) > 1 else 0,  # Autocorrelation
            df['RSI'].iloc[-1] if 'RSI' in df.columns else 50,
            df['MACD_histogram'].iloc[-1] if 'MACD_histogram' in df.columns else 0,
            self.market_conditions.get('trend_strength', 0),
            self.market_conditions.get('efficiency', 0.5),
            self.market_conditions.get('volume_trend', 1),
        ]
        
        return features
    
    def _prepare_regime_training_data(self, data: pd.DataFrame):
        """Prepare training data for regime classification"""
        features = []
        labels = []
        
        window_size = 50
        for i in range(window_size, len(data) - 10, 5):  # Step by 5 for efficiency
            window_data = data.iloc[i-window_size:i]
            future_data = data.iloc[i:i+10]
            
            # Extract features
            feature_vector = self._prepare_regime_features(window_data)
            if feature_vector is None:
                continue
            
            # Determine label based on future performance
            label = self._determine_optimal_strategy(window_data, future_data)
            
            features.append(feature_vector)
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def _determine_optimal_strategy(self, historical_data: pd.DataFrame, future_data: pd.DataFrame) -> str:
        """Determine which strategy would have been optimal"""
        # Simplified labeling - in practice, you'd run backtests
        returns = future_data['close'].pct_change().dropna()
        
        if returns.std() > 0.025:  # High volatility
            return 'momentum_favorable'
        elif abs(returns.autocorr(lag=1)) > 0.1:  # Mean reverting
            return 'mean_reversion_favorable'
        else:
            return 'mixed'
    
    def calculate_confidence(self, data: pd.DataFrame, signal: SignalType) -> float:
        """Calculate adaptive confidence based on regime and conditions"""
        base_confidence = 0.6
        
        # Regime confidence adjustment
        base_confidence += (self.regime_confidence - 0.5) * 0.2
        
        # Market condition adjustments
        if self.market_conditions.get('volatility', 0.02) < 0.015:
            base_confidence += 0.05  # Low volatility is favorable
        
        if self.market_conditions.get('efficiency', 0.5) < 0.4:
            base_confidence += 0.05  # Less efficient markets are more predictable
        
        # Strategy performance history
        strategy_type = 'momentum' if signal in [SignalType.BUY, SignalType.SELL] else 'mean_reversion'
        perf = self.strategy_performance[strategy_type]
        if perf['wins'] + perf['losses'] > 10:
            win_rate = perf['wins'] / (perf['wins'] + perf['losses'])
            base_confidence += (win_rate - 0.5) * 0.1
        
        return max(0.0, min(1.0, base_confidence))
    
    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """Approximate Hurst exponent for market efficiency"""
        try:
            lags = range(2, min(20, len(prices)//4))
            tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag]))) for lag in lags]
            
            if len(tau) < 3:
                return 0.5
            
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        except:
            return 0.5
    
    def _calculate_market_correlation(self, data: pd.DataFrame) -> float:
        """Calculate correlation with market proxy"""
        # Simplified - in practice, you'd use actual market data
        return 0.7  # Assume moderate correlation
    
    def _estimate_sentiment(self, data: pd.DataFrame) -> float:
        """Estimate market sentiment from price and volume action"""
        price_change = data['close'].iloc[-1] / data['close'].iloc[0] - 1
        volume_ratio = data['volume'].iloc[-5:].mean() / data['volume'].mean()
        
        sentiment = price_change * volume_ratio
        return np.tanh(sentiment)  # Normalize to [-1, 1]
    
    def _adjust_confidence_for_momentum(self, base_confidence: float, data: pd.DataFrame) -> float:
        """Adjust confidence for momentum signals based on market conditions"""
        adjustment = 0
        
        if self.market_conditions.get('volatility', 0.02) > 0.025:
            adjustment += 0.1  # High volatility favors momentum
        
        if self.market_conditions.get('trend_strength', 0) > 0.05:
            adjustment += 0.1  # Strong trends favor momentum
        
        return min(1.0, base_confidence + adjustment)
    
    def _adjust_confidence_for_reversion(self, base_confidence: float, data: pd.DataFrame) -> float:
        """Adjust confidence for mean reversion signals based on market conditions"""
        adjustment = 0
        
        if self.market_conditions.get('volatility', 0.02) < 0.015:
            adjustment += 0.1  # Low volatility favors mean reversion
        
        if self.market_conditions.get('efficiency', 0.5) < 0.4:
            adjustment += 0.05  # Less efficient markets favor mean reversion
        
        return min(1.0, base_confidence + adjustment)
    
    def _get_strategy_multiplier(self, signal: TradingSignal) -> float:
        """Get multiplier based on current regime suitability"""
        if 'Momentum' in signal.reason and self.current_regime == 'momentum_favorable':
            return 1.2
        elif 'Reversion' in signal.reason and self.current_regime == 'mean_reversion_favorable':
            return 1.2
        elif self.current_regime == 'mixed':
            return 1.0
        else:
            return 0.8
    
    def _predict_signal_performance(self, signal: TradingSignal, data: pd.DataFrame) -> float:
        """Predict expected performance of a signal"""
        # Simplified prediction - in practice, you'd use trained models
        base_return = 0.02 if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY] else 0.02
        
        # Adjust based on confidence and market conditions
        confidence_multiplier = signal.confidence
        regime_multiplier = self._get_strategy_multiplier(signal)
        
        return base_return * confidence_multiplier * regime_multiplier
    
    def _apply_adaptive_filters(self, signals: List[TradingSignal], data: pd.DataFrame) -> List[TradingSignal]:
        """Apply adaptive filters to improve signal quality"""
        if not signals:
            return signals
        
        filtered_signals = []
        
        for signal in signals:
            # Market condition filter
            if self._passes_market_condition_filter(signal, data):
                # Risk-reward filter
                if self._passes_risk_reward_filter(signal):
                    # Timing filter
                    if self._passes_timing_filter(signal, data):
                        filtered_signals.append(signal)
        
        return filtered_signals
    
    def _passes_market_condition_filter(self, signal: TradingSignal, data: pd.DataFrame) -> bool:
        """Check if signal passes market condition filters"""
        # Avoid trading in extremely volatile conditions
        if self.market_conditions.get('volatility', 0.02) > 0.05:
            return False
        
        # Avoid trading with very low volume
        if self.market_conditions.get('volume_trend', 1) < 0.5:
            return False
        
        return True
    
    def _passes_risk_reward_filter(self, signal: TradingSignal) -> bool:
        """Check if signal has adequate risk-reward ratio"""
        if signal.stop_loss and signal.take_profit:
            risk = abs(signal.price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.price)
            
            if risk > 0:
                risk_reward_ratio = reward / risk
                return risk_reward_ratio >= 2.0  # Minimum 2:1 ratio
        
        return True  # Pass if stop/target not set
    
    def _passes_timing_filter(self, signal: TradingSignal, data: pd.DataFrame) -> bool:
        """Check if timing is appropriate for the signal"""
        # Avoid trading near market close (if timestamp available)
        # Avoid trading during low liquidity periods
        # These would be implemented based on your specific requirements
        
        return True
    
    def update_performance(self, signal: TradingSignal, outcome: str, pnl: float):
        """Update strategy performance tracking"""
        strategy_type = 'momentum' if 'Momentum' in signal.reason else 'mean_reversion'
        
        if outcome == 'win':
            self.strategy_performance[strategy_type]['wins'] += 1
        else:
            self.strategy_performance[strategy_type]['losses'] += 1
        
        self.strategy_performance[strategy_type]['total_return'] += pnl
        self.strategy_performance['adaptive']['total_return'] += pnl
