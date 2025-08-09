"""
Universal Strategy Manager
Works with any valid ticker symbol and provides adaptive strategy selection
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class UniversalStrategyManager:
    """
    Advanced strategy manager that can adapt to any ticker symbol
    and automatically select optimal strategies based on market conditions
    """
    
    def __init__(self, data_source, config: Dict[str, Any] = None):
        self.data_source = data_source
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Strategy performance cache
        self.strategy_cache = {}
        self.performance_cache = {}
        
        # Available strategies
        self.strategies = {
            'momentum': self._momentum_strategy,
            'mean_reversion': self._mean_reversion_strategy,
            'breakout': self._breakout_strategy,
            'trend_following': self._trend_following_strategy,
            'volatility_based': self._volatility_strategy,
            'adaptive': self._adaptive_strategy
        }
        
        # Market regime detection
        self.regime_detector = None
        self.current_regime = 'unknown'
        
        self.logger.info("Universal Strategy Manager initialized")
    
    def analyze_symbol_characteristics(self, symbol: str, period: str = '1y') -> Dict[str, Any]:
        """
        Analyze symbol characteristics to determine optimal strategies
        
        Args:
            symbol: Ticker symbol
            period: Analysis period
            
        Returns:
            Dictionary with symbol characteristics and strategy recommendations
        """
        try:
            # Get historical data
            hist_data = self.data_source.get_historical_data(symbol, period=period)
            
            if hist_data.empty:
                return {'error': 'No historical data available for analysis'}
            
            # Calculate key characteristics
            returns = hist_data['close'].pct_change().dropna()
            
            characteristics = {
                'symbol': symbol,
                'analysis_period': period,
                'total_data_points': len(hist_data),
                
                # Return characteristics
                'mean_return': returns.mean() * 252,  # Annualized
                'volatility': returns.std() * np.sqrt(252),  # Annualized
                'sharpe_ratio': (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
                
                # Distribution characteristics
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'max_drawdown': self._calculate_max_drawdown(hist_data['close']),
                
                # Trend characteristics
                'trend_strength': self._calculate_trend_strength(hist_data),
                'trend_direction': self._determine_trend_direction(hist_data),
                'trend_consistency': self._calculate_trend_consistency(hist_data),
                
                # Volatility characteristics
                'volatility_regime': self._classify_volatility_regime(returns),
                'volatility_clustering': self._detect_volatility_clustering(returns),
                
                # Mean reversion characteristics
                'mean_reversion_score': self._calculate_mean_reversion_score(returns),
                'autocorrelation': returns.autocorr(lag=1),
                
                # Market microstructure
                'liquidity_score': self._calculate_liquidity_score(hist_data),
                'price_impact': self._estimate_price_impact(hist_data),
                
                # Momentum characteristics
                'momentum_strength': self._calculate_momentum_strength(hist_data),
                'momentum_persistence': self._calculate_momentum_persistence(hist_data),
            }
            
            # Determine market regime
            characteristics['market_regime'] = self._detect_market_regime(hist_data)
            
            # Get strategy recommendations
            characteristics['strategy_recommendations'] = self._recommend_strategies(characteristics)
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error analyzing symbol {symbol}: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators"""
        try:
            # ADX-like calculation (simplified)
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift(1))
            low_close = abs(data['low'] - data['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            plus_dm = np.where((data['high'] - data['high'].shift(1)) > 
                              (data['low'].shift(1) - data['low']),
                              np.maximum(data['high'] - data['high'].shift(1), 0), 0)
            
            minus_dm = np.where((data['low'].shift(1) - data['low']) > 
                               (data['high'] - data['high'].shift(1)),
                               np.maximum(data['low'].shift(1) - data['low'], 0), 0)
            
            atr = true_range.rolling(window=14).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=14).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=14).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()
            
            return adx.iloc[-1] if not adx.empty else 0
            
        except:
            return 0
    
    def _determine_trend_direction(self, data: pd.DataFrame) -> str:
        """Determine overall trend direction"""
        try:
            if len(data) < 50:
                return 'insufficient_data'
            
            # Use multiple timeframes
            short_ma = data['close'].rolling(window=20).mean()
            long_ma = data['close'].rolling(window=50).mean()
            
            latest_short = short_ma.iloc[-1]
            latest_long = long_ma.iloc[-1]
            latest_price = data['close'].iloc[-1]
            
            if latest_price > latest_short > latest_long:
                return 'bullish'
            elif latest_price < latest_short < latest_long:
                return 'bearish'
            else:
                return 'sideways'
                
        except:
            return 'unknown'
    
    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """Calculate how consistent the trend is"""
        try:
            returns = data['close'].pct_change().dropna()
            
            # Count consecutive positive/negative returns
            signs = np.sign(returns)
            sign_changes = (signs != signs.shift(1)).sum()
            consistency = 1 - (sign_changes / len(returns))
            
            return consistency
            
        except:
            return 0
    
    def _classify_volatility_regime(self, returns: pd.Series) -> str:
        """Classify volatility regime"""
        try:
            vol = returns.std() * np.sqrt(252)  # Annualized volatility
            
            if vol < 0.15:
                return 'low_volatility'
            elif vol < 0.30:
                return 'medium_volatility'
            else:
                return 'high_volatility'
                
        except:
            return 'unknown'
    
    def _detect_volatility_clustering(self, returns: pd.Series) -> float:
        """Detect volatility clustering using ARCH effects"""
        try:
            # Simple ARCH test
            squared_returns = returns ** 2
            autocorr = squared_returns.autocorr(lag=1)
            return autocorr if not np.isnan(autocorr) else 0
            
        except:
            return 0
    
    def _calculate_mean_reversion_score(self, returns: pd.Series) -> float:
        """Calculate mean reversion tendency"""
        try:
            # Hurst exponent approximation
            if len(returns) < 100:
                return 0
                
            # Use variance ratio test
            lags = [2, 4, 8, 16]
            variance_ratios = []
            
            for lag in lags:
                if len(returns) >= lag * 30:  # Ensure sufficient data
                    var_1 = returns.var()
                    var_lag = returns.rolling(window=lag).sum().dropna().var() / lag
                    if var_1 > 0:
                        variance_ratios.append(var_lag / var_1)
            
            if variance_ratios:
                mean_vr = np.mean(variance_ratios)
                # < 1 suggests mean reversion, > 1 suggests momentum
                return 1 - mean_vr  # Convert to mean reversion score
            else:
                return 0
                
        except:
            return 0
    
    def _calculate_liquidity_score(self, data: pd.DataFrame) -> float:
        """Calculate liquidity score based on volume and price impact"""
        try:
            if 'volume' not in data.columns:
                return 0
                
            # Average daily volume in USD
            avg_volume_usd = (data['volume'] * data['close']).mean()
            
            # Normalize to a 0-1 score
            if avg_volume_usd > 100_000_000:  # $100M+
                return 1.0
            elif avg_volume_usd > 10_000_000:   # $10M+
                return 0.8
            elif avg_volume_usd > 1_000_000:    # $1M+
                return 0.6
            elif avg_volume_usd > 100_000:      # $100K+
                return 0.4
            else:
                return 0.2
                
        except:
            return 0.5
    
    def _estimate_price_impact(self, data: pd.DataFrame) -> float:
        """Estimate price impact of trades"""
        try:
            # Simplified Amihud illiquidity measure
            returns = data['close'].pct_change().abs()
            volume = data['volume'] * data['close']  # Dollar volume
            
            # Avoid division by zero
            volume_safe = volume.replace(0, np.nan)
            illiquidity = (returns / volume_safe).dropna()
            
            return illiquidity.mean() if not illiquidity.empty else 0
            
        except:
            return 0
    
    def _calculate_momentum_strength(self, data: pd.DataFrame) -> float:
        """Calculate momentum strength"""
        try:
            # Multiple timeframe momentum
            periods = [10, 20, 50]
            momentum_scores = []
            
            for period in periods:
                if len(data) >= period + 1:
                    momentum = (data['close'].iloc[-1] / data['close'].iloc[-period] - 1)
                    momentum_scores.append(momentum)
            
            if momentum_scores:
                return np.mean(momentum_scores)
            else:
                return 0
                
        except:
            return 0
    
    def _calculate_momentum_persistence(self, data: pd.DataFrame) -> float:
        """Calculate how persistent momentum is"""
        try:
            if len(data) < 50:
                return 0
                
            returns = data['close'].pct_change().dropna()
            
            # Calculate momentum at different lags and check persistence
            momentum_10 = returns.rolling(10).mean()
            momentum_20 = returns.rolling(20).mean()
            
            # Correlation between different momentum measures
            correlation = momentum_10.corr(momentum_20)
            
            return correlation if not np.isnan(correlation) else 0
            
        except:
            return 0
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """Detect current market regime"""
        try:
            if len(data) < 50:
                return 'insufficient_data'
            
            returns = data['close'].pct_change().dropna()
            
            # Recent performance vs long-term
            recent_return = returns.tail(20).mean()
            recent_vol = returns.tail(20).std()
            long_term_vol = returns.std()
            
            vol_ratio = recent_vol / long_term_vol if long_term_vol > 0 else 1
            
            # Classify regime
            if recent_return > 0.001 and vol_ratio < 1.2:
                return 'bull_market'
            elif recent_return < -0.001 and vol_ratio > 1.2:
                return 'bear_market'
            elif vol_ratio > 1.5:
                return 'high_volatility'
            else:
                return 'sideways_market'
                
        except:
            return 'unknown'
    
    def _recommend_strategies(self, characteristics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend optimal strategies based on characteristics"""
        recommendations = []
        
        try:
            # Strategy scoring based on characteristics
            scores = {}
            
            # Momentum strategy
            momentum_score = (
                characteristics.get('momentum_strength', 0) * 0.3 +
                characteristics.get('trend_strength', 0) * 0.01 +  # Scale down ADX
                (1 if characteristics.get('trend_direction') == 'bullish' else 0) * 0.2 +
                characteristics.get('momentum_persistence', 0) * 0.2 +
                (1 - abs(characteristics.get('autocorrelation', 0))) * 0.3  # Low autocorr good for momentum
            )
            scores['momentum'] = max(0, min(1, momentum_score))
            
            # Mean reversion strategy  
            mean_reversion_score = (
                characteristics.get('mean_reversion_score', 0) * 0.4 +
                abs(characteristics.get('autocorrelation', 0)) * 0.3 +  # High autocorr good for mean reversion
                (1 if characteristics.get('volatility_regime') == 'high_volatility' else 0.5) * 0.2 +
                (1 if characteristics.get('market_regime') == 'sideways_market' else 0) * 0.1
            )
            scores['mean_reversion'] = max(0, min(1, mean_reversion_score))
            
            # Breakout strategy
            breakout_score = (
                (characteristics.get('volatility_clustering', 0) + 1) / 2 * 0.3 +  # Normalize to 0-1
                (1 if characteristics.get('volatility_regime') == 'medium_volatility' else 0.5) * 0.3 +
                characteristics.get('liquidity_score', 0) * 0.2 +
                (1 if characteristics.get('trend_direction') != 'sideways' else 0) * 0.2
            )
            scores['breakout'] = max(0, min(1, breakout_score))
            
            # Trend following strategy
            trend_following_score = (
                characteristics.get('trend_strength', 0) * 0.01 +  # Scale down ADX
                characteristics.get('trend_consistency', 0) * 0.3 +
                (1 if characteristics.get('trend_direction') != 'sideways' else 0) * 0.3 +
                (1 if characteristics.get('volatility_regime') == 'low_volatility' else 0.5) * 0.2 +
                characteristics.get('momentum_persistence', 0) * 0.2
            )
            scores['trend_following'] = max(0, min(1, trend_following_score))
            
            # Volatility strategy
            volatility_score = (
                (1 if characteristics.get('volatility_regime') == 'high_volatility' else 0.5) * 0.4 +
                characteristics.get('volatility_clustering', 0) * 0.3 +
                characteristics.get('liquidity_score', 0) * 0.2 +
                (1 - characteristics.get('trend_consistency', 0)) * 0.1  # High vol benefits from inconsistency
            )
            scores['volatility_based'] = max(0, min(1, volatility_score))
            
            # Adaptive strategy (always decent option)
            scores['adaptive'] = 0.6  # Base score for adaptive
            
            # Create recommendations sorted by score
            for strategy, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                confidence = self._calculate_confidence(characteristics, strategy, score)
                
                recommendations.append({
                    'strategy': strategy,
                    'score': round(score, 3),
                    'confidence': round(confidence, 3),
                    'rationale': self._get_strategy_rationale(characteristics, strategy)
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in strategy recommendation: {str(e)}")
            return [{'strategy': 'adaptive', 'score': 0.5, 'confidence': 0.5, 
                    'rationale': 'Default recommendation due to analysis error'}]
    
    def _calculate_confidence(self, characteristics: Dict[str, Any], 
                            strategy: str, score: float) -> float:
        """Calculate confidence in strategy recommendation"""
        try:
            base_confidence = score
            
            # Adjust based on data quality
            data_points = characteristics.get('total_data_points', 0)
            if data_points < 100:
                base_confidence *= 0.7
            elif data_points < 252:
                base_confidence *= 0.85
            
            # Adjust based on market conditions
            if characteristics.get('market_regime') == 'high_volatility':
                base_confidence *= 0.9  # Reduce confidence in volatile markets
            
            # Adjust based on liquidity
            liquidity_score = characteristics.get('liquidity_score', 0.5)
            base_confidence *= (0.7 + 0.3 * liquidity_score)
            
            return max(0, min(1, base_confidence))
            
        except:
            return 0.5
    
    def _get_strategy_rationale(self, characteristics: Dict[str, Any], 
                              strategy: str) -> str:
        """Get human-readable rationale for strategy recommendation"""
        
        rationales = {
            'momentum': f"Strong momentum (trend strength: {characteristics.get('trend_strength', 0):.1f}, "
                       f"direction: {characteristics.get('trend_direction', 'unknown')})",
            
            'mean_reversion': f"Mean reversion indicators (autocorr: {characteristics.get('autocorrelation', 0):.2f}, "
                             f"reversion score: {characteristics.get('mean_reversion_score', 0):.2f})",
            
            'breakout': f"Volatility clustering and medium volatility regime "
                       f"({characteristics.get('volatility_regime', 'unknown')})",
            
            'trend_following': f"Consistent trend (consistency: {characteristics.get('trend_consistency', 0):.2f}, "
                              f"direction: {characteristics.get('trend_direction', 'unknown')})",
            
            'volatility_based': f"High volatility environment "
                               f"({characteristics.get('volatility_regime', 'unknown')})",
            
            'adaptive': "Balanced approach suitable for uncertain market conditions"
        }
        
        return rationales.get(strategy, "Strategy selected based on market analysis")
    
    # Strategy implementations (simplified for demonstration)
    def _momentum_strategy(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Simple momentum strategy"""
        try:
            returns = data['close'].pct_change()
            momentum = returns.rolling(window=20).mean()
            signals = (momentum > 0.001).astype(int)  # Buy when positive momentum
            return signals
        except:
            return pd.Series([0] * len(data))
    
    def _mean_reversion_strategy(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Simple mean reversion strategy"""
        try:
            prices = data['close']
            ma = prices.rolling(window=20).mean()
            std = prices.rolling(window=20).std()
            
            upper_band = ma + 2 * std
            lower_band = ma - 2 * std
            
            # Buy when price below lower band, sell when above upper band
            signals = pd.Series(0, index=data.index)
            signals[prices < lower_band] = 1  # Buy signal
            signals[prices > upper_band] = -1  # Sell signal
            
            return signals
        except:
            return pd.Series([0] * len(data))
    
    def _breakout_strategy(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Simple breakout strategy"""
        try:
            prices = data['close']
            high_20 = data['high'].rolling(window=20).max()
            low_20 = data['low'].rolling(window=20).min()
            
            signals = pd.Series(0, index=data.index)
            signals[prices > high_20.shift(1)] = 1  # Breakout above resistance
            signals[prices < low_20.shift(1)] = -1  # Breakdown below support
            
            return signals
        except:
            return pd.Series([0] * len(data))
    
    def _trend_following_strategy(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Simple trend following strategy"""
        try:
            short_ma = data['close'].rolling(window=10).mean()
            long_ma = data['close'].rolling(window=30).mean()
            
            signals = (short_ma > long_ma).astype(int)
            return signals
        except:
            return pd.Series([0] * len(data))
    
    def _volatility_strategy(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Simple volatility-based strategy"""
        try:
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=20).std()
            vol_threshold = volatility.quantile(0.8)
            
            # Trade in high volatility periods
            signals = (volatility > vol_threshold).astype(int)
            return signals
        except:
            return pd.Series([0] * len(data))
    
    def _adaptive_strategy(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Adaptive strategy that combines multiple approaches"""
        try:
            # Get signals from multiple strategies
            momentum_signals = self._momentum_strategy(data)
            mean_rev_signals = self._mean_reversion_strategy(data)
            trend_signals = self._trend_following_strategy(data)
            
            # Combine with equal weights (simplified)
            combined = (momentum_signals + mean_rev_signals + trend_signals) / 3
            
            # Convert to binary signals
            signals = (combined > 0.5).astype(int)
            return signals
        except:
            return pd.Series([0] * len(data))


# Example usage
def test_universal_strategy_manager():
    """Test the universal strategy manager"""
    print("🧪 Testing Universal Strategy Manager")
    print("=" * 50)
    
    # This would normally use the enhanced data source
    # For testing, we'll use a mock data source
    class MockDataSource:
        def get_historical_data(self, symbol, period='1y'):
            # Generate sample data
            dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            np.random.seed(42)
            
            prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates)))
            volumes = np.random.randint(1000000, 10000000, len(dates))
            
            data = pd.DataFrame({
                'close': prices,
                'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
                'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
                'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
                'volume': volumes
            }, index=dates)
            
            return data
    
    mock_data_source = MockDataSource()
    strategy_manager = UniversalStrategyManager(mock_data_source)
    
    # Test different symbols
    test_symbols = ['AAPL', 'TSLA', 'RELIANCE.NS', 'BTC-USD']
    
    for symbol in test_symbols:
        print(f"\n📊 Analyzing {symbol}")
        print("-" * 30)
        
        characteristics = strategy_manager.analyze_symbol_characteristics(symbol)
        
        if 'error' not in characteristics:
            print(f"Market Regime: {characteristics['market_regime']}")
            print(f"Volatility: {characteristics['volatility']:.2%}")
            print(f"Trend Direction: {characteristics['trend_direction']}")
            print(f"Liquidity Score: {characteristics['liquidity_score']:.2f}")
            
            print("\n🎯 Strategy Recommendations:")
            for i, rec in enumerate(characteristics['strategy_recommendations'][:3], 1):
                print(f"  {i}. {rec['strategy'].title()}")
                print(f"     Score: {rec['score']:.3f} | Confidence: {rec['confidence']:.3f}")
                print(f"     Rationale: {rec['rationale']}")
        else:
            print(f"❌ Error: {characteristics['error']}")


if __name__ == '__main__':
    test_universal_strategy_manager()
