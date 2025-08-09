"""
Portfolio Optimizer Methods
Contains optimization methods for the PortfolioOptimizer class
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable
from scipy.optimize import differential_evolution
import logging

class PortfolioOptimizerMethods:
    """Additional optimization methods for portfolio optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters
        self.max_iterations = config.get('max_iterations', 100)
        self.max_position_size = config.get('max_position_size', 0.1)
    
    def _bayesian_optimization(self, returns: pd.DataFrame, objective_func: Callable) -> Dict[str, float]:
        """Bayesian optimization for portfolio weights"""
        
        try:
            from skopt import gp_minimize
            from skopt.space import Real
            
            n_assets = len(returns.columns)
            
            # Define search space - each weight between 0 and 1
            space = [Real(0.0, self.max_position_size) for _ in range(n_assets)]
            
            def objective(weights):
                # Normalize weights to sum to 1
                weights = np.array(weights)
                if np.sum(weights) == 0:
                    return float('inf')
                weights = weights / np.sum(weights)
                
                return -objective_func(weights, returns)  # Minimize negative
            
            # Run optimization
            result = gp_minimize(
                objective, space, n_calls=self.max_iterations,
                random_state=42, acq_func='EI'
            )
            
            # Normalize final weights
            optimal_weights = np.array(result.x)
            if np.sum(optimal_weights) > 0:
                optimal_weights = optimal_weights / np.sum(optimal_weights)
            else:
                optimal_weights = np.ones(n_assets) / n_assets
            
            return dict(zip(returns.columns, optimal_weights))
            
        except ImportError:
            self.logger.warning("scikit-optimize not available, using equal weights")
            return self._equal_weights(returns.columns)
        except Exception as e:
            self.logger.error(f"Bayesian optimization failed: {str(e)}")
            return self._equal_weights(returns.columns)
    
    def _genetic_optimization(self, returns: pd.DataFrame, objective_func: Callable) -> Dict[str, float]:
        """Genetic algorithm optimization for portfolio weights"""
        
        try:
            n_assets = len(returns.columns)
            
            def objective(weights):
                # Normalize weights
                if np.sum(weights) == 0:
                    return float('inf')
                weights = weights / np.sum(weights)
                return -objective_func(weights, returns)  # Minimize negative
            
            # Define bounds - each weight between 0 and max_position_size
            bounds = [(0.0, self.max_position_size) for _ in range(n_assets)]
            
            # Run optimization
            result = differential_evolution(
                objective, bounds, 
                maxiter=self.max_iterations,
                seed=42, polish=True
            )
            
            if result.success:
                # Normalize weights
                optimal_weights = result.x
                if np.sum(optimal_weights) > 0:
                    optimal_weights = optimal_weights / np.sum(optimal_weights)
                else:
                    optimal_weights = np.ones(n_assets) / n_assets
                return dict(zip(returns.columns, optimal_weights))
            else:
                self.logger.warning("Genetic optimization did not converge")
                return self._equal_weights(returns.columns)
                
        except Exception as e:
            self.logger.error(f"Genetic optimization failed: {str(e)}")
            return self._equal_weights(returns.columns)
    
    def _equal_weights(self, symbols: List[str]) -> Dict[str, float]:
        """Fallback equal weight allocation"""
        weight = 1.0 / len(symbols)
        return {symbol: weight for symbol in symbols}
    
    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply portfolio constraints"""
        
        # Apply maximum position size constraint
        constrained_weights = {}
        total_weight = 0
        
        for symbol, weight in weights.items():
            constrained_weight = min(weight, self.max_position_size)
            constrained_weights[symbol] = constrained_weight
            total_weight += constrained_weight
        
        # Renormalize if needed
        if total_weight > 0 and total_weight != 1.0:
            for symbol in constrained_weights:
                constrained_weights[symbol] /= total_weight
        
        return constrained_weights
    
    def _sharpe_objective(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """Sharpe ratio objective function"""
        
        portfolio_returns = (returns * weights).sum(axis=1)
        
        if len(portfolio_returns) < 2:
            return 0.0
        
        mean_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        
        if volatility == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        return (mean_return / volatility) * np.sqrt(252)
    
    async def calculate_portfolio_metrics(self, weights: Dict[str, float],
                                        returns: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive portfolio metrics"""
        
        try:
            weight_array = np.array([weights.get(col, 0) for col in returns.columns])
            portfolio_returns = (returns * weight_array).sum(axis=1)
            
            # Basic metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Risk metrics
            sharpe_ratio = (annualized_return / volatility) if volatility > 0 else 0
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            
            # Advanced metrics
            sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
            calmar_ratio = (annualized_return / abs(max_drawdown)) if max_drawdown < 0 else 0
            
            # Risk measures
            var_95 = portfolio_returns.quantile(0.05) if len(portfolio_returns) > 20 else 0
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if len(portfolio_returns) > 20 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'weights': weights
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio metrics calculation failed: {str(e)}")
            return {}
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        
        mean_return = returns.mean()
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_deviation = negative_returns.std()
        
        if downside_deviation == 0:
            return 0.0
        
        return (mean_return / downside_deviation) * np.sqrt(252)
    
    async def rebalance_portfolio(self, current_positions: Dict[str, float],
                                target_weights: Dict[str, float],
                                total_value: float) -> List[Dict[str, Any]]:
        """
        Generate rebalancing trades
        
        Args:
            current_positions: Current position weights by symbol
            target_weights: Target allocation weights
            total_value: Total portfolio value
            
        Returns:
            List of trade recommendations
        """
        
        trades = []
        threshold = self.config.get('rebalance_threshold', 0.05)  # 5% threshold
        
        try:
            all_symbols = set(current_positions.keys()) | set(target_weights.keys())
            
            for symbol in all_symbols:
                current_weight = current_positions.get(symbol, 0.0)
                target_weight = target_weights.get(symbol, 0.0)
                
                weight_diff = target_weight - current_weight
                
                # Only trade if difference exceeds threshold
                if abs(weight_diff) > threshold:
                    dollar_amount = weight_diff * total_value
                    
                    trade = {
                        'symbol': symbol,
                        'action': 'buy' if weight_diff > 0 else 'sell',
                        'weight_change': weight_diff,
                        'dollar_amount': abs(dollar_amount),
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'priority': abs(weight_diff)  # Larger changes have higher priority
                    }
                    
                    trades.append(trade)
            
            # Sort by priority (largest weight changes first)
            trades.sort(key=lambda x: x['priority'], reverse=True)
            
            self.logger.info(f"Generated {len(trades)} rebalancing trades")
            return trades
            
        except Exception as e:
            self.logger.error(f"Rebalancing calculation failed: {str(e)}")
            return []
