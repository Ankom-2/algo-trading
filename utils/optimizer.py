"""
Advanced Portfolio Optimizer for Algorithmic Trading
Implements multiple optimization techniques for maximum performance
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Callable
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')
import logging


class PortfolioOptimizer:
    """
    Advanced portfolio optimization system using multiple methodologies
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters
        self.optimization_method = config.get('optimization_method', 'bayesian')
        self.max_iterations = config.get('max_iterations', 100)
        self.cv_folds = config.get('cv_folds', 5)
        self.lookback_period = config.get('lookback_period', 252)
        
        # Risk constraints
        self.max_position_size = config.get('max_position_size', 0.1)
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.15)
        self.target_sharpe = config.get('target_sharpe', 2.0)
        
        # Optimization history
        self.optimization_history = []
        
        self.logger.info("Portfolio Optimizer initialized")
    
    async def optimize_portfolio(self, returns_data: pd.DataFrame, 
                               method: str = None) -> Dict[str, Any]:
        """
        Optimize portfolio weights using specified method
        
        Args:
            returns_data: Historical returns data
            method: Optimization method ('bayesian', 'genetic', 'mean_variance')
            
        Returns:
            Optimization results with weights and metrics
        """
        
        method = method or self.optimization_method
        self.logger.info(f"Starting portfolio optimization using {method}")
        
        try:
            # Prepare data
            clean_returns = self._prepare_returns_data(returns_data)
            
            if len(clean_returns) < self.lookback_period:
                self.logger.warning(f"Insufficient data: {len(clean_returns)} < {self.lookback_period}")
            
            # Select optimization method
            if method == 'bayesian':
                weights = self._bayesian_optimization(clean_returns, self._sharpe_objective)
            elif method == 'genetic':
                weights = self._genetic_optimization(clean_returns, self._sharpe_objective)
            elif method == 'mean_variance':
                weights = self._mean_variance_optimization(clean_returns)
            elif method == 'risk_parity':
                weights = self._risk_parity_optimization(clean_returns)
            elif method == 'max_diversification':
                weights = self._max_diversification_optimization(clean_returns)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Calculate portfolio metrics
            metrics = await self.calculate_portfolio_metrics(weights, clean_returns)
            
            # Validate constraints
            weights = self._apply_constraints(weights)
            
            result = {
                'weights': weights,
                'metrics': metrics,
                'method': method,
                'timestamp': pd.Timestamp.now(),
                'data_points': len(clean_returns)
            }
            
            # Store in history
            self.optimization_history.append(result)
            
            self.logger.info(f"Optimization completed. Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {str(e)}")
            return {'weights': {}, 'metrics': {}, 'error': str(e)}
    
    def _prepare_returns_data(self, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare returns data for optimization"""
        
        # Remove NaN values
        clean_data = returns_data.dropna()
        
        # Remove columns with insufficient data
        min_periods = max(30, self.lookback_period // 10)
        clean_data = clean_data.loc[:, clean_data.count() >= min_periods]
        
        # Use only recent data if specified
        if len(clean_data) > self.lookback_period:
            clean_data = clean_data.tail(self.lookback_period)
        
        return clean_data
    
    def _mean_variance_optimization(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Classic mean-variance optimization"""
        
        try:
            from scipy.optimize import minimize
            
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            n_assets = len(returns.columns)
            
            # Objective: maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.sum(weights * mean_returns)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                return -portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            
            # Bounds
            bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
            
            # Initial guess
            x0 = np.array([1/n_assets] * n_assets)
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return dict(zip(returns.columns, result.x))
            else:
                return self._equal_weights(returns.columns)
                
        except Exception as e:
            self.logger.error(f"Mean-variance optimization failed: {str(e)}")
            return self._equal_weights(returns.columns)
    
    def _risk_parity_optimization(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Risk parity optimization"""
        
        try:
            from scipy.optimize import minimize
            
            cov_matrix = returns.cov().values
            n_assets = len(returns.columns)
            
            def risk_parity_objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                
                if portfolio_vol == 0:
                    return float('inf')
                
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                risk_contrib = weights * marginal_contrib
                
                # Target equal risk contribution
                target_risk = np.ones(n_assets) / n_assets
                
                return np.sum((risk_contrib - target_risk) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
            x0 = np.array([1/n_assets] * n_assets)
            
            result = minimize(risk_parity_objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return dict(zip(returns.columns, result.x))
            else:
                return self._equal_weights(returns.columns)
                
        except Exception as e:
            self.logger.error(f"Risk parity optimization failed: {str(e)}")
            return self._equal_weights(returns.columns)
    
    def _max_diversification_optimization(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Maximum diversification optimization"""
        
        try:
            from scipy.optimize import minimize
            
            individual_vols = returns.std().values
            cov_matrix = returns.cov().values
            n_assets = len(returns.columns)
            
            def max_div_objective(weights):
                weighted_avg_vol = np.sum(weights * individual_vols)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                
                if portfolio_vol == 0:
                    return 0
                
                return -weighted_avg_vol / portfolio_vol  # Negative for maximization
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
            x0 = np.array([1/n_assets] * n_assets)
            
            result = minimize(max_div_objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return dict(zip(returns.columns, result.x))
            else:
                return self._equal_weights(returns.columns)
                
        except Exception as e:
            self.logger.error(f"Max diversification optimization failed: {str(e)}")
            return self._equal_weights(returns.columns)
