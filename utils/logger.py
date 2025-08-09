"""
Advanced Logging System for Algorithmic Trading
Provides comprehensive logging with performance tracking and analysis
"""
import logging
import logging.handlers
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import json
import traceback


class TradingLogger:
    """
    Advanced logging system for trading applications
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_logging()
        
        # Performance tracking
        self.trade_count = 0
        self.error_count = 0
        self.warning_count = 0
        
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        
        # Create logs directory
        log_dir = os.path.dirname(self.config.get('file', 'logs/trading.log'))
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger
        self.logger = logging.getLogger('AlgoTrading')
        self.logger.setLevel(getattr(logging, self.config.get('level', 'INFO')))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.config.get('file', 'logs/trading.log'),
            maxBytes=self.config.get('max_size', 10 * 1024 * 1024),  # 10MB
            backupCount=self.config.get('backup_count', 5)
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Custom formatter
        formatter = self.CustomFormatter()
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Performance logger (separate file)
        self.perf_logger = logging.getLogger('Performance')
        perf_handler = logging.FileHandler('logs/performance.log')
        perf_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)
        
        # Trade logger (separate file)
        self.trade_logger = logging.getLogger('Trades')
        trade_handler = logging.FileHandler('logs/trades.log')
        trade_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        self.trade_logger.addHandler(trade_handler)
        self.trade_logger.setLevel(logging.INFO)
        
        self.logger.info("Advanced Logging System initialized")
    
    class CustomFormatter(logging.Formatter):
        """Custom formatter with colors and enhanced information"""
        
        # Color codes
        COLORS = {
            logging.DEBUG: '\033[36m',    # Cyan
            logging.INFO: '\033[32m',     # Green
            logging.WARNING: '\033[33m',  # Yellow
            logging.ERROR: '\033[31m',    # Red
            logging.CRITICAL: '\033[35m', # Magenta
        }
        RESET = '\033[0m'
        
        def format(self, record):
            # Add color to console output
            if hasattr(record, 'stream') and record.stream == sys.stdout:
                color = self.COLORS.get(record.levelno, self.RESET)
                record.levelname = f"{color}{record.levelname}{self.RESET}"
            
            # Enhanced format with more context
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            return formatter.format(record)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.warning_count += 1
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.error_count += 1
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.error_count += 1
        self.logger.critical(message, extra=kwargs)
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade information"""
        self.trade_count += 1
        
        # Format trade data
        trade_msg = json.dumps({
            'trade_id': trade_data.get('id', f'trade_{self.trade_count}'),
            'timestamp': datetime.now().isoformat(),
            'symbol': trade_data.get('symbol', 'UNKNOWN'),
            'side': trade_data.get('side', 'UNKNOWN'),
            'quantity': trade_data.get('quantity', 0),
            'price': trade_data.get('price', 0),
            'strategy': trade_data.get('strategy', 'UNKNOWN'),
            'confidence': trade_data.get('confidence', 0),
            'reason': trade_data.get('reason', 'No reason provided')
        })
        
        self.trade_logger.info(trade_msg)
        self.info(f"Trade executed: {trade_data.get('symbol')} {trade_data.get('side')} {trade_data.get('quantity')} @ {trade_data.get('price')}")
    
    def log_performance(self, perf_data: Dict[str, Any]):
        """Log performance metrics"""
        
        perf_msg = json.dumps({
            'timestamp': datetime.now().isoformat(),
            'total_return': perf_data.get('total_return', 0),
            'sharpe_ratio': perf_data.get('sharpe_ratio', 0),
            'max_drawdown': perf_data.get('max_drawdown', 0),
            'win_rate': perf_data.get('win_rate', 0),
            'total_trades': perf_data.get('total_trades', 0),
            'portfolio_value': perf_data.get('portfolio_value', 0)
        })
        
        self.perf_logger.info(perf_msg)
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """Log trading signal"""
        
        signal_msg = (f"SIGNAL: {signal_data.get('symbol', 'UNKNOWN')} "
                     f"{signal_data.get('signal', 'HOLD')} "
                     f"Confidence: {signal_data.get('confidence', 0):.2f} "
                     f"Price: {signal_data.get('price', 0):.4f} "
                     f"Reason: {signal_data.get('reason', 'No reason')}")
        
        self.info(signal_msg)
    
    def log_risk_event(self, event_type: str, details: Dict[str, Any]):
        """Log risk management events"""
        
        risk_msg = f"RISK EVENT: {event_type} - {json.dumps(details)}"
        
        if event_type in ['STOP_LOSS_TRIGGERED', 'DRAWDOWN_LIMIT']:
            self.warning(risk_msg)
        elif event_type in ['POSITION_SIZE_EXCEEDED', 'CORRELATION_LIMIT']:
            self.error(risk_msg)
        else:
            self.info(risk_msg)
    
    def log_system_status(self, status: str, details: Optional[Dict[str, Any]] = None):
        """Log system status changes"""
        
        status_msg = f"SYSTEM: {status}"
        if details:
            status_msg += f" - {json.dumps(details)}"
        
        if status in ['STARTING', 'RUNNING', 'OPTIMIZATION_COMPLETE']:
            self.info(status_msg)
        elif status in ['WARNING', 'DEGRADED_PERFORMANCE']:
            self.warning(status_msg)
        elif status in ['ERROR', 'EMERGENCY_SHUTDOWN']:
            self.error(status_msg)
        else:
            self.info(status_msg)
    
    def log_exception(self, exception: Exception, context: Optional[str] = None):
        """Log exception with full traceback"""
        
        exc_msg = f"EXCEPTION: {type(exception).__name__}: {str(exception)}"
        if context:
            exc_msg = f"{context} - {exc_msg}"
        
        self.error(exc_msg)
        self.error(f"Traceback: {traceback.format_exc()}")
    
    def log_backtest_results(self, results: Dict[str, Any]):
        """Log backtest results"""
        
        backtest_msg = (f"BACKTEST RESULTS: "
                       f"Return: {results.get('total_return', 0):.2%} "
                       f"Sharpe: {results.get('sharpe_ratio', 0):.2f} "
                       f"Max DD: {results.get('max_drawdown', 0):.2%} "
                       f"Win Rate: {results.get('win_rate', 0):.2%} "
                       f"Trades: {results.get('total_trades', 0)}")
        
        self.info(backtest_msg)
        self.log_performance(results)
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            'total_trades_logged': self.trade_count,
            'total_errors': self.error_count,
            'total_warnings': self.warning_count,
            'log_level': self.config.get('level', 'INFO')
        }
    
    def create_context_logger(self, context: str) -> 'ContextLogger':
        """Create a context-specific logger"""
        return ContextLogger(self, context)
    
    def flush_logs(self):
        """Flush all log handlers"""
        for handler in self.logger.handlers:
            handler.flush()
        for handler in self.perf_logger.handlers:
            handler.flush()
        for handler in self.trade_logger.handlers:
            handler.flush()


class ContextLogger:
    """Context-specific logger wrapper"""
    
    def __init__(self, parent_logger: TradingLogger, context: str):
        self.parent = parent_logger
        self.context = context
    
    def info(self, message: str, **kwargs):
        self.parent.info(f"[{self.context}] {message}", **kwargs)
    
    def debug(self, message: str, **kwargs):
        self.parent.debug(f"[{self.context}] {message}", **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.parent.warning(f"[{self.context}] {message}", **kwargs)
    
    def error(self, message: str, **kwargs):
        self.parent.error(f"[{self.context}] {message}", **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.parent.critical(f"[{self.context}] {message}", **kwargs)


def setup_logger(config: Optional[Dict[str, Any]] = None) -> TradingLogger:
    """Setup and return configured logger"""
    
    if config is None:
        config = {
            'level': 'INFO',
            'file': 'logs/trading.log',
            'max_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5
        }
    
    return TradingLogger(config)
