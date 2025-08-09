"""
Advanced Broker Interface for Live Trading
Unified interface for multiple brokerage APIs with advanced features
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import broker APIs with error handling
broker_modules = {}
try:
    import alpaca_trade_api as tradeapi
    broker_modules['alpaca'] = tradeapi
except ImportError:
    logging.warning("Alpaca Trade API not available")

try:
    import yfinance as yf
    broker_modules['yahoo'] = yf
except ImportError:
    logging.warning("Yahoo Finance not available")


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted" 
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


@dataclass
class BrokerOrder:
    """Unified order representation"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'day'  # 'day', 'gtc', 'ioc', 'fok'
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    commission: float = 0.0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass 
class BrokerPosition:
    """Unified position representation"""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    side: str  # 'long' or 'short'
    last_updated: datetime


@dataclass
class AccountInfo:
    """Account information"""
    account_id: str
    buying_power: float
    cash: float
    portfolio_value: float
    day_trade_count: int = 0
    is_pattern_day_trader: bool = False
    positions: Dict[str, BrokerPosition] = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}


class BrokerInterface(ABC):
    """Abstract base class for broker implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.connected = False
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker API"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker API"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get account information"""
        pass
    
    @abstractmethod
    async def submit_order(self, order: BrokerOrder) -> str:
        """Submit an order and return order ID"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> BrokerOrder:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, BrokerPosition]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str, period: str = '1d') -> pd.DataFrame:
        """Get market data for a symbol"""
        pass


class BrokerManager:
    """
    Advanced broker management with multiple broker support
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.brokers: Dict[str, BrokerInterface] = {}
        self.active_broker = None
        self.order_callbacks: List[Callable] = []
        
    def get_active_broker(self) -> Optional[BrokerInterface]:
        """Get the active broker instance"""
        
        if self.active_broker and self.active_broker in self.brokers:
            return self.brokers[self.active_broker]
        return None
