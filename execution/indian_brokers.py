"""
Indian Brokers Implementation
Support for Zerodha Kite and IIFL APIs
"""
import asyncio
import logging
import requests
import json
import hmac
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np
from urllib.parse import urlencode
from .broker import BrokerOrder, BrokerPosition, AccountInfo, OrderStatus, OrderType


class ZerodhaBroker:
    """Zerodha Kite Connect API Implementation"""
    
    def __init__(self, api_key: str, api_secret: str, access_token: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.base_url = "https://api.kite.trade"
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
        if access_token:
            self.session.headers.update({
                'X-Kite-Version': '3',
                'Authorization': f'token {api_key}:{access_token}'
            })
    
    def generate_session(self, request_token: str) -> str:
        """Generate session using request token"""
        checksum = hashlib.sha256(
            f"{self.api_key}{request_token}{self.api_secret}".encode()
        ).hexdigest()
        
        url = f"{self.base_url}/session/token"
        data = {
            'api_key': self.api_key,
            'request_token': request_token,
            'checksum': checksum
        }
        
        response = requests.post(url, data=data)
        if response.status_code == 200:
            result = response.json()
            self.access_token = result['data']['access_token']
            self.session.headers.update({
                'X-Kite-Version': '3',
                'Authorization': f'token {self.api_key}:{self.access_token}'
            })
            return self.access_token
        else:
            raise Exception(f"Session generation failed: {response.text}")
    
    def get_profile(self) -> Dict[str, Any]:
        """Get user profile"""
        response = self.session.get(f"{self.base_url}/user/profile")
        if response.status_code == 200:
            return response.json()['data']
        else:
            raise Exception(f"Profile fetch failed: {response.text}")
    
    def get_positions(self) -> List[BrokerPosition]:
        """Get current positions"""
        response = self.session.get(f"{self.base_url}/portfolio/positions")
        if response.status_code == 200:
            positions = []
            for pos in response.json()['data']:
                if pos['quantity'] != 0:
                    position = BrokerPosition(
                        symbol=pos['tradingsymbol'],
                        quantity=pos['quantity'],
                        avg_price=pos['average_price'],
                        market_value=pos['last_price'] * pos['quantity'],
                        unrealized_pnl=pos['unrealised'],
                        realized_pnl=pos['realised'],
                        side='long' if pos['quantity'] > 0 else 'short',
                        last_updated=datetime.now()
                    )
                    positions.append(position)
            return positions
        else:
            raise Exception(f"Positions fetch failed: {response.text}")
    
    def place_order(self, order: BrokerOrder) -> str:
        """Place an order"""
        url = f"{self.base_url}/orders/regular"
        
        # Map order type
        variety = "regular"
        if order.order_type == OrderType.MARKET:
            order_type = "MARKET"
        elif order.order_type == OrderType.LIMIT:
            order_type = "LIMIT"
        elif order.order_type == OrderType.STOP:
            order_type = "SL"
        else:
            order_type = "LIMIT"
        
        data = {
            'exchange': 'NSE',
            'tradingsymbol': order.symbol,
            'transaction_type': order.side.upper(),
            'quantity': int(order.quantity),
            'order_type': order_type,
            'product': 'MIS',  # Intraday
            'validity': 'DAY'
        }
        
        if order.price:
            data['price'] = order.price
        if order.stop_price:
            data['trigger_price'] = order.stop_price
        
        response = self.session.post(url, data=data)
        if response.status_code == 200:
            result = response.json()
            order.order_id = result['data']['order_id']
            order.status = OrderStatus.SUBMITTED
            return order.order_id
        else:
            raise Exception(f"Order placement failed: {response.text}")
    
    def get_orders(self) -> List[BrokerOrder]:
        """Get order history"""
        response = self.session.get(f"{self.base_url}/orders")
        if response.status_code == 200:
            orders = []
            for order_data in response.json()['data']:
                order = BrokerOrder(
                    symbol=order_data['tradingsymbol'],
                    side=order_data['transaction_type'].lower(),
                    quantity=order_data['quantity'],
                    order_type=OrderType.MARKET if order_data['order_type'] == 'MARKET' else OrderType.LIMIT,
                    price=order_data.get('price'),
                    order_id=order_data['order_id'],
                    status=self._map_order_status(order_data['status']),
                    filled_quantity=order_data['filled_quantity'],
                    avg_fill_price=order_data.get('average_price')
                )
                orders.append(order)
            return orders
        else:
            raise Exception(f"Orders fetch failed: {response.text}")
    
    def get_instruments(self, exchange: str = "NSE") -> pd.DataFrame:
        """Get all instruments for an exchange"""
        url = f"{self.base_url}/instruments/{exchange}"
        response = self.session.get(url)
        
        if response.status_code == 200:
            # Convert CSV response to DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            return df
        else:
            raise Exception(f"Instruments fetch failed: {response.text}")
    
    def get_historical_data(self, instrument_token: str, from_date: str, 
                          to_date: str, interval: str = "day") -> pd.DataFrame:
        """Get historical data"""
        url = f"{self.base_url}/instruments/historical/{instrument_token}/{interval}"
        params = {
            'from': from_date,
            'to': to_date
        }
        
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()['data']['candles']
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.set_index('timestamp')
        else:
            raise Exception(f"Historical data fetch failed: {response.text}")
    
    def _map_order_status(self, status: str) -> OrderStatus:
        """Map Zerodha order status to internal status"""
        status_map = {
            'OPEN': OrderStatus.SUBMITTED,
            'COMPLETE': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED
        }
        return status_map.get(status, OrderStatus.PENDING)


class IIFLBroker:
    """IIFL Markets API Implementation"""
    
    def __init__(self, api_key: str, api_secret: str, client_id: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.client_id = client_id
        self.base_url = "https://ttblaze.iifl.com/interactive"
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        self.auth_token = None
    
    def login(self, password: str) -> str:
        """Login to IIFL"""
        url = f"{self.base_url}/user/session"
        
        # Generate authentication
        timestamp = str(int(time.time()))
        signature = hmac.new(
            self.api_secret.encode(),
            f"{self.api_key}{timestamp}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            'X-API-KEY': self.api_key,
            'X-TIMESTAMP': timestamp,
            'X-SIGNATURE': signature,
            'Content-Type': 'application/json'
        }
        
        data = {
            'clientId': self.client_id,
            'password': password
        }
        
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            self.auth_token = result['result']['token']
            self.session.headers.update({
                'Authorization': f'Bearer {self.auth_token}'
            })
            return self.auth_token
        else:
            raise Exception(f"Login failed: {response.text}")
    
    def get_positions(self) -> List[BrokerPosition]:
        """Get current positions"""
        if not self.auth_token:
            raise Exception("Not authenticated")
        
        response = self.session.get(f"{self.base_url}/portfolio/positions")
        if response.status_code == 200:
            positions = []
            for pos in response.json()['result']:
                if pos['netQuantity'] != 0:
                    position = BrokerPosition(
                        symbol=pos['symbol'],
                        quantity=pos['netQuantity'],
                        avg_price=pos['avgPrice'],
                        market_value=pos['marketValue'],
                        unrealized_pnl=pos['unrealizedPnL'],
                        realized_pnl=pos['realizedPnL'],
                        side='long' if pos['netQuantity'] > 0 else 'short',
                        last_updated=datetime.now()
                    )
                    positions.append(position)
            return positions
        else:
            raise Exception(f"Positions fetch failed: {response.text}")
    
    def place_order(self, order: BrokerOrder) -> str:
        """Place an order"""
        if not self.auth_token:
            raise Exception("Not authenticated")
        
        url = f"{self.base_url}/orders"
        
        # Map order type
        if order.order_type == OrderType.MARKET:
            order_type = "MKT"
        elif order.order_type == OrderType.LIMIT:
            order_type = "L"
        else:
            order_type = "L"
        
        data = {
            'exchange': 'NSE',
            'symbol': order.symbol,
            'side': order.side.upper(),
            'quantity': int(order.quantity),
            'orderType': order_type,
            'productType': 'MIS',  # Intraday
            'validity': 'DAY'
        }
        
        if order.price:
            data['price'] = order.price
        if order.stop_price:
            data['stopPrice'] = order.stop_price
        
        response = self.session.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            order.order_id = result['result']['orderId']
            order.status = OrderStatus.SUBMITTED
            return order.order_id
        else:
            raise Exception(f"Order placement failed: {response.text}")
    
    def get_orders(self) -> List[BrokerOrder]:
        """Get order history"""
        if not self.auth_token:
            raise Exception("Not authenticated")
        
        response = self.session.get(f"{self.base_url}/orders")
        if response.status_code == 200:
            orders = []
            for order_data in response.json()['result']:
                order = BrokerOrder(
                    symbol=order_data['symbol'],
                    side=order_data['side'].lower(),
                    quantity=order_data['quantity'],
                    order_type=OrderType.MARKET if order_data['orderType'] == 'MKT' else OrderType.LIMIT,
                    price=order_data.get('price'),
                    order_id=order_data['orderId'],
                    status=self._map_order_status(order_data['status']),
                    filled_quantity=order_data['filledQuantity'],
                    avg_fill_price=order_data.get('avgPrice')
                )
                orders.append(order)
            return orders
        else:
            raise Exception(f"Orders fetch failed: {response.text}")
    
    def _map_order_status(self, status: str) -> OrderStatus:
        """Map IIFL order status to internal status"""
        status_map = {
            'NEW': OrderStatus.SUBMITTED,
            'FILLED': OrderStatus.FILLED,
            'CANCELLED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED
        }
        return status_map.get(status, OrderStatus.PENDING)


class IndianBrokerFactory:
    """Factory class for Indian brokers"""
    
    @staticmethod
    def create_broker(broker_name: str, config: Dict[str, Any]):
        """Create broker instance"""
        if broker_name.lower() == 'zerodha':
            return ZerodhaBroker(
                api_key=config['api_key'],
                api_secret=config['api_secret'],
                access_token=config.get('access_token')
            )
        elif broker_name.lower() == 'iifl':
            return IIFLBroker(
                api_key=config['api_key'],
                api_secret=config['api_secret'],
                client_id=config['client_id']
            )
        else:
            raise ValueError(f"Unsupported broker: {broker_name}")
