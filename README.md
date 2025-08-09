algo-trading/
│
├── data/ # Raw and processed market data
│ ├── raw/
│ └── processed/
│
├── strategies/ # Strategy modules
│ ├── base.py
│ ├── momentum.py
│ ├── mean_reversion.py
│ └── adaptive.py
│
├── risk_management/ # Risk management modules
│ ├── stop_loss.py
│ ├── position_sizing.py
│ └── drawdown_control.py
│
├── backtest/ # Backtesting framework
│ ├── engine.py
│ └── results.py
│
├── execution/ # Order execution logic
│ ├── broker.py
│ └── paper_trader.py
│
├── utils/ # Utility functions
│ ├── data_loader.py
│ ├── logger.py
│ └── optimizer.py
│
├── config.py # Configuration file
├── main.py # Entry point
└── requirements.txt # Project dependencies
