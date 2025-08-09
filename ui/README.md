# 🚀 Algorithmic Trading System - User Interfaces

This directory contains multiple user interfaces for interacting with the algorithmic trading system. Choose the interface that best fits your needs and workflow.

## 🌐 Available Interfaces

### 1. Web Dashboard (`dashboard.py`)
**Best for**: Comprehensive analysis, real-time monitoring, and visual portfolio management

**Features:**
- Interactive web-based dashboard using Dash/Plotly
- Real-time portfolio performance charts
- Strategy backtesting interface
- Portfolio optimization tools
- Risk metrics visualization
- Position management table
- Mobile-responsive design

**Launch:** 
```bash
python ui/dashboard.py
# or
python ui/launcher.py  # Option 1
```

**Access:** Open http://localhost:8050 in your browser

### 2. Command Line Interface (`cli.py`)
**Best for**: Quick testing, scripting, and terminal-based workflows

**Features:**
- Interactive menu-driven interface
- Strategy backtesting
- Portfolio optimization
- Market data viewing
- Paper trading demo
- Configuration management
- Non-interactive mode for scripting

**Launch:**
```bash
python ui/cli.py
# or
python ui/launcher.py  # Option 2
```

### 3. Jupyter Notebook (`trading_analysis.ipynb`)
**Best for**: Research, analysis, and experimentation

**Features:**
- Interactive data analysis environment
- Pre-built analysis templates
- Visualization capabilities
- Step-by-step strategy development
- Research notebook format

**Launch:**
```bash
python ui/launcher.py  # Option 3
# or directly
jupyter notebook ui/trading_analysis.ipynb
```

### 4. UI Launcher (`launcher.py`)
**Best for**: Easy interface selection and configuration

**Features:**
- Single entry point for all interfaces
- Configuration management
- Interface comparison
- Quick setup and launch

**Launch:**
```bash
python ui/launcher.py
```

## 🛠 Prerequisites

### Required Packages
All interfaces require the packages listed in `requirements.txt`. Additional UI-specific requirements:

```bash
# For web dashboard
pip install dash plotly

# For Jupyter notebook
pip install jupyter matplotlib seaborn

# Core packages (already in requirements.txt)
pip install pandas numpy scipy scikit-learn
```

### Installation
1. Install all requirements:
```bash
pip install -r requirements.txt
```

2. Verify installation by running the launcher:
```bash
python ui/launcher.py
```

## 📱 Interface Comparison

| Feature | Web Dashboard | CLI | Jupyter | Launcher |
|---------|---------------|-----|---------|----------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Visualization** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Real-time Updates** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐ |
| **Automation** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Research** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Quick Testing** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🚦 Quick Start Guide

### For First-Time Users
1. **Start with the Launcher:**
```bash
python ui/launcher.py
```

2. **Try the Web Dashboard:**
   - Select option 1 from launcher
   - Open http://localhost:8050
   - Click "Run Backtest" to see sample results

3. **Explore the CLI:**
   - Select option 2 from launcher
   - Choose "Run Strategy Backtest"
   - Follow the interactive prompts

### For Developers
1. **Use Jupyter for Research:**
```bash
python ui/launcher.py
# Select option 3
```

2. **Integrate with CLI for Automation:**
```bash
python ui/cli.py --strategy momentum --symbols AAPL GOOGL --days 180
```

## 🎯 Common Use Cases

### Portfolio Manager
```bash
# Launch web dashboard for monitoring
python ui/dashboard.py
```
- Monitor real-time performance
- Rebalance portfolios
- Analyze risk metrics

### Quantitative Researcher  
```bash
# Launch Jupyter for analysis
python ui/launcher.py  # Option 3
```
- Develop new strategies
- Backtest hypotheses
- Create custom visualizations

### Algorithmic Trader
```bash
# Use CLI for quick operations
python ui/cli.py
```
- Run quick backtests
- Check portfolio status
- Execute paper trades

### Strategy Developer
- **Research phase:** Jupyter notebook
- **Testing phase:** CLI for quick iterations  
- **Production phase:** Web dashboard for monitoring

## 📊 Sample Workflows

### 1. Complete Strategy Analysis
```bash
# Step 1: Research in Jupyter
python ui/launcher.py  # Select Jupyter

# Step 2: Quick testing in CLI
python ui/cli.py  # Run backtests

# Step 3: Production monitoring in Dashboard
python ui/dashboard.py  # Monitor live performance
```

### 2. Daily Portfolio Management
```bash
# Morning: Check positions and performance
python ui/dashboard.py

# Midday: Quick CLI check
python ui/cli.py  # Option 4: View positions

# Evening: Analysis in Jupyter
jupyter notebook ui/trading_analysis.ipynb
```

## 🔧 Customization

### Web Dashboard
- Modify `COLORS` dictionary in `dashboard.py` for custom themes
- Add new charts by creating callback functions
- Customize layout in the main layout section

### CLI Interface
- Add new menu options in `AlgoTradingCLI.run()`
- Modify display formats in `display_*_results()` methods
- Add command-line arguments in `main()`

### Jupyter Notebook
- Edit `trading_analysis.ipynb` directly
- Add new cells for custom analysis
- Create new notebooks for specific research

## 🐛 Troubleshooting

### Common Issues

1. **Dashboard won't load:**
```bash
# Check if port 8050 is available
netstat -an | find "8050"

# Try different port
python ui/dashboard.py --port 8051
```

2. **Import errors:**
```bash
# Make sure you're in the right directory
cd algo-trading
python ui/launcher.py
```

3. **Missing packages:**
```bash
# Install missing packages
pip install -r requirements.txt
pip install dash plotly jupyter
```

4. **Data loading errors:**
- Check your API keys in `config.py`
- Verify internet connection
- Try with sample data first

### Getting Help
- Check logs in the `logs/` directory
- Run with debug mode: `python ui/dashboard.py --debug`
- Review configuration in `config.py`

## 📈 Performance Tips

### Web Dashboard
- Use data caching for better performance
- Limit real-time updates for large datasets
- Use aggregated data for historical charts

### CLI
- Use command-line arguments for batch operations
- Cache frequently accessed data
- Use async operations for data loading

### Jupyter
- Use `%matplotlib inline` for inline plots
- Cache expensive calculations
- Use profiling for performance optimization

## 🔒 Security Notes

- Web dashboard runs on localhost by default
- No sensitive data is exposed in logs
- API keys are stored in configuration files
- Use environment variables for production deployments

---

## 🎉 Happy Trading!

Choose your interface and start exploring the world of algorithmic trading. Each interface offers unique advantages - feel free to switch between them based on your current needs.

For questions or contributions, please refer to the main README.md file.
