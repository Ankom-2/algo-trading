#!/usr/bin/env python3
"""
NSE/BSE Trading Platform Launcher
Clean, comprehensive trading application for Indian stock markets
"""

import os
import sys
import logging

def main():
    """Launch the NSE/BSE Trading Application"""
    
    print("\n" + "="*60)
    print("🚀 NSE/BSE Trading Platform")
    print("="*60)
    print("✨ Features:")
    print("  📈 Real-time NSE/BSE stock data")
    print("  🔍 Smart stock search & discovery") 
    print("  📊 Advanced technical analysis")
    print("  🔥 Trending stocks identification")
    print("  💹 Professional trading interface")
    print("  🔐 API credentials management")
    print("  📱 Responsive mobile-friendly UI")
    print("  💰 Paper trading simulation")
    print("="*60)
    
    try:
        # Import and run the main application
        from nse_bse_trading_app import NSEBSETradingApp
        
        print("\n🔄 Initializing trading platform...")
        app = NSEBSETradingApp()
        
        print("✅ Platform ready!")
        print("🌐 Access your trading dashboard at: http://localhost:8050")
        print("⚠️  Press Ctrl+C to stop the server\n")
        
        # Run the application
        app.run(debug=False, port=8050)
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("💡 Make sure all required packages are installed:")
        print("   pip install dash plotly pandas yfinance requests")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
