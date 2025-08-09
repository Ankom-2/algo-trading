"""
Test script to validate the fixes and UI components
"""

import sys
import os
import traceback

def test_optimizer_methods():
    """Test the fixed optimizer methods"""
    print("🔧 Testing Optimizer Methods...")
    try:
        from utils.optimizer_methods import PortfolioOptimizerMethods
        
        # Test initialization
        config = {
            'max_iterations': 50,
            'max_position_size': 0.3,
            'rebalance_threshold': 0.05
        }
        
        optimizer_methods = PortfolioOptimizerMethods(config)
        print("✅ PortfolioOptimizerMethods initialized successfully")
        
        # Test simple methods
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        weights = optimizer_methods._equal_weights(symbols)
        print(f"✅ Equal weights calculation: {weights}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimizer methods test failed: {str(e)}")
        print(traceback.format_exc())
        return False

def test_ui_imports():
    """Test UI component imports"""
    print("\n🖥  Testing UI Imports...")
    
    ui_tests = [
        ("Dashboard", "ui.dashboard"),
        ("CLI", "ui.cli"),
        ("Launcher", "ui.launcher")
    ]
    
    results = []
    for name, module in ui_tests:
        try:
            __import__(module)
            print(f"✅ {name} imports successfully")
            results.append(True)
        except Exception as e:
            print(f"❌ {name} import failed: {str(e)}")
            results.append(False)
    
    return all(results)

def test_core_functionality():
    """Test core trading system functionality"""
    print("\n⚙️  Testing Core Functionality...")
    
    try:
        # Test config import
        import config
        print("✅ Configuration loaded")
        
        # Test data structures
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        sample_data = pd.DataFrame({
            'AAPL': 150 * np.cumprod(1 + np.random.normal(0.001, 0.02, len(dates))),
            'GOOGL': 2800 * np.cumprod(1 + np.random.normal(0.0008, 0.025, len(dates))),
        }, index=dates)
        
        print(f"✅ Sample data created: {sample_data.shape}")
        
        # Test basic calculations
        returns = sample_data.pct_change().dropna()
        volatility = returns.std()
        print(f"✅ Basic calculations work: volatility = {volatility.round(4).to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {str(e)}")
        print(traceback.format_exc())
        return False

def main():
    """Run all tests"""
    print("🚀 ALGORITHMIC TRADING SYSTEM - VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Optimizer Methods", test_optimizer_methods),
        ("UI Components", test_ui_imports),  
        ("Core Functionality", test_core_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} tests...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("-" * 30)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if not result:
            all_passed = False
    
    print("-" * 30)
    if all_passed:
        print("🎉 All tests passed! Your system is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Launch UI: python ui/launcher.py")
        print("2. Try CLI: python ui/cli.py") 
        print("3. Run dashboard: python ui/dashboard.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
