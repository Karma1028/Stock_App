
import sys
import os
sys.path.append(os.getcwd())

from modules.data.manager import StockDataManager
from modules.ml.engine import MLEngine
from modules.ml.ensemble import MLEnsemble
from config import Config

def test_system():
    print("Testing StockDataManager...")
    dm = StockDataManager()
    # Test with a known symbol
    symbol = "RELIANCE.NS"
    
    print("\n1. Testing Live Data...")
    try:
        live = dm.get_live_data(symbol)
        print(f"Live Data Data Keys: {list(live.keys())}")
    except Exception as e:
        print(f"FAIL: Live Data error: {e}")
        import traceback
        traceback.print_exc()

    print("\n2. Testing Historical Data...")
    try:
        hist = dm.get_historical_data(symbol, period="1y")
        print(f"Historical Shape: {hist.shape}")
        if not hist.empty:
            print(f"Columns: {hist.columns.tolist()}")
    except Exception as e:
        print(f"FAIL: Historical Data error: {e}")
        import traceback
        traceback.print_exc()

    print("\n3. Testing MLEngine...")
    try:
        ml = MLEngine()
        if not hist.empty:
            print("Predicting...")
            preds = ml.predict(hist)
            print(f"Prediction: {preds.tail() if preds is not None else 'None'}")
    except Exception as e:
        print(f"FAIL: MLEngine error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()
