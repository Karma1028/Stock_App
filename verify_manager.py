
from modules.data.manager import StockDataManager
import pandas as pd

def verify_fix():
    print("Initializing Manager...")
    dm = StockDataManager()
    symbol = "RELIANCE.NS"
    print(f"Fetching data for {symbol}...")
    df = dm.get_historical_data(symbol, period="1y")
    
    if df.empty:
        print("FAIL: DataFrame is empty.")
    else:
        print(f"SUCCESS: DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:5]}...")

if __name__ == "__main__":
    verify_fix()
