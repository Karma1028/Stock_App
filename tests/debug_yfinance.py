import yfinance as yf
import pandas as pd

def debug_yfinance():
    symbol = "RELIANCE.NS"
    print(f"Fetching data for {symbol}...")
    
    # Test 1: Historical Data
    try:
        df = yf.download(symbol, period="1mo", interval="1d", progress=False)
        print("\n--- Historical Data Shape ---")
        print(df.shape)
        print("\n--- Columns ---")
        print(df.columns)
        print("\n--- Head ---")
        print(df.head())
        
        # Check if columns are MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            print("\n[WARN] Columns are MultiIndex!")
            # Simulate fix
            df.columns = df.columns.get_level_values(0)
            print("--- Flattened Columns ---")
            print(df.columns)
            
    except Exception as e:
        print(f"\n[ERROR] Failed to fetch historical data: {e}")

    # Test 2: Ticker Info
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        print("\n--- Ticker Info Keys ---")
        print(list(info.keys())[:10]) # Print first 10 keys
        print(f"Current Price: {info.get('currentPrice')}")
    except Exception as e:
        print(f"\n[ERROR] Failed to fetch ticker info: {e}")

if __name__ == "__main__":
    debug_yfinance()
