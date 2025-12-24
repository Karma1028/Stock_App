
import yfinance as yf
import pandas as pd

def test_history():
    symbol = "RELIANCE.NS"
    t = yf.Ticker(symbol)
    print("Fetching history...")
    hist = t.history(period="1d")
    print(f"Shape: {hist.shape}")
    print(f"Columns: {hist.columns}")
    print(f"Index: {hist.index}")
    
    if not hist.empty:
        try:
            price = hist["Close"].iloc[-1]
            print(f"Price: {price}, Type: {type(price)}")
        except Exception as e:
            print(f"Accessing ['Close'] failed: {e}")
            
        # Check if direct access works if columns are weird
        if isinstance(hist.columns, pd.MultiIndex):
            print("Detected MultiIndex in history!")

if __name__ == "__main__":
    test_history()
