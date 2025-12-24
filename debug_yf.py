
import yfinance as yf
import pandas as pd
import traceback

def test_yf():
    print(f"YFinance Version: {yf.__version__}")
    symbol = "RELIANCE.NS"
    print(f"Testing {symbol}...")
    
    try:
        print("1. Testing yf.download...")
        df = yf.download(symbol, period="1d", progress=False)
        print(f"Download shape: {df.shape}")
        print(f"Download columns: {df.columns}")
        if not df.empty:
            print(df.head())
    except Exception:
        traceback.print_exc()

    try:
        print("\n2. Testing Ticker.info...")
        t = yf.Ticker(symbol)
        info = t.info
        print(f"Info keys: {list(info.keys())[:5]}")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_yf()
