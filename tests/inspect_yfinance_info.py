import yfinance as yf
import json

def inspect_info():
    symbol = "RELIANCE.NS"
    ticker = yf.Ticker(symbol)
    info = ticker.info
    
    print("--- Available Info Keys ---")
    for key, value in info.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    inspect_info()
