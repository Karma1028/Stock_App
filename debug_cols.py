
import yfinance as yf
import pandas as pd

def debug_columns():
    symbol = "RELIANCE.NS"
    print("Downloading...")
    df = yf.download(symbol, period="1d", progress=False)
    print("Download Complete.")
    print(f"Is MultiIndex? {isinstance(df.columns, pd.MultiIndex)}")
    if isinstance(df.columns, pd.MultiIndex):
        print(f"Levels: {df.columns.levels}")
        print(f"Values: {df.columns.values}")
    else:
        print(f"Columns: {df.columns}")

if __name__ == "__main__":
    debug_columns()
