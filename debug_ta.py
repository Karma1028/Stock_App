
import pandas as pd
import yfinance as yf
import ta
import traceback

def test_ta_integration():
    symbol = "RELIANCE.NS"
    print("Downloading data...")
    df = yf.download(symbol, period="1y", interval="1d", progress=False)
    print(f"Index type: {type(df.index)}")
    print(f"Columns types: {type(df.columns)}")
    print(df.columns)

    # Flatten logic from manager.py
    if isinstance(df.columns, pd.MultiIndex):
        print("Flattening MultiIndex...")
        if 'Close' in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        elif df.columns.nlevels > 1 and 'Close' in df.columns.get_level_values(1):
            df.columns = df.columns.get_level_values(1)
    
    print("Columns after flatten:", df.columns)
    print("Head:", df.head(2))

    print("Running TA...")
    try:
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = ta.add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )
        print("TA Success!")
        print("New Columns:", df.columns[-5:])
    except Exception as e:
        print("TA Failed!")
        traceback.print_exc()

if __name__ == "__main__":
    test_ta_integration()
