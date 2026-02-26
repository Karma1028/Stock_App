import pandas as pd
import ta
import yfinance as yf

print("Fetching data...")
df = yf.download("RELIANCE.NS", period="1y", interval="1d", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

print("Data fetched. Shape:", df.shape)
print("Columns:", df.columns)
if 'Close' not in df.columns:
    print("ERROR: 'Close' column missing!")


print("Adding TA features...")
try:
    df = ta.add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
    )
    print("TA features added successfully.")
    print("New Shape:", df.shape)
except Exception as e:
    print(f"Error adding TA features: {e}")
