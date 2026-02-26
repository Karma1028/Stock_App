from data_manager import StockDataManager
import pandas as pd

dm = StockDataManager()
symbol = "RELIANCE.NS"
bs = dm.get_balance_sheet(symbol)

for idx in bs.index:
    if "Asset" in str(idx) or "Liabilit" in str(idx) or "Inventory" in str(idx):
        print(idx)

print("\n--- Sample Data (First Column) ---")
if not bs.empty:
    print(bs.iloc[:, 0])
