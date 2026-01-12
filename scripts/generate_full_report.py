import pandas as pd
import sys
import os
import time

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data.manager import StockDataManager

def fetch_and_stack(dm, tickers, func_name, sheet_name):
    print(f"Fetching {sheet_name}...")
    all_dfs = []
    
    # Process in chunks or one by one
    for ticker in tickers:
        try:
            # Dynamically call the method (get_balance_sheet, etc.)
            func = getattr(dm, func_name)
            df = func(ticker)
            
            if not df.empty:
                # Transpose so dates are rows or columns? 
                # yfinance returns dates as columns usually for financials.
                # Let's text transpose to have Dates as Index, Metrics as Columns
                df = df.T
                df['Ticker'] = ticker
                df = df.reset_index().rename(columns={'index': 'Date'})
                all_dfs.append(df)
            print(f".", end="", flush=True)
        except Exception as e:
            print(f"x", end="", flush=True)
    print("\n")
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()

def main():
    dm = StockDataManager()
    tickers = dm.get_stock_list()
    
    # Limit for testing/speed if needed, but user wants 'all'. 
    # There are ~50 tickers. 150 requests. Should be fine in < 2 mins.
    # tickers = tickers[:5] 
    
    output_file = os.path.join("data", "Stock_Data_Consolidated.xlsx")
    
    print(f"Starting consolidation for {len(tickers)} stocks...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # 1. Sentiments (if CSV exists)
        sentiments_path = os.path.join("data", "consolidated_sentiments.csv")
        if os.path.exists(sentiments_path):
            print("Adding Sentiments...")
            sent_df = pd.read_csv(sentiments_path)
            sent_df.to_excel(writer, sheet_name='Sentiments', index=False)
        
        # 2. Balance Sheet
        bs_df = fetch_and_stack(dm, tickers, "get_balance_sheet", "Balance Sheet")
        if not bs_df.empty:
            bs_df.to_excel(writer, sheet_name='Balance Sheet', index=False)
            
        # 3. Income Statement
        is_df = fetch_and_stack(dm, tickers, "get_income_statement", "Income Statement")
        if not is_df.empty:
            is_df.to_excel(writer, sheet_name='Income Statement', index=False)
            
        # 4. Cash Flow
        cf_df = fetch_and_stack(dm, tickers, "get_cash_flow", "Cash Flow")
        if not cf_df.empty:
            cf_df.to_excel(writer, sheet_name='Cash Flow', index=False)
            
        # 5. Live Metrics Summary
        print("Fetching Live Metrics...")
        metrics = []
        for ticker in tickers:
            try:
                info = dm.get_live_data(ticker)
                metrics.append(info)
                print(f".", end="", flush=True)
            except:
                print(f"x", end="", flush=True)
        print("\n")
        
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            # Reorder cols to put Symbol first
            cols = ['symbol'] + [c for c in metrics_df.columns if c != 'symbol']
            metrics_df = metrics_df[cols]
            metrics_df.to_excel(writer, sheet_name='Live Metrics', index=False)

    print(f"Consolidation Complete! Saved to {output_file}")

if __name__ == "__main__":
    main()
