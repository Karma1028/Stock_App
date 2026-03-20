import yfinance as yf
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

# Configuration
warnings.filterwarnings('ignore')
RAW_DIR = r'd:\stock\stock project dnyanesh\stock_app\data\raw'
PROCESSED_DIR = r'd:\stock\stock project dnyanesh\stock_app\data\processed'
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Environment ready")
print(f"yfinance version: {yf.__version__}")

def fetch_stock_data(ticker, start='2019-01-01', end=None, name=None):
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    display_name = name or ticker
    print(f"\n{'='*60}")
    print(f"  Fetching: {display_name} ({ticker})")
    print(f"  Period: {start} -> {end}")
    print(f"{'='*60}")
    
    # LEVEL 1: Try primary ticker
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            print(f"  Level 1 SUCCESS | Rows: {len(df)} | Date Range: {df.index[0].date()} to {df.index[-1].date()}")
            return df
        else:
            print(f"  Level 1 returned empty DataFrame")
    except Exception as e:
        print(f"  Level 1 FAILED: {e}")
    
    # LEVEL 3: Try max period
    try:
        df = yf.download(ticker, period='max', progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[df.index >= start]
            print(f"  Level 3 SUCCESS (max period) | Rows: {len(df)}")
            return df
    except Exception as e:
        print(f"  Level 3 FAILED: {e}")
    
    print(f"  ALL LEVELS FAILED for {display_name}")
    return pd.DataFrame()

# Define Universe
START_DATE = '2020-01-01'

# Define Universe
START_DATE = '2020-01-01'

# For Tata Motors, we use a "Stitching Strategy"
# 1. Try TTM (ADR) as first choice for history
# 2. If TTM fails, use ^CNXAUTO (Nifty Auto Index) as proxy (high correlation)
# 3. Stitch with TMCV.NS (Current)
HISTORY_CANDIDATES = ['TTM', '^CNXAUTO']
stock_data = {}

print("Fetching Tata Motors Data (Synthetic Stitching)...")

# 1. Fetch History Proxy
history_df = pd.DataFrame()
history_source = ""
for ticker in HISTORY_CANDIDATES:
    print(f"Trying history source: {ticker}")
    df = fetch_stock_data(ticker, start=START_DATE, name=f'History Proxy ({ticker})')
    if not df.empty and len(df) > 100: # Ensure substantial history
        history_df = df
        history_source = ticker
        break

# 2. Fetch Current (TMCV)
tmcv = fetch_stock_data('TMCV.NS', start='2025-10-01', name='Tata Motors CV (Current)')

# 3. Stitch
if not history_df.empty and not tmcv.empty:
    print(f"\nCreating Synthetic History using {history_source}...")
    
    # Ensure timezone naive
    if history_df.index.tz is not None: history_df.index = history_df.index.tz_localize(None)
    if tmcv.index.tz is not None: tmcv.index = tmcv.index.tz_localize(None)
    
    # Find overlap or join point
    join_date = tmcv.index[0]
    
    # Get history before join date
    history = history_df[history_df.index < join_date].copy()
    
    if not history.empty:
        # Calculate Scale Factor
        last_history_price = history['Close'].iloc[-1]
        first_current_price = tmcv['Close'].iloc[0]
        
        scale_factor = first_current_price / last_history_price
        print(f"  Stitching Date: {join_date.date()}")
        print(f"  Proxy Close: {last_history_price:.2f} | TMCV Open: {first_current_price:.2f}")
        print(f"  Scale Factor: {scale_factor:.4f}")
        
        # Apply scaling
        cols_to_scale = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in cols_to_scale:
            if col in history.columns:
                history[col] = history[col] * scale_factor
        
        # Combine
        combined_df = pd.concat([history, tmcv])
        print(f"  ✅ Synthetic Dataset Created: {len(combined_df)} rows")
        stock_data['Tata Motors CV'] = combined_df
    else:
        print("  ❌ No history found before split. Using TMCV only.")
        stock_data['Tata Motors CV'] = tmcv
else:
    print("  ❌ Failed to stitch. Using components.")
    stock_data['Tata Motors CV'] = tmcv if not tmcv.empty else history_df

# Fetch others normally
OTHER_TICKERS = {
    'Tata Motors PV': 'TMPV.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Mahindra & Mahindra': 'M&M.NS',
    'Bajaj Auto': 'BAJAJ-AUTO.NS',
    'NIFTY 50': '^NSEI',
    'NIFTY Auto': '^CNXAUTO'
}

for name, ticker in OTHER_TICKERS.items():
    df = fetch_stock_data(ticker, start=START_DATE, name=name)
    stock_data[name] = df

# Save Data
for name, df in stock_data.items():
    if not df.empty:
        filename = name.lower().replace(' ', '_').replace('&', 'and') + '_prices.csv'
        filepath = os.path.join(RAW_DIR, filename)
        df.to_csv(filepath)
        print(f"Saved {filename}")

# Create Merged Close Prices
close_prices = pd.DataFrame()
for name, df in stock_data.items():
    if not df.empty and 'Close' in df.columns:
        close_prices[name] = df['Close']
close_prices.to_csv(os.path.join(PROCESSED_DIR, 'all_close_prices.csv'))

# Save Tata Motors Clean (Primary)
tata = stock_data.get('Tata Motors CV', pd.DataFrame())
if not tata.empty:
    tata.to_csv(os.path.join(PROCESSED_DIR, 'tata_motors_clean.csv'))
    print(f"\nSaved tata_motors_clean.csv with {len(tata)} rows")
else:
    print("\nERROR: Tata Motors CV data not found!")
