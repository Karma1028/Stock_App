import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import os

# Set up logging
os.makedirs('Universal_Engine_Workspace/mas_logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataEngineer')

class AdvancedDeltaScraper:
    def __init__(self):
        self.macro_tickers = {
            'India VIX': '^INDIAVIX',
            'Nifty 50': '^NSEI',
            'Brent Crude': 'BZ=F',
            'USD/INR': 'USDINR=X',
            'GBP/INR': 'GBPINR=X'
        }
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        }
    
    def fetch_macro_anchors(self):
        """Fetches macro indicators for the last 5 years."""
        logger.info("Initiating 5-year historical sweep for Macro Anchors...")
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        macro_df = pd.DataFrame()
        
        for name, ticker in self.macro_tickers.items():
            try:
                data = yf.download(ticker, start=start_date, progress=False)
                if not data.empty:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                    closes = data[['Close']].copy()
                    closes.columns = [name]
                    if macro_df.empty: macro_df = closes
                    else: macro_df = macro_df.join(closes, how='outer')
            except Exception as e:
                logger.error(f"Failed {name}: {e}")

        # Specialized Fallback for India 10Y Bond Yield
        bond_data = self.fetch_bond_yield_marketwatch()
        if not bond_data.empty:
            macro_df = macro_df.join(bond_data, how='outer')

        return macro_df.ffill().reset_index()

    def fetch_bond_yield_marketwatch(self):
        try:
            url = "https://www.marketwatch.com/investing/bond/tmbmkint-10y?countrycode=in"
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                soup = BeautifulSoup(res.text, 'html.parser')
                val = soup.find('bg-quote', {'class': 'value'})
                if val:
                    yield_val = float(val.text.replace('%', '').strip())
                    return pd.DataFrame({'10-Yr India Bond Yield': [yield_val]}, index=[pd.Timestamp.now().normalize()])
        except Exception as e:
            logger.error(f"Bond fallback failed: {e}")
        return pd.DataFrame()

    def fetch_all_equity_data(self, equity_csv_path, chunk_size=50):
        """Processes 1900+ stocks in robust chunks of 50 to avoid rate limits."""
        logger.info(f"Executing massive 5-year historical scrape from {equity_csv_path}")
        try:
            equity_df = pd.read_csv(equity_csv_path)
            all_symbols = [str(s).strip() + '.NS' for s in equity_df['SYMBOL'].tolist()]
            
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            
            all_prices = []
            all_fundamentals = []

            # Chunked processing for robustness
            for i in range(0, len(all_symbols), chunk_size):
                chunk = all_symbols[i:i + chunk_size]
                logger.info(f"Processing Chunk {i//chunk_size + 1}/{len(all_symbols)//chunk_size + 1} (Size: {len(chunk)})")
                
                try:
                    raw_prices = yf.download(chunk, start=start_date, threads=True, progress=False, group_by='ticker')
                    if not raw_prices.empty:
                        # Reshape chunked prices
                        if len(chunk) > 1:
                            prices_chunk = raw_prices.stack(level=0, future_stack=True).reset_index()
                            prices_chunk.rename(columns={'level_0': 'Date', 'level_1': 'Ticker'}, inplace=True)
                        else:
                            prices_chunk = raw_prices.reset_index()
                            prices_chunk['Ticker'] = chunk[0]
                        all_prices.append(prices_chunk)

                    # Chunked Fundamentals
                    for sym in chunk:
                        try:
                            t = yf.Ticker(sym)
                            info = t.info
                            all_fundamentals.append({
                                'Ticker': sym,
                                'PE': info.get('trailingPE'),
                                'PB': info.get('priceToBook'),
                                'ROE': info.get('returnOnEquity'),
                                'PEG': info.get('pegRatio'),
                                'DebtToEquity': info.get('debtToEquity')
                            })
                        except: pass
                    
                    # Small delay between chunks to be polite to Yahoo Finance
                    time.sleep(2)
                except Exception as e:
                    logger.error(f"Error in chunk starting at index {i}: {e}")
            
            final_prices = pd.concat(all_prices) if all_prices else pd.DataFrame()
            final_funds = pd.DataFrame(all_fundamentals) if all_fundamentals else pd.DataFrame()
            
            return final_prices, final_funds
            
        except Exception as e:
            logger.error(f"Massive equity pipeline failed: {e}")
            return pd.DataFrame(), pd.DataFrame()

if __name__ == "__main__":
    scraper = AdvancedDeltaScraper()
    
    # 1. Macro Data
    print("Fetching 5-Year Macro Baseline...")
    macros = scraper.fetch_macro_anchors()
    macros.to_csv('data/macro_anchors_5y.csv', index=False)
    
    # 2. Equity Data (Running for all 1900+ stocks)
    print("\nInitiating massive 5-year historical download for all tickers...")
    prices, funds = scraper.fetch_all_equity_data('data/EQUITY.csv', chunk_size=100)
    
    os.makedirs('data/processed', exist_ok=True)
    if not prices.empty:
        prices.to_csv('data/processed/equity_prices_5y.csv', index=False)
        print(f"Price Capture: {len(prices)} rows.")
    
    if not funds.empty:
        funds.to_csv('data/processed/equity_fundamentals_5y.csv', index=False)
        print(f"Fundamentals Capture: {len(funds)} tickers.")

    print("\n[COMPLETE] Robust 5-Year Historical Matrix Locked.")
