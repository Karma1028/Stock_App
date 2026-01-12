import requests
import feedparser
import urllib.parse
import pandas as pd
import time
import random
import io
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
NEWS_HISTORY_DIR = DATA_DIR / "news_history"
NEWS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

class HistoricalNewsScraper:
    def __init__(self):
        self.base_url = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'
        ]
        self.session = self._create_session()

    def _create_session(self):
        """Creates a requests session with retry logic."""
        session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def get_random_header(self):
        return {'User-Agent': random.choice(self.user_agents)}

    def get_nifty500_tickers(self):
        """Fetches Nifty 500 tickers from Wikipedia, returning list of {'symbol': ..., 'name': ...}."""
        print("Fetching Nifty 500 tickers from Wikipedia...")
        url = 'https://en.wikipedia.org/wiki/NIFTY_500'
        
        try:
            response = requests.get(url, headers=self.get_random_header())
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table', {'class': 'wikitable'})
            
            companies = []
            
            for table in tables:
                try:
                    df = pd.read_html(io.StringIO(str(table)), header=0)[0]
                    
                    # Identify Symbol Column
                    ticker_col = None
                    possible_ticker_cols = ['Symbol', 'Ticker Symbol', 'NSE Symbol', 'BSE Scrip Code', 'Symbol (NSE)', 'Symbol (BSE)']
                    for col in possible_ticker_cols:
                        if col in df.columns:
                            ticker_col = col
                            break
                    
                    # Identify Name Column
                    name_col = None
                    possible_name_cols = ['Company Name', 'Company', 'Name']
                    for col in possible_name_cols:
                        if col in df.columns:
                            name_col = col
                            break
                    
                    if ticker_col:
                        # Iterate rows
                        for idx, row in df.iterrows():
                            ticker = str(row[ticker_col]).strip()
                            name = str(row[name_col]).strip() if name_col else ticker
                            
                            if ticker and ticker.upper() != 'NAN' and pd.isna(ticker) == False and ticker != '-':
                                if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
                                    ticker += '.NS'
                                
                                companies.append({'symbol': ticker, 'name': name})
                        
                        break # Found the main table
                except Exception as e:
                    print(f"Error parsing table: {e}")
                    continue
                    
            # Remove duplicates based on symbol
            seen = set()
            unique_companies = []
            for c in companies:
                if c['symbol'] not in seen:
                    unique_companies.append(c)
                    seen.add(c['symbol'])
                    
            print(f"Found {len(unique_companies)} unique Nifty 500 companies.")
            return unique_companies
        except Exception as e:
            print(f"Error fetching Nifty 500 tickers: {e}")
            return []

    def fetch_news_chunk(self, ticker, queries, start_date, end_date):
        """Fetches news trying multiple queries until results found."""
        
        date_query = f"after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}"
        
        for q_base in queries:
            query = f"{q_base} {date_query}"
            encoded_query = urllib.parse.quote(query)
            url = self.base_url.format(query=encoded_query)

            try:
                resp = self.session.get(url, headers=self.get_random_header(), timeout=10)
                if resp.status_code != 200:
                    continue
                    
                feed = feedparser.parse(resp.content)
                
                if not feed.entries:
                    continue # Try next query
                
                news_items = []
                for entry in feed.entries:
                     try:
                        pub_date_struct = entry.published_parsed
                        if pub_date_struct:
                            pub_date = datetime(*pub_date_struct[:6])
                        else:
                            pub_date = start_date 

                        news_items.append({
                            'Ticker': ticker,
                            'Date': pub_date,
                            'Title': entry.title,
                            'Source': entry.source.title if 'source' in entry else 'Google News',
                            'Link': entry.link
                        })
                     except Exception:
                         continue
                
                # If we got results, valid results, return them.
                if news_items:
                    # print(f"  > Found {len(news_items)} items with query: '{q_base}'")
                    return news_items
                    
            except Exception as e:
                print(f"Error fetching: {e}")
                continue
        
        return [] # No results with any query

    def fetch_news_history(self, company_data, years=5):
        """Fetches news history for the last N years."""
        ticker = company_data['symbol']
        name = company_data['name']
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
        
        # Prioritize queries: 
        # 1. "Ticker stock news" (most precise)
        # 2. "Company Name stock news" (good fallback)
        # 3. "Ticker" (broad)
        queries = [
            f"{clean_ticker} stock news",
            f"{name} stock news",
            f"{clean_ticker}"
        ]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        
        all_news = []
        
        current_start = start_date
        while current_start < end_date:
            current_end = current_start + timedelta(days=30)
            if current_end > end_date:
                current_end = end_date
            
            # Rate limiting
            time.sleep(random.uniform(1.5, 3.5))
            
            print(f"Fetching {ticker} ({current_start.date()} to {current_end.date()})...")
            news_chunk = self.fetch_news_chunk(ticker, queries, current_start, current_end)
            if news_chunk:
                print(f"  > Found {len(news_chunk)} items.")
            
            all_news.extend(news_chunk)
            
            current_start = current_end
            
        return pd.DataFrame(all_news)

    def scrape_ticker_and_save(self, company_data, years=5):
        """Worker function."""
        ticker = company_data['symbol']
        save_path = NEWS_HISTORY_DIR / f"{ticker}_news.csv"
        
        if save_path.exists():
            return # Skip

        print(f"Starting ID: {ticker}")
        df = self.fetch_news_history(company_data, years=years)
        
        if not df.empty:
            df.to_csv(save_path, index=False)
            print(f"SUCCESS: Saved {len(df)} items for {ticker}")
        else:
            print(f"WARNING: No news found for {ticker} ({company_data['name']})")

    def run_bulk_scrape(self, workers=3):
        companies = self.get_nifty500_tickers()
        if not companies:
            print("No tickers found. Aborting.")
            return

        print(f"Starting bulk scrape for {len(companies)} companies with {workers} workers.")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self.scrape_ticker_and_save, c) for c in companies]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Worker exception: {e}")

if __name__ == "__main__":
    scraper = HistoricalNewsScraper()
    print(f"Saving data to: {NEWS_HISTORY_DIR}")
    scraper.run_bulk_scrape(workers=5)
