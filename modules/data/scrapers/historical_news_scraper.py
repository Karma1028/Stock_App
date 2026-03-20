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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console

console = Console()

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

    def get_all_equity_tickers(self):
        """Fetches all equity tickers from data/EQUITY.csv."""
        equity_csv = DATA_DIR / 'EQUITY.csv'
        print(f"Reading tickers from {equity_csv}...")
        try:
            df = pd.read_csv(equity_csv)
            companies = []
            for _, row in df.iterrows():
                ticker = str(row['SYMBOL']).strip() + '.NS'
                name = str(row['NAME OF COMPANY']).strip()
                companies.append({'symbol': ticker, 'name': name})
            print(f"Loaded {len(companies)} companies from EQUITY.csv.")
            return companies
        except Exception as e:
            print(f"Error reading EQUITY.csv: {e}")
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

    def fetch_news_history(self, company_data, years=5, progress_ctx=None):
        """Fetches news history for the last N years."""
        ticker = company_data['symbol']
        name = company_data['name']
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
        
        def log(msg):
            if progress_ctx:
                pass # Skip verbose internal logs during massive bulk run
            else:
                print(msg)
        
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
            
            log(f"Fetching {ticker} ({current_start.date()} to {current_end.date()})...")
            news_chunk = self.fetch_news_chunk(ticker, queries, current_start, current_end)
            if news_chunk:
                log(f"  > Found {len(news_chunk)} items.")
            
            all_news.extend(news_chunk)
            
            current_start = current_end
            
        return pd.DataFrame(all_news)

    def scrape_ticker_and_save(self, company_data, progress_ctx=None, years=5):
        """Worker function."""
        ticker = company_data['symbol']
        save_path = NEWS_HISTORY_DIR / f"{ticker}_news.csv"
        five_years_ago = datetime.now() - timedelta(days=years*365)
        
        def log(msg, style=""):
            if progress_ctx:
                progress_ctx.console.print(f"[{style}]{msg}[/{style}]" if style else msg)
            else:
                print(msg)

        if save_path.exists():
            try:
                df_existing = pd.read_csv(save_path)
                if not df_existing.empty and 'Date' in df_existing.columns:
                    df_existing['Date'] = pd.to_datetime(df_existing['Date'], errors='coerce', utc=True).dt.tz_localize(None)
                    valid = df_existing.dropna(subset=['Date'])
                    if not valid.empty:
                        min_date = valid['Date'].min()
                        if min_date <= pd.Timestamp(five_years_ago) + pd.Timedelta(days=60):
                            log(f"Skipping {ticker}: 5-year history already present (oldest: {min_date.date()})", "dim")
                            return # Skip, already has ~5 years of data
                        else:
                            log(f"Re-scraping {ticker}: history incomplete (oldest is {min_date.date()})", "yellow")
            except Exception:
                pass

        log(f"Starting ID: {ticker}", "cyan")
        df = self.fetch_news_history(company_data, years=years, progress_ctx=progress_ctx)
        
        if not df.empty:
            df.to_csv(save_path, index=False)
            log(f"SUCCESS: Saved {len(df)} items for {ticker}", "green")
        else:
            log(f"WARNING: No news found for {ticker} ({company_data['name']})", "yellow")

    def run_bulk_scrape(self, workers=3):
        companies = self.get_all_equity_tickers()
        if not companies:
            console.print("[red]No tickers found. Aborting.[/red]")
            return

        console.print(f"[bold green]Starting bulk scrape for {len(companies)} companies with {workers} workers.[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            main_task = progress.add_task("[cyan]Sourcing RSS URLs for all equities...", total=len(companies))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(self.scrape_ticker_and_save, c, progress): c for c in companies}
                
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        progress.console.print(f"[red]Worker exception: {e}[/red]")
                    finally:
                        progress.advance(main_task)

if __name__ == "__main__":
    scraper = HistoricalNewsScraper()
    print(f"Saving data to: {NEWS_HISTORY_DIR}")
    scraper.run_bulk_scrape(workers=5)
