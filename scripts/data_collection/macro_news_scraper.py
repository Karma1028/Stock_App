import pandas as pd
import feedparser
import urllib.parse
import time
import random
import os
from datetime import datetime, timedelta
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

DATA_DIR = Path('data')
NEWS_HISTORY_DIR = DATA_DIR / 'news_history'
NEWS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

class MacroNewsScraper:
    def __init__(self):
        self.base_url = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        self.session = self._create_session()

    def _create_session(self):
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

    def fetch_news_chunk(self, queries, start_date, end_date):
        date_query = f"after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}"
        news_items = []
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
                    continue
                
                for entry in feed.entries:
                     try:
                        pub_date_struct = entry.published_parsed
                        pub_date = datetime(*pub_date_struct[:6]) if pub_date_struct else start_date 
                        news_items.append({
                            'Ticker': 'MACRO',
                            'Date': pub_date,
                            'Title': entry.title,
                            'Source': entry.source.title if 'source' in entry else 'Google News',
                            'Link': entry.link,
                            'Category': q_base
                        })
                     except Exception:
                         continue
            except Exception as e:
                print(f"Error fetching macro news: {e}")
        return news_items

    def fetch_macro_history(self, years=5):
        queries = [
            "Reserve Bank of India interest rates OR RBI repo rate",
            "India inflation data OR CPI India",
            "India GDP growth benchmark",
            "BSE Sensex Nifty 50 market trends"
        ]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        all_news = []
        
        current_start = start_date
        while current_start < end_date:
            current_end = current_start + timedelta(days=30)
            if current_end > end_date:
                current_end = end_date
            
            time.sleep(random.uniform(2, 4))
            print(f"Fetching Macro ({current_start.date()} to {current_end.date()})...")
            
            chunk = self.fetch_news_chunk(queries, current_start, current_end)
            if chunk:
                print(f"  > Found {len(chunk)} items.")
            
            all_news.extend(chunk)
            current_start = current_end
            
        return pd.DataFrame(all_news)

if __name__ == "__main__":
    scraper = MacroNewsScraper()
    print("Scraping MACRO news history for the last 5 years...")
    df = scraper.fetch_macro_history(years=5)
    
    save_path = NEWS_HISTORY_DIR / "MACRO_news.csv"
    if not df.empty:
        # Sort and drop duplicates based on title
        df = df.drop_duplicates(subset=['Title'])
        df.to_csv(save_path, index=False)
        print(f"SUCCESS: Saved {len(df)} macro news items to {save_path}")
    else:
        print("WARNING: No macro news found.")
