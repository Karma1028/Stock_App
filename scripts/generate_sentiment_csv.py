import pandas as pd
import sys
import os

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data.manager import StockDataManager
from modules.data.scrapers.news_scraper import NewsScraper

def main():
    dm = StockDataManager()
    ns = NewsScraper()
    tickers = dm.get_stock_list()
    
    all_news = []
    
    print(f"Fetching news for {len(tickers)} stocks (limiting to top 15 for speed)...")
    
    # Limiting to first 15 tickers to avoid timeout during agent execution.
    # In production, remove the slice.
    target_tickers = tickers[:15]
    
    for ticker in target_tickers:
        try:
            print(f"Processing {ticker}...")
            # Fetch last 14 days of news
            df = ns.fetch_news_history(ticker, days=14)
            if not df.empty:
                df = ns.analyze_sentiment(df)
                df['Stock Index Name'] = ticker
                all_news.append(df)
        except Exception as e:
            print(f"Failed {ticker}: {e}")
            
    if all_news:
        final_df = pd.concat(all_news)
        # Ensure 'Stock Index Name' is the first column
        cols = ['Stock Index Name', 'date', 'title', 'sentiment', 'source', 'link']
        final_df = final_df[cols]
        
        output_path = os.path.join("data", "consolidated_sentiments.csv")
        os.makedirs("data", exist_ok=True)
        final_df.to_csv(output_path, index=False)
        print(f"Scraping complete. Saved {len(final_df)} rows to {output_path}")
    else:
        print("No news found.")

if __name__ == "__main__":
    main()
