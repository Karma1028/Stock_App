from modules.data.scrapers.historical_news_scraper import HistoricalNewsScraper
import pandas as pd
from pathlib import Path

def test_scraper():
    print("Initializing scraper...")
    scraper = HistoricalNewsScraper()
    
    # Test 1: Nifty 500 Tickers
    print("\nTesting Ticker Fetch...")
    tickers = scraper.get_nifty500_tickers()
    if not tickers:
        print("FAILED: Could not fetch tickers.")
        return
    print(f"SUCCESS: Fetched {len(tickers)} tickers.")
    print(f"Sample: {tickers[:5]}")
    
    # Test 2: Scrape single ticker for short duration
    test_ticker = tickers[0] if tickers else 'RELIANCE.NS'
    print(f"\nTesting News Scrape for {test_ticker} (1 month)...")
    
    # Temporarily override fetch_news_history to just do 1 month for testing
    # Or just call fetch_news_chunk manually, but we want to test the flow
    # Let's just call scrape_ticker_and_save but maybe mock the years or modify the method?
    # Actually, let's just use the public method fetch_news_chunk for testing logic, 
    # then try a small full run.
    
    from datetime import datetime, timedelta
    start = datetime.now() - timedelta(days=30)
    end = datetime.now()
    
    news = scraper.fetch_news_chunk(test_ticker, start, end)
    print(f"Fetched {len(news)} items.")
    if news:
        print(f"Sample item: {news[0]}")
    
    # Test 3: CSV Save (Simulated)
    print("\nTesting CSV Save Logic...")
    if news:
        df = pd.DataFrame(news)
        save_path = Path("test_output.csv")
        df.to_csv(save_path, index=False)
        print(f"Saved to {save_path.absolute()}")
        if save_path.exists():
            print("SUCCESS: File created.")
            # clean up
            save_path.unlink()
        else:
            print("FAILED: File not created.")
    else:
        print("WARNING: No news to save, skipping CSV test.")

if __name__ == "__main__":
    test_scraper()
