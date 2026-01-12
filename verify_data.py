from modules.data.scrapers.historical_news_scraper import HistoricalNewsScraper
import pandas as pd
import sys

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

scraper = HistoricalNewsScraper()

# Mock company data
company = {'symbol': 'RELIANCE.NS', 'name': 'Reliance Industries'}

print(f"Verifying data for {company['name']}...")
df = scraper.fetch_news_history(company, years=5)

print(f"Scraping complete.")
print(f"Total Rows: {len(df)}")
if not df.empty:
    print("Sample Data:")
    print(df.head())
    df.to_csv("verification_reliance.csv", index=False)
else:
    print("ERROR: DATA IS EMPTY")
