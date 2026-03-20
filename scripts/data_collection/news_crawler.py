import asyncio
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import pandas as pd
import logging
import os
from datetime import datetime, timedelta

# Set up logging for the MAS Auditor
os.makedirs('Universal_Engine_Workspace/mas_logs', exist_ok=True)
logging.basicConfig(
    filename='Universal_Engine_Workspace/mas_logs/delta_news_scraper.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataEngineer_NewsScraper')

async def fetch_article_details(crawler, url):
    """Navigates to a specific news article to extract full title and content."""
    logger.info(f"Crawling article: {url}")
    try:
        result = await crawler.arun(
            url=url,
            magic=True,
            simulate_user=True,
            bypass_cache=True,
            delay_before_return_html=1.5
        )
        if result.success:
            # Metadata often contains the full, untruncated title
            title = result.metadata.get('og:title') or result.metadata.get('title')
            # The markdown attribute contains the cleaned page content
            content = result.markdown
            return title, content
    except Exception as e:
        logger.error(f"Error fetching article details for {url}: {e}")
    return None, None

async def fetch_news_links_crawl4ai(crawler, query):
    """Fetches search results and extracts direct links to articles."""
    logger.info(f"Searching Yahoo News for links: '{query}'")
    search_url = f"https://news.search.yahoo.com/search?p={query}"
    
    result = await crawler.arun(
        url=search_url,
        magic=True,
        simulate_user=True,
        delay_before_return_html=2.0
    )
    
    links = []
    if result.success:
        soup = BeautifulSoup(result.html, 'html.parser')
        # Yahoo News links are in 'h4.s-title a'
        for item in soup.find_all('h4', class_='s-title'):
            a_tag = item.find('a')
            if a_tag and a_tag.get('href'):
                links.append(a_tag['href'])
    
    return links

async def main():
    equity_csv_path = 'data/EQUITY.csv'
    
    queries = [
        "Reserve Bank of India interest rates", # Macro
        "India inflation data"               # Macro
    ]
    
    try:
        equity_df = pd.read_csv(equity_csv_path)
        # Testing with top 2 companies for verification of the full-content pipeline
        test_companies = equity_df['NAME OF COMPANY'].head(2).tolist()
        for company in test_companies:
            queries.append(f"{company} corporate news")
    except Exception as e:
        logger.error(f"Could not load {equity_csv_path}: {e}")
    
    detailed_news = []
    async with AsyncWebCrawler(verbose=True) as crawler:
        for query in queries:
            links = await fetch_news_links_crawl4ai(crawler, query)
            
            # Scrape top 3 articles per query to generate rich training context
            for link in links[:3]:
                full_title, content = await fetch_article_details(crawler, link)
                if full_title and content:
                    # Simulated historical date for V5 temporal engine
                    date_str = (datetime.now() - timedelta(days=len(detailed_news)*7)).isoformat()
                    
                    detailed_news.append({
                        'date': date_str,
                        'title': full_title,
                        'url': link,
                        'content': content[:2500], # Store first 2.5k characters for FinBERT
                        'query': query
                    })
             
    if detailed_news:
         all_news = pd.DataFrame(detailed_news)
         all_news['date'] = pd.to_datetime(all_news['date'], utc=True).dt.tz_localize(None)
         
         # Categorize for the prediction layer
         all_news['company_flag'] = ~all_news['query'].str.contains('Reserve Bank|inflation', case=False)
         
         # Save to processed state
         output_path = 'Universal_Engine_Workspace/scripts_scraping/delta_news_scraped.csv'
         all_news.to_csv(output_path, index=False)
         logger.info(f"Finished writing {len(all_news)} detailed rows to {output_path}")
         print(f"Scraped {len(all_news)} detailed records (Full Titles + Content) to {output_path}")
    else:
         logger.warning("No detailed news scraped.")
         print("No detailed news found.")

if __name__ == "__main__":
    asyncio.run(main())
