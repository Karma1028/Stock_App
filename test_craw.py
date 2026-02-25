import asyncio
from modules.data.scrapers.news_analysis import _crawl_article_text

def test():
    url = "https://finance.yahoo.com/news/stock-market-today-dow-nasdaq-sp500-latest-news"
    print("Testing crawl4ai on:", url)
    res = _crawl_article_text(url)
    print("Result length:", len(res))
    if len(res) > 0:
        print("Preview:", res[:100])
    else:
        print("Failed to extract text.")

if __name__ == "__main__":
    test()
