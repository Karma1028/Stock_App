import requests
import feedparser
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob
import time
import urllib.parse

class NewsScraper:
    def __init__(self):
        self.base_url = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    def fetch_news_history(self, ticker, days=30):
        """
        Fetches news headlines for the last `days` from Google News RSS.
        Note: RSS feed doesn't strictly support date ranges in the URL for all regions,
        so we fetch the general feed and filter by date, or use specific query modifiers.
        """
        # Clean ticker for search (remove .NS)
        search_term = ticker.replace('.NS', '').replace('.BO', '')
        
        # Construct query with date range (Google Search syntax often works in RSS)
        # q=Reliance+after:2023-10-01+before:2023-11-01
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        query = f"{search_term} stock news after:{start_date.strftime('%Y-%m-%d')}"
        encoded_query = urllib.parse.quote(query)
        url = self.base_url.format(query=encoded_query)
        
        print(f"Fetching news for {ticker} from {url}...")
        
        try:
            feed = feedparser.parse(url)
            news_items = []
            
            for entry in feed.entries:
                try:
                    # Parse published date
                    # format: 'Mon, 02 Dec 2025 07:00:00 GMT'
                    pub_date = datetime(*entry.published_parsed[:6])
                    
                    if pub_date >= start_date:
                        news_items.append({
                            'date': pub_date,
                            'title': entry.title,
                            'link': entry.link,
                            'source': entry.source.title if 'source' in entry else 'Google News'
                        })
                except Exception as e:
                    continue
            
            df = pd.DataFrame(news_items)
            if not df.empty:
                df = df.sort_values('date', ascending=False)
            return df
            
        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return pd.DataFrame()

    def analyze_sentiment(self, df):
        """
        Adds sentiment scores to the news dataframe.
        """
        if df.empty:
            return df
        
        def get_sentiment(text):
            return TextBlob(text).sentiment.polarity

        df['sentiment'] = df['title'].apply(get_sentiment)
        return df

    def get_aggregated_sentiment(self, ticker, days=30):
        """
        Fetches news and returns daily aggregated sentiment.
        """
        df = self.fetch_news_history(ticker, days)
        if df.empty:
            return pd.DataFrame()
        
        df = self.analyze_sentiment(df)
        
        # Group by date (daily)
        df['date_only'] = df['date'].dt.date
        daily_sentiment = df.groupby('date_only')['sentiment'].agg(['mean', 'count', 'std']).reset_index()
        daily_sentiment = daily_sentiment.rename(columns={'mean': 'sentiment_score', 'count': 'news_count', 'std': 'sentiment_volatility'})
        daily_sentiment = daily_sentiment.fillna(0) # std might be NaN if count=1
        
        return daily_sentiment
