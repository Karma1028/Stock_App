import pandas as pd
from ntscraper import Nitter
from textblob import TextBlob
import random
import time

class TwitterScraper:
    def __init__(self):
        self.scraper = Nitter(log_level=1, skip_instance_check=False)

    def fetch_tweets(self, symbol, limit=20, days=7):
        """
        Fetches tweets for a stock symbol (cashtag) from Nitter.
        """
        try:
            # Clean symbol to get ticker or name
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
            query = f"${clean_symbol}"
            
            # Fetch tweets
            print(f"Fetching tweets for {query}...")
            # mode='term' finds exact match provided
            tweets_data = self.scraper.get_tweets(query, mode='term', number=limit)
            
            if not tweets_data or 'tweets' not in tweets_data:
                return []
                
            tweets = []
            for t in tweets_data['tweets']:
                tweets.append({
                    'text': t['text'],
                    'date': t['date'],
                    'likes': t['stats']['likes'],
                    'retweets': t['stats']['retweets'],
                    'user': t['user']['username']
                })
                
            return tweets
            
        except Exception as e:
            print(f"Error fetching tweets for {symbol}: {e}")
            return []

    def analyze_sentiment(self, tweets):
        """
        Analyzes sentiment of a list of tweets using TextBlob.
        """
        if not tweets:
            return pd.DataFrame()
            
        df = pd.DataFrame(tweets)
        
        def get_sentiment(text):
            blob = TextBlob(text)
            return blob.sentiment.polarity
            
        df['sentiment'] = df['text'].apply(get_sentiment)
        
        # Calculate weighted sentiment if possible (likes/retweets)
        # For now, simple average
        return df

    def get_sentiment_summary(self, symbol, limit=30):
        """
        Returns a summary of twitter sentiment for a stock.
        """
        tweets = self.fetch_tweets(symbol, limit=limit)
        if not tweets:
            return {"score": 0, "status": "Neutral", "count": 0}
            
        df = self.analyze_sentiment(tweets)
        avg_sentiment = df['sentiment'].mean()
        count = len(df)
        
        # Scale -1 to 1 -> 0 to 100
        score = (avg_sentiment + 1) * 50
        
        if score > 60: status = "Bullish"
        elif score < 40: status = "Bearish"
        else: status = "Neutral"
        
        return {
            "score": score,
            "status": status,
            "count": count,
            "latest_tweets": tweets[:3]
        }
