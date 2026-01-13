"""
Script to train all enhanced ML models with sentiment integration.
Saves models in PKL format for deployment.
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.ml.sentiment_model import SentimentModel
from modules.ml.engine import MLEngine
from modules.ml.features import FeatureEngineer
from modules.data.manager import StockDataManager
from modules.data.scrapers.news_scraper import NewsScraper
from config import Config
import pandas as pd

def train_sentiment_model():
    """Train the sentiment analysis model."""
    print("\n" + "="*60)
    print("1. TRAINING SENTIMENT ANALYSIS MODEL")
    print("="*60)
    
    news_path = Config.DATA_DIR / "all_stocks_news_consolidated.csv"
    
    if not news_path.exists():
        print(f"⚠️  News data not found at {news_path}")
        print("Skipping sentiment model training...")
        return None
    
    model = SentimentModel()
    accuracy = model.train(news_path)
    
    return model

def train_price_prediction_model(include_sentiment=True):
    """Train the enhanced price prediction model with sentiment features."""
    print("\n" + "="*60)
    print("2. TRAINING PRICE PREDICTION MODEL (XGBoost)")
    print("="*60)
    
    dm = StockDataManager()
    fe = FeatureEngineer()
    ns = NewsScraper()
    
    # Get top stocks for training
    tickers = dm.default_tickers[:10]  # Start with Nifty 50 top 10 for speed
    
    print(f"Fetching data for {len(tickers)} stocks...")
    df = dm.get_cached_data(tickers, period="2y")
    
    if df.empty:
        print("No data available for training")
        return None
    
    print("Computing technical features...")
    
    # Process each ticker
    all_features = []
    
    for ticker in tickers:
        try:
            # Extract ticker data
            if isinstance(df.columns, pd.MultiIndex):
                ticker_df = df[ticker].copy()
            else:
                ticker_df = df.copy()
            
            # Compute technical features
            features_df = fe.compute_all_features(ticker_df)
            
            # Add sentiment if requested
            if include_sentiment:
                print(f"Fetching sentiment for {ticker}...")
                try:
                    news_df = ns.fetch_news_history(ticker, days=365)
                    
                    if not news_df.empty:
                        # Analyze sentiment with both TextBlob and ML model
                        news_df = ns.analyze_sentiment(news_df)
                        
                        # Load sentiment model if available
                        sent_model = SentimentModel()
                        if sent_model.load():
                            ml_sentiments = sent_model.predict(news_df['title'].tolist())
                            news_df['ml_sentiment'] = ml_sentiments
                            # Average TextBlob and ML model sentiments
                            news_df['combined_sentiment'] = (news_df['sentiment'] + news_df['ml_sentiment']) / 2
                        else:
                            news_df['combined_sentiment'] = news_df['sentiment']
                        
                        # Aggregate to daily
                        news_df['date_only'] = news_df['date'].dt.date
                        sentiment_daily = news_df.groupby('date_only').agg({
                            'combined_sentiment': ['mean', 'std', 'count']
                        }).reset_index()
                        
                        sentiment_daily.columns = ['date_only', 'sentiment_score', 'sentiment_volatility', 'news_count']
                        sentiment_daily = sentiment_daily.fillna(0)
                        
                        # Merge with features
                        features_df = fe.merge_sentiment(features_df, sentiment_daily)
                except Exception as e:
                    print(f"Error processing sentiment for {ticker}: {e}")
                    # Add zero sentiment
                    features_df['sentiment_score'] = 0
                    features_df['sentiment_volatility'] = 0
            else:
                features_df['sentiment_score'] = 0
                features_df['sentiment_volatility'] = 0
            
            features_df['ticker'] = ticker
            all_features.append(features_df)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    if not all_features:
        print("No features computed")
        return None
    
    # Combine all tickers
    combined_df = pd.concat(all_features, ignore_index=False)
    
    print(f"\nTraining data shape: {combined_df.shape}")
    print(f"Features: {combined_df.columns.tolist()}")
    
    # Train model
    ml = MLEngine(model_name="xgb_model_global.pkl")
    model = ml.train_model(combined_df, model_type="xgboost")
    
    return model

def main():
    """Main training pipeline."""
    print("\n" + "🚀 " * 20)
    print("ENHANCED ML MODEL TRAINING PIPELINE")
    print("🚀 " * 20)
    
    # Step 1: Train Sentiment Model
    sentiment_model = train_sentiment_model()
    
    # Step 2: Train Price Prediction Model (with sentiment)
    price_model = train_price_prediction_model(include_sentiment=True)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    
    print("\nTrained Models:")
    print(f"  - sentiment_model.pkl: {'✓' if sentiment_model else '✗'}")
    print(f"  - xgb_model_global.pkl: {'✓' if price_model else '✗'}")
    
    print(f"\nModels saved in: {Config.MODELS_DIR}")
    
    # Test sentiment model
    if sentiment_model:
        print("\n" + "-"*60)
        print("Testing Sentiment Model:")
        print("-"*60)
        test_headlines = [
            "Company beats earnings estimates, stock soars 15%",
            "Major scandal uncovered, shares crash to year low",
            "Quarterly results meet expectations"
        ]
        
        for headline in test_headlines:
            score = sentiment_model.predict([headline])[0]
            sentiment_label = "Positive" if score > 0.1 else ("Negative" if score < -0.1 else "Neutral")
            print(f"'{headline}'")
            print(f"  → Sentiment: {sentiment_label} (Score: {score:.3f})\n")

if __name__ == "__main__":
    main()
