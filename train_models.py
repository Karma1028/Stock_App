import pandas as pd
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

from modules.data.manager import StockDataManager
from modules.ml.features import FeatureEngineer
from modules.ml.engine import MLEngine
from modules.data.scrapers.news_scraper import NewsScraper
from config import Config

def train_global_model():
    print("🚀 Starting Global Model Training...")
    
    dm = StockDataManager()
    fe = FeatureEngineer()
    ns = NewsScraper()
    # Initialize with global model name
    ml = MLEngine(model_name="xgb_model_global.pkl")
    
    tickers = dm.get_stock_list()
    print(f"Found {len(tickers)} stocks to train on.")
    
    all_features = []
    
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Processing {ticker}...")
        try:
            # 1. Fetch 5 years of data
            # get_cached_data now defaults to 5y
            df = dm.get_cached_data([ticker], period="5y")
            
            if df.empty:
                print(f"  No data for {ticker}")
                continue
            
            # Handle MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                if ticker in df.columns:
                    df = df[ticker]
                else:
                    # Fallback if ticker format differs
                    df = df.xs(ticker, level=0, axis=1, drop_level=True)
                
            # 2. Fetch News (Last 30 days)
            # Note: We only have recent news. Historical sentiment will be 0 (neutral).
            sentiment_daily = ns.get_aggregated_sentiment(ticker, days=30)
            
            # 3. Compute Features
            df_features = fe.compute_all_features(df, sentiment_daily)
            if not sentiment_daily.empty:
                df_features = fe.merge_sentiment(df_features, sentiment_daily)
            
            # Add Ticker column for reference (optional, model doesn't use it yet but good for debugging)
            df_features['Ticker'] = ticker
            
            # Add to collection
            all_features.append(df_features)
            
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
    
    if not all_features:
        print("❌ No data collected. Check internet connection or data source.")
        return
    
    # Concatenate all data
    print("Merging all data...")
    full_df = pd.concat(all_features)
    print(f"Total dataset size: {full_df.shape}")
    
    # Save CSV
    csv_path = Config.DATA_DIR / "training_data.csv"
    full_df.to_csv(csv_path)
    print(f"Training data saved to {csv_path}")
    
    # Train Model
    print("Training Global XGBoost Model...")
    model = ml.train_model(full_df, model_type="xgboost")
    
    if model:
        print(f"✅ Global Model Trained and Saved to {ml.model_file}")
    else:
        print("❌ Training Failed.")

if __name__ == "__main__":
    train_global_model()
