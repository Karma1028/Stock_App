from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import pandas as pd

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data.manager import StockDataManager
from modules.data.scrapers.news_scraper import NewsScraper

app = FastAPI(title="Stock App API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "app": "Stock App API"}

@app.get("/api/dashboard")
def get_dashboard_data():
    dm = StockDataManager()
    try:
        sentiment = dm.get_market_sentiment()
        gainers = dm.get_top_gainers(limit=5)
        stock_count = len(dm.get_stock_list())
        return {
            "sentiment": sentiment,
            "gainers": gainers,
            "stock_count": stock_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{symbol}")
def get_stock_details(symbol: str):
    dm = StockDataManager()
    try:
        data = dm.get_live_data(symbol)
        if not data:
            raise HTTPException(status_code=404, detail="Stock not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{symbol}/history")
def get_stock_history(symbol: str, period: str = "1y"):
    dm = StockDataManager()
    try:
        df = dm.get_cached_data([symbol], period=period)
        if df.empty:
             return []
        
        # Handle MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns:
                df = df[symbol]
            elif df.columns.nlevels > 1 and symbol in df.columns.get_level_values(1):
                df = df.xs(symbol, axis=1, level=1)
            elif 'Close' in df.columns.get_level_values(0):
                 df.columns = df.columns.get_level_values(0)
        
        # Reset index to make Date a column
        df_reset = df.reset_index()
        # Convert Timestamp to str
        df_reset['Date'] = df_reset['Date'].astype(str)
        # Convert to dict
        return df_reset.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/news")
def get_news(limit: int = 6):
    ns = NewsScraper()
    try:
        # Fetch news for generic major stocks for dashboard
        n1_df = ns.fetch_news_history("RELIANCE.NS", days=2)
        n2_df = ns.fetch_news_history("HDFCBANK.NS", days=2)
        combined_news = pd.concat([n1_df, n2_df], ignore_index=True)
        
        if not combined_news.empty:
            combined_news = combined_news.sort_values('date', ascending=False).head(limit)
            # Serialize date
            combined_news['date'] = combined_news['date'].astype(str)
            return combined_news.to_dict('records')
        return []
    except Exception as e:
        print(f"News error: {e}")
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
