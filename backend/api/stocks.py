from fastapi import APIRouter, HTTPException
from modules.data.manager import StockDataManager
from modules.data.scrapers.news_scraper import NewsScraper
from modules.ml.engine import MLEngine
from modules.ml.features import FeatureEngineer
from modules.ml.prediction import StockPredictor
import pandas as pd

router = APIRouter()

@router.get("/stocks")
def get_stock_list():
    dm = StockDataManager()
    try:
        stocks = dm.get_stock_list()
        stocks = [s for s in stocks if s]
        return {"stocks": stocks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock/{symbol}")
def get_stock_details(symbol: str):
    dm = StockDataManager()
    try:
        data = dm.get_live_data(symbol)
        if not data:
            raise HTTPException(status_code=404, detail="Stock not found")
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock/{symbol}/history")
def get_stock_history(symbol: str, period: str = "1y"):
    dm = StockDataManager()
    try:
        df = dm.get_cached_data([symbol], period=period)
        if df.empty:
             return []
        
        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns:
                df = df[symbol]
            elif df.columns.nlevels > 1 and symbol in df.columns.get_level_values(1):
                df = df.xs(symbol, axis=1, level=1)
            elif 'Close' in df.columns.get_level_values(0):
                 df.columns = df.columns.get_level_values(0)
        
        df_reset = df.reset_index()
        df_reset['Date'] = df_reset['Date'].astype(str)
        return df_reset.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/news")
def get_news(limit: int = 6):
    ns = NewsScraper()
    try:
        n1_df = ns.fetch_news_history("RELIANCE.NS", days=2)
        n2_df = ns.fetch_news_history("HDFCBANK.NS", days=2)
        combined_news = pd.concat([n1_df, n2_df], ignore_index=True)
        
        if not combined_news.empty:
            combined_news = combined_news.sort_values('date', ascending=False).head(limit)
            combined_news['date'] = combined_news['date'].astype(str)
            return combined_news.to_dict('records')
        return []
    except Exception as e:
        print(f"News error: {e}")
        return []

@router.get("/stock/{symbol}/predict")
def get_stock_prediction(symbol: str, days: int = 30):
    dm = StockDataManager()
    fe = FeatureEngineer()
    ml = MLEngine()
    sp = StockPredictor()
    ns = NewsScraper()
    
    try:
        # 1. Fetch Data
        df = dm.get_cached_data([symbol], period="2y")
        if df.empty:
             raise HTTPException(status_code=404, detail="No data found for prediction")
        
        # Handle MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns: df = df[symbol]
            elif df.columns.nlevels > 1 and symbol in df.columns.get_level_values(1): df = df.xs(symbol, axis=1, level=1)
            elif 'Close' in df.columns.get_level_values(0): df.columns = df.columns.get_level_values(0)

        # 2. Features & KPI Scores
        # Fetch sentiment
        news_df = ns.fetch_news_history(symbol, days=30)
        sentiment_daily = pd.DataFrame()
        if not news_df.empty:
            news_df = ns.analyze_sentiment(news_df)
            df_sent = news_df.copy()
            df_sent['date_only'] = df_sent['date'].dt.date
            sentiment_daily = df_sent.groupby('date_only')['sentiment'].agg(['mean', 'count', 'std']).reset_index()
            sentiment_daily = sentiment_daily.rename(columns={'mean': 'sentiment_score', 'count': 'news_count', 'std': 'sentiment_volatility'}).fillna(0)
        
        df_features = fe.compute_all_features(df, sentiment_daily)
        if not sentiment_daily.empty:
            df_features = fe.merge_sentiment(df_features, sentiment_daily)
            
        # Predict Score
        preds = None
        try:
             # Try global model first
             ml_global = MLEngine(model_name="xgb_model_global.pkl")
             if ml_global.load_model():
                 preds = ml_global.predict(df_features)
             else:
                 # Minimal training if no model (fallback)
                 ml.train_model(df_features, model_type="xgboost")
                 preds = ml.predict(df_features)
        except Exception:
             pass 

        kpi = {}
        if preds is not None:
            kpi = ml.calculate_combined_score(df_features, preds)
            
        # 3. Forecast (Prophet)
        forecast, _ = sp.train_and_predict(df.copy(), periods=days)
        forecast_data = []
        if forecast is not None:
             last_date = df.index.max()
             if last_date.tzinfo: last_date = last_date.tz_localize(None)
             forecast['ds'] = forecast['ds'].astype(str)
             forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
        
        return {
            "kpi": kpi,
            "forecast": forecast_data
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
