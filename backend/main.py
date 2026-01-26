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
def get_dashboard_data(symbols: str = None):
    """
    Get dashboard data.
    If symbols parameter is provided (comma-separated), returns data for those stocks.
    Otherwise returns default stocks (top gainers + market metrics).
    """
    dm = StockDataManager()
    try:
        # Get market sentiment
        sentiment = dm.get_market_sentiment()
        
        # Get stock count
        stock_count = len(dm.get_stock_list())
        
        # Get top gainers (always show these)
        gainers = dm.get_top_gainers(limit=5)
        
        # Get specific stocks if requested
        selected_stocks = []
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(',')]
            for symbol in symbol_list[:10]:  # Limit to 10 stocks
                try:
                    stock_data = dm.get_live_data(symbol)
                    if stock_data:
                        selected_stocks.append(stock_data)
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
        
        return {
            "sentiment": sentiment,
            "gainers": gainers,
            "stock_count": stock_count,
            "selected_stocks": selected_stocks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks")
def get_stock_list():
    dm = StockDataManager()
    try:
        stocks = dm.get_stock_list()
        # Clean up filtered out None or empty
        stocks = [s for s in stocks if s]
        return {"stocks": stocks}
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


@app.get("/api/stock/{symbol}/news")
def get_stock_news_specific(symbol: str, days: int = 7):
    ns = NewsScraper()
    try:
        df = ns.fetch_news_history(symbol, days=days)
        if not df.empty:
            df['date'] = df['date'].astype(str)
            return df.to_dict('records')
        return []
    except Exception as e:
        print(f"News error for {symbol}: {e}")
        return []

@app.post("/api/ai/summary")
def get_ai_summary(payload: dict):
    symbol = payload.get("symbol")
    model = payload.get("model", "google/gemini-2.0-flash-exp:free")
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    
    from modules.utils.ai_insights import generate_company_summary
    dm = StockDataManager()
    
    try:
        # Fetch real data to result in grounded AI response
        stock_data = dm.get_live_data(symbol) or {}
        ratios = dm.get_key_ratios(symbol) or {}
        
        # Combine into a context dict
        context_data = {
            "profile": stock_data,
            "ratios": ratios
        }
        
        summary_str = generate_company_summary(symbol, context_data=context_data, model=model)
        import json
        try:
             # Try to parse the AI response as JSON if possible
             # It might be wrapped in markdown code blocks
             clean_str = summary_str.replace('```json', '').replace('```', '').strip()
             summary_data = json.loads(clean_str)
             return summary_data # Return directly as JSON object
        except json.JSONDecodeError:
             return {"summary": summary_str}
    except Exception as e:
        print(f"AI Summary Error: {e}")
        # Fallback to simple generation if data fetch fails
        try:
             from modules.utils.ai_insights import generate_company_summary
             return {"summary": generate_company_summary(symbol, model=model)}
        except:
             raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{symbol}/financials")
def get_stock_financials(symbol: str):
    dm = StockDataManager()
    try:
        financials = dm.get_financials(symbol)
        result = {}
        for key, df in financials.items():
            if not df.empty:
                result[key] = df.fillna(0).reset_index().to_dict(orient="records")
            else:
                result[key] = []
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{symbol}/ratios")
def get_stock_ratios(symbol: str):
    dm = StockDataManager()
    try:
        ratios = dm.get_key_ratios(symbol)
        if not ratios:
            return {}
        return ratios
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{symbol}/sentiment/social")
def get_social_sentiment(symbol: str):
    dm = StockDataManager()
    try:
        sentiment = dm.get_twitter_sentiment(symbol)
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/portfolio/backtest")
def run_backtest(payload: dict):
    from modules.utils.quant import QuantEngine
    dm = StockDataManager()
    qe = QuantEngine(dm)
    
    try:
        result = qe.run_pipeline(payload)
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/investment-plan")
def get_investment_plan(payload: dict):
    from modules.utils.ai_insights import generate_investment_plan
    try:
        plan = generate_investment_plan(
            amount=payload.get("amount"),
            duration_years=payload.get("duration"),
            expected_return=payload.get("expected_return"),
            risk_profile=payload.get("risk_profile"),
            market_context=payload.get("market_context", "Indian markets are trading at all-time highs with high volatility."),
            model=payload.get("model", "google/gemini-2.0-flash-exp:free")
        )
        return {"plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/{symbol}/predict")
def get_stock_prediction(symbol: str, days: int = 30):
    from modules.ml.engine import MLEngine
    from modules.ml.features import FeatureEngineer
    from modules.ml.prediction import StockPredictor
    
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
             # Try global model first, ideally we shouldn't train on every request but for demo/completeness:
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
             # Filter for future only or include history? User wants chart. Return last 90 days + future.
             last_date = df.index.max()
             if last_date.tzinfo: last_date = last_date.tz_localize(None)
             
             # Convert ds to str
             forecast['ds'] = forecast['ds'].astype(str)
             forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
        
        return {
            "kpi": kpi,
            "forecast": forecast_data
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/portfolio/analyze")
def analyze_custom_portfolio(payload: dict):
    from modules.utils.ai_insights import analyze_portfolio
    try:
        analysis = analyze_portfolio(
            portfolio_data=payload.get("portfolio"),
            model=payload.get("model", "google/gemini-2.0-flash-exp:free")
        )
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
