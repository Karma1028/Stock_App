from fastapi import APIRouter, HTTPException
from modules.data.manager import StockDataManager

router = APIRouter()

@router.get("/dashboard")
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
