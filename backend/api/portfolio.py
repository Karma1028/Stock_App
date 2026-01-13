from fastapi import APIRouter, HTTPException
from modules.utils.ai_insights import generate_company_summary, generate_investment_plan, analyze_portfolio
from modules.utils.quant import QuantEngine
from modules.data.manager import StockDataManager

router = APIRouter()

@router.post("/ai/summary")
def get_ai_summary(payload: dict):
    symbol = payload.get("symbol")
    model = payload.get("model", "google/gemini-2.0-flash-exp:free")
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    
    try:
        summary = generate_company_summary(symbol, model=model)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolio/backtest")
def run_backtest(payload: dict):
    dm = StockDataManager()
    qe = QuantEngine(dm)
    
    try:
        result = qe.run_pipeline(payload)
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/investment-plan")
def get_investment_plan(payload: dict):
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

@router.post("/portfolio/analyze")
def analyze_custom_portfolio(payload: dict):
    try:
        analysis = analyze_portfolio(
            portfolio_data=payload.get("portfolio"),
            model=payload.get("model", "google/gemini-2.0-flash-exp:free")
        )
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
