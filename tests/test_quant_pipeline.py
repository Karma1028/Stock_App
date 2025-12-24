from quant_engine import QuantEngine
from data_manager import StockDataManager
import pandas as pd

print("🚀 Testing Quant Pipeline...")
dm = StockDataManager()
qe = QuantEngine(dm)

user_profile = {
    "investment_amount": 100000,
    "duration_years": 5,
    "risk_profile": "Moderate",
    "investment_type": "Growth",
    "expected_annual_return_pct": 15
}

try:
    print("Running pipeline...")
    payload = qe.run_pipeline(user_profile)
    if "error" in payload:
        print(f"❌ Pipeline Error: {payload['error']}")
    else:
        print("✅ Pipeline Success!")
        print("Keys in payload:", payload.keys())
        if 'backtest_summary' in payload:
            print("Backtest Summary:", payload['backtest_summary'])
except Exception as e:
    print(f"❌ Exception: {e}")
    import traceback
    traceback.print_exc()
