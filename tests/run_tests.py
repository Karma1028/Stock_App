import sys
from config import Config
from data_manager import StockDataManager
from prediction_model import StockPredictor

def run_smoke_test():
    print("Starting Smoke Test...")
    
    # 1. Check Directories
    try:
        Config.ensure_directories()
        print("[OK] Directories verified.")
    except Exception as e:
        print(f"[FAIL] Directory check failed: {e}")
        return

    # 2. Check Data Manager (yfinance)
    dm = StockDataManager()
    try:
        tickers = dm.get_stock_list()
        if not tickers:
            print("[FAIL] Stock list is empty.")
            return
        print(f"[OK] Stock list fetched: {len(tickers)} tickers.")
        
        # Fetch historical data
        test_symbol = tickers[0]
        print(f"Fetching data for {test_symbol}...")
        df = dm.get_historical_data(test_symbol, period="1mo")
        if df.empty:
            print(f"[WARN] Historical data empty for {test_symbol}. Network issue?")
        else:
            print(f"[OK] Historical data fetched: {len(df)} rows.")
            
            # 3. Check Prediction
            print("Testing Prediction Model...")
            predictor = StockPredictor()
            forecast, model = predictor.train_and_predict(df, periods=5)
            if forecast is not None:
                print("[OK] Prediction model ran successfully.")
            else:
                print("[FAIL] Prediction model failed.")

    except Exception as e:
        print(f"[FAIL] Data Manager test failed: {e}")
        return

    print("SMOKE OK")

if __name__ == "__main__":
    run_smoke_test()
