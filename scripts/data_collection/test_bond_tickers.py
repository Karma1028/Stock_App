import yfinance as yf
import pandas as pd

def test_tickers():
    tickers = ['^IN10YN', 'IN10Y.NS', 'IND10Y=R', 'INDIA10Y=RR']
    results = {}
    for t in tickers:
        try:
            data = yf.download(t, period='1mo', progress=False)
            results[t] = "Success" if not data.empty else "Empty"
        except Exception as e:
            results[t] = f"Error: {e}"
    
    print(results)

if __name__ == "__main__":
    test_tickers()
