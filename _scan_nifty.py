import sys, os, warnings
sys.path.insert(0, '.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from modules.ml.features import FeatureEngineer
from modules.ml.engine import MLEngine

fe = FeatureEngineer()
ml = MLEngine(model_name='xgb_regressor_v5.pkl')

nifty50 = [
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','INFY.NS','ICICIBANK.NS',
    'ITC.NS','SBIN.NS','BHARTIARTL.NS','LT.NS','KOTAKBANK.NS',
    'WIPRO.NS','HCLTECH.NS','MARUTI.NS','SUNPHARMA.NS','AXISBANK.NS',
    'TITAN.NS','BAJFINANCE.NS','ULTRACEMCO.NS','NESTLEIND.NS','POWERGRID.NS'
]

results = []
for t in nifty50:
    try:
        df = yf.download(t, period='1y', progress=False)
        if hasattr(df.columns, 'levels') and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df_feat = fe._compute_single_ticker_features(df.copy())
        if 'sentiment_score' not in df_feat.columns:
            df_feat['sentiment_score'] = 0.0
        preds = ml.predict(df_feat)
        if preds is not None and len(preds) > 0:
            latest_pred = preds.iloc[-1] * 100
            scores = ml.calculate_combined_score(df_feat, preds)
            results.append({
                'Ticker': t.replace('.NS', ''),
                'Pred_5d_%': round(latest_pred, 3),
                'Score': scores['combined_score'],
                'Tech': scores['technical_score'],
                'Signal': 'BUY' if scores['combined_score'] > 60 else ('HOLD' if scores['combined_score'] > 40 else 'SELL')
            })
    except Exception as e:
        pass

df_r = pd.DataFrame(results).sort_values('Score', ascending=False)
print(df_r.to_string(index=False))
buy = len(df_r[df_r['Signal'] == 'BUY'])
hold = len(df_r[df_r['Signal'] == 'HOLD'])
sell = len(df_r[df_r['Signal'] == 'SELL'])
pos = len(df_r[df_r['Pred_5d_%'] > 0])
print(f"\n--- Summary ---")
print(f"Positive predicted 5d return: {pos}/{len(df_r)} stocks")
print(f"BUY: {buy} | HOLD: {hold} | SELL: {sell}")
if len(df_r) > 0:
    best = df_r.iloc[0]
    print(f"Top pick: {best['Ticker']} (Score {best['Score']}, {best['Pred_5d_%']:+.3f}% predicted 5d return)")
