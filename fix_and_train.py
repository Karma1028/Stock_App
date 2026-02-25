import pandas as pd
import yfinance as yf
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
from pathlib import Path

# Add root to path
sys.path.append(os.getcwd())

from config import Config
from modules.ml.features import FeatureEngineer
from xgboost import XGBRegressor
from agentic_backend import MonteCarloLSTM

# Explicit Nifty 50 List (Hardcoded to ensure reliability)
NIFTY_50 = [
    'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS',
    'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
    'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS',
    'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
    'MARUTI.NS', 'M&M.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS',
    'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS',
    'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS',
    'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'
]

def fetch_and_train():
    print("🚀 Starting Reliable Fetch & Train Sequence...")
    
    # 1. Fetch Data
    print(f"📡 Fetching data for {len(NIFTY_50)} Nifty 50 stocks (2y)...")
    try:
        # Force fresh download, bypassing manager logic for now to ensure we get data
        data = yf.download(NIFTY_50, period="2y", progress=True, threads=True)
        
        if data.empty:
            print("❌ Download returned empty dataframe.")
            return

        print(f"✅ Data fetched. Shape: {data.shape}")
        
        # Save to cache manually to fix the "manager" looking for it
        cache_dir = Config.DATA_DIR / "raw"
        cache_dir.mkdir(exist_ok=True, parents=True)
        data.to_pickle(cache_dir / "bulk_data_2y.pkl")
        print("💾 Cached to data/raw/bulk_data_2y.pkl")
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return

    # 2. Train XGBoost (Global Model)
    print("\n🌲 Training XGBoost Model...")
    fe = FeatureEngineer()
    
    # Feature Engineering
    print("⚙️ Computing features...")
    df_features = fe.compute_all_features(data)
    
    print("Features Shape (Pre-Dropna):", df_features.shape)
    if not df_features.empty:
        print("NaN Counts:\n", df_features.isna().sum().sort_values(ascending=False).head(10))
        print("Head:\n", df_features.head())
    
    # Replace inf with nan
    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop NaNs created by lagging/rolling
    df_features = df_features.dropna()
    print("Features Shape (Post-Dropna):", df_features.shape)
    
    if df_features.empty:
        print("❌ No valid features after dropna.")
        return

    target_col = 'Target_5d'
    target_col = 'Target_5d'
    # Exclude non-numeric/target cols
    features_col = [c for c in df_features.columns if c not in [target_col, 'Ticker', 'date_only', 'Close', 'Open', 'High', 'Low', 'Volume']]
    
    X = df_features[features_col]
    y = df_features[target_col]
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6)
    model.fit(X, y)
    
    unique_tickers = df_features['Ticker'].unique() if 'Ticker' in df_features.columns else ["Nifty50"]
    
    model_path = Config.MODELS_DIR / "xgb_regressor_v5.pkl"
    joblib.dump(model, model_path)
    print(f"✅ XGBoost Model Saved to {model_path}")

    # 3. Train LSTM (Monte Carlo)
    print("\n🧠 Training LSTM Model...")
    
    # Scale
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(X)
    
    # Save Scaler
    scaler_path = Config.MODELS_DIR / "lstm_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    # Create Sequences
    SEQ_LENGTH = 60
    X_lstm, y_lstm = [], []
    
    # We need to preserve sequences per ticker. 
    # Re-iterating over unique tickers in df_features is safer
    if 'Ticker' in df_features.columns:
        for ticker in df_features['Ticker'].unique():
            idx = df_features['Ticker'] == ticker
            sub_X = scaled_features[idx]
            sub_y = (y[idx] > 0).astype(int).values # Binary Target
            
            if len(sub_X) < SEQ_LENGTH:
                continue
                
            for i in range(len(sub_X) - SEQ_LENGTH):
                X_lstm.append(sub_X[i:i+SEQ_LENGTH])
                y_lstm.append(sub_y[i+SEQ_LENGTH])
    else:
        # Fallback if no ticker column (single index?)
        # But compute_all_features adds 'Ticker' if multiindex. 
        # If single index, we treat as one sequence
        pass # Todo handle single

    X_lstm = np.array(X_lstm)
    y_lstm = np.array(y_lstm)
    
    if len(X_lstm) == 0:
        print("❌ No sequences generated for LSTM.")
        return
        
    print(f"LSTM Training Data: {X_lstm.shape}")
    
    X_tensor = torch.FloatTensor(X_lstm)
    y_tensor = torch.FloatTensor(y_lstm).reshape(-1, 1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Model
    input_size = X_lstm.shape[2]
    lstm_model = MonteCarloLSTM(input_size=input_size, hidden_size=64, num_layers=2, dropout_rate=0.2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    
    lstm_model.train()
    for epoch in range(5):
        epoch_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = lstm_model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss {epoch_loss/len(loader):.4f}")
        
    lstm_path = Config.MODELS_DIR / "lstm_model.pth"
    torch.save(lstm_model.state_dict(), lstm_path)
    print(f"✅ LSTM Model Saved to {lstm_path}")

if __name__ == "__main__":
    fetch_and_train()
