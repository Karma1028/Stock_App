import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add root to path
sys.path.append(os.getcwd())

from modules.data.manager import StockDataManager
from modules.ml.features import FeatureEngineer
from agentic_backend import MonteCarloLSTM
from config import Config

def create_sequences(data, seq_length, target_col_idx):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, target_col_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm_model():
    print("🚀 Starting LSTM Model Training (Nifty 50)...")
    
    dm = StockDataManager()
    fe = FeatureEngineer()
    
    # 1. Fetch Data
    tickers = dm.get_stock_list()
    # For speed/demo, maybe limit to top 10 liquid stocks if Nifty 50 checks take too long
    # But user asked for Nifty 50.
    # We'll use get_cached_data which handles bulk
    print(f"Fetching data for {len(tickers)} tickers...")
    df = dm.get_cached_data(tickers, period="2y") # 2y is enough for LSTM demo, 5y might be huge
    
    if df.empty:
        print("❌ No data found.")
        return

    # 2. Features
    print("Computing features...")
    # Need to handle MultiIndex correctly
    # compute_all_features handles it
    feature_df = fe.compute_all_features(df)
    
    if feature_df.empty:
        print("❌ Feature computation failed.")
        return

    # Drop NaNs
    feature_df = feature_df.dropna()
    print(f"Total samples: {len(feature_df)}")

    # Features to use
    exclude_cols = ['Target_5d', 'Ticker', 'date_only', 'Close', 'Open', 'High', 'Low', 'Volume']
    feature_cols = [c for c in feature_df.columns if c not in exclude_cols]
    
    # Ensure all are numeric
    feature_df = feature_df[feature_cols + ['Target_5d']]
    
    # 3. Scale
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(feature_df[feature_cols])
    
    # Save Scaler
    scaler_path = Config.MODELS_DIR / "lstm_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # 4. Prepare Sequences
    SEQ_LENGTH = 60
    # Target is Target_5d. We want to predict if it's > 0 (Binary classification) or Regression?
    # MonteCarloLSTM ends with Sigmoid -> Binary Classification (Probability of Up)
    # Let's check agentic_backend.py -> it ends with Sigmoid. 
    # Logic: if Target_5d > 0: 1 else 0
    
    targets = (feature_df['Target_5d'] > 0).astype(int).values
    
    # We need to map scaled_data + targets
    # But creating sequences from concatenated heterogeneous stocks is tricky (boundary effects).
    # Correct way: Group by Ticker and create sequences per ticker.
    # Re-fetch with Ticker column
    # feature_df was processed with compute_all_features which adds 'Ticker' but we dropped it above.
    
    # Let's redo with Ticker preservation
    feature_df_full = fe.compute_all_features(df)
    feature_df_full = feature_df_full.dropna()
    
    X_list, y_list = [], []
    
    print("Generating sequences...")
    for ticker in feature_df_full['Ticker'].unique():
        sub_df = feature_df_full[feature_df_full['Ticker'] == ticker]
        if len(sub_df) < SEQ_LENGTH + 5:
            continue
            
        sub_feats = sub_df[feature_cols].values
        sub_targets = (sub_df['Target_5d'] > 0).astype(int).values
        
        # Scale per ticker? Or Global? Global is better for generalization if features are normalized (like RSI)
        # We used global scaler above. Applying transformation.
        sub_feats_scaled = scaler.transform(sub_feats)
        
        xs, ys = create_sequences(sub_feats_scaled, SEQ_LENGTH, -1) # -1 is dummy, we have separate targets
        # wait create_sequences extracts y from data.
        # Custom loop needed
        for i in range(len(sub_feats) - SEQ_LENGTH):
            X_list.append(sub_feats_scaled[i:i+SEQ_LENGTH])
            y_list.append(sub_targets[i+SEQ_LENGTH])
            
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Training Data Shape: {X.shape}")
    
    # Tensor conversion
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 5. Model
    input_size = X.shape[2]
    model = MonteCarloLSTM(input_size=input_size, hidden_size=64, num_layers=2, dropout_rate=0.2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Train
    model.train()
    EPOCHS = 5 # Short for demo
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")
        
    # 7. Save
    model_path = Config.MODELS_DIR / "lstm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ LSTM Model saved to {model_path}")
    print(f"Input Dimension: {input_size}")

if __name__ == "__main__":
    train_lstm_model()
