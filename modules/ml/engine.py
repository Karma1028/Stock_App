
import pandas as pd
import numpy as np
import pickle
from config import Config
from datetime import datetime
import os

# Define standard features expected by the model
FEATURE_COLS = [
    'Returns_1d', 'Returns_5d', 'Returns_21d', 'Log_Returns',
    'Vol_21d', 'MA_21d',
    'SMA_50', 'SMA_200',
    'MACD', 'MACD_Signal', 'MACD_Diff',
    'RSI',
    'BB_High', 'BB_Low', 'ATR',
    'Vol_Change', 'Vol_MA_20', 'Vol_Z',
    'sentiment_score'
]

class MLEngine:
    def __init__(self, model_name="lgb_model_global.pkl"):
        self.models_dir = Config.MODELS_DIR
        self.model_file = self.models_dir / model_name
        
    def prepare_data(self, df):
        """
        Prepares data for training or prediction by selecting features.
        """
        # Ensure columns exist
        available_cols = [c for c in FEATURE_COLS if c in df.columns]
        
        if not available_cols:
            return pd.DataFrame(), None # Return empty if no features match
            
        data = df[available_cols].copy()
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # If training, we need Target (Target_5d)
        target = None
        if 'Target_5d' in df.columns:
            target = df['Target_5d']
            # Align data and target (drop NaNs)
            # Use intersection of indices where both are valid
            valid_idx = data.dropna().index.intersection(target.dropna().index)
            data = data.loc[valid_idx]
            target = target.loc[valid_idx]
        else:
            # Prediction mode: just drop NaNs in features
            data = data.dropna()
            
        return data, target

    def train_model(self, df, model_type="xgboost"):
        """
        Trains a model (XGBoost or LightGBM) to predict 5-day returns.
        """
        print(f"Training {model_type} Model...")
        X, y = self.prepare_data(df)
        
        if X.empty or len(X) < 100:
            print("Not enough data to train.")
            return None
        
        if y is None:
            print("No target (Target_5d) found in data. Cannot train.")
            return None
        
        # Simple Train/Test Split (Time-based)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if model_type == "xgboost":
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    early_stopping_rounds=10
                )
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    verbose=False
                )
            except ImportError:
                print("XGBoost not installed. Skipping.")
                return None
            
        else: # LightGBM
            try:
                import lightgbm as lgb
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'verbosity': -1
                }
                
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[valid_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=10)]
                )
            except ImportError:
                print("LightGBM not installed. Skipping.")
                return None
        
        # Save model
        if not self.models_dir.exists():
             self.models_dir.mkdir(parents=True, exist_ok=True)

        with open(self.model_file, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"Model saved to {self.model_file}")
        return model

    def load_model(self):
        if self.model_file.exists():
            with open(self.model_file, 'rb') as f:
                return pickle.load(f)
        return None

    def predict(self, df):
        """
        Generates predictions for the latest data points.
        """
        model = self.load_model()
        if not model:
            print("Model not found. Please train first.")
            return None
            
        X, _ = self.prepare_data(df)
        if X.empty:
            return None
            
        # Handle different model types
        try:
            # XGBoost (sklearn API)
            preds = model.predict(X)
        except Exception:
            # LightGBM (native API) might need different input? 
            # lgb.train returns a Booster, which accepts numpy or DF.
            try:
                preds = model.predict(X)
            except Exception as e:
                print(f"Prediction error: {e}")
                return None
            
        return pd.Series(preds, index=X.index)

    def calculate_combined_score(self, df, predictions):
        """
        Calculates a 0-100 score based on:
        1. Model Prediction (Predicted Return)
        2. Technical Strength (RSI, MACD)
        3. Sentiment Score
        """
        if df.empty or predictions is None or predictions.empty:
            return {
                "combined_score": 50.0,
                "prediction_score": 50.0,
                "technical_score": 50.0,
                "sentiment_score": 50.0,
                "predicted_return_pct": 0.0
            }

        # Get latest row
        latest = df.iloc[-1]
        try:
             pred_return = predictions.iloc[-1]
        except:
             pred_return = 0

        # 1. Prediction Score (Normalize -5% to +5% -> 0 to 100)
        # Make it more aggressive: 1% return -> High Score
        # Sigmoid-like scaling centered at 0
        score_scaling = 1500 # Multiplier for return
        pred_score = 50 + (pred_return * score_scaling)
        pred_score = np.clip(pred_score, 0, 100)
        
        # 2. Technical Score
        rsi = latest.get('RSI', 50)
        if pd.isna(rsi): rsi = 50
        
        macd_diff = latest.get('MACD_Diff', 0)
        if pd.isna(macd_diff): macd_diff = 0
        
        sma_50 = latest.get('SMA_50', 0)
        sma_200 = latest.get('SMA_200', 0)
        
        tech_score = rsi 
        if macd_diff > 0:
            tech_score += 15
        else:
            tech_score -= 10
            
        if sma_50 > sma_200:
            tech_score += 10
            
        tech_score = np.clip(tech_score, 0, 100)
        
        # 3. Sentiment Score (-1 to 1 -> 0 to 100)
        sent = latest.get('sentiment_score', 0)
        if pd.isna(sent): sent = 0
        
        # Boost sentiment impact
        sent_score = (sent + 1) / 2 * 100
        if sent > 0.1:
            sent_score += 10
        sent_score = np.clip(sent_score, 0, 100)
        
        # Weighted Average
        # Prediction: 40%, Technical: 40%, Sentiment: 20%
        final_score = (pred_score * 0.4) + (tech_score * 0.4) + (sent_score * 0.2)
        
        return {
            "combined_score": float(round(final_score, 2)),
            "prediction_score": float(round(pred_score, 2)),
            "technical_score": float(round(tech_score, 2)),
            "sentiment_score": float(round(sent_score, 2)),
            "predicted_return_pct": float(round(pred_return * 100, 2))
        }
