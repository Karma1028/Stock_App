"""
Ensemble ML Engine with XGBoost, CatBoost, and SHAP explainability.
"""
import pandas as pd
import numpy as np
import pickle
from config import Config
from datetime import datetime

try:
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False
    print("⚠️ Install xgboost and catboost for full ensemble: pip install xgboost catboost")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ Install shap for explainability: pip install shap")

class MLEnsemble:
    def __init__(self):
        self.models_dir = Config.MODELS_DIR
        self.models = {}
        self.meta_model = None
        self.shap_explainer = None
        
    def prepare_data(self, df):
        """
        Prepares data for training/prediction (same as ml_engine).
        """
        feature_cols = [
            'Returns_1d', 'Returns_5d', 'Returns_21d', 'Log_Returns',
            'Vol_21d', 'MA_21d',
            'SMA_50', 'SMA_200', 'MACD', 'MACD_Signal', 'MACD_Diff',
            'RSI', 'BB_High', 'BB_Low', 'ATR',
            'Vol_Change', 'Vol_Z',
            'DayOfWeek', 'Month',
            'sentiment_score', 'sentiment_volatility'
        ]
        
        # Add embedding features if available
        emb_cols = [c for c in df.columns if c.startswith('emb_')]
        feature_cols.extend(emb_cols)
        
        available_cols = [c for c in feature_cols if c in df.columns]
        data = df[available_cols].copy()
        
        if 'Target_5d' in df.columns:
            target = df['Target_5d']
            valid_idx = data.dropna().index.intersection(target.dropna().index)
            return data.loc[valid_idx], target.loc[valid_idx]
        
        return data.dropna(), None

    def train_ensemble(self, df):
        """
        Trains an ensemble of LightGBM, XGBoost, and CatBoost.
        """
        if not HAS_ENSEMBLE:
            print("Ensemble libraries not available. Using LightGBM only.")
            return self._train_single_lgb(df)
        
        print("Training Ensemble Models...")
        X, y = self.prepare_data(df)
        
        if X.empty or len(X) < 100:
            print("Not enough data to train.")
            return None
        
        # Time-based split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 1. LightGBM
        print("Training LightGBM...")
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        
        self.models['lgb'] = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=100,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        
        # 2. XGBoost
        print("Training XGBoost...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9
        }
        
        self.models['xgb'] = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=100,
            evals=[(dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # 3. CatBoost
        print("Training CatBoost...")
        self.models['cat'] = cb.CatBoostRegressor(
            iterations=100,
            learning_rate=0.05,
            depth=6,
            verbose=0
        )
        self.models['cat'].fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=10)
        
        # 4. Meta-learner (simple average for now, can upgrade to stacking)
        print("Creating Meta-Model...")
        # Get out-of-fold predictions
        pred_lgb = self.models['lgb'].predict(X_test)
        pred_xgb = self.models['xgb'].predict(dtest)
        pred_cat = self.models['cat'].predict(X_test)
        
        # Simple average
        self.meta_model = 'average'
        
        # Save models
        self._save_models()
        
        print("✅ Ensemble Training Complete!")
        return self.models
    
    def _train_single_lgb(self, df):
        """Fallback to single LightGBM if ensemble libs not available."""
        from modules.ml.engine import MLEngine
        ml = MLEngine()
        model = ml.train_model(df, model_type="lightgbm")
        self.models['lgb'] = model
        return self.models
    
    def predict_ensemble(self, df):
        """
        Generates predictions using the ensemble.
        """
        if not self.models:
            self._load_models()
        
        X, _ = self.prepare_data(df)
        if X.empty:
            return None
        
        predictions = []
        
        if 'lgb' in self.models:
            pred_lgb = self.models['lgb'].predict(X)
            predictions.append(pred_lgb)
        
        if 'xgb' in self.models:
            dmatrix = xgb.DMatrix(X)
            pred_xgb = self.models['xgb'].predict(dmatrix)
            predictions.append(pred_xgb)
        
        if 'cat' in self.models:
            pred_cat = self.models['cat'].predict(X)
            predictions.append(pred_cat)
        
        # Average predictions
        final_pred = np.mean(predictions, axis=0)
        
        return pd.Series(final_pred, index=X.index)
    
    def explain_prediction(self, df, sample_idx=-1):
        """
        Uses SHAP to explain the model's prediction.
        """
        if not HAS_SHAP:
            return None
        
        X, _ = self.prepare_data(df)
        if X.empty:
            return None
        
        # Use LightGBM for SHAP (fastest)
        if 'lgb' not in self.models:
            self._load_models()
        
        if 'lgb' not in self.models:
            return None
        
        # Create SHAP explainer
        if self.shap_explainer is None:
            self.shap_explainer = shap.TreeExplainer(self.models['lgb'])
        
        # Get SHAP values
        shap_values = self.shap_explainer.shap_values(X)
        
        # Return explanation for the specified sample
        sample = X.iloc[sample_idx]
        sample_shap = shap_values[sample_idx]
        
        explanation = pd.DataFrame({
            'feature': X.columns,
            'value': sample.values,
            'shap_value': sample_shap
        }).sort_values('shap_value', key=abs, ascending=False)
        
        return explanation
    
    def _save_models(self):
        """Saves all models to disk."""
        for name, model in self.models.items():
            model_file = self.models_dir / f"ensemble_{name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
        print(f"Models saved to {self.models_dir}")
    
    def _load_models(self):
        """Loads models from disk."""
        for name in ['lgb', 'xgb', 'cat']:
            model_file = self.models_dir / f"ensemble_{name}.pkl"
            if model_file.exists():
                with open(model_file, 'rb') as f:
                    self.models[name] = pickle.load(f)
