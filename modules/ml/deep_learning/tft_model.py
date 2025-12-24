"""
Temporal Fusion Transformer for time-series forecasting.
Based on the TFT architecture for multi-horizon predictions.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import Config
import pickle

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping
    HAS_TFT = True
except ImportError:
    HAS_TFT = False
    print("⚠️ Install pytorch-forecasting for TFT: pip install pytorch-forecasting pytorch-lightning")

class TFTForecaster:
    """
    Temporal Fusion Transformer implementation for stock price forecasting.
    """
    def __init__(self):
        self.models_dir = Config.MODELS_DIR
        self.model = None
        self.training = None
        
    def prepare_dataset(self, df_features, max_encoder_length=60, max_prediction_length=5):
        """
        Prepares TimeSeriesDataSet for TFT training.
        """
        if not HAS_TFT:
            raise ImportError("pytorch-forecasting not installed")
        
        # Reset index to have a time index
        df = df_features.reset_index()
        df['time_idx'] = np.arange(len(df))
        
        # Add group identifier (for multiple stocks, but single stock for now)
        df['group'] = 'stock_1'
        
        # Define static and time-varying features
        time_varying_known = ['DayOfWeek', 'Month']  # Known future values
        time_varying_unknown = [
            'Returns_1d', 'Vol_21d', 'RSI', 'MACD', 'sentiment_score'
        ]  # Unknown future values
        
        # Filter to available columns
        time_varying_known = [c for c in time_varying_known if c in df.columns]
        time_varying_unknown = [c for c in time_varying_unknown if c in df.columns]
        
        # Create TimeSeriesDataSet
        training = TimeSeriesDataSet(
            df[lambda x: x.time_idx <= df['time_idx'].max() - max_prediction_length],
            time_idx='time_idx',
            target='Target_5d',
            group_ids=['group'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=time_varying_known,
            time_varying_unknown_reals=time_varying_unknown,
            target_normalizer=None,  # Will use default
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        return training, df
    
    def train_tft(self, df_features, epochs=10, gpus=0):
        """
        Trains the Temporal Fusion Transformer.
        """
        if not HAS_TFT:
            print("TFT not available. Install pytorch-forecasting first.")
            return None
        
        print("🚀 Training Temporal Fusion Transformer...")
        
        # Prepare dataset
        self.training, df_prepared = self.prepare_dataset(df_features)
        
        # Create validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            self.training, 
            df_prepared, 
            predict=True, 
            stop_randomization=True
        )
        
        # Create dataloaders
        batch_size = 64
        train_dataloader = self.training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        
        # Configure TFT
        self.model = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=0.03,
            hidden_size=16,  # Smaller for faster training
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        
        # Train
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=False, mode="min")
        
        trainer = Trainer(
            max_epochs=epochs,
            gpus=gpus,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback],
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        
        trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # Save model
        model_path = self.models_dir / "tft_model.ckpt"
        trainer.save_checkpoint(str(model_path))
        
        print(f"✅ TFT model saved to {model_path}")
        return self.model
    
    def predict_tft(self, df_features):
        """
        Generates predictions using trained TFT.
        """
        if self.model is None:
            print("No model loaded. Train or load a model first.")
            return None
        
        # Prepare dataset for prediction
        _, df_prepared = self.prepare_dataset(df_features)
        
        # Get predictions
        predictions = self.model.predict(
            df_prepared,
            mode="prediction",
            return_x=True
        )
        
        return predictions
    
    def load_model(self, checkpoint_path=None):
        """
        Loads a saved TFT model.
        """
        if checkpoint_path is None:
            checkpoint_path = self.models_dir / "tft_model.ckpt"
        
        if not checkpoint_path.exists():
            print(f"Model not found at {checkpoint_path}")
            return None
        
        self.model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path))
        print(f"✅ Model loaded from {checkpoint_path}")
        return self.model
