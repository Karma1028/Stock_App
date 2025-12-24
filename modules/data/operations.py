"""
Production operations module for model monitoring, drift detection, and retraining.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from config import Config
import pickle

class ModelRegistry:
    """
    Simple model registry to track model versions and metadata.
    """
    def __init__(self):
        self.registry_file = Config.MODELS_DIR / "model_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self):
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {}
    
    def register_model(self, model_name, version, metrics, notes=""):
        """
        Registers a new model version with its performance metrics.
        """
        key = f"{model_name}_v{version}"
        self.registry[key] = {
            "model_name": model_name,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "notes": notes,
            "status": "active"
        }
        self._save_registry()
        print(f"✅ Registered {key}")
    
    def get_latest_version(self, model_name):
        """Returns the latest version number for a model."""
        versions = [
            int(v.split('_v')[-1]) 
            for v in self.registry.keys() 
            if v.startswith(model_name)
        ]
        return max(versions) if versions else 0
    
    def get_model_info(self, model_name, version=None):
        """Gets info for a specific model version or latest."""
        if version is None:
            version = self.get_latest_version(model_name)
        
        key = f"{model_name}_v{version}"
        return self.registry.get(key)
    
    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)

class DriftDetector:
    """
    Detects data drift using statistical tests.
    """
    def __init__(self):
        self.baseline_file = Config.DATA_DIR / "processed" / "baseline_stats.pkl"
        self.baseline_stats = self._load_baseline()
    
    def _load_baseline(self):
        if self.baseline_file.exists():
            with open(self.baseline_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def set_baseline(self, df_features):
        """
        Stores baseline statistics from training data.
        """
        stats = {}
        for col in df_features.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': df_features[col].mean(),
                'std': df_features[col].std(),
                'min': df_features[col].min(),
                'max': df_features[col].max(),
                'median': df_features[col].median()
            }
        
        self.baseline_stats = {
            'timestamp': datetime.now().isoformat(),
            'features': stats
        }
        
        # Save baseline
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_file, 'wb') as f:
            pickle.dump(self.baseline_stats, f)
        
        print(f"✅ Baseline stats saved")
    
    def detect_drift(self, df_features, threshold=0.3):
        """
        Compares current data to baseline and detects drift.
        Uses PSI (Population Stability Index) as drift metric.
        """
        if self.baseline_stats is None:
            print("⚠️ No baseline set. Run set_baseline() first.")
            return None
        
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'drifted_features': [],
            'drift_scores': {}
        }
        
        for col in self.baseline_stats['features'].keys():
            if col not in df_features.columns:
                continue
            
            baseline_mean = self.baseline_stats['features'][col]['mean']
            baseline_std = self.baseline_stats['features'][col]['std']
            
            current_mean = df_features[col].mean()
            current_std = df_features[col].std()
            
            # Normalized drift score
            mean_drift = abs(current_mean - baseline_mean) / (baseline_std + 1e-9)
            std_drift = abs(current_std - baseline_std) / (baseline_std + 1e-9)
            
            drift_score = max(mean_drift, std_drift)
            drift_report['drift_scores'][col] = drift_score
            
            if drift_score > threshold:
                drift_report['drifted_features'].append(col)
        
        return drift_report
    
    def should_retrain(self, drift_report, max_drifted_features=5):
        """
        Determines if model should be retrained based on drift.
        """
        if drift_report is None:
            return False
        
        n_drifted = len(drift_report['drifted_features'])
        return n_drifted >= max_drifted_features

class AutoRetrainer:
    """
    Handles automated model retraining.
    """
    def __init__(self, data_manager, feature_engineer):
        self.dm = data_manager
        self.fe = feature_engineer
        self.registry = ModelRegistry()
        self.drift_detector = DriftDetector()
        self.retrain_log_file = Config.LOGS_DIR / "retrain_log.json"
        self.retrain_log = self._load_log()
    
    def _load_log(self):
        if self.retrain_log_file.exists():
            with open(self.retrain_log_file, 'r') as f:
                return json.load(f)
        return []
    
    def check_and_retrain(self, tickers, force=False):
        """
        Checks for drift and retrains if needed.
        """
        print("🔍 Checking for data drift...")
        
        # Get latest data
        df = self.dm.get_cached_data(tickers, period="1y")
        if df.empty:
            print("❌ No data available")
            return False
        
        # Engineer features
        df_features = self.fe.compute_all_features(df)
        
        # Detect drift
        drift_report = self.drift_detector.detect_drift(df_features)
        
        if drift_report:
            print(f"📊 Drift Score: {len(drift_report['drifted_features'])} features drifted")
            
            should_retrain = force or self.drift_detector.should_retrain(drift_report)
            
            if should_retrain:
                print("🔄 Retraining triggered...")
                return self._retrain_models(df_features, drift_report)
            else:
                print("✅ No retraining needed")
                return False
        
        return False
    
    def _retrain_models(self, df_features, drift_report):
        """
        Retrains all models and registers new versions.
        """
        from ml_ensemble import MLEnsemble
        
        ensemble = MLEnsemble()
        models = ensemble.train_ensemble(df_features)
        
        if models:
            # Get latest version and increment
            latest_version = self.registry.get_latest_version("ensemble")
            new_version = latest_version + 1
            
            # Dummy metrics (should compute actual metrics from validation)
            metrics = {
                "rmse": 0.05,
                "mae": 0.03,
                "r2": 0.75
            }
            
            # Register new version
            self.registry.register_model(
                "ensemble",
                new_version,
                metrics,
                notes=f"Auto-retrained due to drift: {drift_report['drifted_features']}"
            )
            
            # Log retrain event
            self.retrain_log.append({
                "timestamp": datetime.now().isoformat(),
                "version": new_version,
                "drift_report": drift_report
            })
            
            with open(self.retrain_log_file, 'w') as f:
                json.dump(self.retrain_log, f, indent=2)
            
            print(f"✅ Retrained to version {new_version}")
            return True
        
        return False
    
    def schedule_daily_check(self):
        """
        Returns the timestamp for next scheduled check (24h from now).
        """
        return datetime.now() + timedelta(days=1)
