# Refactoring and Training Complete

## 1. Code Reorganization
The codebase has been refactored into a modular structure under `modules/`:
- **`modules/ml/`**: Machine Learning engines, features, prediction models, and deep learning.
- **`modules/data/`**: Data managers, portfolio operations, and data fetching logic.
- **`modules/ui/`**: UI components like sidebar and plots.
- **`modules/utils/`**: Helper functions, PDF generation, AI insights, and Quant engine.

## 2. Model Training
A global LightGBM model has been trained on **5 years of historical stock data** and recent news sentiment.
- **Script**: `train_models.py`
- **Output**: `models/lgb_model_global.pkl`
- **Data Source**: 48 Nifty 50 stocks.

## 3. How to Retrain
To retrain the model with fresh data:
1. Open a terminal.
2. Run: `python train_models.py`
   (Ensure you are in the `stock_app` directory and have the environment activated).

## 4. Application Updates
- `pages/stock_analysis.py` now uses the pre-trained global model for faster and more robust predictions.
- Imports across the application have been updated to reflect the new directory structure.

## 5. Next Steps
- You can now run the application using `streamlit run app.py`.
- The "Stock Analysis" page will automatically load the global model.
