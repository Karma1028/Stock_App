# AI Model Creation Process

This document details the end-to-end process used to create the AI models for the Stock Analysis application, specifically the **Global XGBoost Model** (`xgb_model_global.pkl`).

## 1. Data Acquisition
**Source:** Yahoo Finance (`yfinance` library).

**Process:**
- The script `train_models.py` initializes the `StockDataManager`.
- It iterates through a predefined list of stock tickers (e.g., NIFTY 50 stocks).
- For each ticker, it fetches **5 years** of historical OHLCV (Open, High, Low, Close, Volume) data.
- **News Data:** It also fetches recent news (last 30 days) using `NewsScraper` to calculate sentiment scores. Note that for historical training data beyond 30 days, sentiment is assumed to be neutral (0) as historical news is not scraped in bulk.

## 2. Feature Engineering
**Module:** `modules/ml/features.py` (`FeatureEngineer` class)

Raw data is transformed into meaningful features for the model:

### A. Technical Indicators
- **Moving Averages:** SMA_50, SMA_200 (Trend indicators).
- **Momentum:** RSI (Relative Strength Index).
- **MACD:** MACD Line, Signal Line, MACD Histogram (Trend and Momentum).
- **Volatility:** Bollinger Bands (High/Low), ATR (Average True Range).
- **Volume:** Volume Change, Volume Z-Score.

### B. Lag & Return Features
- **Returns:** 1-day, 5-day, and 21-day percentage returns.
- **Log Returns:** Logarithmic returns for better statistical properties.
- **Volatility:** 21-day rolling volatility (standard deviation of returns).
- **Moving Average of Returns:** 21-day rolling mean of returns.

### C. Time Features
- **DayOfWeek**: Cyclical patterns in the week.
- **Month**: Seasonal patterns.

### D. Sentiment Features
- **Sentiment Score**: Aggregated daily sentiment from news ( -1 to 1).
- **Sentiment Volatility**: Stability of sentiment over time.

### E. Target Variable
- **Target_5d**: The model is trained to predict the **5-day future return** of the stock.
  - Calculated as: `(Close_t+5 - Close_t) / Close_t`

## 3. Data Processing & Cleaning
**Module:** `modules/ml/engine.py` (`MLEngine.prepare_data`)

Before training, the data undergoes strict cleaning:
1.  **Feature Selection**: Only the specific list of engineered features is selected.
2.  **Infinity Handling**: Any infinite values (`inf`, `-inf`) resulting from division by zero are replaced with `NaN`.
3.  **NaN Removal**: Rows containing `NaN` values (common at the start of data due to rolling windows like SMA_200) are dropped.
4.  **Alignment**: The Feature set (X) and Target set (y) are aligned to ensure no data leakage or mismatch.

## 4. Model Training
**Script:** `train_models.py`
**Engine:** `modules/ml/engine.py`

### Global XGBoost Model (`xgb_model_global.pkl`)
- **Algorithm**: **XGBoost Regressor** (`xgboost` library).
- **Objective**: `reg:squarederror` (Minimizing Mean Squared Error).
- **Training Data**: Aggregated data from all 50+ stocks is concatenated into a single large dataset (`data/training_data.csv`).
- **Train/Test Split**: 
  - **Time-based split**: The first 80% of data (chronologically) is used for training, and the last 20% is used for validation. This respects the temporal nature of stock data.
- **Hyperparameters**:
  - `n_estimators`: 500 (Maximum trees).
  - `learning_rate`: 0.05.
  - `max_depth`: 6.
  - `early_stopping_rounds`: 10 (Stops training if validation error doesn't improve).
- **Output**: The trained model is saved as a pickle file: `models/xgb_model_global.pkl`.

## 5. How to Re-create the Model
To generate the model files yourself:

1.  **Ensure Dependencies**: Install `yfinance`, `pandas`, `ta`, `xgboost`, `scikit-learn`.
2.  **Run the Script**:
    ```bash
    python train_models.py
    ```
3.  **Outputs**:
    - `models/xgb_model_global.pkl`: The trained model file.
    - `data/training_data.csv`: The raw training data used (for inspection).

This process ensures a robust, data-driven model that learns from a diverse set of market conditions across multiple stocks.
