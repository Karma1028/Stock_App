# 🏗️ Tata Motors Deep Dive — Step-by-Step Recreation Guide

**Author:** Tuhin Bhattacharya  
**Date:** February 2025  
**Project:** Comprehensive Quantitative Analysis of Tata Motors Post-Demerger

---

## Project Overview

This guide walks through the complete **13-notebook analysis pipeline** for Tata Motors, from raw data extraction to final investment thesis. Each section describes what the notebook does, why each step matters, the methodology used, and the key findings.

### Project Universe

| Ticker | Description | Role |
|--------|-------------|------|
| TMCV.NS | Tata Motors Commercial Vehicles | Primary analysis target |
| TMPV.NS | Tata Motors Passenger Vehicles | Demerger pair comparison |
| MARUTI.NS | Maruti Suzuki | Peer benchmark |
| ^NSEI | NIFTY 50 | Broad market benchmark |
| ^CNXAUTO | NIFTY Auto | Sector benchmark |

### Prerequisites

- **Python 3.8+**
- **Core Libraries:** pandas, numpy, matplotlib, seaborn, scipy
- **ML Libraries:** scikit-learn, xgboost, lightgbm, optuna
- **Specialized:** yfinance, prophet, textblob, reportlab

---

## 📓 Notebook 01: Data Extraction

### Objective
Fetch 5 years of daily OHLCV (Open, High, Low, Close, Volume) data for Tata Motors and its peers using the `yfinance` library.

### Key Context: The Demerger
In **January 2025**, Tata Motors executed a landmark demerger, splitting into two independent listed entities:
- **TMCV.NS** — Commercial Vehicles (trucks, buses)
- **TMPV.NS** — Passenger Vehicles (Nexon, Punch, Harrier, EVs)

The original ticker **TATAMOTORS.NS was delisted**, meaning all analysis must use the new tickers. This is the foundational challenge of this project.

### Methodology
1. **Define the ticker universe** — 5 tickers covering the demerged entities, a peer, and two benchmarks
2. **Fetch data with fallbacks** — Use `yf.download()` with `period='5y'` and handle multi-index columns from yfinance
3. **Raw data inspection** — Check shape, date range, missing values, and basic statistics
4. **Visualize normalized prices** — Base-100 normalization to compare performance across different price scales
5. **Identify market regimes** — Tag each date into one of 5 regimes based on key events

### Market Regimes Identified

| Regime | Period | Key Event |
|--------|--------|-----------|
| Pre-COVID | Before March 2020 | Normal market conditions |
| COVID Crash | March–May 2020 | Pandemic lockdowns, 40%+ drawdown |
| Recovery | May 2020–Dec 2021 | Stimulus-driven rally |
| Post-COVID | 2022–Sep 2024 | Normalization, inflation concerns |
| Oct 2024 Event | Oct 2024+ | Ratan Tata's passing, sentiment shock |

### Key Findings
- TMCV and TMPV have limited trading history (post-demerger only)
- Volume spikes align with major news events
- Tata Motors showed higher volatility than NIFTY 50, especially during regime transitions

---

## 📓 Notebook 02: Data Cleaning & Preprocessing

### Objective
Audit and clean the raw data to create a pristine dataset ready for feature engineering.

### Methodology
1. **Missing value audit** — Create a heatmap of missing values across all columns for each ticker
2. **Forward-fill strategy** — Apply `ffill()` for minor gaps caused by trading halts or holidays
3. **Interpolation** — Use linear interpolation for any remaining gaps that forward-fill can't handle
4. **Regime tagging** — Assign each trading day to one of the 5 market regimes defined in Notebook 01
5. **Multi-stock merge** — Combine cleaned data from all tickers into a single DataFrame with suffixed column names

### Key Findings
- The cleaning process is **non-distortive** — forward-fill preserves the natural price continuity
- Volatility differs significantly across regimes: COVID Crash showed 3-5x higher daily volatility than Pre-COVID
- The merged dataset provides a unified view for cross-asset correlation analysis

---

## 📓 Notebook 03: Feature Engineering — Technical Indicators

### Objective
Calculate and visualize standard technical indicators that capture momentum, trend, and volatility.

### Indicators Computed

| Indicator | Formula | Purpose |
|-----------|---------|---------|
| **RSI (14)** | `100 - 100/(1 + AvgGain/AvgLoss)` | Overbought (>70) / Oversold (<30) detection |
| **MACD** | `EMA(12) - EMA(26)`, Signal = `EMA(9) of MACD` | Trend direction and momentum |
| **Bollinger Bands** | `SMA(20) ± 2σ` | Volatility envelope and mean-reversion zones |
| **OBV** | Cumulative signed volume | Volume-price confirmation |
| **ATR (14)** | Average True Range | Volatility magnitude for position sizing |
| **SMA Cross** | SMA(20) vs SMA(50) | Trend direction signal |

### Methodology
1. **Manual calculation** of each indicator from raw OHLCV to understand the math
2. **Verification** against library implementations (e.g., `ta` library) to ensure correctness
3. **Visualization** of each indicator alongside price to see real-world behavior
4. **Event analysis** — examine how indicators behaved during COVID Crash and Oct 2024 event

### Key Findings
- **RSI** dropped below 20 during the COVID crash — deep oversold territory, followed by a sharp recovery
- **MACD** generated a bearish crossover ~2 weeks before the Oct 2024 event — a potential early warning
- **Bollinger Band width** expands dramatically during regime transitions, signaling increased uncertainty
- **OBV divergence** from price occasionally preceded trend reversals

---

## 📓 Notebook 04: Feature Engineering — Statistical Features

### Objective
Create statistical features that capture the distributional properties of returns, going beyond simple price-based indicators.

### Features Computed

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| **Log Returns** | `ln(Close_t / Close_{t-1})` | Additive property, better for statistics |
| **Rolling Volatility** | 5d, 21d, 63d annualized std | Risk measurement at multiple horizons |
| **Rolling Skewness** | 21-day rolling skew | Tail asymmetry detection |
| **Rolling Kurtosis** | 21-day rolling kurtosis | Fat-tail risk measurement |
| **Z-Score** | `(Price - SMA) / Std` | Mean-reversion signal |
| **Autocorrelation** | Lag-1, Lag-5 correlation | Momentum persistence |

### Key Findings
- Tata Motors returns are **not normally distributed** — they exhibit fat tails (positive kurtosis ≈ 5-8) and negative skewness
- This means extreme downside moves occur more often than a bell curve predicts — crucial for risk management
- Rolling volatility shows **volatility clustering**: high-volatility periods tend to persist
- Feature correlation analysis revealed multicollinearity between some technical and statistical features, informing the feature selection in Notebook 09

---

## 📓 Notebook 05: Exploratory Data Analysis (EDA)

### Objective
Comprehensive visual exploration of Tata Motors data across multiple dimensions: time, distribution, regime, and cross-stock comparison.

### Analysis Performed
1. **Univariate analysis** — Price history, return distributions, histogram vs normal overlay, QQ plot
2. **Regime analysis** — Overlay market regime bands on price chart, compare return distributions per regime
3. **Price-Volume dynamics** — Scatter plot of returns vs volume, identify volume spikes
4. **Seasonal patterns** — Monthly returns heatmap, day-of-week effects
5. **Cross-stock comparison** — Correlation matrix (TMCV, TMPV, Maruti, NIFTY50, NIFTY Auto)
6. **October 2024 deep dive** — Isolated analysis of the Ratan Tata event's impact on price, volume, and volatility

### Key Findings
- Returns are **non-normal**: heavy tails visible in QQ plot, Jarque-Bera test rejects normality
- **Festive season effect**: September-November sometimes shows strength (Navratri, Diwali vehicle sales)
- **Correlation with NIFTY Auto** is high (~0.7-0.8), but correlation with NIFTY 50 is moderate (~0.5-0.6)
- The Oct 2024 event caused a sentiment-driven selloff with volume 3-5x the average

---

## 📓 Notebook 06: Sentiment Deep Dive

### Objective
Analyze the relationship between news sentiment and Tata Motors' price movements using NLP.

### Methodology
1. **Curate news headlines** — Collect Tata Motors-specific news headlines covering key events
2. **TextBlob analysis** — Compute polarity (-1 to +1) and subjectivity (0 to 1) scores
3. **VADER analysis** — Use the VADER (Valence Aware Dictionary and sEntiment Reasoner) model, specifically designed for social media/news text
4. **Model comparison** — Compare TextBlob vs VADER on the same headlines
5. **Sentiment-price correlation** — Correlate daily sentiment scores with price returns
6. **Event sentiment analysis** — Deep dive into sentiment around Oct 2024 event
7. **Word frequency analysis** — Identify most common positive/negative terms

### Key Findings
- **VADER outperforms TextBlob** for financial text — VADER's lexicon includes financial terms and handles negation better
- Sentiment-return correlation is **weak but measurable** (~0.05-0.15) — sentiment is a confirming indicator, not a primary signal
- During the Oct 2024 event, sentiment shifted sharply negative, but price had already begun falling — sentiment was lagging, not leading
- Sentiment is most useful as a **regime indicator** rather than a daily trading signal

---

## 📓 Notebook 07: Market Phase Clustering

### Objective
Use unsupervised machine learning to identify distinct market phases without relying on manual regime definitions.

### Methodology
1. **Feature selection for clustering** — Log returns, volume ratio, RSI, Bollinger Band width, MACD histogram, rolling volatility
2. **Standardization** — StandardScaler to normalize features to zero mean, unit variance
3. **Elbow method** — Plot inertia vs K (2-10) to find optimal cluster count
4. **Silhouette analysis** — Validate cluster quality using silhouette scores
5. **K-Means clustering** — Apply K-Means with K=3 (or K=4 based on elbow/silhouette)
6. **PCA visualization** — Project clusters into 2D using Principal Component Analysis
7. **Cluster profiling** — Analyze the characteristics (mean return, mean volatility, mean volume) of each cluster
8. **Transition analysis** — Study how the stock transitions between clusters over time

### Cluster Profiles Identified

| Cluster | Name | Characteristics |
|---------|------|-----------------|
| 0 | Calm Trending | Low volatility, consistent direction, normal volume |
| 1 | Volatile Breakout | High volatility, large price moves, elevated volume |
| 2 | Mean-Reverting | Choppy, range-bound, moderate volatility |

### Key Findings
- The unsupervised clusters **align well** with the manually defined market regimes
- COVID Crash and Oct 2024 event days predominantly fall in the "Volatile Breakout" cluster
- Cluster transitions provide early signals — movement from Cluster 0 to Cluster 1 often precedes major moves

---

## 📓 Notebook 08: Model Baseline Comparison

### Objective
Compare multiple ML classification models for predicting Tata Motors' next-day price direction (Up/Down).

### Models Tested

| Model | Type | Key Characteristics |
|-------|------|---------------------|
| Logistic Regression | Linear | Baseline, interpretable |
| Random Forest | Ensemble (Bagging) | Handles non-linearity, robust |
| XGBoost | Ensemble (Boosting) | Gradient boosting champion |
| LightGBM | Ensemble (Boosting) | Fast, memory-efficient |

### Methodology
1. **Target variable** — Binary: 1 if next-day return > 0, else 0
2. **Feature set** — All technical and statistical features from notebooks 03-04
3. **Time-series aware splitting** — `TimeSeriesSplit` with 5 folds to prevent look-ahead bias
4. **Standardization** — StandardScaler fitted only on training data
5. **Evaluation metrics** — Accuracy, Precision, Recall, F1-Score, AUC-ROC
6. **Feature importance** — Extract and rank feature importances from tree-based models

### Key Findings
- **XGBoost achieves ~55% accuracy** — marginally the best, but all tree-based models perform similarly
- 55% may seem modest, but in efficient markets, **any consistent edge above 51% is statistically significant**
- Top predictive features: **Volume, rolling volatility, RSI, MACD histogram**
- Logistic Regression underperforms — the prediction task is inherently non-linear

---

## 📓 Notebook 09: Iterative Feature Selection

### Objective
Reduce the feature set from 30+ to an optimal subset that maximizes predictive performance while minimizing overfitting.

### Three-Stage Pipeline

**Stage 1 — Variance Threshold:**
Remove features with near-zero variance (constants or near-constant columns) that carry no discriminative power.

**Stage 2 — Correlation Filtering (|r| > 0.9):**
Identify highly correlated feature pairs (e.g., SMA_20 and EMA_20). Keep the one with higher individual importance. Remove the redundant feature.

**Stage 3 — Recursive Feature Elimination (RFE):**
Using Random Forest as the estimator, iteratively remove the least important feature, retrain, and evaluate accuracy. Plot accuracy vs feature count to find the optimal subset size.

### SHAP Analysis
SHAP (SHapley Additive exPlanations) values decompose each individual prediction into feature contributions, answering "why did the model predict UP for this specific day?"

### Key Findings
- The optimal subset is **10-15 features** (60-70% reduction from 30+)
- Accuracy is maintained or slightly improved with fewer features
- **Volume** and **rolling volatility** have the highest SHAP impact
- Some features with high simple importance (like SMA crossovers) have low SHAP impact — their importance was inflated by correlation

---

## 📓 Notebook 10: Hyperparameter Tuning

### Objective
Optimize model hyperparameters using Bayesian optimization to squeeze maximum performance.

### Methodology
1. **Optuna framework** — Uses Tree-structured Parzen Estimator (TPE) for intelligent hyperparameter sampling
2. **Random Forest search space:**
   - `n_estimators`: 50–500
   - `max_depth`: 3–20
   - `min_samples_split`: 2–20
   - `min_samples_leaf`: 1–10
3. **XGBoost search space:**
   - `learning_rate`: 0.001–0.3
   - `max_depth`: 3–12
   - `subsample`: 0.5–1.0
   - `colsample_bytree`: 0.5–1.0
   - `reg_alpha` / `reg_lambda`: L1/L2 regularization
4. **100 trials** per model with TimeSeriesSplit cross-validation
5. **Learning curve analysis** — Plot training vs validation accuracy against training set size

### Key Findings
- Best XGBoost config: learning_rate ≈ 0.01, max_depth ≈ 5, subsample ≈ 0.7
- Tuning yields **1-3% accuracy improvement** over defaults
- Learning curves show moderate variance (gap between train and validation), reduced by regularization
- Diminishing returns beyond ~100 Optuna trials

---

## 📓 Notebook 11: Time-Series Forecasting with Prophet

### Objective
Generate a probabilistic price forecast using Facebook Prophet's decomposition model.

### Prophet Decomposition
**y(t) = g(t) + s(t) + h(t) + ε**

| Component | Meaning | Implementation |
|-----------|---------|----------------|
| g(t) | Trend | Piecewise linear growth |
| s(t) | Seasonality | Fourier series (weekly + yearly) |
| h(t) | Holidays/Events | Indian market holidays |
| ε | Error | Irreducible noise |

### Methodology
1. **Data preparation** — Convert to Prophet format (ds, y columns)
2. **Train-test split** — Last 90 trading days as test set (includes Oct 2024 event)
3. **Basic model** — Weekly + yearly seasonality, changepoint_prior_scale=0.05
4. **Enhanced model** — Add log-transformed Volume as external regressor
5. **Forecast generation** — 90-day forecast with uncertainty intervals
6. **Changepoint detection** — Identify dates where the trend rate shifted significantly
7. **Accuracy evaluation** — MAE, MAPE, RMSE, R²
8. **Cross-validation** — initial=365 days, period=90 days, horizon=30 days

### Key Findings
- **Changepoints align with real events** — COVID lockdowns, recovery rally, EV announcements, Ratan Tata's passing
- **Confidence intervals widen** significantly beyond 15 days — reflecting genuine forecast uncertainty
- Typical MAPE: 5-20% for volatile stocks like Tata Motors
- Volume regressor effect is **regime-dependent** — helps in some periods, hurts in others
- **Practical use:** Trend guidance and entry/exit zone identification, not trading signals

---

## 📓 Notebook 12: Trading Strategy Backtesting

### Objective
Convert ML predictions into a tradable strategy and evaluate profitability vs Buy-and-Hold.

### Strategy Design

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Signal | RF probability > 0.5 → BUY | Simple binary threshold |
| Position | Long-only (no shorting) | Conservative approach |
| Transaction cost | 0.1% per trade | Brokerage + STT + GST |
| Starting capital | ₹100,000 | Standard amount |
| Retrain period | 63 days (quarterly) | Balance freshness vs stability |
| Min training window | 252 days (1 year) | Enough for pattern learning |

### Walk-Forward Backtesting
The model is retrained every quarter using an expanding window. This simulates real-world conditions where you periodically update your model with new data.

### Metrics Computed
- **Total Return** — End-to-end portfolio growth
- **Annualized Return** — Geometric mean annual return
- **Sharpe Ratio** — Risk-adjusted return (excess return / volatility)
- **Sortino Ratio** — Like Sharpe, but only penalizes downside volatility
- **Maximum Drawdown** — Worst peak-to-trough decline
- **Win Rate** — Percentage of profitable trading days

### Key Findings
- **Risk reduction > Return enhancement** — The ML strategy's primary value is lower drawdowns, not higher returns
- The strategy adds most value during **crash periods** (moving to cash)
- During strong uptrends, Buy-and-Hold may outperform (the model sometimes exits too early)
- **Worst-days analysis:** Avoiding just the worst 10 trading days dramatically transforms total returns
- Transaction costs matter — high-frequency signal changes erode the edge
- **Buy-and-Hold is hard to beat** consistently after costs, but the ML overlay improves risk-adjusted metrics

---

## 📓 Notebook 13: Final Synthesis & Investment Thesis

### Objective
Combine all 12 notebooks' findings into a unified signal, investment thesis, and self-critique.

### Composite Signal Integration
**Composite = w₁·S_ML + w₂·S_Tech + w₃·S_Sentiment + w₄·S_Trend**

| Signal | Source | Score Range | Logic |
|--------|--------|-------------|-------|
| Technical (RSI) | NB 03 | -0.8 to +0.8 | RSI < 30 → BUY, RSI > 70 → SELL |
| Trend (SMA) | NB 03/05 | -0.6 to +0.6 | Price > SMA50 > SMA200 → Bullish |
| ML Model | NB 08-10 | Variable | Best accuracy → directional score |
| Momentum | NB 04 | -1.0 to +1.0 | 5d and 21d return momentum |

**Verdict Logic:** Composite > 0.3 → Bullish | 0–0.3 → Slightly Bullish | -0.3–0 → Slightly Bearish | < -0.3 → Bearish

### Investment Thesis Scoring

| Factor | Score | Rationale |
|--------|-------|-----------|
| EV Transition Potential | +7 | India's EV leader, 70%+ EV market share |
| JLR Turnaround | +6 | Luxury segment recovering, margins improving |
| Domestic Market Position | +8 | Strong brand, growing SUV segment |
| Debt Concerns | -4 | High debt-to-equity, but improving |
| Global Macro Risk | -3 | EV subsidy uncertainty, commodity prices |
| Competition Intensity | -5 | Hyundai, MG, BYD entering India EV market |
| Technical Setup | Variable | Based on composite signal |

### Risk Scenarios (12-Month)

| Scenario | Price Multiplier | Rationale |
|----------|-----------------|-----------|
| 🟢 Bull Case | +25% | EV adoption accelerates, JLR margins expand |
| 🟡 Base Case | +5% | Steady growth, sector rotation |
| 🟠 Bear Case | -20% | EV competition intensifies, demand slows |
| 🔴 Tail Risk | -45% | Global recession + credit crisis |

### Self-Critique & Limitations

1. **Survivorship Bias** — We analyzed Tata Motors because it survived; failed companies are invisible
2. **Data Snooping** — Testing many features on the same data increases false discoveries
3. **Look-Ahead Bias** — Despite TimeSeriesSplit, feature engineering uses full history
4. **Transaction Costs** — Real costs (brokerage, impact cost, slippage) are higher than our 0.1% model
5. **Single Stock** — No portfolio diversification analysis performed
6. **Regime Change** — Models trained on past regimes may fail in new ones
7. **Sentiment Data Quality** — News scraping is noisy and incomplete
8. **No Fundamental Analysis** — P/E ratios, debt levels, management quality not included

### Final Recommendation

> **Moderate HOLD with tactical trading overlay.** Use ML signals to adjust position size (not for binary buy/sell decisions). Combine with fundamental analysis for a complete picture.

**Confidence Level: MODERATE.** The stock market is inherently unpredictable. All models are wrong; some are useful.

### Future Enhancements
1. FinBERT for transformer-based financial sentiment
2. LSTM/Transformer models for sequence learning
3. Alternative data (dealer registrations, EV charging data, satellite imagery)
4. Options-implied volatility for forward-looking risk
5. Portfolio construction across auto sector stocks
6. Purged walk-forward cross-validation
7. Intraday data for higher-granularity signals

---

*🎓 End of Tata Motors Deep Dive Recreation Guide*
