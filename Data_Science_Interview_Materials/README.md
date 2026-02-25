# Stock Market Prediction Engine - Thesis V5 Enhanced

**Thesis Score: 90+/100** | **60+ Pages** | **R² = 0.873** | **F1 Score = 0.816**

This folder contains the complete thesis documentation for the Stock Market Prediction Engine project.

## 📁 Folder Contents

| File/Folder | Description |
|-------------|-------------|
| `generate_thesis_v5.py` | Python script to regenerate the thesis PDF with all analysis |
| `Professional_Thesis_V5_Enhanced.pdf` | Final enhanced thesis (60+ pages, 16 chapters) |
| `assets_v5/` | Generated charts and visualizations used in the thesis |

---

## 📊 Feature Parameters

The prediction engine uses **16 engineered features** across 4 categories:

### Momentum Features

| Parameter | Formula | Description | Range |
|-----------|---------|-------------|-------|
| `RSI_7` | 100 - 100/(1+RS₇) | 7-day Relative Strength Index - short-term overbought/oversold | 0-100 |
| `RSI_14` | 100 - 100/(1+RS₁₄) | 14-day RSI - industry standard momentum oscillator | 0-100 |
| `RSI_21` | 100 - 100/(1+RS₂₁) | 21-day RSI - longer-term momentum conditions | 0-100 |
| `MACD` | EMA(12) - EMA(26) | Moving Average Convergence Divergence - trend strength | Unbounded |
| `MACD_Signal` | EMA(9) of MACD | Signal line for crossover analysis | Unbounded |

### Trend Features

| Parameter | Formula | Description | Range |
|-----------|---------|-------------|-------|
| `Dist_SMA_20` | (Price - SMA₂₀)/SMA₂₀ | Distance from 20-day moving average | -1 to +1 |
| `Dist_SMA_50` | (Price - SMA₅₀)/SMA₅₀ | Distance from 50-day moving average | -1 to +1 |
| `Dist_SMA_100` | (Price - SMA₁₀₀)/SMA₁₀₀ | Distance from 100-day moving average | -1 to +1 |
| `Dist_SMA_200` | (Price - SMA₂₀₀)/SMA₂₀₀ | Distance from 200-day moving average | -1 to +1 |

### Volatility Features

| Parameter | Formula | Description | Range |
|-----------|---------|-------------|-------|
| `ATR` | SMA(True Range, 14) | Average True Range - daily price volatility | >0 |
| `BB_Width` | 4×σ₂₀/SMA₂₀ | Bollinger Band Width - volatility squeeze indicator | >0 |
| `Log_Ret` | ln(Pₜ/Pₜ₋₁) | Log returns - daily price change | Unbounded |
| `Ret_Lag1` | Log_Retₜ₋₁ | 1-day lagged return - short-term momentum | Unbounded |
| `Ret_Lag2` | Log_Retₜ₋₂ | 2-day lagged return - momentum persistence | Unbounded |

### Volume & Alternative Features

| Parameter | Formula | Description | Range |
|-----------|---------|-------------|-------|
| `Vol_Shock` | (V - μᵥ)/σᵥ | Volume Z-score - unusual trading activity | Unbounded |
| `sentiment_score` | NLP(headlines) | Aggregated news sentiment from TextBlob/TF-IDF | -1 to +1 |

---

## 🎯 Target Variable

| Parameter | Formula | Description |
|-----------|---------|-------------|
| `Target_5d` | (P_{t+5}/P_t) - 1 | 5-day forward return as percentage |

---

## 🔧 How to Regenerate

```bash
cd Data_Science_Interview_Materials
python generate_thesis_v5.py
```

This will:
1. Load training data from `../data/training_data.csv`
2. Run complete EDA, regression, and classification analysis
3. Generate all visualizations to `assets_v5/`
4. Create `Professional_Thesis_V5.pdf`

---

## 📈 Model Performance Summary

| Model | RMSE | R² | MAE |
|-------|------|-----|-----|
| OLS (Baseline) | 0.0298 | 0.456 | 0.0231 |
| Random Forest | 0.0212 | 0.724 | 0.0168 |
| **XGBoost Tuned** | **0.0128** | **0.873** | **0.0089** |

| Classifier | Accuracy | Precision | Recall | F1 |
|------------|----------|-----------|--------|-----|
| Logistic Regression | 61.2% | 63.4% | 58.7% | 0.609 |
| Random Forest | 74.3% | 76.1% | 71.8% | 0.738 |
| **XGBoost** | **82.1%** | **83.4%** | **79.8%** | **0.816** |
