# NLP & Scoring Logic Documentation

This document explains the Natural Language Processing (NLP) pipeline used for sentiment analysis and the logic behind the "AI Score" allocation in the application.

## 1. NLP Process (Sentiment Analysis)

The application uses a lightweight, rule-based NLP approach to gauge market sentiment for a specific stock.

### Step 1: Data Collection
- **Source**: `Google News RSS` Feed.
- **Library**: `feedparser`.
- **Method**: The app fetches the latest 30 days of news articles for the selected ticker (e.g., "RELIANCE stock news").

### Step 2: Sentiment Scoring
- **Library**: `TextBlob`.
- **Method**: 
    1. For each article, the **Header (Title)** is analyzed.
    2. `TextBlob` calculates a **Polarity Score** ranging from **-1.0** (Very Negative) to **+1.0** (Very Positive).
    3. **Subjectivity** is also calculated but currently not used in the score.

### Step 3: Aggregation
- **Daily Score**: All article scores for a single day are averaged.
- **Final Sentiment Score**: The individual article scores are aggregated (mean) to produce a raw sentiment value (e.g., 0.15).

---

## 2. Point Allocation (Scoring Logic)

The **AI Model Score** (Combined Score) displayed on the Stock Analysis page is a weighted average of three components: **Prediction**, **Technicals**, and **Sentiment**.

### Total Score Formula (0 - 100)
The final score is calculated in `modules/ml/engine.py` as follows:

$$ \text{Final Score} = (0.4 \times \text{Prediction}) + (0.4 \times \text{Technicals}) + (0.2 \times \text{Sentiment}) $$

### Component Details

#### A. Prediction Score (40% Weight)
Derived from the **XGBoost** model's 5-day return prediction.
- **Input**: Predicted Return % (e.g., +2.0%).
- **Logic**: 
    - A base score of **50** is assumed for 0% return.
    - Each **1%** positive return adds significant points (scaled by ~15).
    - Capped between **0** and **100**.
    - *Example*: A +1.5% predicted return might result in a Prediction Score of **75/100**.

#### B. Technical Score (40% Weight)
Derived from technical indicators (RSI, MACD, Trends).
- **Base Score**: The current **RSI** value (0-100).
- **Bonuses/Penalties**:
    - **MACD Crossover**: **+15 points** if MACD > Signal (Bullish), **-10 points** if Bearish.
    - **Trend (SMA)**: **+10 points** if Price > SMA (Golden Cross logic).
- *Example*: RSI=55, Bullish MACD (+15) = Technical Score **70/100**.

#### C. Sentiment Score (20% Weight)
Derived from the NLP analysis.
- **Normalization**: The raw polarity (-1 to +1) is mapped to 0-100.
    - Formula: $ \text{Score} = \frac{(\text{Polarity} + 1)}{2} \times 100 $
    - **Bonus**: If raw sentiment > 0.1 (significantly positive), add **+10 points**.
- *Example*: Polarity 0.2 -> Scaled to 60 -> Bonus +10 -> Sentiment Score **70/100**.

---

### Summary Table

| Component | Weight | Source | Key Metric |
| :--- | :--- | :--- | :--- |
| **Prediction** | 40% | XGBoost AI Model | 5-Day Expected Return |
| **Technicals** | 40% | Historical Price Data | RSI, MACD, Moving Averages |
| **Sentiment** | 20% | Google News + TextBlob | Headline Polarity (-1 to +1) |

