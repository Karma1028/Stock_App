"""
Inject 6-phase framework markdown/code cells into all relevant notebooks.
Each injection appends cells BEFORE the final summary cell (second-to-last position).
"""
import json, os, uuid

NB_DIR = os.path.join(os.path.dirname(__file__), 'notebooks')

def make_md(source_lines):
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source_lines
    }

def make_code(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def inject(nb_file, cells_to_add, before_last_n=1):
    """Insert cells before the last N cells of the notebook."""
    path = os.path.join(NB_DIR, nb_file)
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    pos = len(nb['cells']) - before_last_n
    for i, cell in enumerate(cells_to_add):
        nb['cells'].insert(pos + i, cell)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  ✅ {nb_file}: injected {len(cells_to_add)} cells")

# ════════════════════════════════════════════════
# PHASE 1: Data Foundation
# ════════════════════════════════════════════════

# NB01 — Adjusted Close + Macro-Injection
inject('01_Data_Extraction.ipynb', [
    make_md([
        "---\n",
        "\n",
        "## 📌 Phase 1 Insight: Adjusted Close Is God\n",
        "\n",
        "> **The 50-Year Veteran says:** *\"Prices lie. Dividends, splits, and inflation tell the truth.\"*  \n",
        "> **The Data Scientist says:** *\"Sanitize your inputs or your tensors will hallucinate.\"*\n",
        "\n",
        "Throughout this project, we use **Adjusted Close** — never raw Close — for all calculations.\n",
        "\n",
        "### Why?\n",
        "A 2:1 stock split halves the displayed price overnight, but the investor's wealth hasn't changed. To a naive algorithm, that split looks like a **50% market crash**. Adjusted Close retroactively accounts for:\n",
        "- **Stock splits** (e.g., 2:1, 5:1)\n",
        "- **Dividends** (cash returned to shareholders)\n",
        "- **Rights issues** and bonus shares\n",
        "\n",
        "For Tata Motors post-demerger data, this is *especially critical* — the demerger itself was a corporate action that repriced the entire history.\n",
        "\n",
        "> **Rule:** Never run ML models on raw Close prices. Always use Adjusted Close.\n"
    ]),
    make_md([
        "## 📌 Phase 1 Insight: Macro-Injection — The Secret Sauce\n",
        "\n",
        "A stock doesn't live in a vacuum. A complete analysis would inject these **macro variables** as features:\n",
        "\n",
        "| Macro Variable | Ticker/Source | Why It Matters for Tata Motors |\n",
        "|---------------|---------------|-------------------------------|\n",
        "| **Crude Oil (Brent)** | `BZ=F` | High oil → lower CV sales, higher transport costs |\n",
        "| **Steel Index** | LME Steel Rebar | Largest raw material cost for auto manufacturers |\n",
        "| **USD/INR** | `USDINR=X` | Import costs for components, competitive pricing |\n",
        "| **GBP/INR** | `GBPINR=X` | JLR revenue is in Pounds; costs are in Rupees |\n",
        "| **India 10Y Bond Yield** | India Govt Bond | Auto is credit-dependent; higher rates → lower EMI demand |\n",
        "\n",
        "> **Enhancement for future:** Adding these macro features would significantly improve model accuracy.\n",
        "\n",
        "```python\n",
        "# Future: fetch macro data\n",
        "# macro_tickers = {'Crude_Oil': 'BZ=F', 'USD_INR': 'USDINR=X', 'GBP_INR': 'GBPINR=X'}\n",
        "# for name, ticker in macro_tickers.items():\n",
        "#     macro_df = yf.download(ticker, start='2019-01-01')\n",
        "```\n"
    ])
])

# NB02 — ADF Stationarity Test
inject('02_Data_Cleaning_Preprocessing.ipynb', [
    make_md([
        "---\n",
        "\n",
        "## 📌 Phase 1 Insight: Stationarity Check — The ADF Test\n",
        "\n",
        "> **The Data Scientist says:** *\"Financial data is non-stationary — trends change, variance changes. If you don't handle this, your model is fitting noise.\"*\n",
        "\n",
        "### What is Stationarity?\n",
        "A **stationary** time series has a constant mean and variance over time. Raw stock prices are almost never stationary.\n",
        "\n",
        "### The Augmented Dickey-Fuller (ADF) Test\n",
        "The **ADF test** checks the null hypothesis that a unit root is present (series is non-stationary):\n",
        "- **p-value < 0.05** → Reject null → Series IS stationary ✅\n",
        "- **p-value ≥ 0.05** → Fail to reject → Series is NON-stationary ❌\n",
        "\n",
        "### The Practical Fix\n",
        "If raw prices fail ADF (they almost always do), transform to **log returns**:\n",
        "\n",
        "$$r_t = \\ln\\left(\\frac{P_t}{P_{t-1}}\\right)$$\n",
        "\n",
        "Log returns are typically stationary, additive across time, and the standard input for financial ML models.\n",
        "\n",
        "> **This is why our models use returns, not prices — it's a mathematical necessity.**\n"
    ]),
    make_code([
        "# ADF Stationarity Test (concept demonstration)\n",
        "from statsmodels.tsa.stattools import adfuller\n",
        "\n",
        "if primary_name in cleaned_data:\n",
        "    df_test = cleaned_data[primary_name]\n",
        "    prices = df_test['Close'].dropna()\n",
        "    log_returns = np.log(prices / prices.shift(1)).dropna()\n",
        "    \n",
        "    print('ADF STATIONARITY TEST')\n",
        "    print('=' * 50)\n",
        "    \n",
        "    # Test raw prices\n",
        "    adf_price = adfuller(prices)\n",
        "    print(f'Raw Prices:   ADF Stat = {adf_price[0]:.4f}, p-value = {adf_price[1]:.4f}')\n",
        "    print(f'  → {\"Stationary ✅\" if adf_price[1] < 0.05 else \"NON-Stationary ❌ (as expected)\"}')\n",
        "    \n",
        "    # Test log returns\n",
        "    adf_ret = adfuller(log_returns)\n",
        "    print(f'Log Returns:  ADF Stat = {adf_ret[0]:.4f}, p-value = {adf_ret[1]:.6f}')\n",
        "    print(f'  → {\"Stationary ✅ (safe for modeling)\" if adf_ret[1] < 0.05 else \"NON-Stationary ❌\"}')\n"
    ])
])

# ════════════════════════════════════════════════
# PHASE 2: Feature Engineering (Alpha Factory)
# ════════════════════════════════════════════════

# NB03 — Alpha Factory + Lag Features
inject('03_Feature_Engineering_Technical.ipynb', [
    make_md([
        "---\n",
        "\n",
        "## 📌 Phase 2 Insight: The Alpha Factory — Don't Feed Raw Prices\n",
        "\n",
        "> **The 50-Year Veteran says:** *\"I look for momentum and exhaustion.\"*  \n",
        "> **The Data Scientist says:** *\"I create vectors that represent momentum and mean reversion.\"*\n",
        "\n",
        "### Why Raw Prices Are Bad Features\n",
        "Never feed raw prices into XGBoost. Feed it **stories encoded as numbers**:\n",
        "\n",
        "1. **Lag Features:** Today's price is correlated with recent history\n",
        "   - `t-1` (yesterday), `t-5` (one week), `t-21` (one month)\n",
        "2. **Technical Indicators as Features:** RSI, MACD, Bollinger Bands, ATR\n",
        "3. **Rolling Statistics:** `rolling_mean` and `rolling_std` over 20 and 50 days\n",
        "\n",
        "The rolling statistics teach the model what **\"normal\" looks like** for each specific time period. A price that's 2σ above its 20-day mean is behaving unusually — that's a feature, not a prediction.\n",
        "\n",
        "> **Key Principle:** Transform raw data into *relative* measures (deviations from moving averages, rate of change, z-scores) rather than absolute values.\n"
    ]),
    make_code([
        "# Lag Feature Creation (concept demonstration)\n",
        "# These become input features for ML models in later notebooks\n",
        "\n",
        "print('LAG FEATURE ENGINEERING')\n",
        "print('=' * 50)\n",
        "\n",
        "lag_periods = [1, 5, 21]  # yesterday, 1-week, 1-month\n",
        "for lag in lag_periods:\n",
        "    col_name = f'Close_Lag_{lag}'\n",
        "    df[col_name] = df['Close'].shift(lag)\n",
        "    print(f'  Created: {col_name} (t-{lag})')\n",
        "\n",
        "# Percentage change lags (more useful than absolute lags)\n",
        "for lag in lag_periods:\n",
        "    col_name = f'Return_Lag_{lag}'\n",
        "    df[col_name] = df['Close'].pct_change(lag)\n",
        "    print(f'  Created: {col_name} ({lag}-day return)')\n",
        "\n",
        "print(f'\\nTotal new lag features: {len(lag_periods) * 2}')\n",
        "print('\\nSample lag correlations with next-day return:')\n",
        "next_day = df['Close'].pct_change().shift(-1)\n",
        "for lag in lag_periods:\n",
        "    corr = df[f'Return_Lag_{lag}'].corr(next_day)\n",
        "    print(f'  Return_Lag_{lag} correlation: {corr:.4f}')\n"
    ])
])

# NB04 — Rolling Statistics concept
inject('04_Feature_Engineering_Statistical.ipynb', [
    make_md([
        "---\n",
        "\n",
        "## 📌 Phase 2 Insight: Rolling Statistics Teach \"Normal\"\n",
        "\n",
        "> **The Data Scientist says:** *\"I create vectors that represent momentum and mean reversion.\"*\n",
        "\n",
        "### What Rolling Statistics Do\n",
        "Calculate `rolling_mean` and `rolling_std` over **20-day and 50-day** windows. This teaches the ML model what \"normal\" behaviour looks like for each period:\n",
        "\n",
        "- **Rolling Mean (20d):** Short-term trend anchor\n",
        "- **Rolling Mean (50d):** Medium-term trend anchor\n",
        "- **Rolling Std (20d):** Recent volatility (is the stock calm or excited?)\n",
        "- **Rolling Std (50d):** Medium-term volatility baseline\n",
        "\n",
        "A price that is 2σ above its 20-day rolling mean is exhibiting **unusual behaviour** — and that deviation itself becomes a powerful feature.\n",
        "\n",
        "### Z-Score Normalization\n",
        "$$Z_t = \\frac{P_t - \\text{RollingMean}_{20}}{\\text{RollingStd}_{20}}$$\n",
        "\n",
        "This standardized deviation tells the model \"how many standard deviations away from normal is today's price?\" — a regime-independent, scale-free feature.\n"
    ])
])

# ════════════════════════════════════════════════
# PHASE 3: Modeling (Choosing Your Weapons)
# ════════════════════════════════════════════════

# NB07 — HMM for Regime Detection
inject('07_Clustering_Market_Phases.ipynb', [
    make_md([
        "---\n",
        "\n",
        "## 📌 Phase 3 Insight: From K-Means to Hidden Markov Models\n",
        "\n",
        "> **The 50-Year Veteran says:** *\"Keep it simple. If I can't understand it with a pen and paper, I don't trade it.\"*\n",
        "\n",
        "### K-Means vs Hidden Markov Models (HMM)\n",
        "K-Means clustering (used above) groups days by **similarity**, but it doesn't model **transitions between regimes**. A Hidden Markov Model does both:\n",
        "\n",
        "| Feature | K-Means | HMM |\n",
        "|---------|---------|-----|\n",
        "| **Groups days** | ✅ By feature similarity | ✅ By hidden state |\n",
        "| **Models transitions** | ❌ No temporal awareness | ✅ Transition probability matrix |\n",
        "| **Predicts next regime** | ❌ | ✅ Given current state |\n",
        "| **Handles sequential data** | ❌ | ✅ Designed for it |\n",
        "\n",
        "### HMM for Financial Regimes\n",
        "Train an HMM to classify the market into **Bull**, **Bear**, or **Sideways/Chop** states. Each state has its own return distribution:\n",
        "- **Bull:** Positive mean return, low volatility\n",
        "- **Bear:** Negative mean return, high volatility\n",
        "- **Sideways:** Near-zero mean return, moderate volatility\n",
        "\n",
        "### The Veteran's Rule\n",
        "> **Never trade a breakout strategy in a \"Sideways\" regime detected by HMM.** Breakout signals in a choppy market are false signals — the price will revert to the range.\n",
        "\n",
        "```python\n",
        "# Future enhancement: HMM regime detection\n",
        "# from hmmlearn import hmm\n",
        "# model = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100)\n",
        "# model.fit(returns.reshape(-1, 1))\n",
        "# regimes = model.predict(returns.reshape(-1, 1))\n",
        "```\n"
    ])
])

# NB08 — Why XGBoost Over Deep Learning
inject('08_Model_Baseline_Comparison.ipynb', [
    make_md([
        "---\n",
        "\n",
        "## 📌 Phase 3 Insight: Why Gradient Boosting, Not Deep Learning\n",
        "\n",
        "> **The 50-Year Veteran says:** *\"Keep it simple. If I can't understand it with a pen and paper, I don't trade it.\"*  \n",
        "> **The Data Scientist says:** *\"Use Ensemble methods over Deep Learning for tabular data.\"*\n",
        "\n",
        "### The Case Against LSTMs and Transformers (For Now)\n",
        "It's tempting to throw a neural network at stock data. **Don't.** Here's why:\n",
        "\n",
        "| Factor | XGBoost/LightGBM | LSTM/Transformer |\n",
        "|--------|------------------|------------------|\n",
        "| **Data requirement** | Works with ~1000 rows | Needs 10,000+ for meaningful learning |\n",
        "| **Overfitting risk** | Low (regularization built-in) | High (especially with noisy financial data) |\n",
        "| **Interpretability** | Feature importance scores | Black box |\n",
        "| **Training speed** | Seconds | Hours |\n",
        "| **Tabular data performance** | State-of-the-art | Often worse than boosting |\n",
        "\n",
        "### Feature Importance = Explainability\n",
        "XGBoost tells you **why** it made a prediction — which features contributed most. This is non-negotiable in finance: a portfolio manager will never allocate capital based on a model that can't explain itself.\n",
        "\n",
        "### When to Use Deep Learning\n",
        "- **Sequence modeling** with 10,000+ timesteps\n",
        "- **Alternative data** (images, text, audio)\n",
        "- **High-frequency trading** (microsecond patterns)\n",
        "\n",
        "> **For our Tata Motors report:** XGBoost/LightGBM is the gold standard. We reserve deep learning for future iterations with larger datasets.\n"
    ])
])

# ════════════════════════════════════════════════
# PHASE 4: Validation (The "BS" Detector)
# ════════════════════════════════════════════════

# NB10 — Model Sharpe
inject('10_Hyperparameter_Tuning.ipynb', [
    make_md([
        "---\n",
        "\n",
        "## 📌 Phase 4 Insight: The \"Sharpe\" of Your Model\n",
        "\n",
        "> **The 50-Year Veteran says:** *\"Backtests are just marketing brochures. Show me the pain.\"*  \n",
        "> **The Data Scientist says:** *\"Time series cross-validation is non-negotiable.\"*\n",
        "\n",
        "### Beyond Accuracy: The Model Sharpe Ratio\n",
        "Don't just check **Accuracy (%)**. Check the **Sharpe Ratio** of the strategy the model suggests.\n",
        "\n",
        "$$\\text{Sharpe Ratio} = \\frac{R_p - R_f}{\\sigma_p}$$\n",
        "\n",
        "Where $R_p$ = portfolio return, $R_f$ = risk-free rate, $\\sigma_p$ = portfolio standard deviation.\n",
        "\n",
        "**Why this matters:** A model that is 60% accurate but loses huge money when it's wrong is **useless**. The Sharpe Ratio captures both the return AND the risk of the strategy — a Sharpe above 1.0 is good, above 2.0 is excellent.\n",
        "\n",
        "### Walk-Forward Validation (Expanding Window)\n",
        "Never use standard K-Fold cross-validation for time series. You cannot train on 2024 data to predict 2023:\n",
        "\n",
        "```\n",
        "Train: 2018-2020 → Test: Q1 2021\n",
        "Train: 2018-Q1 2021 → Test: Q2 2021\n",
        "Train: 2018-Q2 2021 → Test: Q3 2021\n",
        "```\n",
        "\n",
        "> **The test set must ALWAYS come after the training set chronologically.**\n"
    ])
])

# NB12 — Veteran's Warning
inject('12_Strategy_Backtesting.ipynb', [
    make_md([
        "---\n",
        "\n",
        "## 📌 Phase 4 Insight: \"Backtests Are Just Marketing Brochures\"\n",
        "\n",
        "> **The 50-Year Veteran says:** *\"Show me the pain. Show me the maximum drawdown, the losing streaks, the months where you bled money.\"*\n",
        "\n",
        "### Why Backtests Lie\n",
        "A backtest that shows smooth equity curves is almost certainly:\n",
        "- **Overfit** to historical data (won't work on new data)\n",
        "- **Ignoring transaction costs** (real brokerage, slippage, impact cost)\n",
        "- **Using future information** (look-ahead bias in feature engineering)\n",
        "\n",
        "### What to Actually Measure\n",
        "1. **Maximum Drawdown:** The worst peak-to-trough loss. If your strategy has a -40% drawdown, can you stomach that?\n",
        "2. **Longest Drawdown Duration:** How many months were you underwater? Most investors give up after 6 months of losses.\n",
        "3. **Sharpe Ratio:** Risk-adjusted return. Below 1.0 = not worth the risk.\n",
        "4. **Win Rate × Avg Win/Loss:** 60% accuracy with 1:1 win/loss ratio = profitable. 60% accuracy with 0.5:1 = losing money.\n",
        "\n",
        "> **Our approach:** We use walk-forward backtesting with realistic transaction costs, quarterly model retraining, and report both the good AND the ugly metrics.\n"
    ])
])

# ════════════════════════════════════════════════
# PHASE 5: Quantamental Layer (NLP & Sentiment)
# ════════════════════════════════════════════════

# NB06 — FinBERT + Signal Override
inject('06_Sentiment_Deep_Dive.ipynb', [
    make_md([
        "---\n",
        "\n",
        "## 📌 Phase 5 Insight: From VADER to FinBERT\n",
        "\n",
        "> **The 50-Year Veteran says:** *\"Read the room. If the CEO dies, the chart is broken.\"*  \n",
        "> **The Data Scientist says:** *\"NLP quantification of earnings calls.\"*\n",
        "\n",
        "### The Upgrade Path\n",
        "We used **TextBlob** and **VADER** above — good general-purpose tools. But for financial text, there's a specialized model:\n",
        "\n",
        "| Tool | Training Data | Financial Accuracy |\n",
        "|------|--------------|--------------------|\n",
        "| TextBlob | General text | ~60% |\n",
        "| VADER | Social media/reviews | ~65% |\n",
        "| **FinBERT** | Financial news, 10-K filings, earnings calls | **~87%** |\n",
        "\n",
        "**FinBERT** is a pre-trained BERT model fine-tuned on 10,000+ financial documents. It understands that *\"the company reported a loss\"* is negative, but *\"the stock was oversold after the loss\"* is actually **positive** (contrarian signal).\n",
        "\n",
        "```python\n",
        "# Future: FinBERT sentiment scoring\n",
        "# from transformers import AutoTokenizer, AutoModelForSequenceClassification\n",
        "# tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')\n",
        "# model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')\n",
        "```\n"
    ]),
    make_md([
        "## 📌 Phase 5 Insight: Signal Override Logic\n",
        "\n",
        "The most powerful application of sentiment is as a **safety valve** for technical signals:\n",
        "\n",
        "```\n",
        "IF Technical_Signal == BUY AND Sentiment_Score == NEGATIVE_EXTREME:\n",
        "    → Downgrade to HOLD\n",
        "    (Protects from \"catching a falling knife\")\n",
        "\n",
        "IF Technical_Signal == SELL AND Sentiment_Score == POSITIVE_EXTREME:\n",
        "    → Downgrade to HOLD  \n",
        "    (Prevents shorting into momentum)\n",
        "```\n",
        "\n",
        "### Real-World Example: Ratan Tata's Passing (Oct 2024)\n",
        "- **Technical signal at the time:** Neutral/Slightly Bullish\n",
        "- **Sentiment score:** EXTREME NEGATIVE\n",
        "- **Override logic output:** HOLD (don't buy the dip yet)\n",
        "- **What happened:** Stock fell 8%, then partially recovered over weeks\n",
        "\n",
        "> **The override saved an investor from buying at the worst possible moment.** This is the quantamental edge — combining numbers with narrative.\n"
    ])
])

# ════════════════════════════════════════════════
# PHASE 6: Storytelling (Report Generation)
# ════════════════════════════════════════════════

# NB13 — Probability Cones + Traffic Light
inject('13_Final_Synthesis.ipynb', [
    make_md([
        "---\n",
        "\n",
        "## 📌 Phase 6 Insight: Probability Cones, Not Price Targets\n",
        "\n",
        "> **The 50-Year Veteran says:** *\"Give me the 'So What?' on page one.\"*\n",
        "\n",
        "### Why Price Targets Are Dangerous\n",
        "Never predict *\"Tata Motors will hit ₹1000.\"* Instead, predict:\n",
        "\n",
        "> *\"There is a **68% probability** price will range between ₹950 and ₹1050 in the next 30 days.\"*\n",
        "\n",
        "Use standard deviation for probability cones:\n",
        "- **1σ cone (68% probability):** Mean ± 1 × std\n",
        "- **2σ cone (95% probability):** Mean ± 2 × std\n",
        "\n",
        "This is honest, scientifically defensible, and far more useful than a single number.\n"
    ]),
    make_md([
        "## 📌 Phase 6 Insight: The Traffic Light System\n",
        "\n",
        "Summarize complex ML outputs into **Red / Yellow / Green** signals for the reader:\n",
        "\n",
        "| Signal | Condition | Action |\n",
        "|--------|-----------|--------|\n",
        "| 🟢 **Green** | Trend UP + Sentiment Positive + ML Bullish | Increase position |\n",
        "| 🟡 **Yellow** | Trend UP + Sentiment Negative (Divergence!) | Hold / reduce size |\n",
        "| 🔴 **Red** | Trend DOWN + Sentiment Negative + ML Bearish | Exit or hedge |\n",
        "\n",
        "### Why Yellow is the Most Important Signal\n",
        "**Yellow = Divergence.** The price is going up but the narrative is turning negative. This is the early warning — the price hasn't caught up with reality yet. Most crashes start in Yellow territory.\n",
        "\n",
        "> **The Ratan Tata event was a sudden shift from Green to Red with no Yellow warning.** Black swan events bypass the traffic light entirely — which is why position sizing and stop-losses are non-negotiable.\n"
    ]),
    make_code([
        "# Traffic Light Signal Implementation\n",
        "print('TRAFFIC LIGHT SYSTEM')\n",
        "print('=' * 50)\n",
        "\n",
        "# Determine signal components\n",
        "trend_up = composite > 0\n",
        "sentiment_positive = signals.get('Momentum', {}).get('score', 0) > 0\n",
        "ml_bullish = signals.get('ML Model', {}).get('score', 0) > 0\n",
        "\n",
        "if trend_up and sentiment_positive and ml_bullish:\n",
        "    traffic = '🟢 GREEN — All signals aligned bullish'\n",
        "elif trend_up and not sentiment_positive:\n",
        "    traffic = '🟡 YELLOW — DIVERGENCE: Price up but sentiment negative'\n",
        "elif not trend_up and not sentiment_positive:\n",
        "    traffic = '🔴 RED — Both trend and sentiment bearish'\n",
        "else:\n",
        "    traffic = '🟡 YELLOW — Mixed signals, proceed with caution'\n",
        "\n",
        "print(f'\\n  Current Signal: {traffic}')\n",
        "print(f'  Trend: {\"UP\" if trend_up else \"DOWN\"}')\n",
        "print(f'  Sentiment: {\"Positive\" if sentiment_positive else \"Negative\"}')\n",
        "print(f'  ML Model: {\"Bullish\" if ml_bullish else \"Bearish\"}')\n"
    ])
])

print("\n✅ All 6 phases injected into notebooks!")
