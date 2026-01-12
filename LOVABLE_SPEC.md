# 📘 Comprehensive Stock Analysis App Logic & Migration Guide

**Version**: 2.0 (Migration to Next.js + FastAPI)
**Status**: Reference Standard (Source of Truth)

This document provides an exhaustive specification of the "Smart Stock Analytics" application logic, derived directly from the original production-grade Streamlit codebase (`streamlit_app.py`, `data_manager.py`). It is intended to be the **absolute reference** for AI agents (Lovable, Bolt, v0) rebuilding this application in a modern structure.

---

## 🏗️ 1. Core Architecture & Design System

### 1.1 Aesthetic Philosophy ("Pro Trader")
The application uses a "Pro Trader" financial dashboard aesthetic.
- **Theme**: Dark Mode Only (`#0f111a`).
- **Primary Accent**: "Grow Green" `rgb(0, 179, 134)` (Success/Call-to-Action).
- **Secondary Accent**: `teal-400` / `orange` (Technicals).
- **Surface**: Glassmorphism (`bg-opacity-70` + `backdrop-blur-xl`).
- **Typography**: Monospaced for financials/numbers, Sans-serif (Inter) for text.

### 1.2 Tech Stack Transition
- **Legacy**: Streamlit (Python-only, immediate mode).
- **Target**: 
  - **Frontend**: Next.js 14+ (React Server Components, TailwindCSS, Framer Motion, Recharts/Plotly).
  - **Backend**: FastAPI (Python, Pandas, yfinance, Scikit-learn).
  - **State**: Client-side (Zustand/Context) + Server State (React Query / SWR).

---

## 🚀 2. Feature Specification: Dashboard (Home)

**Legacy Source**: `render_dashboard()` in `streamlit_app.py`

### 2.1 KPI Cards (Metrics)
Four key metrics displayed at the top.
1.  **Stocks Tracked**:
    - **Logic**: Count of total tickers in `StockDataManager.get_stock_list()`.
    - **Display**: Number with `+` suffix. 
2.  **Market Score**:
    - **Logic**: A derived 0-100 score based on Nifty 50 breadth (Advances / Total Stocks * 100).
    - **Coloring**: Green (>60), Yellow (40-60), Red (<40).
3.  **AI Accuracy**:
    - **Logic**: Hardcoded placeholder `92%` (or derived from valid backtest logs in `MLEngine`).
4.  **Last Updated**:
    - **Logic**: Current System Time (`HH:MM`).

### 2.2 Market Sentiment Gauge
- **Visual**: A semi-circular gauge chart.
- **Data Source**: `dm.get_market_sentiment()`
- **Calculation**: 
  - `Advances = (Current Close > Previous Close)`
  - `Declines = (Current Close < Previous Close)`
  - `Score = (Advances / Total) * 100`
- **Ranges**:
  - 0-40: **Bearish** (Red)
  - 40-60: **Neutral** (Yellow)
  - 60-100: **Bullish** (Green)
- **Insight Text**: "Market Breadth: X Advances vs Y Declines."

### 2.3 Top Gainers List
- **Data Source**: `dm.get_top_gainers(limit=5)`
- **Logic**: 
  - Fetches last 5 days of data for *all* tracked stocks.
  - Calculates `% Change` = `(Last Close - Previous Close) / Previous Close`.
  - Sorts descending and takes top 5.
- **UI Item**:
  - Symbol Name (Bold).
  - Change % (Green text, + sign).
  - Current Price ("₹" formatted).

### 2.4 News Grid
- **Data Source**: `dm.get_news()` / `NewsScraper`.
- **Logic**: Fetches top news for generic market movers (e.g., RELIANCE, HDFCBANK) if no specific stock is selected.
- **UI**: Grid of 3-6 cards with Date, Source, Title, and "Read More" link.

---

## � 3. Feature Specification: Stock Analysis

**Legacy Source**: `render_stock_analysis()` in `streamlit_app.py`

### 3.1 Header & Ticker Stats
- **Inputs**: User selects stock from Sidebar.
- **Live Data**: `dm.get_live_data(symbol)`
- **Metrics Displayed**:
  - **Current Price**: Large font, Green/Red delta indicator.
  - **Day High/Low**: Range context.
  - **Volume**: Formatted (e.g., 1.2M).
  - **Market Cap**: Categorized (Large/Mid/Small) or raw value.

### 3.2 Company Fundamental Overview
A grid of fundamental ratios fetched via `yfinance.info`.
- **Key Metrics**:
  - P/E Ratio, P/B Ratio.
  - Dividend Yield (formatted %).
  - EPS (Trailing).
  - Profit Margins, ROE, Debt/Equity.
- **Business Summary**: A text block describing the company operations (`longBusinessSummary`).

### 3.3 Interactive Price Chart via Plotly/Recharts
**Legacy Logic**: `plot_stock_chart` function.
- **Subplots**: Top (Price + Technicals), Middle (Volume), Bottom (Sentiment - optional).
- **Technicals (Toggleable)**:
  - **SMA 50**: Orange Line.
  - **SMA 200**: Green Line.
  - **Bollinger Bands**: Gray dashed lines (High/Low).
- **Volume**: Bar chart below price.

### 3.4 AI Prediction Engine
**Legacy Logic**: `MLEngine` & `StockPredictor`.
- **XGBoost Model**:
  - Predicts `Next 5 Days Return %`.
  - Display: Large colored percentage (Green > 0, Red < 0).
- **Time Series Forecast (Prophet)**:
  - Plots historical close + `yhat` (Predicted) with `yhat_lower`/`yhat_upper` confidence intervals.
  - User Control: Slider for forecast days (7-90).

### 3.5 AI Executive Summary (GenAI)
**Legacy Logic**: `ai_insights.py`.
- **Function**: `generate_company_summary` & `get_ai_insights`.
- **Models**: Gemini 2.0 / OpenAI / DeepSeek.
- **Content**:
  - **Business Overview**: What they do.
  - **Moat**: Competitive advantage.
  - **Risks**: Bear case.
  - **Technical Deep Dive**: AI analysis of RSI/MACD divergence.

---

## 💰 4. Feature Specification: Investment Planner

**Legacy Source**: `render_investment_planner()` in `streamlit_app.py`

### 4.1 Wizard / Inputs
- **Investment Amount**: numeric input (min 1000).
- **Type**: One-time vs SIP.
- **Duration**: Slider (Years).
- **Risk Profile**: Slider (Conservative -> Very Aggressive).
- **Expected Return**: Percentage input.

### 4.2 AI "Get Suggestions" Tab
- **Logic**: `QuantEngine.run_pipeline`.
- **Backtesting**:
  - Runs a mathematical vector backtest based on risk profile logic.
  - **Metrics**: CAGR, Sharpe Ratio, Max Drawdown, Volatility.
- **Outputs**:
  - **Equity Curve**: Line chart of strategy vs benchmark.
  - **AI Strategy Report**: Markdown generated by `generate_quant_investment_plan`.
    - Explains *why* specific assets were chosen based on momentum/value signals.

### 4.3 Custom Portfolio Builder Tab
- **Local State**: User adds stocks manually (`session_state.portfolio`).
- **Features**:
  - **Add Stock**: Searchable dropdown (Name or Ticker).
  - **Live Valuation**: Quantity * Current Price.
- **Visuals**:
  - **Sector Allocation**: Pie Chart (`Symbol` -> `Sector` mapping).
  - **Market Cap Dist**: Bar Chart (Large/Mid/Small).
  - **Risk/Reward Scatter**:
    - X-Axis: Volatility (Std Dev).
    - Y-Axis: 1Y Return.
    - Bubble Size: Position Value.

---

## ⚙️ 5. Data & Logic Dictionary

### 5.1 Market Cap Categorization Logic
Standard Indian Market definitions used in `dm.get_market_cap_category`:
- **Large Cap**: > ₹20,000 Crores.
- **Mid Cap**: ₹5,000 - ₹20,000 Crores.
- **Small Cap**: < ₹5,000 Crores.

### 5.2 Stock List
- The app tracks Nifty 50 stocks by default (`ADANIENT.NS`, `RELIANCE.NS`, etc.).
- Fallback list is hardcoded in `StockDataManager`.

### 5.3 Sentiment Scoring
- **Source**: News headlines via Google Search RSS (`NewsScraper`).
- **Algorithm**: `TextBlob` polarity score (-1 to 1).
- **Aggregation**: Daily average score. Be sure to handle days with 0 news (fill NaNs).

---

## � 6. Full API Schema (FastAPI)

To replicate the Streamlit functionality, the Backend must expose these endpoints:

| Endpoint | Method | Params | Response Description |
| :--- | :--- | :--- | :--- |
| `/api/dashboard` | GET | - | Sentiment Score, Top 5 Gainers, Stock Count. |
| `/api/stock/{symbol}` | GET | - | Live Quote (Price, Change, Vol), Fundamentals (PE, PB, Ratios), Business Summary. |
| `/api/stock/{symbol}/history` | GET | `period` | OHLCV Data + calculated SMAs/BBs. |
| `/api/stock/{symbol}/news` | GET | `days` | List of news articles with sentiment scores. |
| `/api/news` | GET | `limit` | General market news. |
| `/api/ai/summary` | POST | `{symbol}` | GenAI Executive Summary text. |
| `/api/portfolio/backtest` | POST | `{amount, risk...}` | QuantEngine backtest result (Equity Curve, Metrics). |

---

## �️ 7. Implementation Checklist for AI Agent

When rebuilding this, ensure you:

1.  **Replicate the Charts**: Use `Recharts` (React) to match the `Plotly` (Python) visuals.
    - *Tip*: Use `AreaChart` for price and `ComposedChart` for Price + SMA overlays.
2.  **Maintain State**: The Investment Planner relies heavily on `st.session_state`. Use a global store like `Zustand` for the Portfolio Builder.
3.  **Handle Loading**: Streamlit uses `st.spinner`. In Next.js, use `<Suspense>` skeletons (Glassmorphic shimmers) for all data fetching.
4.  **Error Handling**: The `StockDataManager` has try/except blocks returning empty DFs. Ensure the API returns proper `500` or empty JSONs and the Frontend displays "Data Unavailable" cards instead of crashing.

---
**End of Specification**
