# 🚀 AI-Powered Stock Analysis & Investment Planner
## Project Documentation & Technical Overview

### 1. Project Overview
This is a comprehensive, production-ready web application designed for real-time stock analysis, portfolio management, and investment planning. Built with **Python** and **Streamlit**, it leverages advanced **Machine Learning (ML)** and **Generative AI** to provide actionable financial insights.

**Key Objectives:**
-   Democratize access to professional-grade financial analytics.
-   Combine traditional Technical Analysis with modern AI predictions.
-   Provide automated, personalized investment strategies.

---

### 2. Technical Architecture

#### **Tech Stack**
-   **Frontend**: Streamlit (Framework), HTML/CSS (Custom Styling), Plotly (Interactive Charts).
-   **Backend**: Python 3.12+.
-   **Data Processing**: Pandas, NumPy.
-   **Machine Learning**: XGBoost, LightGBM, Prophet (Time-series), Scikit-Learn.
-   **Generative AI**: OpenAI API (integration for LLM insights).
-   **Data Sources**: 
    -   `yfinance`: Real-time stock data.
    -   `feedparser`: News aggregation.
    -   `textblob`: Sentiment analysis.

#### **Directory Structure**
-   `app.py`: Main entry point and Dashboard UI.
-   `pages/`: Modular pages for specific features (`stock_analysis.py`, `investment_planner.py`).
-   `modules/`: Core logic decoupled from UI.
    -   `data/`: Data fetching and caching managers.
    -   `ml/`: Machine Learning training, inference, and feature engineering engines.
    -   `ui/`: Design systems and shared components.
    -   `utils/`: Helper functions and AI integration.

---

### 3. Detailed Features & Functionality

#### **A. Smart Dashboard (`app.py`)**
-   **Real-time Metrics**: Displays market sentiment (Fear & Greed style gauge), top gainers, and tracked stock counts.
-   **News Feed**: Aggregates top headlines from major financial sources (Proxy: Reliance/HDFC for general Indian market news).
-   **UI Design**: Implements a "Dark Premium" theme using Glassmorphism, CSS gradients, and custom typography (Inter, JetBrains Mono) for a high-end fintech feel.

#### **B. Stock Analysis Tool (`pages/stock_analysis.py`)**
This is the core analytical engine.
-   **Interactive Charts**: Plotly Candlestick/Line charts with overlays for SMA (50, 200) and Bollinger Bands.
-   **Deep Financial Data**: Tabs for Balance Sheets, Income Statements, and Cash Flow (visualized).
-   **Educational Mode**: Interactive "Learn" tab explaining financial terms (P/E, Beta, RSI) with live examples from the selected stock.
-   **AI Executive Summary**: Uses LLMs to generate a professional "Analyst Report" summarizing the company's business model, moat, and risks on the fly.

#### **C. Investment Planner (`pages/investment_planner.py`)**
A Wealth Management module.
-   **Goal Planning**: Simulates portfolio growth based on specific scenarios (Conservative, Moderate, Aggressive).
-   **Interactive Projections**: Monte Carlo-style visualization of expected returns over 1-30 years.
-   **Portfolio Construction**: Allows users to build custom portfolios and visualizes Asset Allocation (Treemaps, Sector Pies).

---

### 4. Machine Learning & AI Models

#### **A. Predictive Model (XGBoost/LightGBM)**
-   **Goal**: Predict the "Combined AI Score" and 5-Day Return.
-   **Algorithm**: **XGBoost Regressor** (Gradient Boosting).
-   **Target Variable**: Price return over the next 5 days (`Target_5d`).
-   **Training Process**:
    -   Splits data into Train (80%) and Test (20%).
    -   Uses Early Stopping to prevent overfitting.
    -   Standardizes inputs to handle different price scales.

#### **B. Feature Engineering (`modules/ml/features.py`)**
The model consumes a rich 28-dimensional feature set:
1.  **Momentum**: 1d, 5d, 21d Returns, Log Returns.
2.  **Volatility**: Rolling Standard Deviation (21d), Volatility indices.
3.  **Technical Indicators**:
    -   **Trend**: SMA_50, SMA_200, MACD (Signal/Diff).
    -   **Oscillators**: RSI (Relative Strength Index).
    -   **Bands**: Bollinger Bands (High/Low), ATR (Average True Range).
4.  **Sentiment**: Aggregated daily sentiment scores from news headlines.
5.  **Volume**: Rolling Volume Z-Scores.

#### **C. Forecasting (Prophet)**
-   **Goal**: visual price forecast for N days into the future.
-   **Algorithm**: **Facebook Prophet** (Additive Regression).
-   **Why Prophet?**: Handles seasonality (weekly/yearly cycles) and trend shifts better than simple linear regression for time-series data.

#### **D. Generative AI (LLM)**
-   **Role**: Qualitative Analysis.
-   **Implementation**: `modules/utils/ai_insights.py`.
-   **Prompt Engineering**: Uses "Persona-based Prompting" (e.g., "Act as a Senior Equity Analyst at Goldman Sachs") to ensure high-quality, professional outputs.

---

### 5. Resume / CV Points
*Use these bullet points to showcase this project on your resume:*

-   **Full-Stack Data Application**: Developed a production-grade Financial Analytics platform using **Streamlit** and **Python**, featuring real-time data ingestion and interactive dashboards.
-   **Advanced Machine Learning Pipeline**: Engineered a hybrid ML system using **XGBoost** for predictive scoring and **Facebook Prophet** for time-series forecasting, achieving robust performance on stock trend prediction.
-   **Feature Engineering**: Designed a complex feature extraction pipeline processing **25+ technical indicators** (RSI, MACD, Bollinger Bands) and integrating **NLP-based Sentiment Analysis** from news feeds.
-   **Generative AI Integration**: Integrated **OpenAI GPT** models to automate qualitative investment research, reducing analysis time by generating instant "Executive Summaries" and risk reports.
-   **System Design & Architecture**: Built a modular, scalable codebase with decoupled data scraping, caching layers, and ML inference engines, adhering to **SOLID principles**.
-   **UI/UX Design**: Created a premium, responsive "Dark Mode" user interface with custom CSS/HTML injection, improving user engagement and data visualization clarity.
