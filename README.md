# Stock Analysis & Portfolio Management App

## Overview
This is a comprehensive Stock Analysis and Portfolio Management application built with Streamlit. It allows users to analyze stock performance, view technical indicators, generate AI-driven insights, and manage virtual portfolios.

## Features
- **Live Data**: Real-time stock data fetched via `yfinance`.
- **Interactive Charts**: Dynamic Line and Candlestick charts using Plotly.
- **Technical Analysis**: Automated calculation of RSI, MACD, SMA, and Bollinger Bands.
- **AI Insights**: Integration with OpenRouter (e.g., GPT-3.5) for automated investment summaries.
- **Company Info**: Scrapes Wikipedia for company background using Selenium.
- **Price Prediction**: Uses Prophet to forecast future stock prices.
- **Portfolio Management**: Track holdings and view current valuations.

## Prerequisites
- Python 3.8+
- Chrome Browser (for Selenium scraping)

## Installation

1.  **Create a Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup**:
    - Copy `.env.example` to `.env`.
    - Set `OPENROUTER_API_KEY` if you want AI features.
    - `USE_LOCAL_DATA` defaults to `False` (uses yfinance). Set to `True` only if you have local CSVs in `data/`.

## Usage

1.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

2.  **Run Smoke Tests**:
    ```bash
    python run_tests.py
    ```

## Configuration
- **Data Source**: By default, the app uses `yfinance`. To use local data, place `EQUITY.csv` in `data/` and set `USE_LOCAL_DATA=True` in `.env`.
- **AI Model**: Configure the model in `stock_ai_insights.py` (default: `openai/gpt-3.5-turbo`).

## Troubleshooting
- **Selenium Errors**: Ensure Chrome is installed. The app will fallback to "No summary" if scraping fails.
- **yfinance Errors**: Check your internet connection. API rate limits may apply.

## Assumptions
- The app assumes a stable internet connection for data fetching.
- Stock symbols are expected to be compatible with Yahoo Finance (e.g., `.NS` for NSE).
