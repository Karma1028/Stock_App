# Smart Stock Analytics App 🚀

A modern, "Pro Trader" aesthetic stock analysis dashboard built with **Next.js** (Frontend) and **FastAPI** (Backend).

## 📂 Repository Structure

- **`/frontend`**: Full Next.js 14+ Application.
  - Tech Stack: React, TailwindCSS, Framer Motion, Recharts.
- **`/backend`**: Python FastAPI Server.
  - Tech Stack: FastAPI, Pandas, yfinance, Scikit-learn.
- **`/data`**: Financial reports and sentiment datasets.

## 🤖 AI Development Guide

If you are an AI Agent (Bolt, Lovable, v0) looking to work on this code:
1.  Read **`LOVABLE_SPEC.md`** first. It contains the logic map.
2.  The Frontend code is in the `frontend/` directory.
3.  Global styles are in `frontend/src/app/globals.css`.

## 🛠️ Getting Started

### Prerequisites
- Node.js 18+
- Python 3.9+

### Running the App

1.  **Backend**
    ```bash
    pip install -r requirements.txt
    uvicorn backend.main:app --reload
    ```

2.  **Frontend**
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

## 📊 Features
- Live Dashboard with Market Sentiment Gauge.
- Detailed Stock Analysis with Real-time metrics.
- Interactive Price Charts & AI Predictions.
- Investment Portfolio Planner via Quant Engine.
