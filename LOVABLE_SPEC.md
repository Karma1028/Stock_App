# Stock App - Frontend Specification (Lovable / No-Code Ready)

This document provides the necessary details to build a frontend interface for the Stock App using **Lovable** or any modern frontend framework.

## 🎨 Design System ("Pro Trader" Theme)

### Color Palette
- **Background**: `#0f111a` (Deep Dark Blue/Black)
- **Primary Accent**: `#00b386` (Grow Green)
- **Secondary Accent**: `#00d6a0` (Light Green)
- **Cards/Surfaces**: `rgba(30, 41, 59, 0.7)` (Glassmorphic Slate)
- **Text Primary**: `#f8fafc` (White/Slate-50)
- **Text Secondary**: `#94a3b8` (Slate-400)
- **Positive Trend**: `#00b386`
- **Negative Trend**: `#ef4444` (Red-500)

### UI Components
- **Glass Card**: Backdrop blur (12px), Border (1px solid white/10%), Shadow (lg).
- **Gradients**: Use subtle radial gradients (green/blue) in the background for depth.
- **Typography**: Inter (Google Fonts) or system sans-serif.

---

## 🔌 API Integration

**Base URL**: `http://localhost:8000` (Local Development)

### 1. Dashboard Data
**Endpoint**: `GET /api/dashboard`
**Response Structure**:
```json
{
  "sentiment": {
    "status": "Bullish",
    "score": 75,
    "summary": "Market Breadth: 35 Advances vs 15 Declines.",
    "color": "green"
  },
  "gainers": [
    {
      "symbol": "RELIANCE.NS",
      "change_pct": 2.5,
      "price": 2450.0
    }
  ],
  "stock_count": 50
}
```

### 2. Stock Details
**Endpoint**: `GET /api/stock/{symbol}`
**Example**: `/api/stock/RELIANCE.NS`
**Key Fields**:
- `current_price`: Number
- `previous_close`: Number
- `market_cap`: Number
- `pe_ratio`: Number
- `long_business_summary`: String
- `sector`: String

### 3. Historical Data (Charts)
**Endpoint**: `GET /api/stock/{symbol}/history?period={period}`
**Params**: `period` (1d, 1mo, 1y, 5y)
**Response**: Array of objects.
```json
[
  {
    "Date": "2023-01-01",
    "Close": 2400.0,
    "Volume": 500000
  }
]
```

### 4. News
**Endpoint**: `GET /api/news`
**Response**:
```json
[
  {
    "title": "Reliance announces new unexpected profit",
    "source": "Economic Times",
    "date": "2023-10-25",
    "link": "https://..."
  }
]
```

---

## 📂 Data Assets
- **Sentiment Dataset**: `data/consolidated_sentiments.csv`
  - Columns: `Stock Index Name`, `date`, `title`, `sentiment`, `source`
  - Use this for training custom AI models or displaying historical sentiment trends.

## 🚀 Deployment Instructions
1.  **Backend**: Host the FastAPI app (`backend/main.py`) on a Python runner (e.g., Render, Railway, AWS).
2.  **Frontend**: Connect your Lovable project to the deployed Backend URL.
