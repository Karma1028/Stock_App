# 📘 Stock App - AI Development & Integration Guide

This guide is designed for Web Development AIs (e.g., Bolt, Lovable, v0) to understand the existing codebase, architecture, and design system of the **Stock Analysis App**. Use this to seamlessly enhance the application, integrate the backend, and maintain the "Pro Trader" aesthetic.

---

## 🏗️ Project Architecture

The project is a **Monorepo** consisting of a Next.js Frontend and a FastAPI Backend.

- **Frontend**: `frontend/` (Next.js 14+ App Router, TypeScript, TailwindCSS)
- **Backend**: `backend/` (FastAPI, Python)
- **Data**: `data/` (Consolidated Excel/CSV reports)

### Key Directories
- `frontend/src/app/`: App Router pages (`page.tsx`, `layout.tsx`).
- `frontend/src/components/`: Reusable UI components.
- `backend/main.py`: Main API entry point.

---

## 🎨 Design System: "Pro Trader" Theme

All new UI elements **MUST** adhere to this design system.

- **Theme Base**: Dark Mode Only (`#0f111a`).
- **Primary Color**: Grow Green (`#00b386`) for primary actions and positive trends.
- **Glassmorphism**: Use `bg-opacity` and `backdrop-blur` for cards and sidebars.
  - *Class*: `bg-slate-900/70 backdrop-blur-xl border border-white/10`
- **Typography**: Inter or System Sans. Crisp, readable numbers (Monospaced for financials).

**Global Styles**: Defined in `frontend/src/app/globals.css` (Tailwind v4 syntax).

---

## 🔌 Backend Integration Guide

The frontend communicates with the FastAPI backend via REST APIs.
**Base URL**: `http://localhost:8000`

### 1. Dashboard (`frontend/src/app/page.tsx`)
**Goal**: Display high-level market metrics and top movers.
- **API Endpoint**: `GET /api/dashboard`
- **Frontend Hook**: `fetchDashboardData()` in `frontend/src/utils/api.ts`
- **Data Expected**:
    ```json
    {
      "sentiment": { "score": 75, "status": "Bullish", "summary": "..." },
      "gainers": [{ "symbol": "RELIANCE.NS", "change_pct": 2.4, "price": 2400 }],
      "stock_count": 50
    }
    ```

### 2. Stock Detail Page (`frontend/src/app/stock/[symbol]/page.tsx`)
**Goal**: Deep dive into specific stock performance.
- **API Endpoint**: `GET /api/stock/{symbol}`
- **Frontend Hook**: `fetchStockDetails(symbol)`
- **Key Fields**: `current_price`, `market_cap`, `pe_ratio`, `long_business_summary`.

### 3. Interactive Charts (`frontend/src/components/stock-details/PriceChart.tsx`)
**Goal**: Render historical price trends.
- **API Endpoint**: `GET /api/stock/{symbol}/history?period=1y`
- **Data Expected**: Array of `{ Date: string, Close: number, Volume: number }`.

### 4. Market News (`frontend/src/components/dashboard/NewsGrid.tsx`)
**Goal**: Show latest relevant news.
- **API Endpoint**: `GET /api/news`

---

## 🛠️ Files to Modify

When asked to "Enhance the App" or "Fix X", focus on these files:

### Frontend
1.  **`frontend/src/app/layout.tsx`**: Main layout wrapper. Update for global providers or fonts.
2.  **`frontend/src/components/layout/Sidebar.tsx`**: Navigation menu. Add new pages here.
3.  **`frontend/src/app/globals.css`**: Global styles and Tailwind theme variables.
4.  **`frontend/src/utils/api.ts`**: Axios instance and API fetch functions. Add new endpoints here.

### Backend
1.  **`backend/main.py`**: Add new API routes here.
2.  **`modules/data/manager.py`**: Core data fetching logic (yfinance integration).

---

## 🚀 Development Workflow

1.  **Start Backend**:
    ```bash
    uvicorn backend.main:app --reload --port 8000
    ```
2.  **Start Frontend**:
    ```bash
    cd frontend
    npm run dev
    ```
3.  **Verify**: Open `http://localhost:3000`.

---

## 📝 Critical Rules for AI Agents

1.  **Preserve existing logic**: Do not delete `modules/` in Python or `frontend/src/components/ui` unless replacing.
2.  **Strict Typing**: Ensure all TypeScript interfaces match the API responses.
3.  **Error Handling**: Always wrap API calls efficiently (loading states/error boundaries).
4.  **Aesthetics**: If the UI looks "basic", apply the **Glassmorphism** classes defined in `globals.css`.
