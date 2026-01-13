import type {
  DashboardData,
  StockDetails,
  HistoricalData,
  NewsItem,
  AISummary,
  BacktestResult,
  InvestmentPlanParams,
  PredictionResponse,
} from '@/types/stock';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export async function getStockList(): Promise<{ stocks: string[] }> {
  return fetchApi<{ stocks: string[] }>('/api/stocks');
}

async function fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

// Dashboard API
export async function getDashboard(): Promise<DashboardData> {
  return fetchApi<DashboardData>('/api/dashboard');
}

// Stock Details API
export async function getStockDetails(symbol: string): Promise<StockDetails> {
  return fetchApi<StockDetails>(`/api/stock/${encodeURIComponent(symbol)}`);
}

// Stock History API
export async function getStockHistory(
  symbol: string,
  period: string = '1y'
): Promise<HistoricalData[]> {
  return fetchApi<HistoricalData[]>(
    `/api/stock/${encodeURIComponent(symbol)}/history?period=${period}`
  );
}

// Stock News API
export async function getStockNews(
  symbol: string,
  days: number = 7
): Promise<NewsItem[]> {
  return fetchApi<NewsItem[]>(
    `/api/stock/${encodeURIComponent(symbol)}/news?days=${days}`
  );
}

// General Market News API
export async function getNews(limit: number = 10): Promise<NewsItem[]> {
  return fetchApi<NewsItem[]>(`/api/news?limit=${limit}`);
}

// AI Summary API
// AI Summary API
export async function getAISummary(symbol: string, model?: string): Promise<AISummary> {
  return fetchApi<AISummary>('/api/ai/summary', {
    method: 'POST',
    body: JSON.stringify({ symbol, model }),
  });
}

// Portfolio Backtest API
export async function runBacktest(
  params: InvestmentPlanParams
): Promise<BacktestResult> {
  return fetchApi<BacktestResult>('/api/portfolio/backtest', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}


// Prediction and Forecast API
export async function getStockPrediction(symbol: string, days: number = 30): Promise<PredictionResponse> {
  return fetchApi<PredictionResponse>(`/api/stock/${encodeURIComponent(symbol)}/predict?days=${days}`);
}

// Health Check
export async function healthCheck(): Promise<{ status: string; app: string }> {
  return fetchApi<{ status: string; app: string }>('/health');
}
