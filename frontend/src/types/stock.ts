// API Response Types

export interface DashboardData {
  sentiment: SentimentData;
  gainers: Gainer[];
  stock_count: number;
  selected_stocks?: StockDetails[];
}

export interface SentimentData {
  status: string;
  score: number;
  summary: string;
  color?: string;
}

export interface Gainer {
  symbol: string;
  change_pct: number;
  price: number;
}

export interface StockDetails {
  symbol: string;
  current_price: number;
  previous_close: number;
  day_high: number;
  day_low: number;
  volume: number;
  market_cap: number;
  market_cap_category?: 'Large Cap' | 'Mid Cap' | 'Small Cap';
  pe_ratio: number;
  pb_ratio?: number;
  dividend_yield?: number;
  eps?: number;
  profit_margins?: number;
  roe?: number;
  debt_to_equity?: number;
  sector?: string;
  industry?: string;
  long_business_summary?: string;
  fifty_two_week_high?: number;
  fifty_two_week_low?: number;
}

export interface HistoricalData {
  Date: string;
  Open?: number;
  High?: number;
  Low?: number;
  Close: number;
  Volume: number;
  SMA_50?: number;
  SMA_200?: number;
  BB_Upper?: number;
  BB_Lower?: number;
}

export interface NewsItem {
  title: string;
  source: string;
  date: string;
  link: string;
  sentiment_score?: number;
}

export interface AISummary {
  summary: string;
  technical_analysis?: string; // Optional if we separate them
}

export interface BacktestResult {
  equity_curve: { date: string; value: number; benchmark: number }[];
  metrics: {
    cagr: number;
    sharpe_ratio: number;
    max_drawdown: number;
    volatility: number;
  };
  strategy_report: string;
}

export interface PortfolioStock {
  symbol: string;
  name: string;
  quantity: number;
  price: number;
  sector: string;
  market_cap_category: string;
  volatility?: number;
  one_year_return?: number;
}


export interface InvestmentPlanParams {
  amount: number;
  type: 'one-time' | 'sip';
  duration_years: number;
  risk_profile: 'conservative' | 'moderate' | 'aggressive' | 'very_aggressive';
  expected_return: number;
  model?: string;
}

export interface KPIScores {
  combined_score: number;
  prediction_score: number;
  technical_score: number;
  sentiment_score: number;
  predicted_return_pct: number;
}

export interface ForecastPoint {
  ds: string;
  yhat: number;
  yhat_lower: number;
  yhat_upper: number;
}

export interface PredictionResponse {
  kpi: KPIScores;
  forecast: ForecastPoint[];
}

// Pro Features Types
export interface FinancialsResponse {
  balance_sheet: Record<string, any>[];
  income_stmt: Record<string, any>[];
  cashflow: Record<string, any>[];
  quarterly_financials: Record<string, any>[];
}

export interface RatioData {
  Valuation?: Record<string, number | string>;
  Profitability?: Record<string, number | string>;
  Liquidity_Debt?: Record<string, number | string>;
  Dividends?: Record<string, number | string>;
  [key: string]: Record<string, number | string> | undefined;
}

export interface SocialSentiment {
  score: number;
  status: string;
  count: number;
  latest_tweets: Array<{
    text: string;
    date: string;
    likes: number;
    retweets: number;
    user: string;
  }>;
}

export interface AIChart {
  title: string;
  type: 'bar' | 'line';
  data: any[];
  x_key: string;
  y_keys: string[];
}

export interface AISummary {
  summary: string;
  metrics?: Record<string, string>;
  charts?: AIChart[];
  sentiment?: string;
  technical_analysis?: string;
}
