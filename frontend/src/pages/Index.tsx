import { useState } from "react";
import { BarChart2, Activity, Layers, RefreshCw, Target, Clock } from "lucide-react";
import Header from "@/components/Header";
import SentimentCard from "@/components/SentimentCard";
import TopGainersCard from "@/components/TopGainersCard";
import StockChart from "@/components/StockChart";
import NewsCard from "@/components/NewsCard";
import StatsCard from "@/components/StatsCard";
import { useDashboard, useStockHistory, useNews, useHealthCheck } from "@/hooks/useStockData";
import { Alert, AlertDescription } from "@/components/ui/alert";

// Fallback mock data for when API is unavailable
const mockDashboardData = {
  sentiment: {
    status: "Bullish",
    score: 75,
    summary: "Market Breadth: 35 Advances vs 15 Declines. Strong momentum in IT and Banking sectors.",
  },
  gainers: [
    { symbol: "RELIANCE.NS", change_pct: 2.5, price: 2450.0 },
    { symbol: "TCS.NS", change_pct: 1.8, price: 3580.25 },
    { symbol: "HDFCBANK.NS", change_pct: 1.65, price: 1620.5 },
    { symbol: "INFY.NS", change_pct: 1.42, price: 1485.75 },
    { symbol: "ICICIBANK.NS", change_pct: 1.25, price: 945.3 },
  ],
  stock_count: 50,
};

const mockChartData = [
  { Date: "2024-01-01", Close: 2320, Volume: 500000 },
  { Date: "2024-01-08", Close: 2380, Volume: 520000 },
  { Date: "2024-01-15", Close: 2340, Volume: 480000 },
  { Date: "2024-01-22", Close: 2410, Volume: 550000 },
  { Date: "2024-01-29", Close: 2395, Volume: 510000 },
  { Date: "2024-02-05", Close: 2450, Volume: 580000 },
  { Date: "2024-02-12", Close: 2420, Volume: 490000 },
  { Date: "2024-02-19", Close: 2485, Volume: 600000 },
  { Date: "2024-02-26", Close: 2510, Volume: 620000 },
  { Date: "2024-03-04", Close: 2480, Volume: 560000 },
  { Date: "2024-03-11", Close: 2530, Volume: 640000 },
  { Date: "2024-03-18", Close: 2560, Volume: 680000 },
];

const mockNews = [
  {
    title: "Reliance Industries announces major expansion in renewable energy sector",
    source: "Economic Times",
    date: "2 hours ago",
    link: "#",
  },
  {
    title: "TCS wins multi-billion dollar deal with global banking giant",
    source: "Business Standard",
    date: "4 hours ago",
    link: "#",
  },
  {
    title: "HDFC Bank reports strong quarterly earnings, beats estimates",
    source: "Mint",
    date: "6 hours ago",
    link: "#",
  },
  {
    title: "Nifty 50 hits all-time high amid strong FII inflows",
    source: "Reuters",
    date: "8 hours ago",
    link: "#",
  },
];

const Index = () => {
  const [chartPeriod, setChartPeriod] = useState("1y");
  
  // API hooks
  const { data: dashboardData, isLoading: dashboardLoading, error: dashboardError, refetch } = useDashboard();
  const { data: chartData, isLoading: chartLoading } = useStockHistory("RELIANCE.NS", chartPeriod);
  const { data: newsData, isLoading: newsLoading } = useNews(4);
  const { data: health, isError: apiDown } = useHealthCheck();

  // Use API data or fallback to mock
  const sentiment = dashboardData?.sentiment || mockDashboardData.sentiment;
  const gainers = dashboardData?.gainers || mockDashboardData.gainers;
  const stockCount = dashboardData?.stock_count || mockDashboardData.stock_count;
  const chartDataFinal = chartData || mockChartData;
  const newsFinal = newsData || mockNews;

  const isLoading = dashboardLoading;
  const currentTime = new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });

  return (
    <div className="min-h-screen bg-background">
      {/* Background gradient effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 left-0 w-80 h-80 bg-accent/5 rounded-full blur-3xl" />
      </div>

      <Header />

      <main className="container mx-auto px-4 py-8 relative">
        {/* API Status Alert */}
        {apiDown && (
          <Alert className="mb-6 border-warning/50 bg-warning/10">
            <AlertDescription className="text-warning">
              Backend API unavailable. Showing demo data. Start your FastAPI server at localhost:8000 for live data.
            </AlertDescription>
          </Alert>
        )}

        {/* Page Title */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-2xl font-bold text-foreground">Dashboard</h2>
            <p className="text-muted-foreground">Real-time market insights and analysis</p>
          </div>
          <button
            className="flex items-center gap-2 px-4 py-2 bg-secondary/50 hover:bg-secondary rounded-lg text-sm font-medium text-foreground transition-colors"
            onClick={() => refetch()}
            disabled={isLoading}
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {/* KPI Stats Grid - per spec 2.1 */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatsCard
            title="Stocks Tracked"
            value={`${stockCount}+`}
            subtitle="Nifty 50 Universe"
            icon={BarChart2}
            trend="neutral"
            isLoading={isLoading}
          />
          <StatsCard
            title="Market Score"
            value={sentiment.score}
            subtitle={sentiment.score >= 60 ? "Bullish Zone" : sentiment.score <= 40 ? "Bearish Zone" : "Neutral Zone"}
            icon={Activity}
            trend={sentiment.score >= 60 ? "up" : sentiment.score <= 40 ? "down" : "neutral"}
            isLoading={isLoading}
          />
          <StatsCard
            title="AI Accuracy"
            value="92%"
            subtitle="Model confidence"
            icon={Target}
            trend="up"
            isLoading={isLoading}
          />
          <StatsCard
            title="Last Updated"
            value={currentTime}
            subtitle="Auto-refresh: 30s"
            icon={Clock}
            trend="neutral"
            isLoading={isLoading}
          />
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Chart - Takes 2 columns */}
          <div className="lg:col-span-2">
            <StockChart
              symbol="RELIANCE.NS"
              data={chartDataFinal}
              period={chartPeriod}
              onPeriodChange={setChartPeriod}
              isLoading={chartLoading}
            />
          </div>

          {/* Sentiment Card - per spec 2.2 */}
          <div>
            <SentimentCard
              status={sentiment.status}
              score={sentiment.score}
              summary={sentiment.summary}
              isLoading={isLoading}
            />
          </div>
        </div>

        {/* Bottom Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <TopGainersCard gainers={gainers} isLoading={isLoading} />
          <NewsCard news={newsFinal} isLoading={newsLoading} />
        </div>

        {/* Footer Note */}
        <div className="mt-12 text-center text-sm text-muted-foreground">
          <p>
            {health?.status === 'ok' ? (
              <span className="text-primary">● Connected to FastAPI backend</span>
            ) : (
              <>
                Data refreshes every 30 seconds. Connect to{" "}
                <code className="text-primary font-mono bg-secondary/50 px-1.5 py-0.5 rounded">
                  localhost:8000
                </code>{" "}
                for live data.
              </>
            )}
          </p>
        </div>
      </main>
    </div>
  );
};

export default Index;
