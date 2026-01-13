import { useParams, Link } from "react-router-dom";
import { ArrowLeft, TrendingUp, TrendingDown, Building2 } from "lucide-react";
import Header from "@/components/Header";
import StockChart from "@/components/StockChart";
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";
import AIAnalysis from "@/components/AIAnalysis";
import PredictionBadge from "@/components/stock-details/PredictionBadge";
import ScoreCard from "@/components/stock-details/ScoreCard";
import ForecastChart from "@/components/stock-details/ForecastChart";
import MarketDataGrid from "@/components/stock-details/MarketDataGrid";
import { useStockDetails, useStockHistory, useStockPrediction } from "@/hooks/useStockData";
import { Skeleton } from "@/components/ui/skeleton";
import { useState } from "react";
import { cn } from "@/lib/utils";

const StockAnalysis = () => {
  const { symbol } = useParams<{ symbol: string }>();
  const [chartPeriod, setChartPeriod] = useState("1y");

  const { data: stock, isLoading, error } = useStockDetails(symbol);
  const { data: chartData, isLoading: chartLoading } = useStockHistory(symbol, chartPeriod);
  const { data: prediction, isLoading: predLoading } = useStockPrediction(symbol);

  const priceChange = stock ? stock.current_price - stock.previous_close : 0;
  const priceChangePercent = stock?.previous_close ? (priceChange / stock.previous_close) * 100 : 0;
  const isPositive = priceChange >= 0;

  const mockChartData = [
    { Date: "2024-01-01", Close: 2320, Volume: 500000 },
    { Date: "2024-02-01", Close: 2450, Volume: 580000 },
    { Date: "2024-03-01", Close: 2560, Volume: 680000 },
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
      </div>

      <Header />

      <main className="container mx-auto px-4 py-8 relative">
        <Link to="/" className="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground mb-6 transition-colors">
          <ArrowLeft className="w-4 h-4" /> Back to Dashboard
        </Link>

        {/* Stock Header */}
        <div className="mb-8">
          {isLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-8 w-48" />
              <Skeleton className="h-12 w-64" />
            </div>
          ) : (
            <>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-3xl font-bold text-foreground">{symbol?.replace('.NS', '')}</h1>
                <span className="px-3 py-1 bg-secondary rounded-full text-sm text-muted-foreground">NSE</span>
                {stock?.sector && <span className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm">{stock.sector}</span>}
              </div>
              <div className="flex items-baseline gap-4">
                <span className="text-4xl font-bold font-mono text-foreground">
                  ₹{stock?.current_price?.toLocaleString('en-IN', { minimumFractionDigits: 2 }) || '—'}
                </span>
                <span className={cn("text-lg font-medium", isPositive ? "text-primary" : "text-destructive")}>
                  {isPositive ? <TrendingUp className="inline w-5 h-5 mr-1" /> : <TrendingDown className="inline w-5 h-5 mr-1" />}
                  {isPositive ? '+' : ''}{priceChange.toFixed(2)} ({isPositive ? '+' : ''}{priceChangePercent.toFixed(2)}%)
                </span>
              </div>
            </>
          )}
        </div>

        {/* Chart */}
        <div className="mb-8">
          <StockChart
            symbol={symbol || "STOCK"}
            data={chartData || mockChartData}
            period={chartPeriod}
            onPeriodChange={setChartPeriod}
            isLoading={chartLoading}
          />
        </div>


        {/* KPI & Prediction Section */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="md:col-span-1">
            <PredictionBadge predictedReturn={prediction?.kpi?.predicted_return_pct || 0} />
          </div>
          <div className="md:col-span-3 grid grid-cols-1 md:grid-cols-3 gap-4">
            <ScoreCard
              label="Technical Score"
              score={prediction?.kpi?.technical_score || 0}
              color="warning"
            />
            <ScoreCard
              label="Sentiment Score"
              score={prediction?.kpi?.sentiment_score || 0}
              color="success"
            />
            <ScoreCard
              label="AI Model Score"
              score={prediction?.kpi?.prediction_score || 0}
              color="default"
            />
          </div>
        </div>

        {/* Forecast Section */}
        <div className="mb-8">
          <ForecastChart data={prediction?.forecast || []} isLoading={predLoading} />
        </div>

        {/* Detailed Market Data Grid */}
        {stock && <MarketDataGrid stock={stock} />}

        {/* Business Summary */}
        {stock?.long_business_summary && (
          <GlassCard className="mb-8">
            <GlassCardHeader>
              <GlassCardTitle>Business Overview</GlassCardTitle>
            </GlassCardHeader>
            <GlassCardContent>
              <p className="text-muted-foreground leading-relaxed">{stock.long_business_summary}</p>
            </GlassCardContent>
          </GlassCard>
        )}

        {/* AI Analysis Section */}
        <div className="mb-8">
          <AIAnalysis symbol={symbol || ""} />
        </div>

      </main>
    </div>
  );
};

export default StockAnalysis;
