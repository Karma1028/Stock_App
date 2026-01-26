import { Link } from "react-router-dom";
import { TrendingUp, TrendingDown } from "lucide-react";
import StockChart from "@/components/StockChart";
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";
import AIAnalysis from "@/components/AIAnalysis";
import PredictionBadge from "@/components/stock-details/PredictionBadge";
import ScoreCard from "@/components/stock-details/ScoreCard";
import ForecastChart from "@/components/stock-details/ForecastChart";
import MarketDataGrid from "@/components/stock-details/MarketDataGrid";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { StockDetails, HistoricalData, PredictionResponse } from "@/types/stock";
import { useState } from "react";

interface OverviewTabProps {
    symbol: string;
    stock?: StockDetails;
    chartData?: HistoricalData[];
    prediction?: PredictionResponse;
    isLoading: boolean;
    chartLoading: boolean;
    predLoading: boolean;
}

const OverviewTab = ({
    symbol, stock, chartData, prediction,
    isLoading, chartLoading, predLoading
}: OverviewTabProps) => {
    const [chartPeriod, setChartPeriod] = useState("1y");

    // Calculate price change
    const priceChange = stock ? stock.current_price - stock.previous_close : 0;
    const priceChangePercent = stock?.previous_close ? (priceChange / stock.previous_close) * 100 : 0;
    const isPositive = priceChange >= 0;

    const mockChartData = [
        { Date: "2024-01-01", Close: 2320, Volume: 500000 },
        { Date: "2024-02-01", Close: 2450, Volume: 580000 },
        { Date: "2024-03-01", Close: 2560, Volume: 680000 },
    ];

    if (isLoading) {
        return (
            <div className="space-y-4">
                <Skeleton className="h-12 w-1/2" />
                <Skeleton className="h-[400px] w-full" />
            </div>
        );
    }

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            {/* Header Section (Price & Name) */}
            <div className="mb-6">
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
        </div>
    );
};

export default OverviewTab;
