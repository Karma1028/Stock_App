import { useParams, Link } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import Header from "@/components/Header";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useStockDetails, useStockHistory, useStockPrediction } from "@/hooks/useStockData";
import { useState } from "react";
import OverviewTab from "@/components/stock-details/OverviewTab";
import FinancialsTab from "@/components/stock-details/FinancialsTab";
import RatiosTab from "@/components/stock-details/RatiosTab";
import NewsTab from "@/components/stock-details/NewsTab";

const StockAnalysis = () => {
  const { symbol } = useParams<{ symbol: string }>();
  const [activeTab, setActiveTab] = useState("overview");

  // Fetch critical data at the top level
  const { data: stock, isLoading } = useStockDetails(symbol);
  const { data: chartData, isLoading: chartLoading } = useStockHistory(symbol, "1y");
  const { data: prediction, isLoading: predLoading } = useStockPrediction(symbol);

  if (!symbol) return null;

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
      </div>

      <Header />

      <main className="container mx-auto px-4 py-8 relative">
        <div className="flex justify-between items-center mb-6">
          <Link to="/" className="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors">
            <ArrowLeft className="w-4 h-4" /> Back to Dashboard
          </Link>
        </div>

        <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="mb-8 w-full md:w-auto overflow-x-auto justify-start">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="financials">Financials</TabsTrigger>
            <TabsTrigger value="ratios">Key Ratios</TabsTrigger>
            <TabsTrigger value="news">News & Social</TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            <OverviewTab
              symbol={symbol}
              stock={stock}
              chartData={chartData}
              prediction={prediction}
              isLoading={isLoading}
              chartLoading={chartLoading}
              predLoading={predLoading}
            />
          </TabsContent>

          <TabsContent value="financials">
            <FinancialsTab symbol={symbol} />
          </TabsContent>

          <TabsContent value="ratios">
            <RatiosTab symbol={symbol} />
          </TabsContent>

          <TabsContent value="news">
            <NewsTab symbol={symbol} />
          </TabsContent>
        </Tabs>

      </main>
    </div>
  );
};

export default StockAnalysis;
