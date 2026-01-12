import { fetchDashboardData, fetchNews } from "@/utils/api";
import { MetricsCards } from "@/components/dashboard/MetricsCards";
import { SentimentChart } from "@/components/dashboard/SentimentChart";
import { TopGainers } from "@/components/dashboard/TopGainers";
import { NewsGrid } from "@/components/dashboard/NewsGrid";
import { Suspense } from "react";

export const dynamic = 'force-dynamic';

export default async function Dashboard() {
  // Fetch data in parallel
  const dashboardDataPromise = fetchDashboardData();
  const newsPromise = fetchNews();

  const [dashboardData, news] = await Promise.all([
    dashboardDataPromise.catch(e => ({ sentiment: { score: 50, status: 'Neutral', summary: 'Data unavailable' }, gainers: [], stock_count: 0 })),
    newsPromise.catch(e => [])
  ]);

  const { sentiment, gainers, stock_count } = dashboardData;

  return (
    <div className="space-y-8 animate-in fade-in duration-700 slide-in-from-bottom-4">
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">Market Overview</h1>
          <p className="text-grow-gray">Your daily market insights and performance metrics</p>
        </div>
      </div>

      <MetricsCards
        stockCount={stock_count}
        marketScore={sentiment?.score || 50}
        sentimentColor={sentiment?.color || 'white'}
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <SentimentChart
          score={sentiment?.score || 50}
          status={sentiment?.status || 'Neutral'}
          summary={sentiment?.summary || 'No summary available'}
        />
        <TopGainers gainers={gainers || []} />
      </div>

      <div className="mt-12">
        <h2 className="text-2xl font-bold mb-6 text-white">Latest Market News</h2>
        <NewsGrid news={news || []} />
      </div>
    </div>
  );
}
