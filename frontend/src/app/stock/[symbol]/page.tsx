import { fetchStockDetails, fetchStockHistory } from "@/utils/api";
import { StockHeader } from "@/components/stock-details/StockHeader";
import { PriceChart } from "@/components/stock-details/PriceChart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface PageProps {
    params: Promise<{ symbol: string }>;
}

export default async function StockDetailPage({ params }: PageProps) {
    const { symbol } = await params;

    // Fetch Data
    // We should handle 404 or errors gracefully
    let stockData, historyData;

    try {
        [stockData, historyData] = await Promise.all([
            fetchStockDetails(symbol),
            fetchStockHistory(symbol, "1y")
        ]);
    } catch (e) {
        return (
            <div className="flex h-[50vh] items-center justify-center text-red-400">
                <p>Failed to load data for {symbol}. It might be invalid or unavailable.</p>
            </div>
        );
    }

    if (!stockData) return <div>Loading...</div>; // Should use loading.tsx

    return (
        <div className="space-y-8 animate-in fade-in zoom-in duration-500">
            <StockHeader
                symbol={stockData.symbol}
                longName={stockData.long_name || stockData.symbol}
                price={stockData.current_price}
                change={stockData.current_price - stockData.previous_close}
                changePct={(stockData.current_price - stockData.previous_close) / stockData.previous_close * 100}
                sector={stockData.sector || 'N/A'}
                industry={stockData.industry || 'N/A'}
            />

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 space-y-8">
                    <PriceChart data={historyData} symbol={symbol} />

                    <Card>
                        <CardHeader>
                            <CardTitle>Business Summary</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <p className="text-gray-400 leading-relaxed text-sm">
                                {stockData.long_business_summary || "No summary available."}
                            </p>
                        </CardContent>
                    </Card>
                </div>

                <div className="space-y-6">
                    <Card>
                        <CardHeader><CardTitle>Key Statistics</CardTitle></CardHeader>
                        <CardContent className="space-y-4">
                            <div className="flex justify-between border-b border-white/5 pb-2">
                                <span className="text-gray-400">Market Cap</span>
                                <span className="font-semibold text-white">₹{(stockData.market_cap / 10000000).toFixed(2)} Cr</span>
                            </div>
                            <div className="flex justify-between border-b border-white/5 pb-2">
                                <span className="text-gray-400">P/E Ratio</span>
                                <span className="font-semibold text-white">{stockData.pe_ratio?.toFixed(2) || 'N/A'}</span>
                            </div>
                            <div className="flex justify-between border-b border-white/5 pb-2">
                                <span className="text-gray-400">52W High</span>
                                <span className="font-semibold text-white">₹{stockData.fifty_two_week_high?.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between border-b border-white/5 pb-2">
                                <span className="text-gray-400">52W Low</span>
                                <span className="font-semibold text-white">₹{stockData.fifty_two_week_low?.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between pb-2">
                                <span className="text-gray-400">Volume</span>
                                <span className="font-semibold text-white">{stockData.volume?.toLocaleString()}</span>
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    )
}
