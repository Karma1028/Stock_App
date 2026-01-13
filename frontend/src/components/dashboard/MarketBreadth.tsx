import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowUp, ArrowDown, Activity } from "lucide-react";

interface MarketBreadthProps {
    isLoading?: boolean;
}

const MarketBreadth = ({ isLoading = false }: MarketBreadthProps) => {
    // Mock data - in production, fetch from API
    const breadthData = {
        advances: 312,
        declines: 188,
        unchanged: 50,
        advanceDeclineRatio: 1.66,
        new52WeekHighs: 45,
        new52WeekLows: 12,
        aboveMA200: 340,
        belowMA200: 210,
    };

    const totalStocks = breadthData.advances + breadthData.declines + breadthData.unchanged;
    const advancePercent = (breadthData.advances / totalStocks) * 100;
    const declinePercent = (breadthData.declines / totalStocks) * 100;
    const ma200Percent = (breadthData.aboveMA200 / (breadthData.aboveMA200 + breadthData.belowMA200)) * 100;

    const BreadthBar = ({ positive, negative }: { positive: number; negative: number }) => (
        <div className="w-full h-6 bg-secondary rounded-full overflow-hidden flex">
            <div
                className="bg-green-500 flex items-center justify-center text-xs font-semibold text-white"
                style={{ width: `${positive}%` }}
            >
                {positive > 10 && `${positive.toFixed(0)}%`}
            </div>
            <div
                className="bg-red-500 flex items-center justify-center text-xs font-semibold text-white"
                style={{ width: `${negative}%` }}
            >
                {negative > 10 && `${negative.toFixed(0)}%`}
            </div>
        </div>
    );

    if (isLoading) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>Market Breadth</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="h-24 bg-muted animate-pulse rounded" />
                    <div className="h-24 bg-muted animate-pulse rounded" />
                </CardContent>
            </Card>
        );
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Market Breadth
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Advance/Decline Ratio */}
                <div>
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Advance/Decline</span>
                        <span className="text-sm font-semibold">
                            Ratio: {breadthData.advanceDeclineRatio.toFixed(2)}
                        </span>
                    </div>
                    <BreadthBar positive={advancePercent} negative={declinePercent} />
                    <div className="flex items-center justify-between mt-1 text-xs text-muted-foreground">
                        <span className="flex items-center gap-1">
                            <ArrowUp className="h-3 w-3 text-green-500" />
                            {breadthData.advances} Advances
                        </span>
                        <span>{breadthData.unchanged} Unchanged</span>
                        <span className="flex items-center gap-1">
                            <ArrowDown className="h-3 w-3 text-red-500" />
                            {breadthData.declines} Declines
                        </span>
                    </div>
                </div>

                {/* 52-Week Highs/Lows */}
                <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-green-500/10 border border-green-500/20 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">52-Week Highs</div>
                        <div className="text-2xl font-bold text-green-500">
                            {breadthData.new52WeekHighs}
                        </div>
                    </div>
                    <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">52-Week Lows</div>
                        <div className="text-2xl font-bold text-red-500">
                            {breadthData.new52WeekLows}
                        </div>
                    </div>
                </div>

                {/* Above/Below 200-Day MA */}
                <div>
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">200-Day Moving Average</span>
                        <span className="text-sm font-semibold">
                            {ma200Percent.toFixed(0)}% Above
                        </span>
                    </div>
                    <BreadthBar
                        positive={ma200Percent}
                        negative={100 - ma200Percent}
                    />
                    <div className="flex items-center justify-between mt-1 text-xs text-muted-foreground">
                        <span>{breadthData.aboveMA200} Above 200-MA</span>
                        <span>{breadthData.belowMA200} Below 200-MA</span>
                    </div>
                </div>

                {/* Market Health Indicator */}
                <div className="pt-4 border-t">
                    <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Market Health</span>
                        <span
                            className={`px-3 py-1 rounded-full text-xs font-semibold ${breadthData.advanceDeclineRatio > 1.5
                                    ? "bg-green-500/20 text-green-500"
                                    : breadthData.advanceDeclineRatio > 1
                                        ? "bg-yellow-500/20 text-yellow-500"
                                        : "bg-red-500/20 text-red-500"
                                }`}
                        >
                            {breadthData.advanceDeclineRatio > 1.5
                                ? "Strong"
                                : breadthData.advanceDeclineRatio > 1
                                    ? "Moderate"
                                    : "Weak"}
                        </span>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

export default MarketBreadth;
