// Technical Indicators Panel
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity } from "lucide-react";

interface TechnicalIndicatorsProps {
    symbol?: string;
}

const TechnicalIndicators = ({ symbol = "RELIANCE.NS" }: TechnicalIndicatorsProps) => {
    const indicators = [
        { name: "RSI (14)", value: 58.4, signal: "Neutral", color: "text-yellow-500" },
        { name: "MACD", value: 12.3, signal: "Buy", color: "text-green-500" },
        { name: "Stochastic", value: 72.1, signal: "Overbought", color: "text-red-500" },
        { name: "ADX", value: 28.5, signal: "Trending", color: "text-green-500" },
        { name: "CCI", value: 105, signal: "Overbought", color: "text-red-500" },
        { name: "Williams %R", value: -25, signal: "Buy", color: "text-green-500" },
    ];

    const movingAverages = [
        { period: "SMA 20", value: 2456, distance: "+2.1%", signal: "Above", color: "text-green-500" },
        { period: "SMA 50", value: 2389, distance: "+4.8%", signal: "Above", color: "text-green-500" },
        { period: "SMA 200", value: 2298, distance: "+9.3%", signal: "Above", color: "text-green-500" },
        { period: "EMA 20", value: 2465, distance: "+1.7%", signal: "Above", color: "text-green-500" },
    ];

    const summary = {
        buy: 4,
        neutral: 1,
        sell: 1,
        overall: "BUY",
    };

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Technical Indicators - {symbol.replace('.NS', '')}
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Summary */}
                <div className="p-4 bg-green-500/10 border border-green-500/30 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Overall Signal</span>
                        <span className="text-2xl font-bold text-green-500">{summary.overall}</span>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                        <span className="text-green-500">{summary.buy} Buy</span>
                        <span className="text-yellow-500">{summary.neutral} Neutral</span>
                        <span className="text-red-500">{summary.sell} Sell</span>
                    </div>
                </div>

                {/* Oscillators */}
                <div>
                    <h4 className="text-sm font-semibold mb-3">Oscillators</h4>
                    <div className="grid grid-cols-2 gap-3">
                        {indicators.map((indicator) => (
                            <div key={indicator.name} className="p-3 bg-secondary/50 rounded-lg">
                                <div className="text-xs text-muted-foreground mb-1">{indicator.name}</div>
                                <div className="flex items-center justify-between">
                                    <span className="font-bold">{indicator.value}</span>
                                    <span className={`text-xs font-semibold ${indicator.color}`}>
                                        {indicator.signal}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Moving Averages */}
                <div>
                    <h4 className="text-sm font-semibold mb-3">Moving Averages</h4>
                    <div className="space-y-2">
                        {movingAverages.map((ma) => (
                            <div key={ma.period} className="flex items-center justify-between p-2 bg-secondary/30 rounded">
                                <span className="text-sm">{ma.period}</span>
                                <div className="flex items-center gap-3">
                                    <span className="text-sm font-mono">₹{ma.value}</span>
                                    <span className={`text-xs font-semibold ${ma.color}`}>
                                        {ma.distance} {ma.signal}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

export default TechnicalIndicators;
