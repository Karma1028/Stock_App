import { useEffect, useState } from 'react';
import { getStockRatios } from '@/services/api';
import { RatioData } from '@/types/stock';
import { GlassCard } from "@/components/ui/glass-card";
import { Skeleton } from "@/components/ui/skeleton";

interface RatiosTabProps {
    symbol: string;
}

const RatiosTab = ({ symbol }: RatiosTabProps) => {
    const [data, setData] = useState<RatioData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await getStockRatios(symbol);
                setData(res);
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [symbol]);

    if (loading) return <Skeleton className="h-[200px] w-full" />;

    if (!data || Object.keys(data).length === 0) return <div className="text-muted-foreground p-4 text-center">No ratio data available for {symbol}.</div>;

    return (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="text-xs text-muted-foreground bg-secondary/20 p-2 rounded border border-white/5">
                Note: Ratios are calculated based on TTM (Trailing Twelve Months) data provided by the exchange.
            </div>

            {Object.entries(data).map(([category, metrics]) => (
                <div key={category}>
                    <h3 className="text-sm font-semibold text-primary mb-4 flex items-center gap-2 uppercase tracking-widest">
                        <span className="w-2 h-2 rounded-full bg-primary/50" />
                        {category}
                    </h3>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                        {metrics && Object.entries(metrics).map(([key, value]) => {
                            const isPercent = key.includes("Margin") || key.includes("Return") || key.includes("Yield") || key.includes("ROE") || key.includes("ROA");
                            const isRatio = key.includes("Ratio") || key.includes("Debt");

                            // Check for valid number
                            const isValid = typeof value === 'number' && !isNaN(value);
                            const displayValue = isValid
                                ? (isPercent ? `${(value * 100).toFixed(2)}%` : value.toFixed(2))
                                : '—';

                            // Color coding for certain metrics
                            let colorClass = "text-foreground";
                            if (isValid && isPercent) {
                                if (value > 0.20) colorClass = "text-green-500";
                                else if (value < 0.05) colorClass = "text-yellow-500";
                            }

                            return (
                                <GlassCard key={key} className="p-4 flex flex-col hover:border-primary/30 transition-colors bg-secondary/5">
                                    <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-semibold mb-1">{key}</span>
                                    <span className={`text-xl font-bold font-mono ${colorClass}`}>
                                        {displayValue}
                                    </span>
                                </GlassCard>
                            );
                        })}
                    </div>
                </div>
            ))}
        </div>
    );
};

export default RatiosTab;
