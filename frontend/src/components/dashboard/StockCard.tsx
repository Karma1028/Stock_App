import { TrendingUp, TrendingDown, DollarSign, BarChart3, X, ExternalLink } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";

interface StockCardProps {
    symbol: string;
    name?: string;
    current_price?: number;
    previous_close?: number;
    market_cap?: number;
    pe_ratio?: number;
    change_pct?: number;
    isLoading?: boolean;
    onRemove?: (symbol: string) => void;
}

const StockCard = ({
    symbol,
    name,
    current_price,
    previous_close,
    market_cap,
    pe_ratio,
    change_pct,
    isLoading = false,
    onRemove,
}: StockCardProps) => {
    const navigate = useNavigate();

    // Calculate change percentage if not provided
    const calculatedChange =
        change_pct !== undefined
            ? change_pct
            : current_price && previous_close
                ? ((current_price - previous_close) / previous_close) * 100
                : 0;

    const isPositive = calculatedChange >= 0;

    if (isLoading) {
        return (
            <Card className="relative overflow-hidden border-border/40">
                <CardHeader className="pb-3">
                    <div className="h-4 w-24 bg-muted animate-pulse rounded" />
                </CardHeader>
                <CardContent className="space-y-3">
                    <div className="h-8 w-32 bg-muted animate-pulse rounded" />
                    <div className="h-4 w-20 bg-muted animate-pulse rounded" />
                </CardContent>
            </Card>
        );
    }

    const handleClick = () => {
        navigate(`/analysis/${symbol}`);
    };

    return (
        <Card
            className={`group relative overflow-hidden cursor-pointer border-border/40 transition-all duration-300 hover:scale-[1.02] hover:shadow-2xl hover:border-${isPositive ? 'green' : 'red'}-500/50`}
            onClick={handleClick}
        >
            {/* Background gradient overlay */}
            <div className={`absolute inset-0 opacity-[0.03] transition-opacity group-hover:opacity-[0.08] ${isPositive ? 'bg-gradient-to-br from-green-500 to-emerald-500' : 'bg-gradient-to-br from-red-500 to-rose-500'}`} />

            {/* Animated Glow Border */}
            <div className={`absolute -inset-[1px] opacity-0 group-hover:opacity-100 transition-opacity blur-[2px] pointer-events-none ${isPositive ? 'bg-green-500/20' : 'bg-red-500/20'}`} />

            <CardHeader className="pb-3 relative">
                <CardTitle className="text-sm font-semibold text-muted-foreground flex items-center justify-between">
                    <span className="truncate mr-2 group-hover:text-foreground transition-colors">{name || symbol.replace('.NS', '')}</span>
                    <div className="flex items-center gap-2">
                        {onRemove && (
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity hover:bg-destructive/20 hover:text-destructive"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onRemove(symbol);
                                }}
                            >
                                <X className="h-3 w-3" />
                            </Button>
                        )}
                        <div className={`p-1 rounded-full ${isPositive ? 'bg-green-500/10' : 'bg-red-500/10'}`}>
                            {isPositive ? (
                                <TrendingUp className={`h-3 w-3 ${isPositive ? 'text-green-500' : 'text-red-500'}`} />
                            ) : (
                                <TrendingDown className={`h-3 w-3 ${isPositive ? 'text-green-500' : 'text-red-500'}`} />
                            )}
                        </div>
                    </div>
                </CardTitle>
            </CardHeader>

            <CardContent className="space-y-4 relative">
                {/* Price Section */}
                <div>
                    <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-bold tracking-tight text-foreground">
                            ₹{current_price?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || 'N/A'}
                        </span>
                        <ExternalLink className="h-3 w-3 opacity-0 group-hover:opacity-40 transition-opacity" />
                    </div>
                    <div className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-bold mt-1 ${isPositive ? 'bg-green-500/10 text-green-500' : 'bg-red-500/10 text-red-500'}`}>
                        {isPositive ? '+' : ''}{calculatedChange.toFixed(2)}%
                    </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 gap-3 pt-2 border-t border-border/50">
                    {market_cap && (
                        <div className="flex flex-col gap-0.5">
                            <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">Market Cap</span>
                            <div className="flex items-center gap-1 font-medium text-xs">
                                ₹{(market_cap / 10000000).toFixed(0)}Cr
                            </div>
                        </div>
                    )}
                    {pe_ratio && (
                        <div className="flex flex-col gap-0.5">
                            <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold">P/E Ratio</span>
                            <div className="flex items-center gap-1 font-medium text-xs">
                                {pe_ratio.toFixed(2)}
                            </div>
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    );
};

export default StockCard;
