import { TrendingUp, TrendingDown, DollarSign, BarChart3 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface StockCardProps {
    symbol: string;
    name?: string;
    current_price?: number;
    previous_close?: number;
    market_cap?: number;
    pe_ratio?: number;
    change_pct?: number;
    isLoading?: boolean;
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
}: StockCardProps) => {
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
            <Card className="relative overflow-hidden">
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

    return (
        <Card className="relative overflow-hidden hover:shadow-lg transition-shadow">
            {/* Background gradient */}
            <div className={`absolute inset-0 opacity-5 ${isPositive ? 'bg-gradient-to-br from-green-500 to-emerald-500' : 'bg-gradient-to-br from-red-500 to-rose-500'}`} />

            <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-muted-foreground flex items-center justify-between">
                    <span>{name || symbol.replace('.NS', '')}</span>
                    {isPositive ? (
                        <TrendingUp className="h-4 w-4 text-green-500" />
                    ) : (
                        <TrendingDown className="h-4 w-4 text-red-500" />
                    )}
                </CardTitle>
            </CardHeader>

            <CardContent className="space-y-3">
                {/* Price */}
                <div>
                    <div className="text-2xl font-bold text-foreground">
                        ₹{current_price?.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || 'N/A'}
                    </div>
                    <div className={`text-sm font-medium ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
                        {isPositive ? '+' : ''}{calculatedChange.toFixed(2)}%
                    </div>
                </div>

                {/* Additional metrics */}
                <div className="grid grid-cols-2 gap-2 text-xs">
                    {market_cap && (
                        <div className="flex items-center gap-1">
                            <DollarSign className="h-3 w-3 text-muted-foreground" />
                            <span className="text-muted-foreground">MCap:</span>
                            <span className="font-medium">
                                ₹{(market_cap / 10000000).toFixed(0)}Cr
                            </span>
                        </div>
                    )}
                    {pe_ratio && (
                        <div className="flex items-center gap-1">
                            <BarChart3 className="h-3 w-3 text-muted-foreground" />
                            <span className="text-muted-foreground">P/E:</span>
                            <span className="font-medium">{pe_ratio.toFixed(2)}</span>
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    );
};

export default StockCard;
