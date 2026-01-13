import { ArrowUpRight, ArrowDownRight } from "lucide-react";
import { Link } from "react-router-dom";
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

interface Gainer {
  symbol: string;
  change_pct: number;
  price: number;
}

interface TopGainersCardProps {
  gainers: Gainer[];
  isLoading?: boolean;
}

const TopGainersCard = ({ gainers, isLoading }: TopGainersCardProps) => {
  if (isLoading) {
    return (
      <GlassCard variant="elevated">
        <GlassCardHeader>
          <GlassCardTitle>Top Gainers</GlassCardTitle>
        </GlassCardHeader>
        <GlassCardContent>
          <div className="space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <Skeleton className="w-10 h-10 rounded-lg" />
                  <div className="space-y-1">
                    <Skeleton className="h-4 w-20" />
                    <Skeleton className="h-3 w-10" />
                  </div>
                </div>
                <div className="text-right space-y-1">
                  <Skeleton className="h-4 w-16" />
                  <Skeleton className="h-3 w-12" />
                </div>
              </div>
            ))}
          </div>
        </GlassCardContent>
      </GlassCard>
    );
  }

  return (
    <GlassCard variant="elevated">
      <GlassCardHeader>
        <GlassCardTitle>Top Gainers</GlassCardTitle>
      </GlassCardHeader>
      
      <GlassCardContent>
        <div className="space-y-3">
          {gainers.map((stock, index) => {
            const isPositive = stock.change_pct >= 0;
            return (
              <Link
                to={`/stock/${stock.symbol}`}
                key={stock.symbol}
                className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg hover:bg-secondary/80 transition-colors cursor-pointer group animate-slide-up"
                style={{ animationDelay: `${index * 50}ms` }}
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                    <span className="text-xs font-bold text-primary">
                      {stock.symbol.slice(0, 2)}
                    </span>
                  </div>
                  <div>
                    <p className="font-medium text-foreground group-hover:text-primary transition-colors">
                      {stock.symbol.replace('.NS', '')}
                    </p>
                    <p className="text-xs text-muted-foreground">NSE</p>
                  </div>
                </div>
                
                <div className="text-right">
                  <p className="font-mono font-semibold text-foreground">
                    ₹{stock.price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                  </p>
                  <div
                    className={cn(
                      "flex items-center justify-end gap-1 text-sm font-medium",
                      isPositive ? "text-primary" : "text-destructive"
                    )}
                  >
                    {isPositive ? (
                      <ArrowUpRight className="w-3 h-3" />
                    ) : (
                      <ArrowDownRight className="w-3 h-3" />
                    )}
                    {isPositive ? '+' : ''}{stock.change_pct.toFixed(2)}%
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      </GlassCardContent>
    </GlassCard>
  );
};

export default TopGainersCard;
