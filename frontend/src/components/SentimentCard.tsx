import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";
import { cn } from "@/lib/utils";

interface SentimentCardProps {
  status: string;
  score: number;
  summary: string;
  isLoading?: boolean;
}

const SentimentCard = ({ status, score, summary, isLoading }: SentimentCardProps) => {
  // Determine sentiment level based on score
  const getSentimentLevel = (score: number) => {
    if (score >= 60) return { status: 'Bullish', color: 'primary', icon: TrendingUp };
    if (score <= 40) return { status: 'Bearish', color: 'destructive', icon: TrendingDown };
    return { status: 'Neutral', color: 'warning', icon: Minus };
  };

  const sentiment = getSentimentLevel(score);
  const displayStatus = status || sentiment.status;
  const isBullish = displayStatus.toLowerCase() === 'bullish';
  const isBearish = displayStatus.toLowerCase() === 'bearish';
  const SentimentIcon = sentiment.icon;

  // Calculate gauge rotation (-90deg = 0%, 90deg = 100%)
  const gaugeRotation = (score / 100) * 180 - 90;

  if (isLoading) {
    return (
      <GlassCard variant="elevated" className="relative overflow-hidden">
        <GlassCardHeader>
          <GlassCardTitle>Market Sentiment</GlassCardTitle>
        </GlassCardHeader>
        <GlassCardContent>
          <div className="flex flex-col items-center justify-center py-8 space-y-4">
            <div className="w-32 h-16 bg-secondary/50 rounded-t-full animate-pulse" />
            <div className="h-4 w-20 bg-secondary/50 rounded animate-pulse" />
            <div className="h-3 w-48 bg-secondary/50 rounded animate-pulse" />
          </div>
        </GlassCardContent>
      </GlassCard>
    );
  }

  return (
    <GlassCard variant="elevated" className="relative overflow-hidden">
      {/* Glow effect */}
      <div
        className={cn(
          "absolute -top-20 -right-20 w-40 h-40 rounded-full blur-3xl opacity-20 animate-pulse-glow",
          isBullish ? "bg-primary" : isBearish ? "bg-destructive" : "bg-warning"
        )}
      />
      
      <GlassCardHeader>
        <div className="flex items-center justify-between">
          <GlassCardTitle>Market Sentiment</GlassCardTitle>
          <div
            className={cn(
              "flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium",
              isBullish 
                ? "bg-primary/20 text-primary" 
                : isBearish 
                  ? "bg-destructive/20 text-destructive"
                  : "bg-warning/20 text-warning"
            )}
          >
            <SentimentIcon className="w-4 h-4" />
            {displayStatus}
          </div>
        </div>
      </GlassCardHeader>
      
      <GlassCardContent>
        <div className="space-y-4">
          {/* Semi-circular gauge */}
          <div className="relative flex flex-col items-center py-4">
            {/* Gauge background */}
            <div className="relative w-40 h-20 overflow-hidden">
              <div className="absolute inset-0 w-40 h-40 rounded-full border-8 border-secondary" 
                   style={{ clipPath: 'polygon(0 0, 100% 0, 100% 50%, 0 50%)' }} />
              
              {/* Colored arc based on score */}
              <div 
                className={cn(
                  "absolute inset-0 w-40 h-40 rounded-full border-8 transition-all duration-1000",
                  isBullish ? "border-primary" : isBearish ? "border-destructive" : "border-warning"
                )}
                style={{ 
                  clipPath: 'polygon(0 0, 100% 0, 100% 50%, 0 50%)',
                  transform: `rotate(${gaugeRotation - 90}deg)`,
                  transformOrigin: 'center center'
                }} 
              />
              
              {/* Needle */}
              <div 
                className="absolute bottom-0 left-1/2 w-1 h-16 bg-foreground rounded-full origin-bottom transition-transform duration-1000"
                style={{ transform: `translateX(-50%) rotate(${gaugeRotation}deg)` }}
              >
                <div className="w-3 h-3 bg-foreground rounded-full absolute -top-1 left-1/2 -translate-x-1/2" />
              </div>
            </div>
            
            {/* Score display */}
            <div className={cn(
              "text-3xl font-bold font-mono mt-2",
              isBullish ? "text-primary" : isBearish ? "text-destructive" : "text-warning"
            )}>
              {score}%
            </div>
            
            {/* Labels */}
            <div className="flex justify-between w-40 mt-1 text-xs text-muted-foreground">
              <span>Bearish</span>
              <span>Bullish</span>
            </div>
          </div>
          
          {/* Summary text */}
          <p className="text-sm text-muted-foreground leading-relaxed text-center">{summary}</p>
        </div>
      </GlassCardContent>
    </GlassCard>
  );
};

export default SentimentCard;
