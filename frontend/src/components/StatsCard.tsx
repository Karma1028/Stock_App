import { LucideIcon } from "lucide-react";
import { GlassCard, GlassCardContent } from "@/components/ui/glass-card";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";

interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  trend?: "up" | "down" | "neutral";
  isLoading?: boolean;
  className?: string;
}

const StatsCard = ({ title, value, subtitle, icon: Icon, trend = "neutral", isLoading, className }: StatsCardProps) => {
  if (isLoading) {
    return (
      <GlassCard className={cn("relative overflow-hidden", className)}>
        <GlassCardContent className="p-6">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <Skeleton className="h-4 w-24" />
              <Skeleton className="h-8 w-16" />
              <Skeleton className="h-3 w-20" />
            </div>
            <Skeleton className="h-10 w-10 rounded-lg" />
          </div>
        </GlassCardContent>
      </GlassCard>
    );
  }

  return (
    <GlassCard className={cn("relative overflow-hidden group hover:scale-[1.02] transition-transform", className)}>
      {/* Subtle gradient overlay on hover */}
      <div className={cn(
        "absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity",
        trend === "up" && "bg-gradient-to-br from-primary/5 to-transparent",
        trend === "down" && "bg-gradient-to-br from-destructive/5 to-transparent"
      )} />
      
      <GlassCardContent className="p-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className={cn(
              "text-2xl font-bold font-mono mt-1",
              trend === "up" && "text-primary",
              trend === "down" && "text-destructive",
              trend === "neutral" && "text-foreground"
            )}>
              {typeof value === 'number' ? value.toLocaleString() : value}
            </p>
            {subtitle && (
              <p className={cn(
                "text-xs mt-1",
                trend === "up" && "text-primary/80",
                trend === "down" && "text-destructive/80",
                trend === "neutral" && "text-muted-foreground"
              )}>
                {subtitle}
              </p>
            )}
          </div>
          <div className={cn(
            "p-2.5 rounded-lg",
            trend === "up" && "bg-primary/10 text-primary",
            trend === "down" && "bg-destructive/10 text-destructive",
            trend === "neutral" && "bg-secondary text-muted-foreground"
          )}>
            <Icon className="w-5 h-5" />
          </div>
        </div>
      </GlassCardContent>
    </GlassCard>
  );
};

export default StatsCard;
