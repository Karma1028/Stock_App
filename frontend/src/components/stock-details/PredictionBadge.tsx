
import { GlassCard, GlassCardContent } from "@/components/ui/glass-card";
import { TrendingUp, TrendingDown, Target } from "lucide-react";
import { cn } from "@/lib/utils";

interface PredictionBadgeProps {
    predictedReturn: number;
}

const PredictionBadge = ({ predictedReturn }: PredictionBadgeProps) => {
    const isPositive = predictedReturn >= 0;

    return (
        <GlassCard className="h-full">
            <GlassCardContent className="h-full p-6 flex flex-col items-center justify-center text-center space-y-2">
                <h3 className="text-muted-foreground flex items-center gap-2">
                    <Target className="w-4 h-4" />
                    Expected Return (5 Days)
                </h3>
                <div className="flex items-center gap-2">
                    <span className={cn("text-5xl font-bold font-mono", isPositive ? "text-green-500" : "text-red-500")}>
                        {isPositive ? "+" : ""}{predictedReturn.toFixed(2)}%
                    </span>
                </div>
                <p className="text-xs text-muted-foreground">Based on XGBoost Global Model</p>
            </GlassCardContent>
        </GlassCard>
    );
};

export default PredictionBadge;
