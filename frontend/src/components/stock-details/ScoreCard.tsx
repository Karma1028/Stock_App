
import { GlassCard, GlassCardContent } from "@/components/ui/glass-card";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

interface ScoreCardProps {
    label: string;
    score: number;
    color?: "default" | "success" | "warning" | "destructive";
}

const ScoreCard = ({ label, score, color = "default" }: ScoreCardProps) => {
    const getColorClass = () => {
        switch (color) {
            case "success": return "bg-green-500";
            case "warning": return "bg-yellow-500";
            case "destructive": return "bg-red-500";
            default: return "bg-primary";
        }
    };

    return (
        <GlassCard>
            <GlassCardContent className="p-4 flex flex-col items-center justify-center space-y-3">
                <h3 className="text-sm font-medium text-muted-foreground">{label}</h3>
                <div className="flex flex-col items-center">
                    <span className={cn("text-3xl font-bold font-mono",
                        color === "success" && "text-green-500",
                        color === "warning" && "text-yellow-500",
                        color === "destructive" && "text-red-500",
                        color === "default" && "text-primary"
                    )}>
                        {score.toFixed(0)}/100
                    </span>
                </div>

                <div className="relative h-2 w-full overflow-hidden rounded-full bg-secondary">
                    <div
                        className={cn("h-full flex-1 transition-all", getColorClass())}
                        style={{ width: `${score}%` }}
                    />
                </div>

            </GlassCardContent>
        </GlassCard>
    );
};

export default ScoreCard;
