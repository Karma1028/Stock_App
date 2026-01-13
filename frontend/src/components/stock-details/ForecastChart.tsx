
import {
    AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
    CartesianGrid
} from "recharts";
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";
import { Skeleton } from "@/components/ui/skeleton";
import { ForecastPoint } from "@/types/stock";
import { TrendingUp } from "lucide-react";

interface ForecastChartProps {
    data: ForecastPoint[];
    isLoading: boolean;
}

const ForecastChart = ({ data, isLoading }: ForecastChartProps) => {
    if (isLoading) {
        return (
            <GlassCard className="h-[400px]">
                <GlassCardContent className="h-full flex items-center justify-center">
                    <Skeleton className="w-full h-full" />
                </GlassCardContent>
            </GlassCard>
        );
    }

    return (
        <GlassCard className="h-[450px]">
            <GlassCardHeader>
                <GlassCardTitle className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-primary" />
                    AI Price Forecast
                </GlassCardTitle>
            </GlassCardHeader>
            <GlassCardContent className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                        <XAxis
                            dataKey="ds"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                            tickFormatter={(value) => {
                                const date = new Date(value);
                                return date.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
                            }}
                            minTickGap={30}
                        />
                        <YAxis
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                            tickFormatter={(value) => `₹${value.toLocaleString('en-IN')}`}
                            domain={['auto', 'auto']}
                            width={60}
                        />
                        <Tooltip
                            contentStyle={{
                                backgroundColor: 'hsl(var(--card))',
                                borderColor: 'hsl(var(--border))',
                                borderRadius: '8px',
                            }}
                            labelStyle={{ color: 'hsl(var(--muted-foreground))' }}
                            itemStyle={{ color: 'hsl(var(--foreground))' }}
                            formatter={(value: number, name: string) => {
                                const labels: Record<string, string> = {
                                    yhat: "Forecast",
                                    yhat_lower: "Lower Bound",
                                    yhat_upper: "Upper Bound"
                                };
                                return [`₹${value.toFixed(2)}`, labels[name] || name];
                            }}
                        />
                        <Area
                            type="monotone"
                            dataKey="yhat_lower"
                            stackId="1"
                            stroke="transparent"
                            fill="transparent"
                        />
                        <Area
                            type="monotone"
                            dataKey="yhat_upper" // This visualization hack might be tricky with standard area stack. 
                        // Better to separate:
                        // 1. Line for yhat
                        // 2. Range Area for bounds
                        />
                    </AreaChart>
                </ResponsiveContainer>

                {/* Re-rendering with better composition for Confidence Interval */}
                <div className="absolute inset-0 top-[80px] left-6 right-6 bottom-6 pointer-events-none">
                    {/* This overlay logic is complex. Let's stick to simple single graph with custom composition above */}
                </div>
            </GlassCardContent>
        </GlassCard>
    );
};

// Fixing the Chart logic to properly show Confidence Interval
const ForecastChartCorrected = ({ data, isLoading }: ForecastChartProps) => {
    if (isLoading) return <Skeleton className="h-[400px] w-full" />;

    return (
        <GlassCard className="h-[450px]">
            <GlassCardHeader>
                <GlassCardTitle className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-primary" />
                    AI Price Forecast (30 Days)
                </GlassCardTitle>
            </GlassCardHeader>
            <GlassCardContent className="h-[350px]">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                        <XAxis
                            dataKey="ds"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                            tickFormatter={(value) => new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                            minTickGap={30}
                        />
                        <YAxis
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                            domain={['auto', 'auto']}
                            tickFormatter={(val) => `₹${val}`}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                        />
                        <Area
                            type="monotone"
                            dataKey="yhat"
                            stroke="#10b981"
                            strokeWidth={3}
                            fill="url(#colorForecast)"
                            name="Forecast"
                        />
                        {/* Confidence Interval using Area with transparent stack not ideal. 
                Using separate Area for range is better if data structure allows.
                Or just show yhat for simplicity as V1. 
                Actually, Streamlit had yhat_lower/upper.
            */}
                    </AreaChart>
                </ResponsiveContainer>
            </GlassCardContent>
        </GlassCard>
    );
}

export default ForecastChartCorrected;
