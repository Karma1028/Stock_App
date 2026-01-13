import { useState } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ComposedChart, Line, Bar, Legend
} from "recharts";
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";
import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface ChartData {
  Date: string;
  Open?: number;
  High?: number;
  Low?: number;
  Close: number;
  Volume: number;
  SMA_50?: number;
  SMA_200?: number;
  BB_Upper?: number;
  BB_Lower?: number;
}

interface StockChartProps {
  symbol: string;
  data: ChartData[];
  period: string;
  onPeriodChange: (period: string) => void;
  isLoading?: boolean;
  showTechnicals?: boolean;
}

const periods = ["1d", "1m", "1y", "5y"];
const periodLabels: Record<string, string> = {
  "1d": "1D",
  "1m": "1M",
  "1y": "1Y",
  "5y": "5Y"
};

import { useSettingsStore } from "@/store/settingsStore";

const StockChart = ({ symbol, data, period, onPeriodChange, isLoading, showTechnicals: initialShowTechnicals = false }: StockChartProps) => {
  const { showSMA50, showSMA200, showBollingerBands } = useSettingsStore();
  const [showVolume, setShowVolume] = useState(true);

  const latestPrice = data[data.length - 1]?.Close || 0;
  const firstPrice = data[0]?.Close || 0;
  const priceChange = latestPrice - firstPrice;
  const priceChangePercent = firstPrice > 0 ? (priceChange / firstPrice) * 100 : 0;
  const isPositive = priceChange >= 0;

  // Check if technicals are available
  const hasTechnicals = data.some(d => d.SMA_50 || d.SMA_200 || d.BB_Upper);

  if (isLoading) {
    return (
      <GlassCard variant="elevated" className="h-full">
        <GlassCardHeader>
          <div className="flex items-center justify-between">
            <div className="space-y-2">
              <Skeleton className="h-6 w-32" />
              <Skeleton className="h-8 w-48" />
            </div>
            <Skeleton className="h-10 w-40" />
          </div>
        </GlassCardHeader>
        <GlassCardContent className="h-64">
          <Skeleton className="w-full h-full" />
        </GlassCardContent>
      </GlassCard>
    );
  }

  return (
    <GlassCard variant="elevated" className="h-full">
      <GlassCardHeader>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <GlassCardTitle className="text-xl">{symbol.replace('.NS', '')}</GlassCardTitle>
            <div className="flex items-baseline gap-3 mt-1">
              <span className="text-2xl font-bold font-mono text-foreground">
                ₹{latestPrice.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
              </span>
              <span
                className={cn(
                  "text-sm font-medium",
                  isPositive ? "text-primary" : "text-destructive"
                )}
              >
                {isPositive ? '+' : ''}{priceChange.toFixed(2)} ({isPositive ? '+' : ''}{priceChangePercent.toFixed(2)}%)
              </span>
            </div>
          </div>

          <div className="flex gap-1 bg-secondary/50 p-1 rounded-lg">
            {periods.map((p) => (
              <button
                key={p}
                onClick={() => onPeriodChange(p)}
                className={cn(
                  "px-3 py-1.5 text-sm font-medium rounded-md transition-all",
                  period === p
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                )}
              >
                {periodLabels[p]}
              </button>
            ))}
          </div>
        </div>

        {/* Technical Indicators Toggle */}

      </GlassCardHeader>

      <GlassCardContent className={cn("h-64", hasTechnicals && showVolume && "h-80")}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={isPositive ? "hsl(160, 100%, 35%)" : "hsl(0, 84%, 60%)"}
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor={isPositive ? "hsl(160, 100%, 35%)" : "hsl(0, 84%, 60%)"}
                  stopOpacity={0}
                />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="Date"
              axisLine={false}
              tickLine={false}
              tick={{ fill: 'hsl(215, 20%, 65%)', fontSize: 11 }}
              tickFormatter={(value) => {
                const date = new Date(value);
                return date.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
              }}
            />
            <YAxis
              yAxisId="price"
              orientation="right"
              axisLine={false}
              tickLine={false}
              tick={{ fill: 'hsl(215, 20%, 65%)', fontSize: 11 }}
              tickFormatter={(value) => `₹${value.toLocaleString('en-IN')}`}
              width={70}
              domain={['auto', 'auto']}
            />
            {showVolume && (
              <YAxis
                yAxisId="volume"
                orientation="left"
                axisLine={false}
                tickLine={false}
                tick={{ fill: 'hsl(215, 20%, 65%)', fontSize: 10 }}
                tickFormatter={(value) => `${(value / 1000000).toFixed(1)}M`}
                width={50}
              />
            )}
            <Tooltip
              contentStyle={{
                backgroundColor: 'hsl(222, 47%, 11%)',
                border: '1px solid hsl(215, 28%, 17%)',
                borderRadius: '8px',
                boxShadow: '0 10px 40px rgba(0,0,0,0.4)',
              }}
              labelStyle={{ color: 'hsl(215, 20%, 65%)' }}
              itemStyle={{ color: 'hsl(210, 40%, 98%)' }}
              formatter={(value: number, name: string) => {
                if (name === 'Volume') return [`${(value / 1000000).toFixed(2)}M`, 'Volume'];
                return [`₹${value.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`, name];
              }}
            />

            {/* Bollinger Bands */}
            {showBollingerBands && hasTechnicals && (
              <>
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="BB_Upper"
                  stroke="hsl(215, 20%, 50%)"
                  strokeWidth={1}
                  strokeDasharray="5 5"
                  dot={false}
                  name="BB Upper"
                />
                <Line
                  yAxisId="price"
                  type="monotone"
                  dataKey="BB_Lower"
                  stroke="hsl(215, 20%, 50%)"
                  strokeWidth={1}
                  strokeDasharray="5 5"
                  dot={false}
                  name="BB Lower"
                />
              </>
            )}

            {/* Volume bars */}
            {showVolume && (
              <Bar
                yAxisId="volume"
                dataKey="Volume"
                fill="hsl(215, 20%, 40%)"
                opacity={0.3}
              />
            )}

            {/* Price area */}
            <Area
              yAxisId="price"
              type="monotone"
              dataKey="Close"
              stroke={isPositive ? "hsl(160, 100%, 35%)" : "hsl(0, 84%, 60%)"}
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#colorPrice)"
              name="Price"
            />

            {/* SMAs */}
            {hasTechnicals && (
              <>
                {showSMA50 && (
                  <Line
                    yAxisId="price"
                    type="monotone"
                    dataKey="SMA_50"
                    stroke="hsl(30, 100%, 50%)"
                    strokeWidth={1.5}
                    dot={false}
                    name="SMA 50"
                  />
                )}
                {showSMA200 && (
                  <Line
                    yAxisId="price"
                    type="monotone"
                    dataKey="SMA_200"
                    stroke="hsl(160, 100%, 40%)"
                    strokeWidth={1.5}
                    dot={false}
                    name="SMA 200"
                  />
                )}
              </>
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </GlassCardContent>
    </GlassCard>
  );
};

export default StockChart;
