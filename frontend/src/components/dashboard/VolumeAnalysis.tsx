// Volume Analysis Component
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart3 } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

interface VolumeAnalysisProps {
    symbol?: string;
    isLoading?: boolean;
}

const VolumeAnalysis = ({ symbol = "RELIANCE.NS", isLoading }: VolumeAnalysisProps) => {
    // Mock data - in production, fetch from API
    const volumeData = [
        { date: "Mon", volume: 2400000, avgVolume: 2000000, change: 2.5 },
        { date: "Tue", volume: 1800000, avgVolume: 2000000, change: -1.2 },
        { date: "Wed", volume: 3200000, avgVolume: 2000000, change: 3.8 },
        { date: "Thu", volume: 2800000, avgVolume: 2000000, change: 1.5 },
        { date: "Fri", volume: 3600000, avgVolume: 2000000, change: 4.2 },
    ];

    const volumeProfile = {
        averageVolume: 2560000,
        todayVolume: 3600000,
        volumeChange: ((3600000 - 2560000) / 2560000) * 100,
        highVolumeBar: volumeData.filter(d => d.volume > d.avgVolume).length,
        lowVolumeBar: volumeData.filter(d => d.volume < d.avgVolume).length,
    };

    if (isLoading) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>Volume Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="h-64 bg-muted animate-pulse rounded" />
                </CardContent>
            </Card>
        );
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Volume Analysis - {symbol.replace('.NS', '')}
                </CardTitle>
            </CardHeader>
            <CardContent>
                {/* Volume Chart */}
                <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={volumeData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="date" className="text-xs" />
                        <YAxis className="text-xs" tickFormatter={(value) => `${(value / 1000000).toFixed(1)}M`} />
                        <Tooltip
                            formatter={(value: number) => [`${(value / 1000000).toFixed(2)}M`, "Volume"]}
                            contentStyle={{
                                backgroundColor: 'hsl(var(--card))',
                                border: '1px solid hsl(var(--border))',
                                borderRadius: '8px',
                            }}
                        />
                        <Bar dataKey="volume" radius={[4, 4, 0, 0]}>
                            {volumeData.map((entry, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={entry.volume > entry.avgVolume ? 'hsl(var(--chart-1))' : 'hsl(var(--chart-2))'}
                                />
                            ))}
                        </Bar>
                        <Bar dataKey="avgVolume" fill="hsl(var(--muted-foreground))" opacity={0.3} radius={[4, 4, 0, 0]} />
                    </BarChart>
                </ResponsiveContainer>

                {/* Volume Stats */}
                <div className="grid grid-cols-3 gap-4 mt-6">
                    <div className="text-center p-3 bg-secondary/50 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">Today's Volume</div>
                        <div className="text-lg font-bold">
                            {(volumeProfile.todayVolume / 1000000).toFixed(2)}M
                        </div>
                    </div>
                    <div className="text-center p-3 bg-secondary/50 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">Avg Volume</div>
                        <div className="text-lg font-bold">
                            {(volumeProfile.averageVolume / 1000000).toFixed(2)}M
                        </div>
                    </div>
                    <div className="text-center p-3 bg-secondary/50 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-1">Volume Change</div>
                        <div className={`text-lg font-bold ${volumeProfile.volumeChange > 0 ? 'text-green-500' : 'text-red-500'}`}>
                            {volumeProfile.volumeChange > 0 ? '+' : ''}
                            {volumeProfile.volumeChange.toFixed(1)}%
                        </div>
                    </div>
                </div>

                {/* Volume Bars Count */}
                <div className="mt-4 pt-4 border-t flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-chart-1" />
                        <span>Above Average: {volumeProfile.highVolumeBar} days</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-chart-2" />
                        <span>Below Average: {volumeProfile.lowVolumeBar} days</span>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

export default VolumeAnalysis;
