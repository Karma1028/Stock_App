"use client"

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface SentimentChartProps {
    score: number;
    status: string;
    summary: string;
}

export function SentimentChart({ score, status, summary }: SentimentChartProps) {
    const data = [
        { name: 'Score', value: score },
        { name: 'Remaining', value: 100 - score },
    ];

    // Determine color based on score
    let color = "#10b981"; // Green
    if (score < 40) color = "#ef4444"; // Red
    else if (score < 60) color = "#f59e0b"; // Yellow

    const COLORS = [color, '#334155'];

    return (
        <Card className="h-full">
            <CardHeader>
                <CardTitle>Market Sentiment</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="h-[200px] w-full relative">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={data}
                                cx="50%"
                                cy="50%"
                                startAngle={180}
                                endAngle={0}
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                            >
                                {data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="none" />
                                ))}
                            </Pie>
                        </PieChart>
                    </ResponsiveContainer>
                    <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-0 text-center">
                        <span className="text-3xl font-bold text-white">{score}</span>
                        <p className="text-sm text-grow-gray">{status}</p>
                    </div>
                </div>
                <div className="mt-4 p-4 rounded-lg bg-white/5 border border-white/5">
                    <p className="text-sm text-grow-gray italic">"{summary}"</p>
                </div>
            </CardContent>
        </Card>
    )
}
