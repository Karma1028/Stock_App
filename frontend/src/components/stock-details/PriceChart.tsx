"use client"

import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useState } from 'react';

interface PriceChartProps {
    data: any[]; // Expects { Date: string, Close: number, ... }
    symbol: string;
}

export function PriceChart({ data, symbol }: PriceChartProps) {
    // Determine color based on trend (first vs last)
    const startPrice = data.length > 0 ? data[data.length - 1].Close : 0; // Historically sorted? data usually sorted ascending by date
    // Assuming data is sorted ascending (oldest first).
    // Let's check api.ts history response. Usually ascending.
    const isPositive = data.length > 2 ? data[data.length - 1].Close >= data[0].Close : true;
    const color = isPositive ? "#00b386" : "#ef4444";

    const formatXAxis = (tickItem: string) => {
        const date = new Date(tickItem);
        return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: '2-digit' });
    }

    return (
        <Card className="h-[500px] w-full bg-grow-dark/50 border-white/5 relative overflow-hidden">
            {/* Gradient Background Decoration */}
            <div className={`absolute top-0 right-0 w-[300px] h-[300px] rounded-full blur-[100px] opacity-20 pointer-events-none ${isPositive ? 'bg-grow-green' : 'bg-red-500'}`} />

            <CardHeader>
                <CardTitle>Price History (1 Year)</CardTitle>
            </CardHeader>
            <CardContent className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                        data={data}
                        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                    >
                        <defs>
                            <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={color} stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                        <XAxis
                            dataKey="Date"
                            tickFormatter={formatXAxis}
                            stroke="#64748b"
                            tick={{ fontSize: 12 }}
                            minTickGap={50}
                        />
                        <YAxis
                            domain={['auto', 'auto']}
                            stroke="#64748b"
                            tick={{ fontSize: 12 }}
                            tickFormatter={(val) => `₹${val}`}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1e293b', borderColor: 'rgba(255,255,255,0.1)', color: '#fff' }}
                            itemStyle={{ color: '#fff' }}
                            formatter={(value: any) => [`₹${Number(value).toFixed(2)}`, 'Price']}
                            labelFormatter={(label) => new Date(label).toLocaleDateString()}
                        />
                        <Area
                            type="monotone"
                            dataKey="Close"
                            stroke={color}
                            fillOpacity={1}
                            fill="url(#colorClose)"
                            strokeWidth={2}
                            animationDuration={1500}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </CardContent>
        </Card>
    )
}
