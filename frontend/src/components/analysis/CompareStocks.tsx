// Compare Stocks Feature
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { GitCompare, TrendingUp } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

const CompareStocks = () => {
    const [comparing, setComparing] = useState(false);

    // Mock comparison data
    const comparisonData = [
        { date: "Jan", RELIANCE: 2400, TCS: 3200, INFY: 1400 },
        { date: "Feb", RELIANCE: 2210, TCS: 3100, INFY: 1398 },
        { date: "Mar", RELIANCE: 2290, TCS: 3400, INFY: 1580 },
        { date: "Apr", RELIANCE: 2780, TCS: 3600, INFY: 1680 },
        { date: "May", RELIANCE: 2590, TCS: 3500, INFY: 1520 },
        { date: "Jun", RELIANCE: 2900, TCS: 3800, INFY: 1720 },
    ];

    const metrics = [
        { stock: "RELIANCE", ytd: "+21.5%", volatility: "18%", sharpe: "1.2", beta: "0.95" },
        { stock: "TCS", ytd: "+18.8%", volatility: "15%", sharpe: "1.4", beta: "0.82" },
        { stock: "INFY", ytd: "+22.9%", volatility: "16%", sharpe: "1.3", beta: "0.88" },
    ];

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    <span className="flex items-center gap-2">
                        <GitCompare className="h-5 w-5" />
                        Compare Stocks
                    </span>
                    <Button
                        size="sm"
                        variant={comparing ? "default" : "outline"}
                        onClick={() => setComparing(!comparing)}
                    >
                        {comparing ? "Hide Comparison" : "Show Comparison"}
                    </Button>
                </CardTitle>
            </CardHeader>
            <CardContent>
                {comparing ? (
                    <div className="space-y-6">
                        {/* Performance Chart */}
                        <div>
                            <h4 className="text-sm font-medium mb-3">Relative Performance (Indexed to 100)</h4>
                            <ResponsiveContainer width="100%" height={250}>
                                <LineChart data={comparisonData}>
                                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                    <XAxis dataKey="date" className="text-xs" />
                                    <YAxis className="text-xs" />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: 'hsl(var(--card))',
                                            border: '1px solid hsl(var(--border))',
                                            borderRadius: '8px',
                                        }}
                                    />
                                    <Legend />
                                    <Line type="monotone" dataKey="RELIANCE" stroke="#3b82f6" strokeWidth={2} />
                                    <Line type="monotone" dataKey="TCS" stroke="#10b981" strokeWidth={2} />
                                    <Line type="monotone" dataKey="INFY" stroke="#f59e0b" strokeWidth={2} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Metrics Comparison Table */}
                        <div>
                            <h4 className="text-sm font-medium mb-3">Key Metrics Comparison</h4>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="border-b">
                                            <th className="text-left p-2">Stock</th>
                                            <th className="text-right p-2">YTD Return</th>
                                            <th className="text-right p-2">Volatility</th>
                                            <th className="text-right p-2">Sharpe Ratio</th>
                                            <th className="text-right p-2">Beta</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {metrics.map((metric) => (
                                            <tr key={metric.stock} className="border-b hover:bg-muted/50">
                                                <td className="p-2 font-medium">{metric.stock}</td>
                                                <td className="p-2 text-right text-green-500">{metric.ytd}</td>
                                                <td className="p-2 text-right">{metric.volatility}</td>
                                                <td className="p-2 text-right">{metric.sharpe}</td>
                                                <td className="p-2 text-right">{metric.beta}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        {/* Winner Badge */}
                        <div className="p-4 bg-gradient-to-r from-yellow-500/20 to-orange-500/20 border border-yellow-500/30 rounded-lg">
                            <div className="flex items-center gap-2 font-semibold text-yellow-600 dark:text-yellow-400">
                                <TrendingUp className="h-5 w-5" />
                                Best Performer: INFY (+22.9% YTD)
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="text-center py-8 text-muted-foreground">
                        <GitCompare className="h-12 w-12 mx-auto mb-3 opacity-50" />
                        <p>Click "Show Comparison" to compare stock performance</p>
                    </div>
                )}
            </CardContent>
        </Card>
    );
};

export default CompareStocks;
