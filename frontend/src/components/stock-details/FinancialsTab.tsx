import { useEffect, useState } from 'react';
import { getStockFinancials } from '@/services/api';
import { FinancialsResponse } from '@/types/stock';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { GlassCard } from "@/components/ui/glass-card";
import { Skeleton } from "@/components/ui/skeleton";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';

interface FinancialsTabProps {
    symbol: string;
}

const FinancialsTab = ({ symbol }: FinancialsTabProps) => {
    const [data, setData] = useState<FinancialsResponse | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await getStockFinancials(symbol);
                setData(res);
            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [symbol]);

    if (loading) return <Skeleton className="h-[400px] w-full" />;

    if (!data) return <div className="text-muted-foreground">No financial data available.</div>;

    // Helper to transform data for charts
    // Input: Array of { "index": "MetricName", "Date1": Val1, "Date2": Val2 }
    // Output: Array of { "date": "Date1", "MetricName": Val1, ... }
    const transformDataForChart = (rows: any[], metrics: string[]) => {
        if (!rows || rows.length === 0) return [];

        // Extract dates (keys that are not 'index' or 'level_0' or 'Metric')
        const allKeys = Object.keys(rows[0]);
        const dateKeys = allKeys.filter(k => !['index', 'level_0', 'Metric'].includes(k));

        // Sort dates ascending
        dateKeys.sort();

        return dateKeys.map(date => {
            const point: any = { date: new Date(date).getFullYear() }; // Use Year for label
            metrics.forEach(metric => {
                const row = rows.find(r => r['index'] === metric || r['Metric'] === metric);
                if (row) {
                    point[metric] = row[date];
                }
            });
            return point;
        });
    };

    const incomeChartData = transformDataForChart(data.income_stmt, ['Total Revenue', 'Net Income', 'Operating Income']);
    const balanceChartData = transformDataForChart(data.balance_sheet, ['Total Assets', 'Total Liab', 'Total Stockholder Equity']);
    const cashFlowChartData = transformDataForChart(data.cashflow, ['Total Cash From Operating Activities', 'Capital Expenditures']);

    const renderTable = (rows: any[]) => {
        if (!rows || rows.length === 0) return <div className="p-4">No data</div>;
        const keys = Object.keys(rows[0]).filter(k => k !== 'index' && k !== 'level_0');

        return (
            <div className="overflow-x-auto rounded-lg border border-border/50">
                <table className="w-full text-sm text-left">
                    <thead className="text-xs text-muted-foreground uppercase bg-secondary/50">
                        <tr>
                            <th className="px-6 py-4 sticky left-0 bg-background/95 backdrop-blur z-10">Metric</th>
                            {keys.map(k => <th key={k} className="px-6 py-4 whitespace-nowrap">{k}</th>)}
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row, i) => (
                            <tr key={i} className="border-b border-border/50 hover:bg-secondary/20 transition-colors">
                                <td className="px-6 py-3 font-medium text-foreground sticky left-0 bg-background/95 backdrop-blur border-r border-border/50 min-w-[200px]">
                                    {row['index'] || row['Metric']}
                                </td>
                                {keys.map(k => (
                                    <td key={k} className="px-6 py-3 font-mono text-muted-foreground whitespace-nowrap">
                                        {typeof row[k] === 'number' ?
                                            (Math.abs(row[k]) > 10000000 ? `₹${(row[k] / 10000000).toFixed(2)}Cr` : row[k].toLocaleString())
                                            : row[k]}
                                    </td>
                                ))}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        );
    };

    return (
        <GlassCard className="p-6">
            <Tabs defaultValue="income" className="w-full">
                <TabsList className="mb-6 grid w-full grid-cols-3">
                    <TabsTrigger value="income">Income Statement</TabsTrigger>
                    <TabsTrigger value="balance">Balance Sheet</TabsTrigger>
                    <TabsTrigger value="cashflow">Cash Flow</TabsTrigger>
                </TabsList>

                <TabsContent value="income" className="space-y-6">
                    {incomeChartData.length > 0 && (
                        <div className="h-[300px] w-full p-4 bg-secondary/10 rounded-xl border border-white/5">
                            <h3 className="text-sm font-semibold mb-4 text-center">Revenue vs Net Income Trend</h3>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={incomeChartData}>
                                    <CartesianGrid strokeDasharray="3 3" opacity={0.1} vertical={false} />
                                    <XAxis dataKey="date" stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} />
                                    <YAxis stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `₹${(val / 10000000).toFixed(0)}Cr`} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid rgba(255,255,255,0.1)' }}
                                        formatter={(val: number) => `₹${(val / 10000000).toFixed(2)}Cr`}
                                    />
                                    <Legend />
                                    <Bar dataKey="Total Revenue" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Revenue" />
                                    <Bar dataKey="Net Income" fill="#10b981" radius={[4, 4, 0, 0]} name="Net Profit" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                    {renderTable(data.income_stmt)}
                </TabsContent>

                <TabsContent value="balance" className="space-y-6">
                    {balanceChartData.length > 0 && (
                        <div className="h-[300px] w-full p-4 bg-secondary/10 rounded-xl border border-white/5">
                            <h3 className="text-sm font-semibold mb-4 text-center">Assets vs Liabilities</h3>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={balanceChartData}>
                                    <CartesianGrid strokeDasharray="3 3" opacity={0.1} vertical={false} />
                                    <XAxis dataKey="date" stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} />
                                    <YAxis stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `₹${(val / 10000000).toFixed(0)}Cr`} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid rgba(255,255,255,0.1)' }}
                                        formatter={(val: number) => `₹${(val / 10000000).toFixed(2)}Cr`}
                                    />
                                    <Legend />
                                    <Bar dataKey="Total Assets" fill="#8884d8" radius={[4, 4, 0, 0]} name="Assets" />
                                    <Bar dataKey="Total Liab" fill="#ef4444" radius={[4, 4, 0, 0]} name="Liabilities" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                    {renderTable(data.balance_sheet)}
                </TabsContent>

                <TabsContent value="cashflow" className="space-y-6">
                    {cashFlowChartData.length > 0 && (
                        <div className="h-[300px] w-full p-4 bg-secondary/10 rounded-xl border border-white/5">
                            <h3 className="text-sm font-semibold mb-4 text-center">Operating Cash Flow Trend</h3>
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={cashFlowChartData}>
                                    <CartesianGrid strokeDasharray="3 3" opacity={0.1} vertical={false} />
                                    <XAxis dataKey="date" stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} />
                                    <YAxis stroke="#6b7280" fontSize={12} tickLine={false} axisLine={false} tickFormatter={(val) => `₹${(val / 10000000).toFixed(0)}Cr`} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid rgba(255,255,255,0.1)' }}
                                        formatter={(val: number) => `₹${(val / 10000000).toFixed(2)}Cr`}
                                    />
                                    <Legend />
                                    <Line type="monotone" dataKey="Total Cash From Operating Activities" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} name="Operating Cash Flow" />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    )}
                    {renderTable(data.cashflow)}
                </TabsContent>
            </Tabs>
        </GlassCard>
    );
};

export default FinancialsTab;
