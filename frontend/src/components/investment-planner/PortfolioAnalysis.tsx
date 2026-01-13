
import { usePortfolioStore } from "@/store/portfolioStore";
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";
import {
    PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
    BarChart, Bar, XAxis, YAxis, CartesianGrid,
    ScatterChart, Scatter, ZAxis
} from "recharts";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Trash2 } from "lucide-react";

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

const PortfolioAnalysis = () => {
    const { portfolio, totalValue, removeStock } = usePortfolioStore();

    if (portfolio.length === 0) {
        return (
            <div className="text-center py-10 text-muted-foreground">
                No stocks in portfolio. Add some stocks to see analysis.
            </div>
        );
    }

    // Prepare Data for Charts

    // 1. Sector Allocation
    const sectorData = Object.values(portfolio.reduce((acc, stock) => {
        const val = stock.quantity * stock.average_price;
        if (!acc[stock.sector]) acc[stock.sector] = { name: stock.sector, value: 0 };
        acc[stock.sector].value += val;
        return acc;
    }, {} as Record<string, { name: string; value: number }>));

    // 2. Market Cap Distribution
    const capData = Object.values(portfolio.reduce((acc, stock) => {
        const val = stock.quantity * stock.average_price;
        if (!acc[stock.market_cap_category]) acc[stock.market_cap_category] = { name: stock.market_cap_category, value: 0 };
        acc[stock.market_cap_category].value += val;
        return acc;
    }, {} as Record<string, { name: string; value: number }>));

    // 3. Risk vs Reward
    const scatterData = portfolio.map(s => ({
        x: s.volatility, // Risk
        y: s.one_year_return, // Reward
        z: (s.quantity * s.average_price), // Weight (Size)
        name: s.symbol
    }));

    return (
        <div className="space-y-8">
            {/* Holdings Table */}
            <GlassCard>
                <GlassCardHeader>
                    <GlassCardTitle>Current Holdings (Total: ₹{totalValue.toLocaleString('en-IN', { maximumFractionDigits: 0 })})</GlassCardTitle>
                </GlassCardHeader>
                <GlassCardContent>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Symbol</TableHead>
                                <TableHead>Qty</TableHead>
                                <TableHead>Avg. Price</TableHead>
                                <TableHead>Value</TableHead>
                                <TableHead>Sector</TableHead>
                                <TableHead></TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {portfolio.map((stock) => (
                                <TableRow key={stock.symbol}>
                                    <TableCell className="font-medium">{stock.symbol}</TableCell>
                                    <TableCell>{stock.quantity}</TableCell>
                                    <TableCell>₹{stock.average_price.toFixed(2)}</TableCell>
                                    <TableCell>₹{(stock.quantity * stock.average_price).toFixed(2)}</TableCell>
                                    <TableCell>{stock.sector}</TableCell>
                                    <TableCell>
                                        <Button variant="ghost" size="icon" onClick={() => removeStock(stock.symbol)}>
                                            <Trash2 className="h-4 w-4 text-destructive" />
                                        </Button>
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </GlassCardContent>
            </GlassCard>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Sector Allocation */}
                <GlassCard className="h-[400px]">
                    <GlassCardHeader>
                        <GlassCardTitle>Sector Allocation</GlassCardTitle>
                    </GlassCardHeader>
                    <GlassCardContent className="h-[320px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={sectorData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={100}
                                    fill="#8884d8"
                                    paddingAngle={5}
                                    dataKey="value"
                                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                >
                                    {sectorData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip formatter={(val: number) => `₹${val.toFixed(0)}`} />
                            </PieChart>
                        </ResponsiveContainer>
                    </GlassCardContent>
                </GlassCard>

                {/* Market Cap Distribution */}
                <GlassCard className="h-[400px]">
                    <GlassCardHeader>
                        <GlassCardTitle>Market Cap Distribution</GlassCardTitle>
                    </GlassCardHeader>
                    <GlassCardContent className="h-[320px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={capData}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                                <YAxis tickFormatter={(val) => `₹${(val / 1000).toFixed(0)}k`} />
                                <Tooltip formatter={(val: number) => `₹${val.toFixed(0)}`} />
                                <Bar dataKey="value" fill="#82ca9d" radius={[4, 4, 0, 0]}>
                                    {capData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </GlassCardContent>
                </GlassCard>

                {/* Risk vs Reward */}
                <GlassCard className="h-[400px] md:col-span-2">
                    <GlassCardHeader>
                        <GlassCardTitle>Risk vs Reward Analysis</GlassCardTitle>
                    </GlassCardHeader>
                    <GlassCardContent className="h-[320px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid />
                                <XAxis type="number" dataKey="x" name="Volatility" unit="%" label={{ value: 'Risk (Volatility)', position: 'insideBottom', offset: -10 }} />
                                <YAxis type="number" dataKey="y" name="Return" unit="%" label={{ value: 'Return (1Y)', angle: -90, position: 'insideLeft' }} />
                                <ZAxis type="number" dataKey="z" range={[60, 400]} name="Value" />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                <Scatter name="Stocks" data={scatterData} fill="#8884d8">
                                    {scatterData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </GlassCardContent>
                </GlassCard>
            </div>
        </div>
    );
};

export default PortfolioAnalysis;
