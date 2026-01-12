"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Trash2, Plus, PieChart as PieIcon } from "lucide-react"
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as ReTooltip } from 'recharts';

interface PortfolioItem {
    id: string;
    symbol: string;
    quantity: number;
    avgPrice: number;
}

export default function PortfolioPage() {
    const [items, setItems] = useState<PortfolioItem[]>([]);
    const [symbol, setSymbol] = useState("");
    const [quantity, setQuantity] = useState("");
    const [avgPrice, setAvgPrice] = useState("");

    // Load from local storage
    useEffect(() => {
        const saved = localStorage.getItem("portfolio");
        if (saved) setItems(JSON.parse(saved));
    }, []);

    // Save
    useEffect(() => {
        localStorage.setItem("portfolio", JSON.stringify(items));
    }, [items]);

    const addItem = () => {
        if (!symbol || !quantity || !avgPrice) return;
        const newItem: PortfolioItem = {
            id: Date.now().toString(),
            symbol: symbol.toUpperCase(),
            quantity: Number(quantity),
            avgPrice: Number(avgPrice)
        };
        setItems([...items, newItem]);
        setSymbol("");
        setQuantity("");
        setAvgPrice("");
    };

    const removeItem = (id: string) => {
        setItems(items.filter(i => i.id !== id));
    };

    const totalValue = items.reduce((acc, item) => acc + (item.quantity * item.avgPrice), 0);

    const data = items.map(i => ({
        name: i.symbol,
        value: i.quantity * i.avgPrice
    }));

    const COLORS = ['#00b386', '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#a855f7'];

    return (
        <div className="space-y-8 animate-in fade-in duration-500">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">Portfolio Manager</h1>
                    <p className="text-grow-gray">Track and analyze your investments</p>
                </div>
                <div className="text-right">
                    <p className="text-sm text-grow-gray">Total Invested</p>
                    <h2 className="text-3xl font-bold text-grow-green">₹{totalValue.toLocaleString()}</h2>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Add Form */}
                <Card>
                    <CardHeader><CardTitle>Add Holding</CardTitle></CardHeader>
                    <CardContent className="space-y-4">
                        <div>
                            <label className="text-sm text-gray-400">Symbol</label>
                            <input
                                className="w-full p-2 rounded bg-white/5 border border-white/10 text-white focus:outline-none focus:border-grow-green"
                                placeholder="e.g. RELIANCE"
                                value={symbol}
                                onChange={(e) => setSymbol(e.target.value)}
                            />
                        </div>
                        <div>
                            <label className="text-sm text-gray-400">Quantity</label>
                            <input
                                type="number"
                                className="w-full p-2 rounded bg-white/5 border border-white/10 text-white focus:outline-none focus:border-grow-green"
                                placeholder="0"
                                value={quantity}
                                onChange={(e) => setQuantity(e.target.value)}
                            />
                        </div>
                        <div>
                            <label className="text-sm text-gray-400">Avg Price</label>
                            <input
                                type="number"
                                className="w-full p-2 rounded bg-white/5 border border-white/10 text-white focus:outline-none focus:border-grow-green"
                                placeholder="₹0.00"
                                value={avgPrice}
                                onChange={(e) => setAvgPrice(e.target.value)}
                            />
                        </div>
                        <Button className="w-full mt-4" onClick={addItem} disabled={!symbol}>
                            <Plus className="w-4 h-4 mr-2" /> Add Stock
                        </Button>
                    </CardContent>
                </Card>

                {/* Holdings List */}
                <Card className="lg:col-span-1 border-white/5 bg-grow-dark/30">
                    <CardHeader><CardTitle>Your Holdings</CardTitle></CardHeader>
                    <CardContent>
                        <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                            {items.map(item => (
                                <div key={item.id} className="flex items-center justify-between p-3 rounded-lg bg-white/5">
                                    <div>
                                        <p className="font-bold text-white">{item.symbol}</p>
                                        <div className="flex space-x-2 text-xs text-gray-400">
                                            <span>{item.quantity} shares</span>
                                            <span>@ ₹{item.avgPrice}</span>
                                        </div>
                                    </div>
                                    <div className="flex items-center space-x-3">
                                        <span className="font-semibold text-white">₹{(item.quantity * item.avgPrice).toLocaleString()}</span>
                                        <button onClick={() => removeItem(item.id)} className="text-gray-500 hover:text-red-500">
                                            <Trash2 className="w-4 h-4" />
                                        </button>
                                    </div>
                                </div>
                            ))}
                            {!items.length && <p className="text-center text-gray-500">No holdings added.</p>}
                        </div>
                    </CardContent>
                </Card>

                {/* Chart */}
                <Card>
                    <CardHeader><CardTitle>Allocation</CardTitle></CardHeader>
                    <CardContent className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={data}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={50}
                                    outerRadius={80}
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {data.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} stroke="none" />
                                    ))}
                                </Pie>
                                <ReTooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} itemStyle={{ color: '#fff' }} />
                            </PieChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
}
