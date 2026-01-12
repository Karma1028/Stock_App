import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { ArrowUp, ArrowDown } from "lucide-react"

interface StockHeaderProps {
    symbol: string;
    longName: string;
    price: number;
    change: number;
    changePct: number;
    sector: string;
    industry: string;
}

export function StockHeader({ symbol, longName, price, change, changePct, sector, industry }: StockHeaderProps) {
    const isPositive = change >= 0;

    return (
        <div className="mb-8">
            <div className="flex justify-between items-start">
                <div>
                    <h1 className="text-4xl font-bold text-white mb-2">{longName}</h1>
                    <div className="flex items-center space-x-3 mb-4">
                        <span className="text-xl font-semibold text-grow-gray">{symbol}</span>
                        <div className="px-3 py-1 rounded-full bg-white/5 border border-white/10 text-sm text-grow-gray">
                            {sector} • {industry}
                        </div>
                    </div>
                </div>

                <div className="text-right">
                    <h2 className="text-4xl font-mono font-bold text-white mb-1">
                        ₹{price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                    </h2>
                    <div className={`flex items-center justify-end space-x-2 ${isPositive ? 'text-grow-green' : 'text-red-500'}`}>
                        {isPositive ? <ArrowUp className="w-5 h-5" /> : <ArrowDown className="w-5 h-5" />}
                        <span className="text-xl font-semibold">
                            {Math.abs(change).toFixed(2)} ({Math.abs(changePct).toFixed(2)}%)
                        </span>
                    </div>
                </div>
            </div>
        </div>
    )
}
