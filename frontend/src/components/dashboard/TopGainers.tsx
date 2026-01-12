import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowUpRight } from "lucide-react"
import Link from "next/link"

interface Stock {
    symbol: string;
    price: number;
    change_pct: number;
}

interface TopGainersProps {
    gainers: Stock[];
}

export function TopGainers({ gainers }: TopGainersProps) {
    return (
        <Card className="h-full">
            <CardHeader>
                <CardTitle>Top Performers</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="space-y-4">
                    {gainers.map((stock, i) => (
                        <Link key={stock.symbol} href={`/stock/${stock.symbol}`} className="block">
                            <div className="flex items-center justify-between p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors group cursor-pointer">
                                <div className="flex items-center space-x-3">
                                    <div className="p-2 rounded-full bg-grow-green/20 text-grow-green group-hover:bg-grow-green group-hover:text-white transition-colors">
                                        <ArrowUpRight className="w-4 h-4" />
                                    </div>
                                    <div>
                                        <p className="font-bold text-white">{stock.symbol}</p>
                                        <p className="text-xs text-grow-gray">NSE</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className="font-bold text-grow-green">+{stock.change_pct.toFixed(2)}%</p>
                                    <p className="text-sm text-gray-400">₹{stock.price.toFixed(2)}</p>
                                </div>
                            </div>
                        </Link>
                    ))}
                    {!gainers.length && (
                        <p className="text-center text-grow-gray py-4">No data available</p>
                    )}
                </div>
            </CardContent>
        </Card>
    )
}
