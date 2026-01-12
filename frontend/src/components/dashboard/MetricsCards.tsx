import { Card, CardContent, CardTitle } from "@/components/ui/card"
import { Activity, BarChart3, Clock, Zap } from "lucide-react"

interface MetricsProps {
    stockCount: number;
    marketScore: number;
    sentimentColor: string;
}

export function MetricsCards({ stockCount, marketScore, sentimentColor }: MetricsProps) {
    const formattedTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <Card className="border-l-4 border-l-grow-green bg-gradient-to-br from-grow-surface to-grow-dark">
                <CardContent className="flex items-center p-6">
                    <div className="p-3 rounded-full bg-grow-green/10 mr-4">
                        <BarChart3 className="w-6 h-6 text-grow-green" />
                    </div>
                    <div>
                        <p className="text-sm font-medium text-grow-gray">Stocks Tracked</p>
                        <h3 className="text-2xl font-bold text-white">{stockCount}+</h3>
                    </div>
                </CardContent>
            </Card>

            <Card className="border-l-4 border-l-cyan-500 bg-gradient-to-br from-grow-surface to-grow-dark">
                <CardContent className="flex items-center p-6">
                    <div className="p-3 rounded-full bg-cyan-500/10 mr-4">
                        <Activity className="w-6 h-6 text-cyan-500" />
                    </div>
                    <div>
                        <p className="text-sm font-medium text-grow-gray">Market Score</p>
                        <h3 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500">
                            {marketScore}/100
                        </h3>
                    </div>
                </CardContent>
            </Card>

            <Card className="border-l-4 border-l-purple-500 bg-gradient-to-br from-grow-surface to-grow-dark">
                <CardContent className="flex items-center p-6">
                    <div className="p-3 rounded-full bg-purple-500/10 mr-4">
                        <Zap className="w-6 h-6 text-purple-500" />
                    </div>
                    <div>
                        <p className="text-sm font-medium text-grow-gray">AI Accuracy</p>
                        <h3 className="text-2xl font-bold text-purple-400">92%</h3>
                    </div>
                </CardContent>
            </Card>

            <Card className="border-l-4 border-l-amber-500 bg-gradient-to-br from-grow-surface to-grow-dark">
                <CardContent className="flex items-center p-6">
                    <div className="p-3 rounded-full bg-amber-500/10 mr-4">
                        <Clock className="w-6 h-6 text-amber-500" />
                    </div>
                    <div>
                        <p className="text-sm font-medium text-grow-gray">Last Updated</p>
                        <h3 className="text-2xl font-bold text-amber-400">{formattedTime}</h3>
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}
