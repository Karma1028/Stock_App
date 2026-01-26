import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { useNavigate } from "react-router-dom";

interface SectorData {
    name: string;
    change: number;
    companies: number;
    marketCap: number;
}

const SectorHeatmap = () => {
    const navigate = useNavigate();
    const [sectors, setSectors] = useState<SectorData[]>([
        { name: "IT", change: 2.3, companies: 89, marketCap: 12500000 },
        { name: "Banking", change: 1.8, companies: 45, marketCap: 18500000 },
        { name: "Pharma", change: -0.5, companies: 32, marketCap: 5200000 },
        { name: "Auto", change: 0.8, companies: 28, marketCap: 3800000 },
        { name: "FMCG", change: -1.2, companies: 24, marketCap: 4500000 },
        { name: "Energy", change: 3.1, companies: 18, marketCap: 8900000 },
        { name: "Metals", change: -2.1, companies: 22, marketCap: 3200000 },
        { name: "Realty", change: 0.3, companies: 15, marketCap: 1800000 },
        { name: "Telecom", change: 1.5, companies: 8, marketCap: 4200000 },
        { name: "Infrastructure", change: -0.8, companies: 35, marketCap: 2900000 },
    ]);

    const getColorIntensity = (change: number) => {
        const absChange = Math.abs(change);
        if (change > 0) {
            // Green shades
            if (absChange > 2.5) return "bg-green-600/90";
            if (absChange > 1.5) return "bg-green-500/80";
            if (absChange > 0.5) return "bg-green-400/70";
            return "bg-green-300/60";
        } else if (change < 0) {
            // Red shades
            if (absChange > 2.5) return "bg-red-600/90";
            if (absChange > 1.5) return "bg-red-500/80";
            if (absChange > 0.5) return "bg-red-400/70";
            return "bg-red-300/60";
        }
        return "bg-gray-400/60";
    };

    const getTrendIcon = (change: number) => {
        if (change > 0.1) return <TrendingUp className="h-4 w-4" />;
        if (change < -0.1) return <TrendingDown className="h-4 w-4" />;
        return <Minus className="h-4 w-4" />;
    };

    return (
        <Card className="border-border/40">
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    <span>Sector Performance Heatmap</span>
                    <span className="text-sm text-muted-foreground font-normal">
                        Daily Change %
                    </span>
                </CardTitle>
            </CardHeader>
            <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
                    {sectors.map((sector) => (
                        <div
                            key={sector.name}
                            onClick={() => navigate(`/analysis?sector=${sector.name}`)}
                            className={`group relative p-4 rounded-lg border border-white/10 transition-all duration-300 hover:scale-[1.05] hover:shadow-2xl hover:z-10 cursor-pointer ${getColorIntensity(
                                sector.change
                            )}`}
                        >
                            <div className="flex flex-col gap-2">
                                <div className="flex items-center justify-between">
                                    <span className="font-semibold text-white text-sm">
                                        {sector.name}
                                    </span>
                                    <span className="text-white opacity-80">
                                        {getTrendIcon(sector.change)}
                                    </span>
                                </div>
                                <div className="text-2xl font-bold text-white">
                                    {sector.change > 0 ? "+" : ""}
                                    {sector.change.toFixed(2)}%
                                </div>
                                <div className="flex items-center justify-between text-xs text-white/80">
                                    <span>{sector.companies} stocks</span>
                                    <span>₹{(sector.marketCap / 10000).toFixed(0)}K Cr</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Legend */}
                <div className="mt-4 flex items-center justify-center gap-6 text-xs text-muted-foreground">
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-green-500 rounded" />
                        <span>Gainers</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-gray-400 rounded" />
                        <span>Neutral</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-red-500 rounded" />
                        <span>Losers</span>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

export default SectorHeatmap;
