// Correlation Matrix Component
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Network } from "lucide-react";

interface CorrelationMatrixProps {
    symbols?: string[];
    isLoading?: boolean;
}

const CorrelationMatrix = ({ symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"], isLoading }: CorrelationMatrixProps) => {
    // Mock correlation data - in production, calculate from price data
    const correlationData = [
        { stock: "RELIANCE", values: [1.00, 0.45, 0.38] },
        { stock: "TCS", values: [0.45, 1.00, 0.82] },
        { stock: "INFY", values: [0.38, 0.82, 1.00] },
    ];

    const getCorrelationColor = (value: number) => {
        if (value >= 0.7) return "bg-green-600";
        if (value >= 0.4) return "bg-green-400";
        if (value >= 0) return "bg-yellow-400";
        if (value >= -0.4) return "bg-orange-400";
        return "bg-red-600";
    };

    const getCorrelationText = (value: number) => {
        if (value >= 0.7) return "Strong +";
        if (value >= 0.4) return "Moderate +";
        if (value >= 0) return "Weak +";
        if (value >= -0.4) return "Weak -";
        return "Strong -";
    };

    if (isLoading) {
        return (
            <Card>
                <CardHeader>
                    <CardTitle>Correlation Matrix</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="h-64 bg-muted animate-pulse rounded" />
                </CardContent>
            </Card>
        );
    }

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Network className="h-5 w-5" />
                    Stock Correlation Matrix
                </CardTitle>
            </CardHeader>
            <CardContent>
                {/* Correlation Grid */}
                <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                        <thead>
                            <tr>
                                <th className="p-2 text-left text-sm font-medium"></th>
                                {correlationData.map((stock) => (
                                    <th key={stock.stock} className="p-2 text-center text-sm font-medium">
                                        {stock.stock}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {correlationData.map((row, rowIndex) => (
                                <tr key={row.stock}>
                                    <td className="p-2 font-medium text-sm">{row.stock}</td>
                                    {row.values.map((value, colIndex) => (
                                        <td key={colIndex} className="p-1">
                                            <div
                                                className={`${getCorrelationColor(value)} text-white text-center py-3 px-2 rounded font-semibold text-sm transition-all hover:scale-105 cursor-pointer`}
                                                title={`${getCorrelationText(value)}: ${value.toFixed(2)}`}
                                            >
                                                {value.toFixed(2)}
                                            </div>
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Legend */}
                <div className="mt-6 grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-green-600 rounded" />
                        <span>Strong +ve (≥0.7)</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-green-400 rounded" />
                        <span>Moderate +ve</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-yellow-400 rounded" />
                        <span>Weak +ve</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-orange-400 rounded" />
                        <span>Weak -ve</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-4 h-4 bg-red-600 rounded" />
                        <span>Strong -ve (≤-0.7)</span>
                    </div>
                </div>

                {/* Insights */}
                <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                    <div className="text-sm font-medium mb-1">💡 Diversification Insight</div>
                    <div className="text-xs text-muted-foreground">
                        TCS and INFY show high correlation (0.82) - consider diversifying across different sectors for better risk management.
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

export default CorrelationMatrix;
