// Risk Assessment Component for Portfolio
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Shield, AlertTriangle, TrendingUp } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";

interface RiskAssessmentProps {
    portfolio?: any[];
}

const RiskAssessment = ({ portfolio = [] }: RiskAssessmentProps) => {
    // Mock risk data
    const riskDistribution = [
        { name: "Low Risk", value: 35, color: "hsl(var(--chart-1))" },
        { name: "Medium Risk", value: 45, color: "hsl(var(--chart-3))" },
        { name: "High Risk", value: 20, color: "hsl(var(--chart-5))" },
    ];

    const riskMetrics = {
        portfolioBeta: 1.08,
        standardDeviation: 18.5,
        value AtRisk: 12.3,
        sharpeRatio: 1.45,
        maxDrawdown: -15.2,
        diversificationScore: 78,
    };

    const getRiskLevel = () => {
        if (riskMetrics.portfolioBeta < 0.8) return { level: "Low", color: "text-green-500", icon: Shield };
        if (riskMetrics.portfolioBeta < 1.2) return { level: "Medium", color: "text-yellow-500", icon: AlertTriangle };
        return { level: "High", color: "text-red-500", icon: AlertTriangle };
    };

    const risk = getRiskLevel();
    const RiskIcon = risk.icon;

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Shield className="h-5 w-5" />
                    Portfolio Risk Assessment
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Overall Risk Level */}
                <div className="p-4 bg-secondary/50 rounded-lg">
                    <div className="flex items-center justify-between">
                        <div>
                            <div className="text-sm text-muted-foreground mb-1">Overall Risk Level</div>
                            <div className={`text-2xl font-bold ${risk.color} flex items-center gap-2`}>
                                <RiskIcon className="h-6 w-6" />
                                {risk.level}
                            </div>
                        </div>
                        <div className="text-right">
                            <div className="text-sm text-muted-foreground mb-1">Portfolio Beta</div>
                            <div className="text-2xl font-bold">{riskMetrics.portfolioBeta}</div>
                        </div>
                    </div>
                </div>

                {/* Risk Distribution Pie Chart */}
                <div>
                    <h4 className="text-sm font-medium mb-3">Risk Distribution</h4>
                    <ResponsiveContainer width="100%" height={200}>
                        <PieChart>
                            <Pie
                                data={riskDistribution}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                            >
                                {riskDistribution.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                            </Pie>
                            <Tooltip />
                        </PieChart>
                    </ResponsiveContainer>
                    <div className="flex items-center justify-center gap-4 mt-2 text-xs">
                        {riskDistribution.map((item) => (
                            <div key={item.name} className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                                <span>{item.name}: {item.value}%</span>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Risk Metrics Grid */}
                <div>
                    <h4 className="text-sm font-medium mb-3">Risk Metrics</h4>
                    <div className="grid grid-cols-2 gap-3">
                        <div className="p-3 bg-secondary/30 rounded-lg">
                            <div className="text-xs text-muted-foreground mb-1">Volatility (σ)</div>
                            <div className="text-lg font-bold">{riskMetrics.standardDeviation}%</div>
                        </div>
                        <div className="p-3 bg-secondary/30 rounded-lg">
                            <div className="text-xs text-muted-foreground mb-1">Value at Risk</div>
                            <div className="text-lg font-bold text-red-500">{riskMetrics.valueAtRisk}%</div>
                        </div>
                        <div className="p-3 bg-secondary/30 rounded-lg">
                            <div className="text-xs text-muted-foreground mb-1">Sharpe Ratio</div>
                            <div className="text-lg font-bold text-green-500">{riskMetrics.sharpeRatio}</div>
                        </div>
                        <div className="p-3 bg-secondary/30 rounded-lg">
                            <div className="text-xs text-muted-foreground mb-1">Max Drawdown</div>
                            <div className="text-lg font-bold text-red-500">{riskMetrics.maxDrawdown}%</div>
                        </div>
                    </div>
                </div>

                {/* Diversification Score */}
                <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">Diversification Score</span>
                        <span className="text-2xl font-bold text-blue-500">{riskMetrics.diversificationScore}/100</span>
                    </div>
                    <div className="w-full bg-secondary rounded-full h-2">
                        <div
                            className="bg-blue-500 h-2 rounded-full transition-all"
                            style={{ width: `${riskMetrics.diversificationScore}%` }}
                        />
                    </div>
                    <div className="mt-2 text-xs text-muted-foreground">
                        Good diversification! Your portfolio spans across multiple sectors.
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

export default RiskAssessment;
