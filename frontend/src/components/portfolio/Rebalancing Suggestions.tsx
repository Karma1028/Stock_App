// Rebalancing Suggestions Component
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Scale, ArrowRight, TrendingDown, TrendingUp } from "lucide-react";

const RebalancingSuggestions = () => {
    const suggestions = [
        {
            action: "Reduce",
            stock: "TCS",
            currentAllocation: 25,
            targetAllocation: 20,
            amount: "₹50,000",
            reason: "Overweight in IT sector",
            priority: "High",
        },
        {
            action: "Increase",
            stock: "HDFC Bank",
            currentAllocation: 10,
            targetAllocation: 15,
            amount: "₹25,000",
            reason: "Underweight in Banking",
            priority: "Medium",
        },
        {
            action: "Add",
            stock: "FMCG Sector",
            currentAllocation: 0,
            targetAllocation: 10,
            amount: "₹50,000",
            reason: "Missing defensive allocation",
            priority: "High",
        },
    ];

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    <span className="flex items-center gap-2">
                        <Scale className="h-5 w-5" />
                        Rebalancing Suggestions
                    </span>
                    <Button size="sm">Auto-Rebalance</Button>
                </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                {suggestions.map((suggestion, index) => (
                    <div key={index} className="p-4 border rounded-lg hover:shadow-md transition-shadow">
                        <div className="flex items-start justify-between mb-3">
                            <div className="flex items-center gap-3">
                                {suggestion.action === "Reduce" && <TrendingDown className="h-5 w-5 text-red-500" />}
                                {suggestion.action === "Increase" && <TrendingUp className="h-5 w-5 text-green-500" />}
                                {suggestion.action === "Add" && <ArrowRight className="h-5 w-5 text-blue-500" />}
                                <div>
                                    <div className="font-semibold">{suggestion.action} {suggestion.stock}</div>
                                    <div className="text-sm text-muted-foreground">{suggestion.reason}</div>
                                </div>
                            </div>
                            <span
                                className={`px-2 py-1 text-xs font-semibold rounded ${suggestion.priority === "High"
                                        ? "bg-red-500/20 text-red-500"
                                        : "bg-yellow-500/20 text-yellow-500"
                                    }`}
                            >
                                {suggestion.priority}
                            </span>
                        </div>

                        <div className="flex items-center gap-3 text-sm">
                            <div className="flex items-center gap-2">
                                <span className="text-muted-foreground">Current:</span>
                                <span className="font-semibold">{suggestion.currentAllocation}%</span>
                            </div>
                            <ArrowRight className="h-4 w-4 text-muted-foreground" />
                            <div className="flex items-center gap-2">
                                <span className="text-muted-foreground">Target:</span>
                                <span className="font-semibold text-blue-500">{suggestion.targetAllocation}%</span>
                            </div>
                            <div className="ml-auto font-semibold">{suggestion.amount}</div>
                        </div>

                        {/* Allocation Bar */}
                        <div className="mt-3 flex items-center gap-2">
                            <div className="flex-1 bg-secondary rounded-full h-2 overflow-hidden">
                                <div
                                    className="bg-blue-500 h-2 transition-all"
                                    style={{ width: `${suggestion.currentAllocation}%` }}
                                />
                            </div>
                            <div className="flex-1 bg-secondary rounded-full h-2 overflow-hidden">
                                <div
                                    className="bg-green-500 h-2 transition-all"
                                    style={{ width: `${suggestion.targetAllocation}%` }}
                                />
                            </div>
                        </div>
                    </div>
                ))}

                {/* Summary */}
                <div className="p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                    <div className="text-sm font-medium mb-1">Total Rebalancing Required</div>
                    <div className="text-2xl font-bold text-blue-500">₹1,25,000</div>
                    <div className="text-xs text-muted-foreground mt-1">
                        This will bring your portfolio closer to target allocation by 15%
                    </div>
                </div>
            </CardContent>
        </Card>
    );
};

export default RebalancingSuggestions;
