
import Header from "@/components/Header";
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Calculator, Sparkles, Briefcase, RefreshCcw, AlertTriangle, Bot } from "lucide-react";
import { useState } from "react";
import { runBacktest } from "@/services/api";
import { BacktestResult } from "@/types/stock";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import StockSearch from "@/components/investment-planner/StockSearch";
import PortfolioAnalysis from "@/components/investment-planner/PortfolioAnalysis";

import { useSettingsStore } from '@/store/settingsStore';

const InvestmentPlanner = () => {
  const [amount, setAmount] = useState(100000);
  const [duration, setDuration] = useState(5);
  const [risk, setRisk] = useState(50);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { aiModel } = useSettingsStore();

  const riskLabel = risk <= 25 ? 'Conservative' : risk <= 50 ? 'Moderate' : risk <= 75 ? 'Aggressive' : 'Very Aggressive';

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      // Map slider (0-100) to API enum if needed, or backend handles it.
      // Backend expects string: "Conservative", "Moderate", etc.
      const data = await runBacktest({
        amount,
        duration_years: duration,
        risk_profile: riskLabel.toLowerCase() as any,
        type: 'one-time', // Default to one-time for now
        expected_return: 12, // Default
        model: aiModel
      });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to generate plan");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
      </div>

      <Header />

      <main className="container mx-auto px-4 py-8 relative">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground flex items-center gap-3">
            <Briefcase className="w-8 h-8 text-primary" />
            Investment Planner
          </h1>
          <p className="text-muted-foreground mt-2">AI-powered portfolio suggestions and backtesting</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Input Panel */}
          <GlassCard className="lg:col-span-1 h-fit">
            <GlassCardHeader>
              <GlassCardTitle className="flex items-center gap-2">
                <Calculator className="w-4 h-4" /> Investment Parameters
              </GlassCardTitle>
            </GlassCardHeader>
            <GlassCardContent className="space-y-6">
              <div className="space-y-2">
                <Label>Investment Amount (₹)</Label>
                <Input
                  type="number"
                  value={amount}
                  onChange={(e) => setAmount(Number(e.target.value))}
                  min={1000}
                  className="font-mono"
                />
              </div>

              <div className="space-y-2">
                <Label>Duration: {duration} years</Label>
                <Slider
                  value={[duration]}
                  onValueChange={([v]) => setDuration(v)}
                  min={1}
                  max={30}
                  step={1}
                />
              </div>

              <div className="space-y-2">
                <Label>Risk Profile: {riskLabel}</Label>
                <Slider
                  value={[risk]}
                  onValueChange={([v]) => setRisk(v)}
                  min={0}
                  max={100}
                  step={1}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Safe</span>
                  <span>Risky</span>
                </div>
              </div>

              <Button className="w-full" size="lg" onClick={handleGenerate} disabled={loading}>
                {loading ? <RefreshCcw className="w-4 h-4 mr-2 animate-spin" /> : <Sparkles className="w-4 h-4 mr-2" />}
                {loading ? "Generating..." : "Get AI Suggestions"}
              </Button>

              {error && (
                <div className="text-sm text-destructive flex items-center gap-2 bg-destructive/10 p-2 rounded">
                  <AlertTriangle className="w-4 h-4" /> {error}
                </div>
              )}
            </GlassCardContent>
          </GlassCard>

          {/* Results Panel */}
          <div className="lg:col-span-2">
            <Tabs defaultValue="suggestions" className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-6">
                <TabsTrigger value="suggestions">AI Suggestions</TabsTrigger>
                <TabsTrigger value="portfolio">Portfolio Builder</TabsTrigger>
              </TabsList>

              <TabsContent value="suggestions">
                {result ? (
                  <GlassCard>
                    <GlassCardContent className="p-6 space-y-8">
                      {/* Stats Grid */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-4 bg-secondary/50 rounded-lg text-center">
                          <div className="text-xs text-muted-foreground uppercase">Exp. Return</div>
                          <div className="text-2xl font-bold text-primary">{result.metrics?.cagr?.toFixed(2)}%</div>
                        </div>
                        <div className="p-4 bg-secondary/50 rounded-lg text-center">
                          <div className="text-xs text-muted-foreground uppercase">Sharpe Ratio</div>
                          <div className="text-2xl font-bold">{result.metrics?.sharpe_ratio?.toFixed(2)}</div>
                        </div>
                        <div className="p-4 bg-secondary/50 rounded-lg text-center">
                          <div className="text-xs text-muted-foreground uppercase">Max Drawdown</div>
                          <div className="text-2xl font-bold text-destructive">{result.metrics?.max_drawdown?.toFixed(2)}%</div>
                        </div>
                        <div className="p-4 bg-secondary/50 rounded-lg text-center">
                          <div className="text-xs text-muted-foreground uppercase">Volatility</div>
                          <div className="text-2xl font-bold">{result.metrics?.volatility?.toFixed(2)}%</div>
                        </div>
                      </div>

                      {/* Chart */}
                      <div className="h-[300px] w-full">
                        <h3 className="text-sm font-medium mb-4">Projected Growth (Equity Curve)</h3>
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={result.equity_curve}>
                            <defs>
                              <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                            <XAxis dataKey="date" hide />
                            <YAxis domain={['auto', 'auto']} hide />
                            <Tooltip
                              contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                              itemStyle={{ color: 'hsl(var(--foreground))' }}
                              labelStyle={{ color: 'hsl(var(--muted-foreground))' }}
                            />
                            <Area
                              type="monotone"
                              dataKey="value"
                              stroke="hsl(var(--primary))"
                              fillOpacity={1}
                              fill="url(#colorValue)"
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>

                      {/* Strategy Report */}
                      <div>
                        <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                          <Bot className="w-5 h-5 text-primary" /> AI Strategy Report
                        </h3>
                        <div className="prose prose-invert max-w-none prose-sm bg-muted/30 p-4 rounded-lg">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {result.strategy_report || "No detailed report generated."}
                          </ReactMarkdown>
                        </div>
                      </div>
                    </GlassCardContent>
                  </GlassCard>
                ) : (
                  <GlassCard>
                    <GlassCardHeader>
                      <GlassCardTitle>AI Strategy Recommendations</GlassCardTitle>
                    </GlassCardHeader>
                    <GlassCardContent>
                      <div className="text-center py-12 text-muted-foreground">
                        <Sparkles className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>Configure your parameters and click "Get AI Suggestions"</p>
                        <p className="text-sm mt-2">Connect to backend API for live recommendations</p>
                      </div>
                    </GlassCardContent>
                  </GlassCard>
                )}
              </TabsContent>

              <TabsContent value="portfolio">
                <div className="space-y-6">
                  <GlassCard>
                    <GlassCardHeader>
                      <GlassCardTitle className="flex items-center gap-2">
                        <Briefcase className="w-5 h-5 text-primary" /> Build Your Portfolio
                      </GlassCardTitle>
                    </GlassCardHeader>
                    <GlassCardContent>
                      <StockSearch />
                    </GlassCardContent>
                  </GlassCard>

                  <PortfolioAnalysis />
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </main>
    </div>
  );
};

export default InvestmentPlanner;
