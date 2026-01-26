import { useState } from "react";
import { Briefcase, Calculator, Sparkles, RefreshCcw, ShieldAlert, PieChart } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";
import Header from "@/components/Header";
import { useSettingsStore } from "@/store/settingsStore";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import PortfolioAnalysis from "@/components/investment-planner/PortfolioAnalysis";
import { runBacktest } from "@/services/api";
import { BacktestResult } from "@/types/stock";
import { AlertTriangle } from "lucide-react";

const InvestmentPlanner = () => {
  const [amount, setAmount] = useState(100000);
  const [duration, setDuration] = useState(5);
  const [risk, setRisk] = useState(50);
  const [type, setType] = useState<"one-time" | "sip">("one-time");
  const [experience, setExperience] = useState("Intermediate");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showModal, setShowModal] = useState(false);
  const { aiModel } = useSettingsStore();

  const riskLabel = risk <= 25 ? 'Conservative' : risk <= 50 ? 'Moderate' : risk <= 75 ? 'Aggressive' : 'Very Aggressive';

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await runBacktest({
        amount,
        duration_years: duration,
        risk_profile: (riskLabel.toLowerCase()) as any,
        type: type,
        expected_return: 12, // Default
        model: aiModel
      });

      if (!data || !data.metrics) throw new Error("Invalid backtest data received");

      setResult(data);
      setShowModal(true);
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
          <p className="text-muted-foreground mt-2">Quantitative portfolio backtesting & strategy optimization</p>
        </div>

        <GlassCard className="max-w-2xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-700">
          <GlassCardHeader>
            <GlassCardTitle className="flex items-center gap-2">
              <Calculator className="w-4 h-4" /> Strategy Parameters
            </GlassCardTitle>
          </GlassCardHeader>
          <GlassCardContent className="space-y-6">
            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Investment Type</Label>
                <div className="flex bg-secondary/30 p-1 rounded-lg">
                  <button
                    className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${type === 'one-time' ? 'bg-primary text-primary-foreground shadow' : 'text-muted-foreground hover:text-foreground'}`}
                    onClick={() => setType('one-time')}
                  >
                    Lumpsum
                  </button>
                  <button
                    className={`flex-1 py-2 text-sm font-medium rounded-md transition-all ${type === 'sip' ? 'bg-primary text-primary-foreground shadow' : 'text-muted-foreground hover:text-foreground'}`}
                    onClick={() => setType('sip')}
                  >
                    SIP
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Experience Level</Label>
                  <select
                    className="flex h-10 w-full rounded-md border border-input bg-background/50 px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                    value={experience}
                    onChange={(e) => setExperience(e.target.value)}
                  >
                    <option value="Beginner">Beginner</option>
                    <option value="Intermediate">Intermediate</option>
                    <option value="Advanced">Advanced</option>
                  </select>
                </div>
                <div className="space-y-2">
                  <Label>Investment Amount (₹)</Label>
                  <Input
                    type="number"
                    value={amount}
                    onChange={(e) => setAmount(Number(e.target.value))}
                    min={1000}
                    className="font-mono bg-background/50"
                  />
                </div>
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
                <Label>Risk Apetite: {riskLabel}</Label>
                <Slider
                  value={[risk]}
                  onValueChange={([v]) => setRisk(v)}
                  min={0}
                  max={100}
                  step={1}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Conservative</span>
                  <span>Aggressive</span>
                </div>
              </div>
            </div>

            <Button className="w-full text-md font-semibold h-12" size="lg" onClick={handleGenerate} disabled={loading}>
              {loading ? <RefreshCcw className="w-5 h-5 mr-2 animate-spin" /> : <Sparkles className="w-5 h-5 mr-2" />}
              {loading ? "Simulating Backtest..." : "Run Backtest Analysis"}
            </Button>

            {error && (
              <div className="text-sm text-destructive flex items-center gap-2 bg-destructive/10 p-3 rounded-md">
                <AlertTriangle className="w-4 h-4" /> {error}
              </div>
            )}
          </GlassCardContent>
        </GlassCard>

        {/* Results Modal */}
        <Dialog open={showModal} onOpenChange={setShowModal}>
          <DialogContent className="max-w-4xl h-[90vh] overflow-y-auto bg-background/95 backdrop-blur-xl border-white/10 p-0 sm:rounded-2xl">
            {result && (
              <div className="flex flex-col h-full">
                <div className="p-6 border-b border-border/50 sticky top-0 bg-background/95 backdrop-blur z-10">
                  <DialogHeader>
                    <DialogTitle className="text-2xl font-bold flex items-center gap-2">
                      <Sparkles className="text-primary w-6 h-6" /> Backtest Results
                    </DialogTitle>
                  </DialogHeader>
                </div>

                <div className="p-6 flex-1 overflow-y-auto">
                  <Tabs defaultValue="performance" className="w-full">
                    <TabsList className="mb-6 grid grid-cols-3 w-full h-12 bg-secondary/30">
                      <TabsTrigger value="performance">All Metrics</TabsTrigger>
                      <TabsTrigger value="report">Strategy Details</TabsTrigger>
                      <TabsTrigger value="portfolio">Portfolio</TabsTrigger>
                    </TabsList>

                    <TabsContent value="performance" className="space-y-6">
                      {/* Metrics Grid */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="p-4 bg-secondary/10 rounded-lg text-center border border-white/5">
                          <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">CAGR</div>
                          <div className="text-2xl font-bold text-primary">{result.metrics?.cagr?.toFixed(2)}%</div>
                        </div>
                        <div className="p-4 bg-secondary/10 rounded-lg text-center border border-white/5">
                          <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Sharpe Ratio</div>
                          <div className="text-2xl font-bold">{result.metrics?.sharpe_ratio?.toFixed(2)}</div>
                        </div>
                        <div className="p-4 bg-secondary/10 rounded-lg text-center border border-white/5">
                          <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Max Drawdown</div>
                          <div className="text-2xl font-bold text-destructive">{result.metrics?.max_drawdown?.toFixed(2)}%</div>
                        </div>
                        <div className="p-4 bg-secondary/10 rounded-lg text-center border border-white/5">
                          <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Volatility</div>
                          <div className="text-2xl font-bold">{result.metrics?.volatility?.toFixed(2)}%</div>
                        </div>
                      </div>

                      <div className="h-[400px] w-full p-4 bg-secondary/5 rounded-xl border border-white/5">
                        <h3 className="text-sm font-semibold mb-4 text-center">Equity Curve vs Benchmark</h3>
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart data={result.equity_curve}>
                            <defs>
                              <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" opacity={0.1} vertical={false} />
                            <XAxis dataKey="date" hide />
                            <YAxis domain={['auto', 'auto']} hide />
                            <Tooltip contentStyle={{ backgroundColor: '#1f2937' }} />
                            <Area type="monotone" dataKey="value" stroke="hsl(var(--primary))" fillOpacity={1} fill="url(#colorValue)" name="Strategy" />
                          </AreaChart>
                        </ResponsiveContainer>
                      </div>
                    </TabsContent>

                    <TabsContent value="report" className="animate-in fade-in slide-in-from-right-4 duration-500">
                      <div className="prose prose-invert max-w-none bg-secondary/10 p-6 rounded-xl border border-white/5">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {result.strategy_report || "No detailed report generated."}
                        </ReactMarkdown>
                      </div>
                    </TabsContent>

                    <TabsContent value="portfolio" className="space-y-6">
                      <div className="p-6 bg-secondary/10 rounded-xl border border-white/5 text-center">
                        <Briefcase className="w-12 h-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                        <h3 className="text-lg font-bold">Recommended Allocation</h3>
                        <p className="text-muted-foreground mb-4">Based on {riskLabel} profile</p>
                        <Button variant="outline" onClick={() => setShowModal(false)}>Customize in Portfolio Builder</Button>
                      </div>
                      <PortfolioAnalysis />
                    </TabsContent>

                  </Tabs>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </main>
    </div>
  );
};

export default InvestmentPlanner;
