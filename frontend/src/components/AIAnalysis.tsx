
import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, Sparkles, RefreshCcw, AlertTriangle, TrendingUp } from 'lucide-react';
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { getAISummary } from '@/services/api';
import { AISummary } from '@/types/stock';
import { useSettingsStore } from '@/store/settingsStore';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, CartesianGrid } from 'recharts';

interface AIAnalysisProps {
    symbol: string;
}

const AIAnalysis = ({ symbol }: AIAnalysisProps) => {
    const [data, setData] = useState<AISummary | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const { aiModel } = useSettingsStore();

    const fetchAnalysis = async () => {
        setLoading(true);
        setError(null);
        try {
            const result = await getAISummary(symbol, aiModel);
            // Result is already parsed JSON due to backend change
            setData(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to generate analysis');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (symbol) {
            // Optional: Auto-fetch could be enabled here
        }
    }, [symbol]);

    if (!symbol) return null;

    const renderCharts = () => {
        if (!data?.charts || data.charts.length === 0) return null;

        return (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                {data.charts.map((chart, idx) => (
                    <div key={idx} className="bg-secondary/10 p-4 rounded-xl border border-border/50">
                        <h4 className="text-sm font-semibold mb-4 text-center text-muted-foreground">{chart.title}</h4>
                        <div className="h-[250px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                {chart.type === 'bar' ? (
                                    <BarChart data={chart.data}>
                                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} vertical={false} />
                                        <XAxis dataKey={chart.x_key} tick={{ fontSize: 12 }} stroke="#94a3b8" />
                                        <YAxis tick={{ fontSize: 12 }} stroke="#94a3b8" />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
                                            itemStyle={{ color: '#f8fafc' }}
                                        />
                                        {chart.y_keys.map((key, i) => (
                                            <Bar key={key} dataKey={key} fill={i === 0 ? "#60a5fa" : "#34d399"} radius={[4, 4, 0, 0]} />
                                        ))}
                                    </BarChart>
                                ) : (
                                    <LineChart data={chart.data}>
                                        <CartesianGrid strokeDasharray="3 3" opacity={0.1} vertical={false} />
                                        <XAxis dataKey={chart.x_key} tick={{ fontSize: 12 }} stroke="#94a3b8" />
                                        <YAxis tick={{ fontSize: 12 }} stroke="#94a3b8" />
                                        <Tooltip
                                            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155' }}
                                            itemStyle={{ color: '#f8fafc' }}
                                        />
                                        {chart.y_keys.map((key, i) => (
                                            <Line key={key} type="monotone" dataKey={key} stroke={i === 0 ? "#60a5fa" : "#34d399"} strokeWidth={2} dot={{ r: 4 }} />
                                        ))}
                                    </LineChart>
                                )}
                            </ResponsiveContainer>
                        </div>
                    </div>
                ))}
            </div>
        );
    };

    return (
        <div className="space-y-6 animate-in fade-in duration-700">
            <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold flex items-center gap-2">
                    <Bot className="w-8 h-8 text-primary" />
                    AI Executive Summary
                </h2>

                <div className="flex gap-2">
                    <Button variant="outline" size="sm" onClick={fetchAnalysis} disabled={loading}>
                        {loading ? <RefreshCcw className="w-4 h-4 mr-2 animate-spin" /> : <Sparkles className="w-4 h-4 mr-2" />}
                        {data ? 'Regenerate' : 'Generate Insights'}
                    </Button>
                </div>
            </div>

            {loading && (
                <GlassCard>
                    <GlassCardContent className="pt-6 space-y-4">
                        <div className="flex items-center gap-4">
                            <Skeleton className="w-12 h-12 rounded-full" />
                            <div className="space-y-2">
                                <Skeleton className="h-4 w-[250px]" />
                                <Skeleton className="h-4 w-[200px]" />
                            </div>
                        </div>
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-full" />
                        <Skeleton className="h-4 w-3/4" />
                    </GlassCardContent>
                </GlassCard>
            )}

            {error && (
                <div className="bg-destructive/10 text-destructive p-4 rounded-lg flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5" />
                    {error}
                </div>
            )}

            {data && !loading && (
                <div className="space-y-6">
                    {/* Key Metrics Grid */}
                    {data.metrics && (
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            {Object.entries(data.metrics).map(([key, value]) => (
                                <GlassCard key={key} className="p-4 bg-primary/5 border-primary/20">
                                    <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">{key}</div>
                                    <div className="text-xl font-bold text-foreground">{value}</div>
                                </GlassCard>
                            ))}
                        </div>
                    )}

                    <GlassCard>
                        <GlassCardHeader>
                            <GlassCardTitle className="text-lg text-primary flex items-center gap-2">
                                <Sparkles className="w-5 h-5" />
                                Strategic Analysis for {symbol}
                            </GlassCardTitle>
                        </GlassCardHeader>
                        <GlassCardContent>
                            <div className="prose prose-invert max-w-none prose-sm leading-relaxed">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {data.summary}
                                </ReactMarkdown>
                            </div>

                            {/* Render Charts */}
                            {renderCharts()}

                        </GlassCardContent>
                    </GlassCard>
                </div>
            )}

            {!data && !loading && !error && (
                <GlassCard className="bg-muted/20 border-dashed hover:border-primary/50 transition-colors cursor-pointer" onClick={fetchAnalysis}>
                    <GlassCardContent className="py-12 text-center text-muted-foreground">
                        <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>Click "Generate Insights" to analyze {symbol} using advanced AI models.</p>
                        <p className="text-xs mt-2 opacity-70">Generates executive summary, financial trends, and key metrics.</p>
                    </GlassCardContent>
                </GlassCard>
            )}
        </div>
    );
};

export default AIAnalysis;
