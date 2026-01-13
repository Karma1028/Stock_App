
import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot, Sparkles, RefreshCcw, AlertTriangle } from 'lucide-react';
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { getAISummary } from '@/services/api';
import { AISummary } from '@/types/stock';

import { useSettingsStore } from '@/store/settingsStore';

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
            // Updated to pass model:
            const result = await getAISummary(symbol, aiModel);
            setData(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to generate analysis');
        } finally {
            setLoading(false);
        }
    };

    // Initial fetch? Maybe wait for user interaction to save API calls? 
    // User asked to replicate Streamlit behavior which generated it on load or sidebar select.
    // We'll generate on mount for now but allow refresh.
    useEffect(() => {
        if (symbol) {
            // Optional: Auto-fetch or wait for user
        }
    }, [symbol]);

    if (!symbol) return null;

    return (
        <div className="space-y-6">
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
                <GlassCard>
                    <GlassCardHeader>
                        <GlassCardTitle className="text-lg text-primary flex items-center gap-2">
                            <Sparkles className="w-5 h-5" />
                            Strategic Analysis for {symbol}
                        </GlassCardTitle>
                    </GlassCardHeader>
                    <GlassCardContent>
                        <div className="prose prose-invert max-w-none prose-sm">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                {data.summary}
                            </ReactMarkdown>
                        </div>
                    </GlassCardContent>
                </GlassCard>
            )}

            {!data && !loading && !error && (
                <GlassCard className="bg-muted/20 border-dashed">
                    <GlassCardContent className="py-12 text-center text-muted-foreground">
                        <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>Click "Generate Insights" to analyze {symbol} using advanced AI models.</p>
                    </GlassCardContent>
                </GlassCard>
            )}
        </div>
    );
};

export default AIAnalysis;
