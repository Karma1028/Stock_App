import { useEffect, useState } from 'react';
import { getSocialSentiment, getStockNews } from '@/services/api';
import { SocialSentiment, NewsItem } from '@/types/stock';
import { GlassCard } from "@/components/ui/glass-card";
import { Skeleton } from "@/components/ui/skeleton";
import { Twitter, Newspaper } from "lucide-react";
import { Progress } from "@/components/ui/progress";

interface NewsTabProps {
    symbol: string;
}

const NewsTab = ({ symbol }: NewsTabProps) => {
    const [sentiment, setSentiment] = useState<SocialSentiment | null>(null);
    const [news, setNews] = useState<NewsItem[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                // Fetch independently to avoid one failing blocking the other
                const newsPromise = getStockNews(symbol);
                const sentPromise = getSocialSentiment(symbol);

                const [newsRes, sentRes] = await Promise.allSettled([newsPromise, sentPromise]);

                if (newsRes.status === 'fulfilled') setNews(newsRes.value);
                if (sentRes.status === 'fulfilled') setSentiment(sentRes.value);

            } catch (err) {
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [symbol]);

    if (loading) return (
        <div className="space-y-4">
            <Skeleton className="h-[150px] w-full" />
            <Skeleton className="h-[200px] w-full" />
        </div>
    );

    return (
        <div className="space-y-6 animate-in fade-in transition-all">
            {/* Sentiment Section */}
            <GlassCard className="p-6 overflow-hidden relative">
                <div className="absolute top-0 right-0 p-4 opacity-10">
                    <Twitter className="w-32 h-32" />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative z-10">
                    <div className="col-span-1 border-r border-border/50 pr-6">
                        <h3 className="text-lg font-bold flex items-center gap-2 mb-2">
                            <Twitter className="w-5 h-5 text-blue-400" /> Community Sentiment
                        </h3>
                        <div className="mt-4">
                            <div className="flex items-end gap-2">
                                <span className="text-4xl font-black">{sentiment?.score?.toFixed(0) || 50}</span>
                                <span className="text-sm text-muted-foreground mb-1">/ 100</span>
                            </div>
                            <div className={`text-lg font-bold mt-1 ${sentiment?.status === 'Bullish' ? 'text-green-500' :
                                    sentiment?.status === 'Bearish' ? 'text-red-500' : 'text-yellow-500'
                                }`}>
                                {sentiment?.status || 'Neutral'}
                            </div>
                            <Progress value={sentiment?.score || 50} className="h-2 mt-3" />
                        </div>
                    </div>

                    <div className="col-span-2">
                        <h4 className="text-sm font-medium text-muted-foreground mb-4">Trending Discussions</h4>
                        <div className="space-y-3">
                            {sentiment?.latest_tweets?.length ? sentiment.latest_tweets.slice(0, 2).map((tweet, i) => (
                                <div key={i} className="bg-background/40 p-3 rounded-lg border border-white/5 text-sm">
                                    <div className="flex justify-between items-center mb-1">
                                        <span className="font-bold text-blue-300">@{tweet.user}</span>
                                        <span className="text-xs text-muted-foreground">{new Date(tweet.date).toLocaleDateString()}</span>
                                    </div>
                                    <p className="opacity-90 line-clamp-2">{tweet.text}</p>
                                </div>
                            )) : (
                                <div className="text-muted-foreground text-sm italic py-2">No recent tweets found for this symbol.</div>
                            )}
                        </div>
                    </div>
                </div>
            </GlassCard>

            {/* News Feed */}
            <div className="space-y-4">
                <h3 className="text-xl font-bold flex items-center gap-2">
                    <Newspaper className="w-5 h-5 text-primary" /> Recent News
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {news.length > 0 ? news.map((item, i) => (
                        <GlassCard key={i} className="p-4 flex flex-col h-full hover:bg-secondary/10 transition-all group cursor-pointer" onClick={() => window.open(item.link, '_blank')}>
                            <div className="flex justify-between items-start mb-3">
                                <span className="text-xs font-mono bg-primary/10 text-primary px-2 py-1 rounded">
                                    {item.source}
                                </span>
                                <span className="text-xs text-muted-foreground">
                                    {new Date(item.date).toLocaleDateString()}
                                </span>
                            </div>
                            <h4 className="font-semibold text-sm leading-snug group-hover:text-primary transition-colors mb-4 flex-1">
                                {item.title}
                            </h4>
                            <div className="text-xs text-muted-foreground pt-4 border-t border-border/30 flex justify-between items-center mt-auto w-full">
                                <span>Read full story</span>
                                <span className="opacity-0 group-hover:opacity-100 transition-opacity">→</span>
                            </div>
                        </GlassCard>
                    )) : (
                        <div className="col-span-full text-center py-10 text-muted-foreground">
                            No news articles found in the last 7 days.
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default NewsTab;
