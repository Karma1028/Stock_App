import { ExternalLink, Clock } from "lucide-react";
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";
import { Skeleton } from "@/components/ui/skeleton";

interface NewsItem {
  title: string;
  source: string;
  date: string;
  link: string;
  sentiment_score?: number;
}

interface NewsCardProps {
  news: NewsItem[];
  isLoading?: boolean;
}

const NewsCard = ({ news, isLoading }: NewsCardProps) => {
  if (isLoading) {
    return (
      <GlassCard variant="elevated">
        <GlassCardHeader>
          <GlassCardTitle>Market News</GlassCardTitle>
        </GlassCardHeader>
        <GlassCardContent>
          <div className="space-y-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="p-4 bg-secondary/30 rounded-lg">
                <Skeleton className="h-4 w-full mb-2" />
                <Skeleton className="h-4 w-3/4 mb-3" />
                <div className="flex gap-4">
                  <Skeleton className="h-3 w-20" />
                  <Skeleton className="h-3 w-16" />
                </div>
              </div>
            ))}
          </div>
        </GlassCardContent>
      </GlassCard>
    );
  }

  return (
    <GlassCard variant="elevated">
      <GlassCardHeader>
        <GlassCardTitle>Market News</GlassCardTitle>
      </GlassCardHeader>
      
      <GlassCardContent>
        <div className="space-y-4">
          {news.map((item, index) => (
            <a
              key={index}
              href={item.link}
              target="_blank"
              rel="noopener noreferrer"
              className="block p-4 bg-secondary/30 rounded-lg hover:bg-secondary/60 transition-all group animate-slide-up border border-transparent hover:border-primary/20"
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <h4 className="font-medium text-foreground group-hover:text-primary transition-colors line-clamp-2">
                    {item.title}
                  </h4>
                  <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                    <span className="font-medium text-primary/80">{item.source}</span>
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {item.date}
                    </span>
                    {item.sentiment_score !== undefined && (
                      <span className={item.sentiment_score > 0 ? "text-primary" : item.sentiment_score < 0 ? "text-destructive" : ""}>
                        Sentiment: {item.sentiment_score > 0 ? '+' : ''}{item.sentiment_score.toFixed(2)}
                      </span>
                    )}
                  </div>
                </div>
                <ExternalLink className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors flex-shrink-0" />
              </div>
            </a>
          ))}
        </div>
      </GlassCardContent>
    </GlassCard>
  );
};

export default NewsCard;
