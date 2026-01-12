import { Card, CardContent } from "@/components/ui/card"
import { ExternalLink } from "lucide-react"
import Link from "next/link"

interface NewsItem {
    title: string;
    link: string;
    source: string;
    date: string;
}

interface NewsGridProps {
    news: NewsItem[];
}

export function NewsGrid({ news }: NewsGridProps) {
    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {news.map((item, i) => (
                <Card key={i} className="group hover:border-grow-green/50 transition-all duration-300">
                    <CardContent className="p-5 flex flex-col h-full justify-between">
                        <div>
                            <div className="flex justify-between items-start mb-3">
                                <span className="text-xs font-semibold px-2 py-1 rounded bg-white/10 text-grow-gray">
                                    {item.source}
                                </span>
                                <span className="text-xs text-gray-500">{new Date(item.date).toLocaleDateString()}</span>
                            </div>
                            <h4 className="font-semibold text-white mb-2 line-clamp-2 group-hover:text-grow-green transition-colors">
                                {item.title}
                            </h4>
                        </div>
                        <Link href={item.link} target="_blank" className="inline-flex items-center text-sm text-grow-green mt-4 hover:underline">
                            Read Article <ExternalLink className="w-3 h-3 ml-1" />
                        </Link>
                    </CardContent>
                </Card>
            ))}
        </div>
    )
}
