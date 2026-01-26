import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Search, TrendingUp, TrendingDown, ArrowRight, BarChart2, Bot, PieChart, LayoutDashboard } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { GlassCard } from "@/components/ui/glass-card";
import Header from "@/components/Header";
import { getStockList } from "@/services/api";

const Home = () => {
    const [search, setSearch] = useState("");
    const navigate = useNavigate();
    const [stocks, setStocks] = useState<string[]>([]);

    useEffect(() => {
        const loadStocks = async () => {
            try {
                const res = await getStockList();
                setStocks(res.stocks);
            } catch (e) {
                console.error("Failed to load stocks", e);
            }
        };
        loadStocks();
    }, []);

    const handleSearch = (e: React.FormEvent) => {
        e.preventDefault();
        if (search) {
            navigate(`/analysis/${search.toUpperCase()}`);
        }
    };

    const features = [
        {
            icon: <BarChart2 className="w-8 h-8 text-blue-400" />,
            title: "Advanced Charting",
            desc: "Interactive technical analysis with 50+ indicators and overlay capabilities."
        },
        {
            icon: <Bot className="w-8 h-8 text-purple-400" />,
            title: "AI Analysis",
            desc: "Get instant executive summaries, visual trend analysis, and conversational insights."
        },
        {
            icon: <PieChart className="w-8 h-8 text-green-400" />,
            title: "Quant Planner",
            desc: "Build robust portfolios with AI-driven backtesting using SIP or Lumpsum strategies."
        }
    ];

    const trending = [
        { sym: "RELIANCE", change: "+2.4%", up: true },
        { sym: "HDFCBANK", change: "-0.8%", up: false },
        { sym: "TATASTEEL", change: "+1.2%", up: true },
        { sym: "INFY", change: "+0.5%", up: true },
    ];

    return (
        <div className="min-h-screen bg-background text-foreground flex flex-col">
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl opacity-50" />
                <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl opacity-50" />
            </div>

            <Header />

            <main className="flex-1 container mx-auto px-4 py-12 flex flex-col items-center justify-center relative">

                {/* Hero Section */}
                <div className="text-center max-w-3xl mx-auto mb-16 animate-in fade-in slide-in-from-bottom-8 duration-700">
                    <h1 className="text-5xl md:text-6xl font-bold mb-6 tracking-tight">
                        Intelligent <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">Market Research</span>
                    </h1>
                    <p className="text-xl text-muted-foreground mb-8">
                        Next-generation stock analysis powered by AI. Get deep financial insights, real-time sentiment, and institutional-grade tools.
                    </p>

                    <form onSubmit={handleSearch} className="flex gap-2 max-w-lg mx-auto bg-card/50 p-2 rounded-xl border border-white/10 shadow-2xl backdrop-blur-sm">
                        <Input
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            placeholder="Search symbol (e.g. RELIANCE)..."
                            className="bg-transparent border-none text-lg h-12 focus-visible:ring-0 placeholder:text-muted-foreground/50"
                        />
                        <Button type="submit" size="lg" className="h-12 px-8 bg-primary hover:bg-primary/90">
                            <Search className="w-5 h-5" />
                        </Button>
                    </form>

                    <div className="mt-8 flex justify-center">
                        <Link to="/dashboard">
                            <Button variant="outline" className="gap-2 border-white/20 hover:bg-white/10">
                                <LayoutDashboard className="w-4 h-4" /> Go to Pro Dashboard
                            </Button>
                        </Link>
                    </div>

                    <div className="mt-8 flex gap-3 justify-center text-sm text-muted-foreground">
                        <span>Trending:</span>
                        {["RELIANCE.NS", "TCS.NS", "INFY.NS"].map(s => (
                            <Link key={s} to={`/analysis/${s}`} className="text-primary hover:underline">{s.replace('.NS', '')}</Link>
                        ))}
                    </div>
                </div>

                {/* Market Ticker */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full max-w-4xl mb-24 animate-in fade-in slide-in-from-bottom-12 duration-1000 delay-200">
                    {trending.map((t) => (
                        <div key={t.sym} className="bg-secondary/10 border border-white/5 rounded-lg p-4 flex items-center justify-between hover:bg-secondary/20 transition-colors cursor-pointer group">
                            <span className="font-bold">{t.sym}</span>
                            <span className={`flex items-center text-sm font-semibold ${t.up ? 'text-green-400' : 'text-red-400'}`}>
                                {t.up ? <TrendingUp className="w-4 h-4 mr-1 transition-transform group-hover:-translate-y-1" /> : <TrendingDown className="w-4 h-4 mr-1 transition-transform group-hover:translate-y-1" />}
                                {t.change}
                            </span>
                        </div>
                    ))}
                </div>

                {/* Features Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-6xl animate-in fade-in slide-in-from-bottom-16 duration-1000 delay-300">
                    {features.map((f, i) => (
                        <GlassCard key={i} className="hover:border-primary/30 transition-all hover:-translate-y-2 duration-300">
                            <div className="p-6 h-full flex flex-col items-start text-left">
                                <div className="p-3 bg-secondary/50 rounded-lg mb-4">{f.icon}</div>
                                <h3 className="text-xl font-bold mb-2">{f.title}</h3>
                                <p className="text-muted-foreground">{f.desc}</p>
                            </div>
                        </GlassCard>
                    ))}
                </div>

            </main>

            <footer className="w-full border-t border-white/10 py-8 text-center text-muted-foreground text-sm relative z-10 bg-background/50 backdrop-blur-md">
                <p>© 2026 Antstock AI. Built for the future of finance.</p>
            </footer>
        </div>
    );
};

export default Home;
