import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { BarChart3, Search, Bell, Menu, X, TrendingUp, Briefcase, LayoutDashboard } from "lucide-react";
import { Button } from "@/components/ui/button";
import NavLinkItem from "./NavLinkItem";
import { SettingsSidebar } from "./SettingsSidebar";
import { cn } from "@/lib/utils";

// Nifty 50 stocks for search
const STOCK_LIST = [
  { symbol: "RELIANCE.NS", name: "Reliance Industries" },
  { symbol: "TCS.NS", name: "Tata Consultancy Services" },
  { symbol: "HDFCBANK.NS", name: "HDFC Bank" },
  { symbol: "INFY.NS", name: "Infosys" },
  { symbol: "ICICIBANK.NS", name: "ICICI Bank" },
  { symbol: "HINDUNILVR.NS", name: "Hindustan Unilever" },
  { symbol: "BHARTIARTL.NS", name: "Bharti Airtel" },
  { symbol: "ITC.NS", name: "ITC Limited" },
  { symbol: "SBIN.NS", name: "State Bank of India" },
  { symbol: "BAJFINANCE.NS", name: "Bajaj Finance" },
  { symbol: "KOTAKBANK.NS", name: "Kotak Mahindra Bank" },
  { symbol: "LT.NS", name: "Larsen & Toubro" },
  { symbol: "AXISBANK.NS", name: "Axis Bank" },
  { symbol: "WIPRO.NS", name: "Wipro" },
  { symbol: "ASIANPAINT.NS", name: "Asian Paints" },
  { symbol: "MARUTI.NS", name: "Maruti Suzuki" },
  { symbol: "TATAMOTORS.NS", name: "Tata Motors" },
  { symbol: "SUNPHARMA.NS", name: "Sun Pharma" },
  { symbol: "TITAN.NS", name: "Titan Company" },
  { symbol: "ULTRACEMCO.NS", name: "UltraTech Cement" },
];

const Header = () => {
  const [searchQuery, setSearchQuery] = useState("");
  const [showSearch, setShowSearch] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

  const filteredStocks = STOCK_LIST.filter(
    (stock) =>
      stock.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
      stock.name.toLowerCase().includes(searchQuery.toLowerCase())
  ).slice(0, 5);

  const handleStockSelect = (symbol: string) => {
    navigate(`/stock/${symbol}`);
    setSearchQuery("");
    setShowSearch(false);
  };

  return (
    <header className="sticky top-0 z-50 backdrop-blur-xl bg-background/80 border-b border-border/50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <div className="p-2 bg-primary/10 rounded-lg group-hover:bg-primary/20 transition-colors">
              <BarChart3 className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">StockPro</h1>
              <p className="text-xs text-muted-foreground">Pro Trader Dashboard</p>
            </div>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden lg:flex items-center gap-1">
            <NavLinkItem to="/" icon={LayoutDashboard}>Dashboard</NavLinkItem>
            <NavLinkItem to="/stock/RELIANCE.NS" icon={TrendingUp}>Analysis</NavLinkItem>
            <NavLinkItem to="/planner" icon={Briefcase}>Planner</NavLinkItem>
          </nav>

          {/* Search */}
          <div className="hidden md:flex items-center flex-1 max-w-md mx-8 relative">
            <div className="relative w-full">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search stocks..."
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setShowSearch(e.target.value.length > 0);
                }}
                onFocus={() => searchQuery && setShowSearch(true)}
                onBlur={() => setTimeout(() => setShowSearch(false), 200)}
                className="w-full pl-10 pr-4 py-2.5 bg-secondary/50 border border-border/50 rounded-lg text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50 transition-all"
              />
            </div>

            {/* Search dropdown */}
            {showSearch && filteredStocks.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-2 bg-card border border-border rounded-lg shadow-xl overflow-hidden z-50">
                {filteredStocks.map((stock) => (
                  <button
                    key={stock.symbol}
                    onClick={() => handleStockSelect(stock.symbol)}
                    className="w-full px-4 py-3 text-left hover:bg-secondary/50 transition-colors flex items-center justify-between"
                  >
                    <div>
                      <p className="font-medium text-foreground">{stock.symbol.replace('.NS', '')}</p>
                      <p className="text-xs text-muted-foreground">{stock.name}</p>
                    </div>
                    <TrendingUp className="w-4 h-4 text-muted-foreground" />
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3">
            <button className="relative p-2.5 rounded-lg bg-secondary/50 hover:bg-secondary transition-colors">
              <Bell className="w-5 h-5 text-muted-foreground" />
              <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-primary rounded-full" />
            </button>
            <SettingsSidebar />
            <div className="hidden sm:flex w-9 h-9 rounded-lg bg-primary/20 items-center justify-center">
              <span className="text-sm font-bold text-primary">JD</span>
            </div>

            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <nav className="lg:hidden mt-4 pt-4 border-t border-border/50 flex flex-col gap-2">
            <NavLinkItem to="/" icon={LayoutDashboard} onClick={() => setMobileMenuOpen(false)}>
              Dashboard
            </NavLinkItem>
            <NavLinkItem to="/stock/RELIANCE.NS" icon={TrendingUp} onClick={() => setMobileMenuOpen(false)}>
              Analysis
            </NavLinkItem>
            <NavLinkItem to="/planner" icon={Briefcase} onClick={() => setMobileMenuOpen(false)}>
              Planner
            </NavLinkItem>

            {/* Mobile Search */}
            <div className="relative mt-2">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search stocks..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2.5 bg-secondary/50 border border-border/50 rounded-lg text-sm"
              />
            </div>
          </nav>
        )}
      </div>
    </header>
  );
};

export default Header;
