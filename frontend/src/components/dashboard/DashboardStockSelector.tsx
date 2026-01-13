import { useState } from "react";
import { Check, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
    Command,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList,
} from "@/components/ui/command";
import {
    Popover,
    PopoverContent,
    PopoverTrigger,
} from "@/components/ui/popover";

interface StockSelectorProps {
    onSelectionChange: (symbols: string[]) => void;
    defaultStocks?: string[];
}

const POPULAR_STOCKS = [
    { symbol: "RELIANCE.NS", name: "Reliance Industries" },
    { symbol: "TCS.NS", name: "Tata Consultancy Services" },
    { symbol: "HDFCBANK.NS", name: "HDFC Bank" },
    { symbol: "INFY.NS", name: "Infosys" },
    { symbol: "ICICIBANK.NS", name: "ICICI Bank" },
    { symbol: "HINDUNILVR.NS", name: "Hindustan Unilever" },
    { symbol: "ITC.NS", name: "ITC" },
    { symbol: "SBIN.NS", name: "State Bank of India" },
    { symbol: "BHARTIARTL.NS", name: "Bharti Airtel" },
    { symbol: "KOTAKBANK.NS", name: "Kotak Mahindra Bank" },
];

const DashboardStockSelector = ({ onSelectionChange, defaultStocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS"] }: StockSelectorProps) => {
    const [open, setOpen] = useState(false);
    const [selectedStocks, setSelectedStocks] = useState<string[]>(defaultStocks);

    const toggleStock = (symbol: string) => {
        const newSelection = selectedStocks.includes(symbol)
            ? selectedStocks.filter(s => s !== symbol)
            : [...selectedStocks, symbol].slice(0, 5); // Max 5 stocks

        setSelectedStocks(newSelection);
        onSelectionChange(newSelection);

        // Save to localStorage
        localStorage.setItem('dashboard_stocks', JSON.stringify(newSelection));
    };

    return (
        <div className="flex flex-col gap-2">
            <label className="text-sm font-medium text-foreground">
                Featured Stocks ({selectedStocks.length}/5)
            </label>
            <Popover open={open} onOpenChange={setOpen}>
                <PopoverTrigger asChild>
                    <Button
                        variant="outline"
                        role="combobox"
                        aria-expanded={open}
                        className="w-full justify-between"
                    >
                        {selectedStocks.length > 0
                            ? `${selectedStocks.length} selected`
                            : "Select stocks..."}
                        <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                    </Button>
                </PopoverTrigger>
                <PopoverContent className="w-[400px] p-0">
                    <Command>
                        <CommandInput placeholder="Search stocks..." />
                        <CommandList>
                            <CommandEmpty>No stock found.</CommandEmpty>
                            <CommandGroup className="max-h-[300px] overflow-y-auto">
                                {POPULAR_STOCKS.map((stock) => (
                                    <CommandItem
                                        key={stock.symbol}
                                        value={stock.symbol}
                                        onSelect={() => toggleStock(stock.symbol)}
                                    >
                                        <Check
                                            className={cn(
                                                "mr-2 h-4 w-4",
                                                selectedStocks.includes(stock.symbol)
                                                    ? "opacity-100"
                                                    : "opacity-0"
                                            )}
                                        />
                                        <div className="flex flex-col">
                                            <span className="font-medium">{stock.name}</span>
                                            <span className="text-xs text-muted-foreground">
                                                {stock.symbol}
                                            </span>
                                        </div>
                                    </CommandItem>
                                ))}
                            </CommandGroup>
                        </CommandList>
                    </Command>
                </PopoverContent>
            </Popover>

            {/* Selected stocks display */}
            {selectedStocks.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-2">
                    {selectedStocks.map((symbol) => {
                        const stock = POPULAR_STOCKS.find(s => s.symbol === symbol);
                        return (
                            <div
                                key={symbol}
                                className="flex items-center gap-1 px-2 py-1 bg-primary/10 text-primary rounded-md text-xs"
                            >
                                <span>{stock?.name || symbol}</span>
                                <button
                                    onClick={() => toggleStock(symbol)}
                                    className="hover:text-destructive"
                                >
                                    ×
                                </button>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

export default DashboardStockSelector;
