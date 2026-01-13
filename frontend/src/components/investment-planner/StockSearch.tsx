
import { useState, useEffect } from "react";
import { Check, ChevronsUpDown, Loader2, Plus } from "lucide-react";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { getStockList, getStockDetails, getStockHistory } from "@/services/api";
import { usePortfolioStore } from "@/store/portfolioStore";

const StockSearch = () => {
    const [open, setOpen] = useState(false);
    const [value, setValue] = useState("");
    const [stocks, setStocks] = useState<string[]>([]);
    const [quantity, setQuantity] = useState<number>(1);
    const [loading, setLoading] = useState(false);
    const [adding, setAdding] = useState(false);

    const { addStock } = usePortfolioStore();
    const { toast } = useToast();

    useEffect(() => {
        const fetchStocks = async () => {
            setLoading(true);
            try {
                const res = await getStockList();
                setStocks(res.stocks);
            } catch (error) {
                console.error("Failed to load stocks", error);
            } finally {
                setLoading(false);
            }
        };
        fetchStocks();
    }, []);

    const handleAdd = async () => {
        if (!value) return;
        setAdding(true);
        try {
            // 1. Fetch Details (Price, Sector, Market Cap)
            const details = await getStockDetails(value);

            // 2. Fetch History (1Y Return, Volatility)
            // Ideally backend should provide this, but we calculate for now
            const history = await getStockHistory(value, "1y");

            let oneYearReturn = 0;
            let volatility = 0;

            if (history && history.length > 0) {
                const start = history[0].Close;
                const end = history[history.length - 1].Close;
                oneYearReturn = ((end - start) / start) * 100;

                // Calculate volatility (std dev of daily returns) * sqrt(252)
                // Simplified proxy: Average True Range (ATR) or just simple std dev of close prices? 
                // Let's use simplified daily return std dev.
                const returns = [];
                for (let i = 1; i < history.length; i++) {
                    returns.push((history[i].Close - history[i - 1].Close) / history[i - 1].Close);
                }
                const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
                const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
                volatility = Math.sqrt(variance) * Math.sqrt(252) * 100; // Annualized
            }

            addStock({
                symbol: details.symbol,
                name: details.symbol.replace('.NS', ''), // Simplified name
                quantity: Number(quantity),
                average_price: details.current_price, // Assessing at current price
                sector: details.sector || 'Unknown',
                market_cap: details.market_cap || 0,
                market_cap_category: details.market_cap_category || 'Unknown',
                one_year_return: oneYearReturn,
                volatility: volatility
            });

            toast({
                title: "Stock Added",
                description: `Added ${quantity} qty of ${details.symbol}`,
            });

            setValue("");
            setQuantity(1);

        } catch (error) {
            toast({
                variant: "destructive",
                title: "Error",
                description: "Failed to add stock details",
            });
        } finally {
            setAdding(false);
        }
    };

    return (
        <div className="flex flex-col md:flex-row gap-4 items-end p-4 border rounded-lg bg-card/50">
            <div className="flex flex-col gap-2 w-full md:w-[300px]">
                <Label>Select Stock</Label>
                <Popover open={open} onOpenChange={setOpen}>
                    <PopoverTrigger asChild>
                        <Button
                            variant="outline"
                            role="combobox"
                            aria-expanded={open}
                            className="w-full justify-between"
                        >
                            {value
                                ? stocks.find((stock) => stock === value)
                                : "Select stock..."}
                            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                        </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[300px] p-0">
                        <Command>
                            <CommandInput placeholder="Search stock..." />
                            <CommandList>
                                <CommandEmpty>No stock found.</CommandEmpty>
                                <CommandGroup className="max-h-[300px] overflow-y-auto">
                                    {stocks.map((stock) => (
                                        <CommandItem
                                            key={stock}
                                            value={stock}
                                            onSelect={(currentValue) => {
                                                setValue(currentValue === value ? "" : currentValue);
                                                setOpen(false);
                                            }}
                                        >
                                            <Check
                                                className={cn(
                                                    "mr-2 h-4 w-4",
                                                    value === stock ? "opacity-100" : "opacity-0"
                                                )}
                                            />
                                            {stock}
                                        </CommandItem>
                                    ))}
                                </CommandGroup>
                            </CommandList>
                        </Command>
                    </PopoverContent>
                </Popover>
            </div>

            <div className="flex flex-col gap-2 w-full md:w-[150px]">
                <Label>Quantity</Label>
                <Input
                    type="number"
                    min={1}
                    value={quantity}
                    onChange={(e) => setQuantity(Number(e.target.value))}
                />
            </div>

            <Button onClick={handleAdd} disabled={!value || adding || quantity < 1} className="w-full md:w-auto">
                {adding && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                <Plus className="mr-2 h-4 w-4" /> Add
            </Button>
        </div>
    );
};

export default StockSearch;
