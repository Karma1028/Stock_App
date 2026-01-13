
import { StockDetails } from "@/types/stock";
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from "@/components/ui/glass-card";

interface MarketDataGridProps {
    stock: StockDetails;
}

const Row = ({ label, value }: { label: string; value: string | number | undefined }) => (
    <div className="flex justify-between py-2 border-b border-border/40 last:border-0">
        <span className="text-sm text-muted-foreground">{label}</span>
        <span className="text-sm font-medium font-mono text-foreground">{value ?? '—'}</span>
    </div>
);

const MarketDataGrid = ({ stock }: MarketDataGridProps) => {
    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {/* Trade Information */}
            <GlassCard>
                <GlassCardHeader className="pb-2">
                    <GlassCardTitle className="text-lg">Trade Information</GlassCardTitle>
                </GlassCardHeader>
                <GlassCardContent>
                    <Row label="Traded Volume" value={stock.volume?.toLocaleString('en-IN')} />
                    <Row label="Traded Value" value={stock.volume && stock.current_price ? `₹${((stock.volume * stock.current_price) / 10000000).toFixed(2)} Cr` : '—'} />
                    <Row label="Market Cap" value={stock.market_cap ? `₹${(stock.market_cap / 10000000).toFixed(2)} Cr` : '—'} />
                    <Row label="Free Float" value="—" /> {/* Backend might need update for this */}
                    <Row label="Shares Outstanding" value="—" />
                </GlassCardContent>
            </GlassCard>

            {/* Price Information */}
            <GlassCard>
                <GlassCardHeader className="pb-2">
                    <GlassCardTitle className="text-lg">Price Information</GlassCardTitle>
                </GlassCardHeader>
                <GlassCardContent>
                    <Row label="52 Week High" value={stock.fifty_two_week_high?.toLocaleString('en-IN')} />
                    <Row label="52 Week Low" value={stock.fifty_two_week_low?.toLocaleString('en-IN')} />
                    <Row label="Day High" value={stock.day_high?.toLocaleString('en-IN')} />
                    <Row label="Day Low" value={stock.day_low?.toLocaleString('en-IN')} />
                </GlassCardContent>
            </GlassCard>

            {/* Securities Information */}
            <GlassCard>
                <GlassCardHeader className="pb-2">
                    <GlassCardTitle className="text-lg">Securities Info</GlassCardTitle>
                </GlassCardHeader>
                <GlassCardContent>
                    <Row label="Status" value="Listed" />
                    <Row label="Trading Status" value="Active" />
                    <Row label="Sector" value={stock.sector} />
                    <Row label="Industry" value={stock.industry} />
                    <Row label="P/E Ratio" value={stock.pe_ratio?.toFixed(2)} />
                </GlassCardContent>
            </GlassCard>
        </div>
    );
};

export default MarketDataGrid;
