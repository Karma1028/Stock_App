
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface PortfolioStock {
    symbol: string;
    name: string;
    quantity: number;
    average_price: number;
    sector: string;
    market_cap: number;
    market_cap_category: string;
    volatility: number;
    one_year_return: number;
}

interface PortfolioState {
    portfolio: PortfolioStock[];
    totalValue: number;
    addStock: (stock: PortfolioStock) => void;
    removeStock: (symbol: string) => void;
    updateQuantity: (symbol: string, quantity: number) => void;
    clearPortfolio: () => void;
}

export const usePortfolioStore = create<PortfolioState>()(
    persist(
        (set, get) => ({
            portfolio: [],
            totalValue: 0,

            addStock: (stock) => {
                const { portfolio } = get();
                const existing = portfolio.find(s => s.symbol === stock.symbol);

                let newPortfolio;
                if (existing) {
                    // Update quantity if exists
                    newPortfolio = portfolio.map(s =>
                        s.symbol === stock.symbol
                            ? { ...s, quantity: s.quantity + stock.quantity }
                            : s
                    );
                } else {
                    newPortfolio = [...portfolio, stock];
                }

                const totalValue = newPortfolio.reduce((sum, s) => sum + (s.quantity * s.average_price), 0);
                set({ portfolio: newPortfolio, totalValue });
            },

            removeStock: (symbol) => {
                const { portfolio } = get();
                const newPortfolio = portfolio.filter(s => s.symbol !== symbol);
                const totalValue = newPortfolio.reduce((sum, s) => sum + (s.quantity * s.average_price), 0);
                set({ portfolio: newPortfolio, totalValue });
            },

            updateQuantity: (symbol, quantity) => {
                const { portfolio } = get();
                const newPortfolio = portfolio.map(s =>
                    s.symbol === symbol ? { ...s, quantity } : s
                );
                const totalValue = newPortfolio.reduce((sum, s) => sum + (s.quantity * s.average_price), 0);
                set({ portfolio: newPortfolio, totalValue });
            },

            clearPortfolio: () => set({ portfolio: [], totalValue: 0 }),
        }),
        {
            name: 'portfolio-storage',
        }
    )
);
