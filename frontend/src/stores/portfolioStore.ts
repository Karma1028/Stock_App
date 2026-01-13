import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { PortfolioStock } from '@/types/stock';

interface PortfolioState {
  stocks: PortfolioStock[];
  addStock: (stock: PortfolioStock) => void;
  removeStock: (symbol: string) => void;
  updateQuantity: (symbol: string, quantity: number) => void;
  clearPortfolio: () => void;
  getTotalValue: () => number;
  getSectorAllocation: () => { name: string; value: number }[];
  getMarketCapDistribution: () => { category: string; value: number }[];
}

export const usePortfolioStore = create<PortfolioState>()(
  persist(
    (set, get) => ({
      stocks: [],

      addStock: (stock) =>
        set((state) => {
          const existing = state.stocks.find((s) => s.symbol === stock.symbol);
          if (existing) {
            return {
              stocks: state.stocks.map((s) =>
                s.symbol === stock.symbol
                  ? { ...s, quantity: s.quantity + stock.quantity }
                  : s
              ),
            };
          }
          return { stocks: [...state.stocks, stock] };
        }),

      removeStock: (symbol) =>
        set((state) => ({
          stocks: state.stocks.filter((s) => s.symbol !== symbol),
        })),

      updateQuantity: (symbol, quantity) =>
        set((state) => ({
          stocks: state.stocks.map((s) =>
            s.symbol === symbol ? { ...s, quantity } : s
          ),
        })),

      clearPortfolio: () => set({ stocks: [] }),

      getTotalValue: () => {
        return get().stocks.reduce(
          (total, stock) => total + stock.price * stock.quantity,
          0
        );
      },

      getSectorAllocation: () => {
        const stocks = get().stocks;
        const sectorMap = new Map<string, number>();

        stocks.forEach((stock) => {
          const value = stock.price * stock.quantity;
          const current = sectorMap.get(stock.sector) || 0;
          sectorMap.set(stock.sector, current + value);
        });

        return Array.from(sectorMap.entries()).map(([name, value]) => ({
          name,
          value,
        }));
      },

      getMarketCapDistribution: () => {
        const stocks = get().stocks;
        const capMap = new Map<string, number>();

        stocks.forEach((stock) => {
          const value = stock.price * stock.quantity;
          const current = capMap.get(stock.market_cap_category) || 0;
          capMap.set(stock.market_cap_category, current + value);
        });

        return Array.from(capMap.entries()).map(([category, value]) => ({
          category,
          value,
        }));
      },
    }),
    {
      name: 'portfolio-storage',
    }
  )
);
