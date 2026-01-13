import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SettingsState {
    aiModel: string;
    apiKey: string;
    chartType: 'candlestick' | 'area';
    showSMA50: boolean;
    showSMA200: boolean;
    showBollingerBands: boolean;

    setAiModel: (model: string) => void;
    setApiKey: (key: string) => void;
    setChartType: (type: 'candlestick' | 'area') => void;
    toggleSMA50: () => void;
    toggleSMA200: () => void;
    toggleBollingerBands: () => void;
}

export const useSettingsStore = create<SettingsState>()(
    persist(
        (set) => ({
            aiModel: 'google/gemini-2.0-flash-exp:free',
            apiKey: '',
            chartType: 'area', // Default to area for aesthetics, user can switch
            showSMA50: false,
            showSMA200: false,
            showBollingerBands: false,

            setAiModel: (model) => set({ aiModel: model }),
            setApiKey: (key) => set({ apiKey: key }),
            setChartType: (type) => set({ chartType: type }),
            toggleSMA50: () => set((state) => ({ showSMA50: !state.showSMA50 })),
            toggleSMA200: () => set((state) => ({ showSMA200: !state.showSMA200 })),
            toggleBollingerBands: () => set((state) => ({ showBollingerBands: !state.showBollingerBands })),
        }),
        {
            name: 'app-settings',
        }
    )
);
