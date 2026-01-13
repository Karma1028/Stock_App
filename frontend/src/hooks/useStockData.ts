import { useQuery, useMutation } from '@tanstack/react-query';
import {
  getDashboard,
  getStockDetails,
  getStockHistory,
  getStockNews,
  getNews,
  getAISummary,
  runBacktest,
  healthCheck,
  getStockPrediction,
} from '@/services/api';
import type { InvestmentPlanParams } from '@/types/stock';

// Dashboard hook with auto-refresh
export function useDashboard(refetchInterval: number = 30000) {
  return useQuery({
    queryKey: ['dashboard'],
    queryFn: getDashboard,
    refetchInterval,
    staleTime: 10000,
  });
}

// Stock details hook
export function useStockDetails(symbol: string | undefined) {
  return useQuery({
    queryKey: ['stock', symbol],
    queryFn: () => getStockDetails(symbol!),
    enabled: !!symbol,
    staleTime: 30000,
  });
}

// Stock history hook
export function useStockHistory(symbol: string | undefined, period: string = '1y') {
  return useQuery({
    queryKey: ['stockHistory', symbol, period],
    queryFn: () => getStockHistory(symbol!, period),
    enabled: !!symbol,
    staleTime: 60000,
  });
}

// Stock news hook
export function useStockNews(symbol: string | undefined, days: number = 7) {
  return useQuery({
    queryKey: ['stockNews', symbol, days],
    queryFn: () => getStockNews(symbol!, days),
    enabled: !!symbol,
  });
}


// General news hook
export function useNews(limit: number = 10) {
  return useQuery({
    queryKey: ['news', limit],
    queryFn: () => getNews(limit),
    staleTime: 60000,
  });
}

// Prediction hook
export function useStockPrediction(symbol: string | undefined, days: number = 30) {
  return useQuery({
    queryKey: ['prediction', symbol, days],
    queryFn: () => getStockPrediction(symbol!, days),
    enabled: !!symbol,
    staleTime: 60000,
  });
}

// AI Summary hook
export function useAISummary(symbol: string | undefined) {
  return useQuery({
    queryKey: ['aiSummary', symbol],
    queryFn: () => getAISummary(symbol!),
    enabled: !!symbol,
    staleTime: 300000, // 5 minutes
  });
}

// Backtest mutation
export function useBacktest() {
  return useMutation({
    mutationFn: (params: InvestmentPlanParams) => runBacktest(params),
  });
}

// Health check hook
export function useHealthCheck() {
  return useQuery({
    queryKey: ['health'],
    queryFn: healthCheck,
    retry: 3,
    retryDelay: 1000,
  });
}
