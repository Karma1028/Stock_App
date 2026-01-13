import React, { lazy, Suspense } from 'react';

// Lazy load heavy chart components
export const LazyStockChart = lazy(() => import('@/components/StockChart'));
export const LazySectorHeatmap = lazy(() => import('@/components/dashboard/SectorHeatmap'));
export const LazyMarketBreadth = lazy(() => import('@/components/dashboard/MarketBreadth'));

// Loading fallback component
export const ChartSkeleton = () => (
    <div className="w-full h-96 bg-muted animate-pulse rounded-lg" />
);

// Wrapper for lazy components
export const LazyChart = ({ component: Component, ...props }: any) => (
    <Suspense fallback={<ChartSkeleton />}>
        <Component {...props} />
    </Suspense>
);
