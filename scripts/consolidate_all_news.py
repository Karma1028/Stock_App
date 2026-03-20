"""
Consolidate all scraped data into unified files.
Can be run standalone or is called automatically at the end of unified_news_scraper.py.

Produces:
  1. data/all_news_with_content.csv       — All articles (stock + geopolitical) WITH content
  2. data/news_numerical_summary.csv      — Per-ticker and per-category statistics
  3. data/all_stocks_news_consolidated.csv — All stock news (including those without content)
"""
import pandas as pd
import os
import glob
from pathlib import Path
from datetime import datetime


def consolidate_news():
    DATA_DIR = Path("data")
    NEWS_DIR = DATA_DIR / "news_history"
    GEO_DIR = DATA_DIR / "geopolitical"

    print("=" * 60)
    print("📊 NEWS DATA CONSOLIDATION")
    print("=" * 60)

    # ── 1. Load all stock news CSVs ──
    stock_files = glob.glob(str(NEWS_DIR / "*_news.csv"))
    print(f"\nFound {len(stock_files)} stock news files")

    stock_dfs = []
    for f in stock_files:
        try:
            d = pd.read_csv(f)
            if 'Ticker' not in d.columns:
                d['Ticker'] = Path(f).stem.replace('_news', '')
            stock_dfs.append(d)
        except Exception as e:
            print(f"  ⚠ Error reading {Path(f).name}: {e}")

    df_stocks = pd.concat(stock_dfs, ignore_index=True) if stock_dfs else pd.DataFrame()
    if not df_stocks.empty:
        df_stocks['Date'] = pd.to_datetime(df_stocks['Date'], errors='coerce')
        df_stocks = df_stocks.sort_values('Date', ascending=False)
    print(f"  Total stock articles: {len(df_stocks)}")

    # ── 2. Load geopolitical news ──
    geo_path = GEO_DIR / "geopolitical_news_scraped.csv"
    df_geo = pd.DataFrame()
    if geo_path.exists():
        df_geo = pd.read_csv(geo_path)
        df_geo['Date'] = pd.to_datetime(df_geo['Date'], errors='coerce')
        print(f"  Geopolitical articles: {len(df_geo)}")
    else:
        print("  No geopolitical data file found")

    # ── 3. Combined review file (articles WITH scraped content) ──
    combined_parts = []

    if not df_stocks.empty and 'Content' in df_stocks.columns:
        df_s = df_stocks[df_stocks['Content'].notna() & (df_stocks['Content'].astype(str).str.len() >= 200)].copy()
        df_s['Type'] = 'Stock'
        combined_parts.append(df_s)
        print(f"  Stock articles with content: {len(df_s)}")

    if not df_geo.empty and 'Content' in df_geo.columns:
        df_g = df_geo[df_geo['Content'].notna() & (df_geo['Content'].astype(str).str.len() >= 200)].copy()
        df_g['Type'] = 'Geopolitical'
        if 'Category' in df_g.columns and 'Ticker' not in df_g.columns:
            df_g['Ticker'] = df_g['Category']
        combined_parts.append(df_g)
        print(f"  Geopolitical articles with content: {len(df_g)}")

    if combined_parts:
        df_combined = pd.concat(combined_parts, ignore_index=True)
        df_combined = df_combined.sort_values('Date', ascending=False)
        combined_path = DATA_DIR / 'all_news_with_content.csv'
        df_combined.to_csv(combined_path, index=False)
        print(f"\n✅ Combined review data → {combined_path} ({len(df_combined)} rows)")
    else:
        print("\n⚠ No articles with content found")

    # ── 4. Numerical summary ──
    summary_rows = []

    if not df_stocks.empty and 'Content' in df_stocks.columns:
        for ticker, grp in df_stocks.groupby('Ticker'):
            has_content = int((grp['Content'].notna() & (grp['Content'].astype(str).str.len() >= 200)).sum())
            summary_rows.append({
                'Ticker': ticker,
                'Type': 'Stock',
                'Total_Articles': len(grp),
                'Articles_With_Content': has_content,
                'Content_Rate_Pct': round(has_content / max(len(grp), 1) * 100, 1),
                'Unique_Sources': ', '.join(grp['Source'].dropna().unique()[:5]),
                'Date_Range': f"{grp['Date'].min()} to {grp['Date'].max()}",
            })

    if not df_geo.empty and 'Content' in df_geo.columns:
        group_col = 'Category' if 'Category' in df_geo.columns else 'Source'
        for cat, grp in df_geo.groupby(group_col):
            has_content = int((grp['Content'].notna() & (grp['Content'].astype(str).str.len() >= 200)).sum())
            summary_rows.append({
                'Ticker': cat,
                'Type': 'Geopolitical',
                'Total_Articles': len(grp),
                'Articles_With_Content': has_content,
                'Content_Rate_Pct': round(has_content / max(len(grp), 1) * 100, 1),
                'Unique_Sources': ', '.join(grp['Source'].dropna().unique()[:5]),
                'Date_Range': f"{grp['Date'].min()} to {grp['Date'].max()}",
            })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows).sort_values('Articles_With_Content', ascending=False)
        summary_path = DATA_DIR / 'news_numerical_summary.csv'
        df_summary.to_csv(summary_path, index=False)
        print(f"✅ Numerical summary → {summary_path} ({len(df_summary)} entries)")

        # Print top-level stats
        total = df_summary['Total_Articles'].sum()
        with_content = df_summary['Articles_With_Content'].sum()
        print(f"\n{'─' * 40}")
        print(f"  Total articles: {total:,}")
        print(f"  With content:   {with_content:,} ({with_content/max(total,1)*100:.1f}%)")
        print(f"  Stock tickers:  {len(df_summary[df_summary['Type']=='Stock'])}")
        print(f"  Geo categories: {len(df_summary[df_summary['Type']=='Geopolitical'])}")
        print(f"{'─' * 40}")

    # ── 5. All stock news ──
    if not df_stocks.empty:
        all_path = DATA_DIR / 'all_stocks_news_consolidated.csv'
        df_stocks.to_csv(all_path, index=False)
        print(f"✅ All stock news → {all_path} ({len(df_stocks)} rows)")

    # ── 6. Day-wise news files ──
    if combined_parts:
        df_all = pd.concat(combined_parts, ignore_index=True) if 'df_combined' not in dir() else df_combined
        df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
        df_all = df_all.dropna(subset=['Date'])
        df_all['DateOnly'] = df_all['Date'].dt.date

        daywise_dir = DATA_DIR / 'daywise_news'
        daywise_dir.mkdir(parents=True, exist_ok=True)

        day_summary = []
        for day, grp in df_all.groupby('DateOnly'):
            day_str = str(day)
            day_file = daywise_dir / f"{day_str}.csv"
            grp_sorted = grp.sort_values('Date', ascending=False)
            grp_sorted.to_csv(day_file, index=False)

            stock_count = len(grp_sorted[grp_sorted['Type'] == 'Stock']) if 'Type' in grp_sorted.columns else len(grp_sorted)
            geo_count = len(grp_sorted[grp_sorted['Type'] == 'Geopolitical']) if 'Type' in grp_sorted.columns else 0

            day_summary.append({
                'Date': day_str,
                'Total_Articles': len(grp_sorted),
                'Stock_Articles': stock_count,
                'Geo_Articles': geo_count,
                'Tickers': ', '.join(grp_sorted['Ticker'].dropna().unique()[:10]),
                'Sources': ', '.join(grp_sorted['Source'].dropna().unique()[:5]),
            })

        if day_summary:
            df_days = pd.DataFrame(day_summary).sort_values('Date', ascending=False)
            df_days.to_csv(DATA_DIR / 'daywise_news_summary.csv', index=False)
            print(f"✅ Day-wise news → {daywise_dir}/ ({len(day_summary)} days)")
            print(f"✅ Day-wise summary → {DATA_DIR / 'daywise_news_summary.csv'}")

        # ── 6b. Per-ticker day-wise folders ──
        # Structure: data/daywise_news/{TICKER}/{YYYY-MM-DD}.csv
        if 'Ticker' in df_all.columns:
            ticker_count = 0
            for ticker, t_grp in df_all.groupby('Ticker'):
                ticker_dir = daywise_dir / str(ticker).replace('.', '_')
                ticker_dir.mkdir(parents=True, exist_ok=True)
                for day, d_grp in t_grp.groupby('DateOnly'):
                    d_grp.sort_values('Date', ascending=False).to_csv(
                        ticker_dir / f"{str(day)}.csv", index=False
                    )
                ticker_count += 1
            print(f"✅ Per-ticker day-wise → {daywise_dir}/<TICKER>/ ({ticker_count} tickers)")

    print(f"\n{'=' * 60}")
    print("CONSOLIDATION COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    consolidate_news()
