"""
Tata Motors Complete Data Pipeline v2
======================================
CORRECTED: Uses TMPV.BO for full 5yr history (original TATAMOTORS scrip was
renamed to TMPV at demerger) + TMCV.NS for post-demerger commercial vehicles.

Key Dates:
  - Demerger effective: Oct 1, 2025
  - Record date: Oct 14, 2025
  - TMLCV (now TMCV.NS) listed: Nov 12, 2025
  - TMPV trades on BSE: continues from original TATAMOTORS ticker
  - No stock split since Sept 2011 (handled by yfinance Adj Close)

What the stitching does:
  - TMPV.BO 2020-01-01 to 2025-10-13 = pre-demerger Tata Motors full data
  - TMCV.NS 2025-11-12 onwards = post-demerger Commercial Vehicles entity
  - Scale pre-demerger prices by ratio to create continuous price series for CV

Author: Auto-generated for Stock Analysis Project
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

BASE_DIR = r'd:\stock\stock project dnyanesh\stock_app'
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def fetch_ticker(ticker, start, end=None, name=None):
    """Fetch OHLCV data from yfinance."""
    display_name = name or ticker
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    print(f"  Fetching {display_name} ({ticker}): {start} -> {end}")
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            print(f"    WARNING: Empty result for {ticker}")
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        print(f"    OK: {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"    FAILED: {e}")
        return pd.DataFrame()


# ============================================================
# PART 1: STITCH TATA MOTORS PRICE DATA
# ============================================================

def stitch_tata_motors():
    """
    Stitch pre-demerger (TMPV.BO, which carries old TATAMOTORS history) 
    with post-demerger (TMCV.NS = Commercial Vehicles).
    """
    print("\n" + "=" * 70)
    print("  PHASE 1: STITCHING TATA MOTORS PRICE DATA")
    print("  TMPV.BO (Pre-demerger history) -> TMCV.NS (Post-demerger CV)")
    print("=" * 70)
    
    # TMPV.BO has the full history because the original TATAMOTORS scrip was renamed to TMPV
    pre = fetch_ticker('TMPV.BO', start='2020-01-01', end='2025-10-14',
                       name='Tata Motors Pre-Demerger (via TMPV.BO)')
    
    # TMCV.NS is the Commercial Vehicles entity listed post-demerger
    post = fetch_ticker('TMCV.NS', start='2025-11-01',
                        name='TMCV.NS (Post-Demerger CV)')
    
    # Also fetch TMPV separately for the post-demerger PV analysis
    tmpv_post = fetch_ticker('TMPV.BO', start='2025-10-14',
                             name='TMPV.BO (Post-Demerger PV)')
    
    if pre.empty:
        print("  FATAL: Cannot fetch pre-demerger data from TMPV.BO")
        if not post.empty:
            return post, pd.DataFrame(), 'TMCV.NS only (no pre-demerger data)'
        return pd.DataFrame(), pd.DataFrame(), 'FAILED'
    
    if post.empty:
        print("  WARNING: Cannot fetch TMCV.NS post-demerger. Using TMPV.BO only.")
        return pre, tmpv_post, 'TMPV.BO only (no post-demerger CV data)'
    
    # ---- Stitching Logic ----
    # The pre-demerger prices represent the COMBINED entity (CV + PV + JLR)
    # The post-demerger TMCV.NS represents ONLY Commercial Vehicles
    # We scale pre-demerger prices so the series is continuous for the CV entity
    
    pre_last_date = pre.index[-1]
    post_first_date = post.index[0]
    
    close_col = 'Close'
    pre_last_price = float(pre[close_col].iloc[-1])
    post_first_price = float(post[close_col].iloc[0])
    
    scale_factor = post_first_price / pre_last_price
    
    print(f"\n  Stitch Details:")
    print(f"    Pre-demerger last date:   {pre_last_date.date()}")
    print(f"    Pre-demerger last close:  Rs {pre_last_price:.2f}")
    print(f"    Post-demerger first date: {post_first_date.date()}")
    print(f"    Post-demerger first close: Rs {post_first_price:.2f}")
    print(f"    Scale Factor: {scale_factor:.6f}")
    print(f"    (Pre-demerger prices will be multiplied by {scale_factor:.4f})")
    
    # Filter pre to only dates before post starts
    pre_filtered = pre[pre.index < post_first_date].copy()
    
    # Scale pre-demerger prices for continuity
    price_cols = ['Open', 'High', 'Low', 'Close']
    adj_cols = ['Adj Close']
    for col in price_cols + adj_cols:
        if col in pre_filtered.columns:
            pre_filtered[col] = pre_filtered[col] * scale_factor
    
    # Combine
    stitched = pd.concat([pre_filtered, post])
    stitched = stitched.sort_index()
    stitched = stitched[~stitched.index.duplicated(keep='last')]
    
    # Validate stitch continuity
    stitch_idx = len(pre_filtered)
    if stitch_idx > 0 and stitch_idx < len(stitched):
        price_before = float(stitched['Close'].iloc[stitch_idx - 1])
        price_after = float(stitched['Close'].iloc[stitch_idx])
        gap_pct = abs(price_after - price_before) / price_before * 100
        print(f"    Stitch gap: {gap_pct:.2f}% {'OK' if gap_pct < 5 else 'WARNING: LARGE GAP'}")
    
    close_data = stitched['Close'].dropna().astype(float)
    print(f"\n  STITCHED DATASET CREATED:")
    print(f"    Total rows: {len(stitched)}")
    print(f"    Date range: {stitched.index[0].date()} to {stitched.index[-1].date()}")
    print(f"    Price range: Rs {float(close_data.min()):.2f} to Rs {float(close_data.max()):.2f}")
    print(f"    Current price: Rs {float(close_data.iloc[-1]):.2f}")
    
    return stitched, tmpv_post, 'Stitched (TMPV.BO pre + TMCV.NS post)'


# ============================================================
# PART 2: PEER DATA
# ============================================================

def fetch_peers():
    """Fetch peer stock data."""
    print("\n" + "=" * 70)
    print("  PHASE 2: FETCHING PEER & BENCHMARK DATA")
    print("=" * 70)
    
    peers = {
        'Maruti Suzuki': 'MARUTI.NS',
        'Mahindra & Mahindra': 'M&M.NS',
        'Ashok Leyland': 'ASHOKLEY.NS',
        'Bajaj Auto': 'BAJAJ-AUTO.NS',
        'NIFTY 50': '^NSEI',
        'NIFTY Auto': '^CNXAUTO',
    }
    
    peer_data = {}
    for name, ticker in peers.items():
        df = fetch_ticker(ticker, start='2020-01-01', name=name)
        if not df.empty:
            peer_data[name] = df
    
    return peer_data


# ============================================================
# PART 3: FINANCIAL ANALYSIS & RATIOS
# ============================================================

def safe_get(d, keys, default=np.nan):
    if isinstance(keys, str):
        keys = [keys]
    for key in keys:
        val = d.get(key, None)
        if val is not None:
            return val
    return default

def safe_col(df, col_names, row_idx=0, default=np.nan):
    if df.empty:
        return default
    if isinstance(col_names, str):
        col_names = [col_names]
    for col in col_names:
        if col in df.columns:
            try:
                val = df[col].iloc[row_idx]
                if pd.notna(val):
                    return float(val)
            except (IndexError, ValueError):
                continue
    return default

def fetch_financials(ticker, name=None):
    """Fetch income statement, balance sheet, cash flow."""
    display_name = name or ticker
    print(f"\n  Fetching financials for {display_name} ({ticker})...")
    try:
        stock = yf.Ticker(ticker)
        income = stock.financials
        if income is not None and not income.empty:
            income = income.T
        else:
            income = pd.DataFrame()
        
        bs = stock.balance_sheet
        if bs is not None and not bs.empty:
            bs = bs.T
        else:
            bs = pd.DataFrame()
        
        cf = stock.cashflow
        if cf is not None and not cf.empty:
            cf = cf.T
        else:
            cf = pd.DataFrame()
        
        info = stock.info or {}
        
        print(f"    Income: {len(income)} yrs | BS: {len(bs)} yrs | CF: {len(cf)} yrs")
        return {'income': income, 'balance_sheet': bs, 'cashflow': cf, 
                'info': info, 'name': display_name, 'ticker': ticker}
    except Exception as e:
        print(f"    FAILED: {e}")
        return {'income': pd.DataFrame(), 'balance_sheet': pd.DataFrame(),
                'cashflow': pd.DataFrame(), 'info': {}, 'name': display_name, 'ticker': ticker}


def compute_ratios(fin):
    """Compute financial ratios from fetched data."""
    info = fin['info']
    income = fin['income']
    bs = fin['balance_sheet']
    cf = fin['cashflow']
    name = fin['name']
    
    r = {'Company': name}
    
    # Market data
    mcap = safe_get(info, ['marketCap'])
    r['Market Cap (Cr)'] = round(mcap / 1e7, 0) if mcap is not np.nan and mcap else np.nan
    r['P/E'] = safe_get(info, ['trailingPE', 'forwardPE'])
    r['P/B'] = safe_get(info, ['priceToBook'])
    r['EV/EBITDA'] = safe_get(info, ['enterpriseToEbitda'])
    r['P/S'] = safe_get(info, ['priceToSalesTrailing12Months'])
    
    div_yield = safe_get(info, ['dividendYield', 'trailingAnnualDividendYield'])
    r['Div Yield (%)'] = round(div_yield * 100, 2) if div_yield is not np.nan and div_yield else np.nan
    
    # From financial statements
    revenue = safe_col(income, ['Total Revenue', 'Revenue', 'Operating Revenue'])
    net_income = safe_col(income, ['Net Income', 'Net Income From Continuing Operations',
                                    'Net Income Common Stockholders'])
    ebitda = safe_col(income, ['EBITDA', 'Normalized EBITDA'])
    op_income = safe_col(income, ['Operating Income', 'EBIT'])
    gross_profit = safe_col(income, ['Gross Profit'])
    
    total_equity = safe_col(bs, ['Stockholders Equity', 'Total Stockholders Equity',
                                  'Common Stock Equity', 'Ordinary Shares Equity'])
    total_assets = safe_col(bs, ['Total Assets'])
    total_debt = safe_col(bs, ['Total Debt', 'Long Term Debt',
                                'Long Term Debt And Capital Lease Obligation'])
    curr_assets = safe_col(bs, ['Current Assets', 'Total Current Assets'])
    curr_liab = safe_col(bs, ['Current Liabilities', 'Total Current Liabilities',
                                'Current Liabilities And Provisions'])
    
    # Profitability
    if revenue and net_income and revenue != 0:
        r['Net Margin (%)'] = round(net_income / revenue * 100, 2)
    else:
        r['Net Margin (%)'] = np.nan
    
    if revenue and ebitda and revenue != 0:
        r['EBITDA Margin (%)'] = round(ebitda / revenue * 100, 2)
    else:
        r['EBITDA Margin (%)'] = np.nan
    
    if revenue and op_income and revenue != 0:
        r['Op Margin (%)'] = round(op_income / revenue * 100, 2)
    else:
        r['Op Margin (%)'] = np.nan
    
    if revenue and gross_profit and revenue != 0:
        r['Gross Margin (%)'] = round(gross_profit / revenue * 100, 2)
    else:
        r['Gross Margin (%)'] = np.nan
    
    # ROE
    if total_equity and net_income and total_equity != 0:
        r['ROE (%)'] = round(net_income / total_equity * 100, 2)
    else:
        roe = safe_get(info, ['returnOnEquity'])
        r['ROE (%)'] = round(roe * 100, 2) if roe is not np.nan and roe else np.nan
    
    # ROA
    if total_assets and net_income and total_assets != 0:
        r['ROA (%)'] = round(net_income / total_assets * 100, 2)
    else:
        roa = safe_get(info, ['returnOnAssets'])
        r['ROA (%)'] = round(roa * 100, 2) if roa is not np.nan and roa else np.nan
    
    # ROCE
    capital_employed = None
    if total_assets is not np.nan and curr_liab is not np.nan:
        capital_employed = total_assets - curr_liab
    if op_income is not np.nan and capital_employed and capital_employed != 0:
        r['ROCE (%)'] = round(op_income / capital_employed * 100, 2)
    else:
        r['ROCE (%)'] = np.nan
    
    # Leverage
    if total_debt is not np.nan and total_equity is not np.nan and total_equity != 0:
        r['D/E'] = round(total_debt / total_equity, 2)
    else:
        de = safe_get(info, ['debtToEquity'])
        r['D/E'] = round(de / 100, 2) if de is not np.nan and de else np.nan
    
    if total_debt is not np.nan and total_assets is not np.nan and total_assets != 0:
        r['Debt/Assets'] = round(total_debt / total_assets, 2)
    else:
        r['Debt/Assets'] = np.nan
    
    # Interest Coverage
    int_exp = safe_col(income, ['Interest Expense', 'Interest Expense Non Operating'])
    if op_income is not np.nan and int_exp is not np.nan and int_exp != 0:
        r['Int Coverage'] = round(abs(op_income / int_exp), 2)
    else:
        r['Int Coverage'] = np.nan
    
    # Liquidity
    if curr_assets is not np.nan and curr_liab is not np.nan and curr_liab != 0:
        r['Current Ratio'] = round(curr_assets / curr_liab, 2)
    else:
        r['Current Ratio'] = safe_get(info, ['currentRatio'])
    
    r['Quick Ratio'] = safe_get(info, ['quickRatio'])
    
    # Efficiency
    if revenue is not np.nan and total_assets is not np.nan and total_assets != 0:
        r['Asset Turnover'] = round(revenue / total_assets, 2)
    else:
        r['Asset Turnover'] = np.nan
    
    r['Beta'] = safe_get(info, ['beta'])
    r['52W High'] = safe_get(info, ['fiftyTwoWeekHigh'])
    r['52W Low'] = safe_get(info, ['fiftyTwoWeekLow'])
    r['Revenue (Cr)'] = round(revenue / 1e7, 0) if revenue is not np.nan else np.nan
    r['Net Income (Cr)'] = round(net_income / 1e7, 0) if net_income is not np.nan else np.nan
    
    return r


def build_multi_year(fin):
    """Build multi-year financial summary."""
    income = fin['income']
    bs = fin['balance_sheet']
    name = fin['name']
    if income.empty:
        return pd.DataFrame()
    
    rows = []
    for i in range(len(income)):
        yr = income.index[i]
        if hasattr(yr, 'strftime'):
            yr = yr.strftime('%Y-%m')
        row = {'Company': name, 'Period': yr}
        row['Revenue'] = safe_col(income, ['Total Revenue', 'Revenue'], i)
        row['Gross Profit'] = safe_col(income, ['Gross Profit'], i)
        row['EBITDA'] = safe_col(income, ['EBITDA', 'Normalized EBITDA'], i)
        row['Operating Income'] = safe_col(income, ['Operating Income', 'EBIT'], i)
        row['Net Income'] = safe_col(income, ['Net Income', 'Net Income From Continuing Operations',
                                                'Net Income Common Stockholders'], i)
        row['EPS'] = safe_col(income, ['Basic EPS', 'Diluted EPS'], i)
        if not bs.empty and i < len(bs):
            row['Total Assets'] = safe_col(bs, ['Total Assets'], i)
            row['Total Debt'] = safe_col(bs, ['Total Debt', 'Long Term Debt'], i)
            row['Equity'] = safe_col(bs, ['Stockholders Equity', 'Total Stockholders Equity',
                                           'Common Stock Equity'], i)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# PART 4: CORRECTED STATISTICS
# ============================================================

def compute_stats(df, name='Tata Motors'):
    """Compute corrected price statistics."""
    close = df['Close'].dropna().astype(float)
    returns = np.log(close / close.shift(1)).dropna()
    
    s = {}
    s['Ticker'] = name
    s['Total Trading Days'] = len(close)
    s['Date Range'] = f"{df.index[0].date()} to {df.index[-1].date()}"
    s['Price Min'] = f"{float(close.min()):.2f}"
    s['Price Max'] = f"{float(close.max()):.2f}"
    s['Current Price'] = f"{float(close.iloc[-1]):.2f}"
    s['Avg Volume'] = f"{df['Volume'].mean():,.0f}"
    
    s['Mean Daily Ret (%)'] = f"{returns.mean() * 100:.4f}"
    s['Daily Std (%)'] = f"{returns.std() * 100:.4f}"
    s['Ann Return (%)'] = f"{returns.mean() * 252 * 100:.2f}"
    s['Ann Volatility (%)'] = f"{returns.std() * np.sqrt(252) * 100:.2f}"
    
    rf = 0.065 / 252  # 6.5% Indian risk-free
    sharpe = (returns.mean() - rf) / returns.std() * np.sqrt(252)
    s['Sharpe Ratio'] = f"{sharpe:.4f}"
    
    s['Skewness'] = f"{returns.skew():.4f}"
    s['Kurtosis (Excess)'] = f"{returns.kurtosis():.4f}"
    
    cum_ret = (1 + returns).cumprod()
    roll_max = cum_ret.cummax()
    dd = (cum_ret - roll_max) / roll_max
    s['Max Drawdown (%)'] = f"{dd.min() * 100:.2f}"
    
    ann_ret = returns.mean() * 252
    calmar = ann_ret / abs(dd.min()) if dd.min() != 0 else np.nan
    s['Calmar Ratio'] = f"{calmar:.4f}" if not np.isnan(calmar) else 'N/A'
    
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (ann_ret - 0.065) / downside if downside != 0 else np.nan
    s['Sortino Ratio'] = f"{sortino:.4f}" if not np.isnan(sortino) else 'N/A'
    
    pos = (returns > 0).sum()
    neg = (returns < 0).sum()
    s['Positive Days'] = f"{pos} ({pos/len(returns)*100:.1f}%)"
    s['Negative Days'] = f"{neg} ({neg/len(returns)*100:.1f}%)"
    s['Avg Gain'] = f"{returns[returns > 0].mean() * 100:.4f}%"
    s['Avg Loss'] = f"{returns[returns < 0].mean() * 100:.4f}%"
    s['Gain/Loss Ratio'] = f"{abs(returns[returns > 0].mean() / returns[returns < 0].mean()):.4f}"
    
    return s


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  TATA MOTORS COMPLETE DATA PIPELINE v2")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # ---- Step 1: Stitch ----
    stitched, tmpv_post, source = stitch_tata_motors()
    
    if stitched.empty:
        print("\n FATAL: Could not fetch any data. Aborting.")
        return
    
    stitched.to_csv(os.path.join(PROCESSED_DIR, 'tata_motors_stitched.csv'))
    print(f"  Saved: tata_motors_stitched.csv ({len(stitched)} rows)")
    
    # Also save TMPV post-demerger separately
    if not tmpv_post.empty:
        tmpv_post.to_csv(os.path.join(PROCESSED_DIR, 'tata_motors_pv_post_demerger.csv'))
        print(f"  Saved: tata_motors_pv_post_demerger.csv ({len(tmpv_post)} rows)")
    
    # ---- Step 2: Peers ----
    peer_data = fetch_peers()
    
    for name, df in peer_data.items():
        fname = name.lower().replace(' ', '_').replace('&', 'and') + '_prices.csv'
        df.to_csv(os.path.join(RAW_DIR, fname))
    
    # Merged close prices
    all_close = pd.DataFrame()
    all_close['Tata Motors (Stitched)'] = stitched['Close']
    for name, df in peer_data.items():
        all_close[name] = df['Close']
    all_close.to_csv(os.path.join(PROCESSED_DIR, 'all_close_prices_v2.csv'))
    print(f"  Saved: all_close_prices_v2.csv ({len(all_close)} rows, {len(all_close.columns)} tickers)")
    
    # ---- Step 3: Stats ----
    stats = compute_stats(stitched, name=source)
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(PROCESSED_DIR, 'corrected_price_stats.csv'), index=False)
    print(f"\n  === CORRECTED PRICE STATISTICS ===")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    
    # ---- Step 4: Financial Ratios ----
    print("\n" + "=" * 70)
    print("  PHASE 4: FINANCIAL STATEMENT ANALYSIS")
    print("=" * 70)
    
    # Use TMPV.BO for Tata Motors financials (it has the full entity data)
    fin_tickers = {
        'Tata Motors': 'TATAMOTORS.NS',
        'Maruti Suzuki': 'MARUTI.NS',
        'Mahindra & Mahindra': 'M&M.NS',
        'Ashok Leyland': 'ASHOKLEY.NS',
        'Bajaj Auto': 'BAJAJ-AUTO.NS',
    }
    
    # Try TATAMOTORS.NS first, fallback to TMPV.BO for financials
    all_ratios = []
    all_multi = []
    
    for name, ticker in fin_tickers.items():
        fin = fetch_financials(ticker, name=name)
        
        # Fallback for Tata Motors if TATAMOTORS.NS financials are empty
        if name == 'Tata Motors' and fin['income'].empty:
            print(f"    Retrying with TMPV.BO...")
            fin = fetch_financials('TMPV.BO', name='Tata Motors (via TMPV.BO)')
        if name == 'Tata Motors' and fin['income'].empty:
            print(f"    Retrying with TMCV.NS...")
            fin = fetch_financials('TMCV.NS', name='Tata Motors (via TMCV.NS)')
        
        ratios = compute_ratios(fin)
        all_ratios.append(ratios)
        
        multi = build_multi_year(fin)
        if not multi.empty:
            all_multi.append(multi)
    
    # Save ratios
    ratio_df = pd.DataFrame(all_ratios)
    ratio_df.to_csv(os.path.join(PROCESSED_DIR, 'financial_ratios.csv'), index=False)
    print(f"\n  Saved: financial_ratios.csv ({len(ratio_df)} companies)")
    
    # Save multi-year
    if all_multi:
        multi_df = pd.concat(all_multi, ignore_index=True)
        multi_df.to_csv(os.path.join(PROCESSED_DIR, 'multi_year_financials.csv'), index=False)
        print(f"  Saved: multi_year_financials.csv ({len(multi_df)} rows)")
    
    # ---- Print Ratio Table ----
    print("\n" + "=" * 70)
    print("  PEER RATIO COMPARISON TABLE")
    print("=" * 70)
    
    display_cols = ['Company', 'Market Cap (Cr)', 'P/E', 'P/B', 'EV/EBITDA',
                    'ROE (%)', 'ROCE (%)', 'Net Margin (%)', 'EBITDA Margin (%)',
                    'D/E', 'Current Ratio', 'Beta', 'Revenue (Cr)', 'Net Income (Cr)']
    avail = [c for c in display_cols if c in ratio_df.columns]
    print(ratio_df[avail].to_string(index=False))
    
    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Tata Motors: {len(stitched)} trading days ({source})")
    print(f"  Peers fetched: {len(peer_data)}")
    print(f"  Financial ratios: {len(all_ratios)} companies")
    print(f"\n  Output files in: {PROCESSED_DIR}")
    for f in ['tata_motors_stitched.csv', 'all_close_prices_v2.csv',
              'corrected_price_stats.csv', 'financial_ratios.csv', 'multi_year_financials.csv']:
        print(f"    {f}")


if __name__ == '__main__':
    main()
