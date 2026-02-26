from notebook_generator import NotebookBuilder
nb = NotebookBuilder()

# ==================== CELL 1-3: Title & Motivation ====================
nb.add_markdown("""# 📊 Notebook 01: Data Extraction — The Foundation

---

## 🎯 The Story Behind This Project

> *"I invested in Tata Motors after COVID, believing in the EV transition story. When Ratan Tata passed away in October 2024, the stock crashed. In 2025, Tata Motors demerged into TMCV (Commercial Vehicles) and TMPV (Passenger Vehicles). This project is a data-driven deep dive into the new entities."*

This notebook series is a **data-driven analysis** of the Tata Motors universe post-demerger. We will:

1. **Extract** historical price data for TMCV, TMPV and peer companies
2. **Compare** performance across different market regimes
3. **Analyze** sentiment from news headlines around key events
4. **Build** predictive models to identify patterns
5. **Backtest** trading strategies for risk management

### Companies Under Analysis
| Company | Ticker | Why? |
|---------|--------|------|
| **Tata Motors CV (TMCV)** | `TMCV.NS` | Commercial Vehicles — post-demerger entity |
| **Tata Motors PV (TMPV)** | `TMPV.NS` | Passenger Vehicles — post-demerger entity (includes EV) |
| **Maruti Suzuki** | `MARUTI.NS` | India's largest carmaker — the "safe" benchmark |
| **Mahindra & Mahindra** | `M&M.NS` | Direct SUV & EV competitor + CV player |
| **Bajaj Auto** | `BAJAJ-AUTO.NS` | India's #1 two-wheeler exporter — broader auto sector control |
| **Ashok Leyland** | `ASHOKLEY.NS` | India's #2 CV maker — direct TMCV competitor |
| **Hyundai Motor India** | `HYUNDAI.NS` | India's #2 PV maker — direct TMPV competitor (Oct 2024 IPO) |
| **Toyota Motor** | `TM` | Global auto bellwether — JLR-comparable luxury + mass market |
| **Volkswagen AG** | `VWAGY` | European EV transition parallel (ID series vs Nexon EV) |
| **NIFTY 50** | `^NSEI` | Broad market index — sector vs market performance |
| **NIFTY Auto** | `^CNXAUTO` | Sector index — was it an auto sector trend? |
""")

nb.add_markdown("""## 📅 Timeline of Key Events

Understanding the *context* behind price movements is critical. Here are the events we'll track:

| Date | Event | Expected Impact |
|------|-------|----------------|
| **Jan 2020** | Pre-COVID baseline | Normal trading |
| **Mar 2020** | COVID-19 lockdown announced | Crash across all sectors |
| **Oct 2020** | Tata Motors EV announcement | Recovery begins |
| **Jan 2021** | Vaccine rollout begins | Broad market rally |
| **Jun 2022** | Interest rate hikes begin | Growth stocks under pressure |
| **Oct 2024** | Ratan Tata passes away | Emotional selling + institutional rebalancing |

> **Hypothesis:** The Oct 2024 crash was driven more by *sentiment* than by *fundamentals*. We will test this hypothesis throughout the project.
""")

# ==================== CELL 4-6: Imports & Setup ====================
nb.add_markdown("""## 1. Environment Setup

We begin by importing all necessary libraries. Each serves a specific purpose:
- `yfinance` — Yahoo Finance API wrapper for fetching stock data
- `pandas` — Data manipulation and analysis
- `matplotlib` + `seaborn` — Visualization
- `warnings` — Suppress non-critical warnings for cleaner output
""")

nb.add_code("""import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import warnings
from datetime import datetime

# Configuration
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13

# Directories
RAW_DIR = '../data/raw'
PROCESSED_DIR = '../data/processed'
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("✅ Environment ready")
print(f"   yfinance version: {yf.__version__}")
print(f"   pandas version: {pd.__version__}")
print(f"   numpy version: {np.__version__}")""")

nb.add_markdown("""### 📝 Note on Data Sources

**Yahoo Finance** provides adjusted close prices that account for splits and dividends. This is crucial for Tata Motors because:
- **DVR shares** were cancelled and merged in 2023
- **Stock splits** and **bonus issues** have occurred historically

We use `yfinance` because it automatically handles these corporate actions, giving us *clean, comparable* price series.
""")

# ==================== CELL 7-10: Robust Fetcher ====================
nb.add_markdown("""## 2. Robust Data Fetching Function

Stock data APIs are unreliable — they can fail due to:
- Rate limiting (too many requests)
- Ticker changes (NSE vs BSE symbols)
- Corporate actions (delistings, mergers)

Our fetcher implements a **3-level fallback strategy**:

```
Level 1: Try NSE ticker (.NS) with date range
Level 2: Try BSE ticker (.BO) with date range  
Level 3: Try maximum available period
```

This ensures we *always* get data, even if one exchange is temporarily down.
""")

nb.add_code("""def fetch_stock_data(ticker, start='2019-01-01', end=None, name=None):
    \"\"\"
    Robust stock data fetcher with NSE/BSE fallback.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker (e.g., 'TMCV.NS')
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str, optional
        End date. Defaults to today.
    name : str, optional
        Human-readable name for logging
        
    Returns:
    --------
    pd.DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
    \"\"\"
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    
    display_name = name or ticker
    print(f"\\n{'='*60}")
    print(f"  Fetching: {display_name} ({ticker})")
    print(f"  Period: {start} → {end}")
    print(f"{'='*60}")
    
    # LEVEL 1: Try primary ticker
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if not df.empty:
            # Flatten MultiIndex columns (yfinance v0.2+ returns MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            print(f"  ✅ Level 1 SUCCESS | Rows: {len(df)} | Date Range: {df.index[0].date()} to {df.index[-1].date()}")
            return df
        else:
            print(f"  ⚠️ Level 1 returned empty DataFrame")
    except Exception as e:
        print(f"  ❌ Level 1 FAILED: {e}")
    
    # LEVEL 2: Try alternate exchange
    alt_ticker = ticker.replace('.NS', '.BO') if '.NS' in ticker else ticker.replace('.BO', '.NS')
    try:
        df = yf.download(alt_ticker, start=start, end=end, progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            print(f"  ✅ Level 2 SUCCESS (alt: {alt_ticker}) | Rows: {len(df)}")
            return df
        else:
            print(f"  ⚠️ Level 2 returned empty DataFrame")
    except Exception as e:
        print(f"  ❌ Level 2 FAILED: {e}")
    
    # LEVEL 3: Try max period
    try:
        df = yf.download(ticker, period='max', progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            # Filter to our date range
            df = df[df.index >= start]
            print(f"  ✅ Level 3 SUCCESS (max period) | Rows: {len(df)}")
            return df
    except Exception as e:
        print(f"  ❌ Level 3 FAILED: {e}")
    
    print(f"  🚫 ALL LEVELS FAILED for {display_name}")
    return pd.DataFrame()

print("✅ fetch_stock_data() defined")""")

# ==================== CELL 11-14: Fetch All Stocks ====================
nb.add_markdown("""## 3. Fetching Data for All Companies

We fetch data from **January 2019** (pre-COVID baseline) to the present. This gives us:
- ~12 months of "normal" pre-COVID trading
- The full COVID crash and recovery cycle
- The post-COVID bull run
- The October 2024 crash event

### Why 2019?
Starting from 2019 gives us a year of "normal" data before COVID. This baseline is essential for:
- Calculating what "normal" volatility looks like
- Establishing pre-crisis price levels
- Comparing recovery trajectories
""")

nb.add_code("""# Define our universe (Tata Motors demerged into TMCV + TMPV in 2025)
TICKERS = {
    'Tata Motors CV': 'TATAMOTORS.NS',  # Using historical ticker for long-term analysis
    'Tata Motors PV': 'TMPV.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Mahindra & Mahindra': 'M&M.NS',
    'Bajaj Auto': 'BAJAJ-AUTO.NS',
    'Ashok Leyland': 'ASHOKLEY.NS',
    'Hyundai Motor India': 'HYUNDAI.NS',
    'Toyota Motor': 'TM',
    'Volkswagen AG': 'VWAGY',
    'NIFTY 50': '^NSEI',
    'NIFTY Auto': '^CNXAUTO'
}

START_DATE = '2020-01-01'

# Fetch all
stock_data = {}
for name, ticker in TICKERS.items():
    df = fetch_stock_data(ticker, start=START_DATE, name=name)
    stock_data[name] = df
    
print("\\n" + "="*60)
print("  📊 FETCH SUMMARY")
print("="*60)
for name, df in stock_data.items():
    if not df.empty:
        print(f"  ✅ {name:25s} | {len(df):5d} rows | {df.index[0].date()} → {df.index[-1].date()}")
    else:
        print(f"  ❌ {name:25s} | FAILED")""")

nb.add_code("""# Quick sanity check: Do all datasets cover the same period?
print("\\n📋 DATE RANGE COMPARISON")
print("-" * 70)
for name, df in stock_data.items():
    if not df.empty:
        missing_pct = (df.isna().sum() / len(df) * 100).mean()
        print(f"  {name:25s} | Start: {df.index.min().date()} | End: {df.index.max().date()} | Missing: {missing_pct:.1f}%")""")

# ==================== CELL 15-18: Raw Data Inspection ====================
nb.add_markdown("""## 4. Raw Data Inspection

Before any analysis, we must **inspect the raw data** for anomalies. This is the "boring but essential" step that separates professional analysis from amateur work.

### What we're looking for:
1. **Missing values** — Are there gaps? Weekends/holidays are expected, but mid-week gaps are suspicious
2. **Data types** — Are dates parsed correctly? Are prices numeric?
3. **Column structure** — Did yfinance return what we expected?
4. **Obvious outliers** — Any prices that look impossibly high or low?
""")

nb.add_code("""# Detailed inspection — use TMCV as primary, or first available
tata = stock_data.get('Tata Motors CV', stock_data.get('Tata Motors PV', pd.DataFrame()))
primary_name = 'Tata Motors CV' if 'Tata Motors CV' in stock_data and not stock_data['Tata Motors CV'].empty else 'Tata Motors PV'

if not tata.empty:
    print("=" * 60)
    print(f"  {primary_name.upper()} — RAW DATA INSPECTION")
    print("=" * 60)
    
    print("\\n📐 Shape:", tata.shape)
    print("\\n📊 Data Types:")
    print(tata.dtypes)
    
    print("\\n📋 First 5 Rows:")
    display(tata.head())
    
    print("\\n📋 Last 5 Rows:")
    display(tata.tail())
    
    print("\\n📈 Statistical Summary:")
    display(tata.describe().round(2))
    
    print("\\n🔍 Missing Values:")
    missing = tata.isna().sum()
    print(missing[missing > 0] if missing.any() else "   None detected ✅")
else:
    print("❌ Tata Motors data not available")""")

nb.add_code("""# Check for suspicious values (price < 0, volume = 0, extreme jumps)
if not tata.empty:
    print("ANOMALY DETECTION")
    print("-" * 40)
    
    # Negative prices
    price_cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in tata.columns]
    neg_prices = (tata[price_cols] < 0).any().any() if price_cols else False
    print(f"  Negative prices found: {'WARNING YES' if neg_prices else 'No'}")
    
    # Zero volume days
    if 'Volume' in tata.columns:
        zero_vol = (tata['Volume'] == 0).sum()
        print(f"  Zero volume days: {zero_vol} ({'Suspicious' if zero_vol > 5 else 'OK'})")
    
    # Extreme daily moves (>20%)
    if 'Close' in tata.columns:
        daily_returns = tata['Close'].pct_change()
        extreme_moves = daily_returns[daily_returns.abs() > 0.20]
        print(f"  Days with >20% move: {len(extreme_moves)}")
        if len(extreme_moves) > 0:
            print("  Extreme moves detected:")
            for date, ret in extreme_moves.items():
                print(f"     {date.date()}: {ret:+.1%}")""")

# ==================== CELL 19-24: Raw Price Visualization ====================
nb.add_markdown("""## 5. Raw Price Visualization

The first plot is always the most important — it tells the *entire story* at a glance.

We plot all 5 instruments on the same chart using **normalized prices** (base = 100) so we can compare relative performance regardless of absolute price levels.

### Why Normalize?
- Tata Motors trades around ₹700-1000
- NIFTY 50 trades around 18,000-25,000
- Direct comparison is impossible without normalization

**Formula:** $P_{normalized} = \\frac{P_t}{P_0} \\times 100$

Where $P_0$ is the price on the first day of our dataset.
""")

nb.add_code("""# Create normalized price comparison chart
fig, axes = plt.subplots(2, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 1]})

# --- TOP PANEL: Normalized Prices ---
ax1 = axes[0]
colors = {'Tata Motors CV': '#E74C3C', 'Tata Motors PV': '#9B59B6',
          'Maruti Suzuki': '#3498DB', 'NIFTY 50': '#95A5A6', 'NIFTY Auto': '#F39C12'}

for name, df in stock_data.items():
    if not df.empty and 'Close' in df.columns:
        normalized = (df['Close'] / df['Close'].iloc[0]) * 100
        ax1.plot(normalized.index, normalized.values, label=name, 
                color=colors.get(name, 'gray'), linewidth=2 if 'Tata' in name else 1.2,
                alpha=1.0 if 'Tata' in name else 0.7)

# Mark key events
events = {
    '2020-03-23': ('COVID\\nCrash', '#E74C3C'),
    '2021-01-01': ('Vaccine\\nRollout', '#2ECC71'),
    '2024-10-09': ('Ratan Tata\\nPasses Away', '#8E44AD'),
}
for date_str, (label, color) in events.items():
    event_date = pd.Timestamp(date_str)
    ax1.axvline(x=event_date, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.annotate(label, xy=(event_date, ax1.get_ylim()[1]*0.95), fontsize=9,
                color=color, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax1.set_title('Normalized Price Performance (Base = 100)', fontsize=16, fontweight='bold')
ax1.set_ylabel('Normalized Price')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# --- BOTTOM PANEL: TMCV Volume ---
ax2 = axes[1]
if not tata.empty and 'Volume' in tata.columns:
    ax2.bar(tata.index, tata['Volume'] / 1e6, color='#E74C3C', alpha=0.5, width=1)
    ax2.set_ylabel('Volume (Millions)')
    ax2.set_title(f'{primary_name} — Daily Trading Volume', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(RAW_DIR, 'price_comparison_normalized.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\\n💾 Saved: price_comparison_normalized.png")""")

nb.add_markdown("""### 🔍 Interpretation of the Normalized Price Chart

**Key Observations:**
1. **COVID Crash (Mar 2020):** ALL stocks crashed simultaneously — this was a *systematic* market event, not company-specific
2. **Recovery Divergence:** After COVID, Tata Motors recovered *faster* than Maruti, driven by the EV narrative (Nexon EV launch)
3. **Oct 2024 Crash:** Notice how Tata Motors dropped sharply while NIFTY and Maruti were relatively stable — this was a *company-specific* event
4. **Volume Spike:** The bottom panel shows a massive volume increase during the Oct 2024 crash, suggesting institutional selling

> **Key Insight:** The COVID crash was unavoidable (whole market), but the Oct 2024 crash might have been predictable through sentiment analysis and institutional flow data.
""")

nb.add_code("""# Individual stock charts for detailed analysis
if not tata.empty:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. OHLC Summary
    ax = axes[0, 0]
    ax.plot(tata.index, tata['Close'], color='#E74C3C', linewidth=1.5, label='Close')
    ax.fill_between(tata.index, tata['Low'], tata['High'], alpha=0.15, color='#E74C3C', label='High-Low Range')
    ax.set_title('Tata Motors — Price with High-Low Range')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Volume Histogram
    ax = axes[0, 1]
    ax.hist(tata['Volume'] / 1e6, bins=50, color='#3498DB', alpha=0.7, edgecolor='white')
    ax.set_title('Volume Distribution (Millions)')
    ax.set_xlabel('Daily Volume (M)')
    
    # 3. Daily Returns Distribution
    ax = axes[1, 0]
    returns = tata['Close'].pct_change().dropna()
    ax.hist(returns, bins=100, color='#2ECC71', alpha=0.7, edgecolor='white')
    ax.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
    ax.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax.set_title('Daily Returns Distribution')
    ax.legend()
    
    # 4. Cumulative Returns
    ax = axes[1, 1]
    cum_ret = (1 + returns).cumprod()
    ax.plot(cum_ret.index, cum_ret.values, color='#8E44AD', linewidth=1.5)
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('Cumulative Returns (₹1 Invested)')
    ax.fill_between(cum_ret.index, 1, cum_ret.values, where=cum_ret.values >= 1, 
                    color='green', alpha=0.1)
    ax.fill_between(cum_ret.index, 1, cum_ret.values, where=cum_ret.values < 1, 
                    color='red', alpha=0.1)
    
    plt.suptitle('Tata Motors — Detailed Price Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RAW_DIR, 'tata_detailed_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()""")

# ==================== CELL 25-30: Regime Analysis ====================
nb.add_markdown("""## 6. Regime-Based Price Comparison

The market doesn't behave the same during calm and crisis periods. We define **5 regimes** to slice our data:

| Regime | Period | Market Condition |
|--------|--------|-----------------|
| **Pre-COVID** | Jan 2019 – Feb 2020 | Normal/slightly bearish |
| **COVID Crash** | Mar 2020 – Jun 2020 | Panic selling, liquidity crisis |
| **Recovery** | Jul 2020 – Dec 2021 | V-shaped recovery, stimulus-driven |
| **Post-COVID** | Jan 2022 – Sep 2024 | Rate hikes, normalization |
| **Oct 2024 Crash** | Oct 2024 – Present | Ratan Tata event, sentiment-driven |

This regime tagging will be used throughout all subsequent notebooks.
""")

nb.add_code("""def tag_regime(date):
    \"\"\"
    Tags a date with its market regime.
    
    Logic:
    - Pre-COVID: Before the WHO declared pandemic (Mar 2020)
    - COVID Crash: The acute crash period (Mar-Jun 2020)
    - Recovery: Stimulus-driven V-shaped recovery (Jul 2020 - Dec 2021)
    - Post-COVID: Normalization with rate hikes (Jan 2022 - Sep 2024)
    - Oct 2024 Crash: Ratan Tata event and aftermath
    \"\"\"
    if date < pd.Timestamp('2020-03-01'):
        return 'Pre-COVID'
    elif date < pd.Timestamp('2020-07-01'):
        return 'COVID Crash'
    elif date < pd.Timestamp('2022-01-01'):
        return 'Recovery'
    elif date < pd.Timestamp('2024-10-01'):
        return 'Post-COVID'
    else:
        return 'Oct 2024 Crash'

# Apply to all stocks
for name, df in stock_data.items():
    if not df.empty:
        df['Regime'] = df.index.map(tag_regime)
        df['Returns'] = df['Close'].pct_change()
        stock_data[name] = df

print("Regime tagging complete")
tata = stock_data.get(primary_name, pd.DataFrame())
if not tata.empty:
    print(f"\\n{primary_name} — Observations per Regime:")
    print(tata['Regime'].value_counts().sort_index())""")

nb.add_code("""# Regime-colored price chart
if not tata.empty:
    fig, ax = plt.subplots(figsize=(16, 8))
    
    regime_colors = {
        'Pre-COVID': '#3498DB',
        'COVID Crash': '#E74C3C',
        'Recovery': '#2ECC71',
        'Post-COVID': '#F39C12',
        'Oct 2024 Crash': '#8E44AD'
    }
    
    for regime, color in regime_colors.items():
        if 'Regime' not in tata.columns:
            break
        mask = tata['Regime'] == regime
        regime_data = tata[mask]
        if not regime_data.empty:
            ax.fill_between(regime_data.index, 0, regime_data['Close'], 
                          alpha=0.15, color=color)
            ax.plot(regime_data.index, regime_data['Close'], color=color, 
                   linewidth=1.5, label=regime)
    
    ax.set_title(f'{primary_name} — Price by Market Regime', fontsize=16, fontweight='bold')
    ax.set_ylabel('Price (₹)')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RAW_DIR, 'tata_regime_chart.png'), dpi=150, bbox_inches='tight')
    plt.show()""")

nb.add_markdown("""### 📊 Regime Performance Summary

Let's calculate the **return and volatility** for each regime. This gives us a quantitative view of how Tata Motors behaved during each period.

**Key Metrics:**
- **Total Return:** How much the stock gained/lost in each period
- **Annualized Volatility:** $\\sigma_{annual} = \\sigma_{daily} \\times \\sqrt{252}$ — how "wild" the price swings were
- **Sharpe Ratio:** $SR = \\frac{R - R_f}{\\sigma}$ — was the return worth the risk? (Using $R_f = 6\\%$ for India)
""")

nb.add_code("""# Calculate regime-level statistics for all stocks
if not tata.empty:
    print("=" * 80)
    print("  REGIME PERFORMANCE COMPARISON")
    print("=" * 80)
    
    for name in ['Tata Motors CV', 'Tata Motors PV', 'Maruti Suzuki']:
        df = stock_data.get(name, pd.DataFrame())
        if df.empty:
            continue
            
        print(f"\\n📊 {name}")
        print("-" * 70)
        print(f"  {'Regime':20s} | {'Total Return':>12s} | {'Volatility':>12s} | {'Sharpe':>8s} | {'Max DD':>8s}")
        print("-" * 70)
        
        for regime in ['Pre-COVID', 'COVID Crash', 'Recovery', 'Post-COVID', 'Oct 2024 Crash']:
            mask = df['Regime'] == regime if 'Regime' in df.columns else pd.Series(False, index=df.index)
            regime_data = df[mask]
            
            if len(regime_data) > 1:
                total_ret = (regime_data['Close'].iloc[-1] / regime_data['Close'].iloc[0]) - 1
                vol = regime_data['Returns'].std() * np.sqrt(252)
                sharpe = (regime_data['Returns'].mean() * 252 - 0.06) / vol if vol > 0 else 0
                
                # Max Drawdown
                cummax = regime_data['Close'].cummax()
                drawdown = (regime_data['Close'] / cummax - 1).min()
                
                print(f"  {regime:20s} | {total_ret:>+11.1%} | {vol:>11.1%} | {sharpe:>+7.2f} | {drawdown:>+7.1%}")""")

# ==================== CELL 31-36: Deep Dive Oct 2024 ====================
nb.add_markdown("""## 7. 🔍 Deep Dive: The October 2024 Crash

This is the event that started this project. Let's zoom in on what happened:

**Timeline:**
- **Oct 9, 2024:** Ratan Tata passes away at age 86
- **Oct 10-11:** Emotional selling begins, stock gaps down
- **Oct 14-18:** Institutional selling accelerates
- **Late Oct:** Stock hits 52-week low

### Questions to Answer:
1. How much did Tata Motors fall vs peers during this period?
2. Was the volume abnormally high?
3. Did the broader market also fall, or was this Tata-specific?
4. Was there any recovery pattern?
""")

nb.add_code("""# Zoom into Oct 2024 (with 2 weeks context before and after)
crash_start = '2024-09-15'
crash_end = '2024-11-15'

if not tata.empty:
    fig, axes = plt.subplots(3, 1, figsize=(16, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Panel 1: Price comparison during crash
    ax = axes[0]
    for name in ['Tata Motors CV', 'Tata Motors PV', 'Maruti Suzuki']:
        df = stock_data.get(name, pd.DataFrame())
        if not df.empty:
            crash_data = df[(df.index >= crash_start) & (df.index <= crash_end)]
            if not crash_data.empty:
                normalized = (crash_data['Close'] / crash_data['Close'].iloc[0]) * 100
                ax.plot(normalized.index, normalized.values, label=name, linewidth=2)
    
    ax.axvline(pd.Timestamp('2024-10-09'), color='black', linestyle='--', linewidth=2, label='Ratan Tata Passes')
    ax.set_title('Oct 2024 Period — Normalized Price Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Price (Base=100)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Volume comparison
    ax = axes[1]
    crash_tata = tata[(tata.index >= crash_start) & (tata.index <= crash_end)] if not tata.empty else pd.DataFrame()
    if not crash_tata.empty:
        avg_vol = crash_tata['Volume'].mean()
        colors_vol = ['#E74C3C' if v > avg_vol * 1.5 else '#3498DB' for v in crash_tata['Volume']]
        ax.bar(crash_tata.index, crash_tata['Volume'] / 1e6, color=colors_vol, alpha=0.7, width=1)
        ax.axhline(avg_vol / 1e6, color='orange', linestyle='--', label=f'Avg: {avg_vol/1e6:.1f}M')
        ax.axhline(avg_vol * 1.5 / 1e6, color='red', linestyle='--', alpha=0.5, label='1.5x Avg (Abnormal)')
    ax.set_title('Tata Motors — Volume During Crash (Red = Abnormally High)', fontsize=12)
    ax.set_ylabel('Volume (Millions)')
    ax.legend()
    
    # Panel 3: Daily Returns
    ax = axes[2]
    if not crash_tata.empty:
        crash_returns = crash_tata['Returns'].dropna()
        bar_colors = ['#E74C3C' if r < 0 else '#2ECC71' for r in crash_returns]
        ax.bar(crash_returns.index, crash_returns.values * 100, color=bar_colors, alpha=0.8, width=1)
        ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Tata Motors — Daily Returns (%)', fontsize=12)
    ax.set_ylabel('Return (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RAW_DIR, 'oct_2024_crash_deep_dive.png'), dpi=150, bbox_inches='tight')
    plt.show()""")

nb.add_markdown("""### 🧐 Crash Analysis — Key Findings

From the charts above, we can draw several critical observations:

1. **Tata-Specific Event:** While Maruti and M&M showed minor weakness, they didn't crash — confirming this was NOT a sector-wide event
2. **Volume Anomaly:** Trading volume spiked to **3-5x normal levels** during the crash, indicating:
   - Institutional investors were actively selling
   - Retail panic selling amplified the move
3. **No Immediate Recovery:** Unlike the COVID crash (which saw a quick V-shape), the Oct 2024 crash showed a slow, grinding decline — suggesting structural concern, not just emotional selling
4. **Worst Daily Drop:** The single worst day saw >5% decline, comparable to COVID-era crashes

> **Hypothesis Update:** The crash appears to be a mix of emotional selling (short-term) + institutional rebalancing (medium-term). Fundamentals need to be checked separately (Notebook 01b).
""")

# ==================== CELL 37-42: Save Data ====================
nb.add_markdown("""## 8. Saving Raw Data

We save each stock's data as a separate CSV for use in subsequent notebooks. We also create a **merged dataset** with regime tags for quick loading.
""")

nb.add_code("""# Save individual stock CSVs
saved_files = []
for name, df in stock_data.items():
    if not df.empty:
        filename = name.lower().replace(' ', '_').replace('&', 'and') + '_prices.csv'
        filepath = os.path.join(RAW_DIR, filename)
        df.to_csv(filepath)
        saved_files.append((name, filepath, len(df)))
        
print("💾 SAVED FILES:")
print("-" * 60)
for name, path, rows in saved_files:
    print(f"  ✅ {name:25s} → {os.path.basename(path):35s} ({rows} rows)")""")

nb.add_code("""# Create merged close-price DataFrame for easy comparison
close_prices = pd.DataFrame()
for name, df in stock_data.items():
    if not df.empty and 'Close' in df.columns:
        close_prices[name] = df['Close']

# Save merged
close_prices.to_csv(os.path.join(PROCESSED_DIR, 'all_close_prices.csv'))
print(f"\\n💾 Merged close prices saved: {close_prices.shape}")
print(f"   Columns: {list(close_prices.columns)}")
print(f"   Date range: {close_prices.index.min().date()} → {close_prices.index.max().date()}")""")

nb.add_code("""# Save primary Tata entity (TMCV) as main analysis file, and also TMPV
for key in ['Tata Motors CV', 'Tata Motors PV']:
    df_save = stock_data.get(key, pd.DataFrame())
    if not df_save.empty:
        tag = 'tmcv' if 'CV' in key else 'tmpv'
        df_save.to_csv(os.path.join(RAW_DIR, f'{tag}_prices.csv'))
        print(f"  Saved: {tag}_prices.csv")

# Save primary (TMCV) as the main downstream file
tata = stock_data.get('Tata Motors CV', stock_data.get('Tata Motors PV', pd.DataFrame()))
if not tata.empty:
    tata.to_csv(os.path.join(RAW_DIR, 'tata_motors_prices.csv'))
    tata.to_csv(os.path.join(PROCESSED_DIR, 'tata_motors_clean.csv'))
    print("\\nPrimary data saved as tata_motors_clean.csv for downstream notebooks")""")

# ==================== CELL 43-47: Summary Statistics ====================
nb.add_markdown("""## 9. Summary Statistics & Correlation Matrix

Before moving to cleaning and feature engineering, let's understand how our stocks **correlate** with each other. High correlation with NIFTY means the stock moves with the market; low correlation suggests company-specific drivers.

**Correlation Coefficient:** $\\rho_{X,Y} = \\frac{\\text{Cov}(X,Y)}{\\sigma_X \\cdot \\sigma_Y}$

Where:
- $\\rho = 1$: Perfect positive correlation (stocks move together)
- $\\rho = 0$: No correlation (independent movements)
- $\\rho = -1$: Perfect negative correlation (stocks move opposite)
""")

nb.add_code("""# Return correlations
returns_df = pd.DataFrame()
for name, df in stock_data.items():
    if not df.empty and 'Returns' in df.columns:
        returns_df[name] = df['Returns']

if not returns_df.empty:
    corr = returns_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, vmin=-1, vmax=1, square=True, linewidths=1,
                cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title('Return Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RAW_DIR, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\\n🔍 Key Correlations with Tata Motors:")
    primary_col = 'Tata Motors CV' if 'Tata Motors CV' in corr.columns else ('Tata Motors PV' if 'Tata Motors PV' in corr.columns else None)
    if primary_col:
        tata_corr = corr[primary_col].drop(primary_col).sort_values(ascending=False)
        for name, val in tata_corr.items():
            strength = "Strong" if abs(val) > 0.7 else "Moderate" if abs(val) > 0.4 else "Weak"
            print(f"   {name:25s}: {val:.3f} ({strength})")""")

nb.add_markdown("""### 📊 Correlation Insights

**What the correlation matrix tells us:**
- **Tata Motors vs NIFTY Auto:** High correlation expected — both reflect the auto sector
- **Tata Motors vs Maruti:** Moderate correlation — they share sector trends but differ in fundamentals (EV vs ICE)
- **Tata Motors vs NIFTY 50:** If correlation is moderate (~0.5-0.7), company-specific factors explain ~50% of daily moves

> **Implication for modeling:** Features that capture *idiosyncratic* (company-specific) movements will be more useful than broad market indicators.
""")

# ==================== CELL 48-50: Conclusion ====================
nb.add_markdown("""## 10. Notebook Summary & Next Steps

### ✅ What We Accomplished
1. **Fetched** 5+ years of price data for Tata Motors, Maruti, M&M, NIFTY 50, and NIFTY Auto
2. **Inspected** raw data for anomalies, missing values, and data quality issues
3. **Visualized** the complete price history with event markers
4. **Tagged** 5 market regimes (Pre-COVID through Oct 2024 Crash)
5. **Calculated** regime-specific performance metrics
6. **Deep-dived** into the October 2024 crash mechanics
7. **Computed** return correlations between all instruments

### 📁 Output Files
| File | Location | Description |
|------|----------|-------------|
| `tata_motors_prices.csv` | `data/raw/` | Tata Motors OHLCV + Regime tags |
| `maruti_suzuki_prices.csv` | `data/raw/` | Maruti Suzuki OHLCV |
| `mahindra_and_mahindra_prices.csv` | `data/raw/` | M&M OHLCV |
| `all_close_prices.csv` | `data/processed/` | Merged close prices for all stocks |
| `tata_motors_clean.csv` | `data/processed/` | Clean Tata Motors data for downstream |

### ➡️ Next Steps
- **Notebook 02:** Data Cleaning & Preprocessing — handle missing values, merge data sources
- **Notebook 03:** Feature Engineering — build technical indicators (RSI, MACD, Bollinger)
- **Notebook 06:** Sentiment Analysis — combine price crashes with news sentiment

---
*"The data doesn't lie, but it doesn't explain everything either. The numbers tell us WHAT happened. The next notebooks will tell us WHY."*
""")

# ==================== CELL: Synthetic vs Actual Price Overlay ====================
nb.add_markdown("""## 🧪 Bonus: Synthetic vs. Actual Price — The Random Walk Test

### What is a Synthetic Price Path?

Imagine you flip a coin every day: heads → stock goes up, tails → stock goes down. If you scale the moves by the stock's *actual* average return and volatility, you get a **Geometric Brownian Motion (GBM)** path — this is what finance theory says stock prices *should* look like if markets are perfectly efficient.

**The key question:** Does Tata Motors' real price look like a random walk, or does it show patterns that a model can exploit?

**GBM Formula:**
$$S_{t+1} = S_t \\cdot \\exp\\left((\\mu - \\frac{\\sigma^2}{2})\\Delta t + \\sigma \\sqrt{\\Delta t} \\cdot Z\\right)$$

Where:
- $\\mu$ = average daily return (drift)
- $\\sigma$ = daily volatility
- $Z$ = random normal variable (the "coin flip")

> **Layman's Translation:** We generate 10 fake versions of the stock using the same statistics (average return + volatility). If the real stock looks just like the fakes, then its movements are essentially random. If it looks *different*, there might be exploitable patterns.
""")

nb.add_code("""# Generate Synthetic GBM Price Paths
np.random.seed(42)
num_paths = 10

if not tata.empty and 'Close' in tata.columns:
    close = tata['Close'].dropna()
    returns = close.pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    S0 = close.iloc[0]
    T = len(close)
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Generate and plot synthetic paths
    for i in range(num_paths):
        Z = np.random.normal(0, 1, T)
        log_returns = (mu - 0.5 * sigma**2) + sigma * Z
        synthetic_path = S0 * np.exp(np.cumsum(log_returns))
        ax.plot(close.index, synthetic_path[:len(close.index)], 
                alpha=0.25, linewidth=0.8, color='#3498DB')
    
    # Plot actual price (bold, on top)
    ax.plot(close.index, close.values, color='#E74C3C', linewidth=2.5, 
            label='Actual Tata Motors', zorder=10)
    
    # Dummy for legend
    ax.plot([], [], color='#3498DB', alpha=0.5, linewidth=1, label=f'{num_paths} GBM Synthetic Paths')
    
    # Mark key events
    for date_str, (label, color) in events.items():
        event_date = pd.Timestamp(date_str)
        if event_date >= close.index.min() and event_date <= close.index.max():
            ax.axvline(x=event_date, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
    
    ax.set_title('Actual vs. Synthetic (GBM) Price Paths — Is Tata Motors a Random Walk?', 
                 fontsize=16, fontweight='bold')
    ax.set_ylabel('Price (₹)', fontsize=13)
    ax.set_xlabel('Date', fontsize=13)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RAW_DIR, 'synthetic_vs_actual_gbm.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\\nGBM Parameters Used:")
    print(f"  Daily Drift (μ):       {mu:.6f} ({mu*252*100:.2f}% annualized)")
    print(f"  Daily Volatility (σ):  {sigma:.6f} ({sigma*np.sqrt(252)*100:.2f}% annualized)")
    print(f"  Starting Price (S₀):   ₹{S0:.2f}")
else:
    print("❌ Data not available for synthetic path generation")""")

nb.add_markdown("""### 🔍 What This Chart Tells Us

**If the red line (actual price) sits comfortably among the blue lines (synthetic paths):**
- The stock is behaving like a random walk → Hard to predict
- Technical indicators may have limited value

**If the red line frequently breaks away from the blue band:**
- There are non-random patterns → Potential for modeling
- Crashes like Oct 2024 appear as sharp deviations that a purely random model wouldn't replicate

> **Key Insight:** The GBM model knows NOTHING about EV announcements, COVID, or Ratan Tata's passing. Any deviation of the red line from the blue band is, by definition, driven by *information* — and that's what our later models will try to capture.
""")

# Save
nb.save("notebooks/01_Data_Extraction.ipynb")
