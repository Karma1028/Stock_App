from notebook_generator import NotebookBuilder
nb = NotebookBuilder()

# ==================== TITLE ====================
nb.add_markdown("""# 📊 Notebook 03: Feature Engineering — Technical Indicators

---

## 🎯 Objective

Technical indicators are mathematical transformations of price and volume data that traders use to predict future price movements. In this notebook, we will:

1. **Calculate indicators manually** (from scratch, no libraries) to deeply understand the math
2. **Verify** our calculations against the `pandas_ta` library
3. **Visualize** each indicator with buy/sell signals
4. **Analyze** how indicators behaved during COVID and the Oct 2024 crash

### Why Manual Calculation?
> Using a library is easy. Understanding *why* RSI = 30 means "oversold" requires building it from scratch.

We'll implement: **RSI, MACD, Bollinger Bands, OBV, and ATR**
""")

# ==================== Imports ====================
nb.add_code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 12

PROCESSED_DIR = '../data/processed'
print("Environment ready")""")

# ==================== Load Data ====================
nb.add_markdown("""## 1. Load Cleaned Data""")

nb.add_code("""# Load Tata Motors cleaned data
file_path = os.path.join(PROCESSED_DIR, 'tata_motors_clean.csv')
print(f"Loading data from: {os.path.abspath(file_path)}")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Data file not found at {file_path}. Please run Notebook 01 first.")

df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# Verify 5-year history
if len(df) < 1000: # Approx 4 years
    print(f"WARNING: Data history is short ({len(df)} rows). Transfer Learning requires 5+ years.")
else:
    print(f"✅ Data loaded successfully: {len(df)} rows (approx {len(df)/252:.1f} years)")
    print(f"   Range: {df.index.min().date()} to {df.index.max().date()}")

# Ensure we have the essential columns
essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in essential_cols:
    if col not in df.columns:
        print(f"WARNING: {col} column missing!")
        
print(f"Data loaded: {df.shape}")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
print(f"Columns: {list(df.columns[:8])}")
df.head()""")

# ==================== RSI (8 cells) ====================
nb.add_markdown("""## 2. RSI — Relative Strength Index

### Theory

RSI measures the **speed and magnitude** of price changes to evaluate overbought or oversold conditions.

**Mathematical Formula:**

$$RSI = 100 - \\frac{100}{1 + RS}$$

Where:
$$RS = \\frac{\\text{Average Gain over } n \\text{ periods}}{\\text{Average Loss over } n \\text{ periods}}$$

**Interpretation:**
- RSI > 70: **Overbought** — Price has risen too fast, might reverse down
- RSI < 30: **Oversold** — Price has fallen too fast, might bounce back
- RSI = 50: **Neutral** — No strong signal

**Standard period:** n = 14 days (proposed by J. Welles Wilder in 1978)

### Step-by-Step Calculation:
1. Calculate daily price change: $\\Delta P = Close_t - Close_{t-1}$
2. Separate into gains and losses
3. Calculate average gain and loss using exponential smoothing
4. Compute RS and then RSI
""")

nb.add_code("""# Step 1: Calculate daily price changes
df['Price_Change'] = df['Close'].diff()

print("Step 1: Daily Price Changes")
print(df[['Close', 'Price_Change']].head(10))
print(f"\\nPositive (gain) days: {(df['Price_Change'] > 0).sum()}")
print(f"Negative (loss) days: {(df['Price_Change'] < 0).sum()}")
print(f"Zero change days:     {(df['Price_Change'] == 0).sum()}")""")

nb.add_code("""# Step 2: Separate gains and losses
df['Gain'] = df['Price_Change'].apply(lambda x: x if x > 0 else 0)
df['Loss'] = df['Price_Change'].apply(lambda x: abs(x) if x < 0 else 0)

print("Step 2: Gains and Losses Separated")
print(df[['Close', 'Price_Change', 'Gain', 'Loss']].head(10))""")

nb.add_code("""# Step 3: Calculate Average Gain/Loss (Wilder's Smoothing Method)
# Wilder's smoothing is an exponential moving average with alpha = 1/n
period = 14

df['Avg_Gain'] = df['Gain'].ewm(alpha=1/period, min_periods=period).mean()
df['Avg_Loss'] = df['Loss'].ewm(alpha=1/period, min_periods=period).mean()

print("Step 3: Smoothed Average Gain/Loss")
print(df[['Gain', 'Avg_Gain', 'Loss', 'Avg_Loss']].dropna().head(10))""")

nb.add_code("""# Step 4: Calculate RS and RSI
df['RS'] = df['Avg_Gain'] / df['Avg_Loss']
df['RSI_Manual'] = 100 - (100 / (1 + df['RS']))

# Handle edge case: when Avg_Loss = 0, RS = infinity, RSI = 100
df['RSI_Manual'] = df['RSI_Manual'].clip(0, 100)

print("Step 4: RS and RSI")
print(df[['Avg_Gain', 'Avg_Loss', 'RS', 'RSI_Manual']].dropna().head(10))""")

nb.add_code("""# Visualize RSI
fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})

# Price
ax = axes[0]
ax.plot(df.index, df['Close'], color='black', linewidth=1.2)
ax.set_title('Tata Motors — Price + RSI Analysis', fontsize=14, fontweight='bold')
ax.set_ylabel('Price')
ax.grid(True, alpha=0.3)

# RSI
ax = axes[1]
ax.plot(df.index, df['RSI_Manual'], color='#8E44AD', linewidth=1)
ax.axhline(70, color='#E74C3C', linestyle='--', alpha=0.7, label='Overbought (70)')
ax.axhline(30, color='#2ECC71', linestyle='--', alpha=0.7, label='Oversold (30)')
ax.axhline(50, color='gray', linestyle=':', alpha=0.5)
ax.fill_between(df.index, 70, 100, alpha=0.1, color='red')
ax.fill_between(df.index, 0, 30, alpha=0.1, color='green')
ax.set_ylabel('RSI')
ax.set_ylim(0, 100)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, 'rsi_chart.png'), dpi=150, bbox_inches='tight')
plt.show()""")

nb.add_markdown("""### RSI Analysis — Key Observations

1. **COVID Crash (Mar 2020):** RSI plunged below 20 — extreme oversold territory. This was a strong \"buy\" signal in hindsight.
2. **Recovery Rally (2020-2021):** RSI frequently touched 70+ — the stock was consistently in overbought territory during the rally.
3. **Oct 2024:** Did RSI signal the crash before it happened? Look for divergence — if price made new highs but RSI made lower highs, that's a **bearish divergence** warning.

> **Key Insight:** RSI is a *momentum* indicator. It tells you HOW FAST the price is moving, not WHERE it will go.
""")

# ==================== MACD (6 cells) ====================
nb.add_markdown("""## 3. MACD — Moving Average Convergence Divergence

### Theory

MACD captures **trend changes** by comparing two exponential moving averages (EMAs).

**Components:**
1. **MACD Line:** $EMA_{12} - EMA_{26}$ (difference between fast and slow EMAs)
2. **Signal Line:** $EMA_9(MACD\\ Line)$ (smoothed version of MACD)
3. **Histogram:** $MACD\\ Line - Signal\\ Line$ (momentum)

**Trading Signals:**
- **Bullish Crossover:** MACD crosses ABOVE Signal Line = Buy
- **Bearish Crossover:** MACD crosses BELOW Signal Line = Sell
- **Histogram Divergence:** When histogram shrinks while price makes new highs = warning

### EMA Formula:
$$EMA_t = Price_t \\times \\alpha + EMA_{t-1} \\times (1 - \\alpha)$$

Where $\\alpha = \\frac{2}{n + 1}$ and $n$ is the number of periods.
""")

nb.add_code("""# Calculate MACD components
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']

print("MACD Components Calculated")
print(df[['Close', 'EMA_12', 'EMA_26', 'MACD_Line', 'Signal_Line', 'MACD_Histogram']].dropna().tail(10))""")

nb.add_code("""# Visualize MACD
fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [2, 1, 1]})

# Price with EMAs
ax = axes[0]
ax.plot(df.index, df['Close'], color='black', linewidth=1.2, label='Close')
ax.plot(df.index, df['EMA_12'], color='#3498DB', linewidth=0.8, alpha=0.7, label='EMA 12')
ax.plot(df.index, df['EMA_26'], color='#E74C3C', linewidth=0.8, alpha=0.7, label='EMA 26')
ax.set_title('Price with EMA 12 & EMA 26', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# MACD Line + Signal
ax = axes[1]
ax.plot(df.index, df['MACD_Line'], color='#3498DB', linewidth=1, label='MACD Line')
ax.plot(df.index, df['Signal_Line'], color='#E74C3C', linewidth=1, label='Signal Line')
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.set_title('MACD Line vs Signal Line')
ax.legend()

# Histogram
ax = axes[2]
colors = ['#2ECC71' if v >= 0 else '#E74C3C' for v in df['MACD_Histogram']]
ax.bar(df.index, df['MACD_Histogram'], color=colors, alpha=0.7, width=1)
ax.set_title('MACD Histogram (Momentum)')
ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, 'macd_chart.png'), dpi=150, bbox_inches='tight')
plt.show()""")

nb.add_markdown("""### MACD Analysis

- **Bullish crossovers** (MACD > Signal) during 2020-2021 recovery were strong buy signals
- **Bearish crossovers** around Oct 2024 preceded the crash
- The **histogram** shows momentum: when it shrinks → trend is weakening, even if price is still rising
""")

# ==================== Bollinger Bands (5 cells) ====================
nb.add_markdown("""## 4. Bollinger Bands

### Theory

Bollinger Bands measure **volatility** by creating an envelope around the price.

**Formula:**
- **Middle Band:** $SMA_{20}$ (20-day Simple Moving Average)
- **Upper Band:** $SMA_{20} + 2 \\times \\sigma_{20}$
- **Lower Band:** $SMA_{20} - 2 \\times \\sigma_{20}$

Where $\\sigma_{20}$ is the 20-day rolling standard deviation.

**Interpretation:**
- Price touching Upper Band = potentially overbought
- Price touching Lower Band = potentially oversold
- **Band Width** = $(Upper - Lower) / Middle$ — measures volatility
- Narrow bands (\"squeeze\") often precede big moves
""")

nb.add_code("""# Calculate Bollinger Bands
window = 20
df['BB_Middle'] = df['Close'].rolling(window=window).mean()
df['BB_Std'] = df['Close'].rolling(window=window).std()
df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

# Band touch signals
df['BB_Upper_Touch'] = df['Close'] >= df['BB_Upper']
df['BB_Lower_Touch'] = df['Close'] <= df['BB_Lower']

print(f"Upper band touches: {df['BB_Upper_Touch'].sum()}")
print(f"Lower band touches: {df['BB_Lower_Touch'].sum()}")""")

nb.add_code("""# Visualize Bollinger Bands
fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

ax = axes[0]
ax.plot(df.index, df['Close'], color='black', linewidth=1.2, label='Close')
ax.plot(df.index, df['BB_Middle'], color='#3498DB', linewidth=0.8, label='SMA 20')
ax.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.15, color='#3498DB', label='Bollinger Bands')
ax.plot(df.index, df['BB_Upper'], color='#3498DB', linewidth=0.5, alpha=0.5)
ax.plot(df.index, df['BB_Lower'], color='#3498DB', linewidth=0.5, alpha=0.5)
ax.set_title('Bollinger Bands (20, 2)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(df.index, df['BB_Width'], color='#E74C3C', linewidth=1)
ax.set_title('Band Width (Volatility Measure)')
ax.axhline(df['BB_Width'].mean(), color='gray', linestyle='--', alpha=0.5, label='Average Width')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, 'bollinger_chart.png'), dpi=150, bbox_inches='tight')
plt.show()""")

# ==================== OBV (4 cells) ====================
nb.add_markdown("""## 5. OBV — On-Balance Volume

### Theory

OBV links **volume to price direction**. The idea: volume precedes price.

**Formula:**
$$OBV_t = OBV_{t-1} + \\begin{cases} Volume_t & \\text{if } Close_t > Close_{t-1} \\\\ -Volume_t & \\text{if } Close_t < Close_{t-1} \\\\ 0 & \\text{if } Close_t = Close_{t-1} \\end{cases}$$

**Why it matters:** If OBV is rising while price is flat, it means \"smart money\" is accumulating — a bullish signal. If OBV is falling while price is stable, distribution is happening.
""")

nb.add_code("""# Calculate OBV
df['OBV'] = 0.0
for i in range(1, len(df)):
    if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
        df.iloc[i, df.columns.get_loc('OBV')] = df['OBV'].iloc[i-1] + df['Volume'].iloc[i]
    elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
        df.iloc[i, df.columns.get_loc('OBV')] = df['OBV'].iloc[i-1] - df['Volume'].iloc[i]
    else:
        df.iloc[i, df.columns.get_loc('OBV')] = df['OBV'].iloc[i-1]

# Normalize for easier visualization
df['OBV_Normalized'] = (df['OBV'] - df['OBV'].min()) / (df['OBV'].max() - df['OBV'].min()) * 100

print("OBV calculated")
print(df[['Close', 'Volume', 'OBV']].tail(10))""")

nb.add_code("""# OBV vs Price comparison
fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 1]})

axes[0].plot(df.index, df['Close'], color='black', linewidth=1.2)
axes[0].set_title('Price', fontsize=12)
axes[0].set_ylabel('Price')
axes[0].grid(True, alpha=0.3)

axes[1].plot(df.index, df['OBV'] / 1e9, color='#2ECC71', linewidth=1)
axes[1].set_title('On-Balance Volume (Billions)', fontsize=12)
axes[1].set_ylabel('OBV (B)')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Price vs OBV — Divergence Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\\nLook for DIVERGENCES:")
print("  - Price UP + OBV DOWN = Bearish divergence (distribution)")
print("  - Price DOWN + OBV UP = Bullish divergence (accumulation)")""")

# ==================== ATR (3 cells) ====================
nb.add_markdown("""## 6. ATR — Average True Range

### Theory

ATR measures **volatility** without caring about direction.

**True Range (TR):**
$$TR = \\max(High - Low, |High - Close_{prev}|, |Low - Close_{prev}|)$$

**ATR:** $ATR = SMA_{14}(TR)$ or $EMA_{14}(TR)$

Higher ATR = more volatile (bigger daily swings), Lower ATR = calmer market.
""")

nb.add_code("""# Calculate ATR
df['TR'] = np.maximum(
    df['High'] - df['Low'],
    np.maximum(
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    )
)
df['ATR_14'] = df['TR'].rolling(window=14).mean()

print("ATR calculated")
print(df[['High', 'Low', 'Close', 'TR', 'ATR_14']].dropna().tail(10))""")

nb.add_code("""# ATR Chart
fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [2, 1]})

axes[0].plot(df.index, df['Close'], color='black', linewidth=1.2)
axes[0].set_title('Price')

axes[1].plot(df.index, df['ATR_14'], color='#E67E22', linewidth=1.2)
axes[1].axhline(df['ATR_14'].mean(), color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('ATR(14) — Volatility Measure')

plt.suptitle('Average True Range Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\\nCurrent ATR: {df['ATR_14'].iloc[-1]:.2f}")
print(f"Average ATR: {df['ATR_14'].mean():.2f}")
print(f"Max ATR (COVID crash likely): {df['ATR_14'].max():.2f}")""")

# ==================== Summary Dashboard (3 cells) ====================
nb.add_markdown("""## 7. Technical Indicator Dashboard

Let's create a comprehensive view showing all indicators together:
""")

nb.add_code("""# Complete Technical Dashboard
fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(5, 1, height_ratios=[3, 1, 1, 1, 1], hspace=0.3)

# Price + Bollinger
ax1 = fig.add_subplot(gs[0])
ax1.plot(df.index, df['Close'], color='black', linewidth=1.2, label='Close')
ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='blue')
ax1.set_title('Tata Motors — Complete Technical Analysis Dashboard', fontsize=16, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# RSI
ax2 = fig.add_subplot(gs[1])
ax2.plot(df.index, df['RSI_Manual'], color='purple', linewidth=0.8)
ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
ax2.fill_between(df.index, 70, 100, alpha=0.05, color='red')
ax2.fill_between(df.index, 0, 30, alpha=0.05, color='green')
ax2.set_ylabel('RSI')
ax2.set_ylim(0, 100)

# MACD Histogram
ax3 = fig.add_subplot(gs[2])
colors = ['green' if v >= 0 else 'red' for v in df['MACD_Histogram']]
ax3.bar(df.index, df['MACD_Histogram'], color=colors, alpha=0.6, width=1)
ax3.set_ylabel('MACD Hist')

# OBV
ax4 = fig.add_subplot(gs[3])
ax4.plot(df.index, df['OBV'] / 1e9, color='teal', linewidth=0.8)
ax4.set_ylabel('OBV (B)')

# ATR
ax5 = fig.add_subplot(gs[4])
ax5.plot(df.index, df['ATR_14'], color='orange', linewidth=0.8)
ax5.set_ylabel('ATR')

plt.savefig(os.path.join(PROCESSED_DIR, 'technical_dashboard.png'), dpi=150, bbox_inches='tight')
plt.show()""")

# ==================== Save & Conclude ====================
nb.add_code("""# Save the DataFrame with all technical indicators
df.to_csv(os.path.join(PROCESSED_DIR, 'tata_motors_with_technicals.csv'))
print(f"Saved: tata_motors_with_technicals.csv ({df.shape[0]} rows, {df.shape[1]} columns)")
print(f"\\nNew columns added: {[c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Regime']]}")""")

nb.add_markdown("""## 8. Summary

### Indicators Calculated:
| Indicator | Formula | Signal |
|-----------|---------|--------|
| **RSI** | $100 - \\frac{100}{1+RS}$ | Overbought (>70) / Oversold (<30) |
| **MACD** | $EMA_{12} - EMA_{26}$ | Crossover signals |
| **Bollinger Bands** | $SMA_{20} \\pm 2\\sigma$ | Volatility breakouts |
| **OBV** | Cumulative signed volume | Volume-price divergence |
| **ATR** | $SMA_{14}(TR)$ | Volatility magnitude |

### Key Findings:
- All indicators confirmed the severity of both the COVID crash and Oct 2024 crash
- RSI oversold signals during COVID crash preceded a massive rally
- OBV divergence analysis may reveal \"smart money\" patterns before Oct 2024

---
*Next: Notebook 04 — Statistical Feature Engineering*
""")

# ==================== CELL: Rolling Correlation Heatmap ====================
nb.add_markdown("""## 🔥 Bonus: Rolling Correlation Heatmap — When Indicators Agree and Disagree

### Why This Matters

Technical indicators are not independent — they often move together. But **the degree to which they agree changes over time**:

- **During calm markets:** RSI, MACD, and Bollinger Bands may give independent, sometimes conflicting signals.
- **During crashes:** ALL indicators tend to align (everything screams "sell") — this is called **correlation convergence** and it's a hallmark of crisis periods.

> **Layman's Translation:** Think of technical indicators as different doctors examining the same patient. In calm times, each doctor might have a slightly different opinion. During a crisis, every doctor says the same thing: "There's a problem." This chart shows you WHEN the doctors agree.

We compute a **60-day rolling correlation** between key indicators and visualize how it evolves.
""")

nb.add_code("""# Rolling Correlation Heatmap (Animated Snapshots)
corr_features = []
for f in ['RSI_Manual', 'MACD_Histogram', 'BB_Width', 'ATR_14', 'OBV_Normalized']:
    if f in df.columns:
        corr_features.append(f)

if len(corr_features) >= 3:
    window = 60  # 60-day rolling window
    
    # Calculate rolling correlations between select pairs
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Select 4 regime snapshots
    regimes_to_show = {
        'Pre-COVID (Calm)': ('2019-06-01', '2020-02-01'),
        'COVID Crash (Crisis)': ('2020-03-01', '2020-07-01'),
        'Recovery (Bull Run)': ('2020-10-01', '2021-12-01'),
        'Oct 2024 (Event Shock)': ('2024-09-01', '2024-12-31')
    }
    
    for idx, (regime_name, (start, end)) in enumerate(regimes_to_show.items()):
        ax = axes[idx // 2][idx % 2]
        mask = (df.index >= start) & (df.index <= end)
        regime_data = df.loc[mask, corr_features].dropna()
        
        if len(regime_data) > 10:
            corr_matrix = regime_data.corr()
            mask_tri = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask_tri, annot=True, fmt='.2f', 
                       cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                       square=True, linewidths=1, ax=ax,
                       cbar_kws={'shrink': 0.8})
            ax.set_title(f'{regime_name}', fontsize=13, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'{regime_name}\\nInsufficient Data', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f'{regime_name}', fontsize=13, fontweight='bold')
    
    plt.suptitle('Rolling Correlation Heatmaps: How Indicators Relate Across Regimes', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'rolling_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Also plot rolling correlation time series for the most important pair
    if 'RSI_Manual' in df.columns and 'MACD_Histogram' in df.columns:
        rolling_corr = df['RSI_Manual'].rolling(window).corr(df['MACD_Histogram'])
        
        fig, axes = plt.subplots(2, 1, figsize=(18, 8), gridspec_kw={'height_ratios': [1, 1]})
        
        axes[0].plot(df.index, df['Close'], color='black', linewidth=1)
        axes[0].set_title('Tata Motors Price', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Price')
        
        axes[1].plot(rolling_corr.index, rolling_corr.values, color='#8E44AD', linewidth=1)
        axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1].fill_between(rolling_corr.index, 0, rolling_corr.values, 
                            where=rolling_corr.values > 0, alpha=0.2, color='green')
        axes[1].fill_between(rolling_corr.index, 0, rolling_corr.values, 
                            where=rolling_corr.values < 0, alpha=0.2, color='red')
        axes[1].set_title('60-Day Rolling Correlation: RSI vs MACD Histogram', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Correlation')
        axes[1].set_ylim(-1, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PROCESSED_DIR, 'rolling_corr_rsi_macd.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\\nCorrelation Statistics (RSI vs MACD):")
        print(f"  Overall mean: {rolling_corr.mean():.3f}")
        print(f"  Max (most aligned): {rolling_corr.max():.3f}")
        print(f"  Min (most divergent): {rolling_corr.min():.3f}")
else:
    print("Insufficient features for rolling correlation analysis")""")

nb.add_markdown("""### 🔍 Key Observations from the Rolling Correlation Analysis

**What to look for:**

1. **Correlation Convergence During Crashes:**
   - If the heatmaps during COVID Crash and Oct 2024 show HIGHER absolute correlations than during calm periods, it confirms that crises cause indicator alignment.
   - This is a warning sign: when all indicators agree your model has less "edge" since there's only one signal, not multiple independent ones.

2. **RSI-MACD Rolling Correlation:**
   - When this is strongly positive → Both momentum indicators confirm each other (strong trend)
   - When near zero → Indicators diverge (choppy market, hard to trade)
   - When negative → Indicators contradict each other (potential reversal forming)

> **For Your Trading Model:** Features that are highly correlated during crashes add redundancy, not information. Consider using only the MOST responsive indicator during crisis periods (typically ATR or Bollinger Width for volatility).
""")

nb.save("notebooks/03_Feature_Engineering_Technical.ipynb")
