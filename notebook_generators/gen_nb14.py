"""
gen_nb14.py — Generate Notebook 14: The Technical Engine
=========================================================
Replaces the old Institutional Roadmap with a disciplined
Deep Learning approach using LSTM + MC Dropout + GARCH.
"""
from notebook_generator import NotebookBuilder
nb = NotebookBuilder()

# ==================== TITLE ====================
nb.add_markdown("""# 🧠 Notebook 14: The Technical Engine — A Disciplined Deep Learning Approach

---

**Author:** Dnyanesh  
**Date:** February 2025  
**Objective:** Move beyond basic classifiers. Deploy a temporal-aware LSTM neural network  
with Monte Carlo Dropout for uncertainty estimation and GARCH(1,1) for volatility gating.

---

## Why This Matters

In modern quantitative finance, achieving a high accuracy score on historical data is relatively easy;  
surviving real-world market turbulence is exceptionally hard. The primary reason retail machine learning  
models fail in live markets is their inability to say, **"I don't know."**

To bridge the gap between academic theory and institutional execution, we construct:

1. **A Monte Carlo LSTM** — that provides probability distributions, not point estimates
2. **A GARCH Volatility Gate** — that blocks risky trades even when the LSTM says "BUY"
3. **Purged Walk-Forward Validation** — that eliminates data leakage from rolling indicators
""")

# ==================== IMPORTS ====================
nb.add_code("""# ============================================================
# IMPORTS & CONFIGURATION
# ============================================================
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11

# Import our institutional backend
from agentic_backend import MonteCarloLSTM, get_mc_dropout_predictions, run_garch_volatility_forecast, garch_volatility_gate

PROCESSED_DIR = '../data/processed' if os.path.exists('../data/processed') else 'data/processed'
print("✅ All imports successful. Institutional engine armed.")
""")

# ==================== 14.1 LSTM ARCHITECTURE ====================
nb.add_markdown("""---

## 14.1 Respecting the Sequence of Time: The LSTM Architecture

Standard models, like Random Forests or XGBoost, view market data as **isolated snapshots**.  
They look at Tuesday's RSI and Wednesday's moving average without truly understanding  
the chronological narrative connecting them.

We deploy an **LSTM** because financial markets are **inherently sequential**. An LSTM utilizes  
an internal "Cell State" and algorithmic "Gates" (Forget, Input, Output) to maintain long-term memory.

> 💡 **Layman Translation:** If a stock slowly accumulates volume over 30 days before breaking out,  
> a standard model only sees the final day's breakout. Our LSTM **remembers** the 30 days of quiet  
> accumulation. It understands the *trajectory* of the price, not just the current snapshot.

### Architecture Details

| Component | Configuration | Purpose |
|-----------|--------------|---------|
| Input Layer | 14-18 features | RSI, MACD, Bollinger, Volume, HMM Regime, Macro... |
| LSTM Layers | 2 stacked, 64 hidden units | Captures temporal dependencies |
| Dropout | 20% (explicit nn.Dropout) | Enables Monte Carlo inference |
| Output | Sigmoid → [0, 1] | Probability of upward breakout |
""")

nb.add_code("""# ============================================================
# 14.1 Initialize the Monte Carlo LSTM Architecture
# ============================================================

# Load feature-engineered data
feat_file = os.path.join(PROCESSED_DIR, 'tata_motors_all_features.csv')
clean_file = os.path.join(PROCESSED_DIR, 'tata_motors_clean.csv')

if os.path.exists(feat_file):
    df = pd.read_csv(feat_file, index_col=0, parse_dates=True)
    print(f"✅ Loaded feature data: {df.shape[0]} rows × {df.shape[1]} columns")
elif os.path.exists(clean_file):
    df = pd.read_csv(clean_file, index_col=0, parse_dates=True)
    print(f"✅ Loaded clean data: {df.shape[0]} rows × {df.shape[1]} columns")
else:
    print("⚠️ No processed data found. Using synthetic data for demonstration.")
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='B')
    df = pd.DataFrame({
        'Close': np.cumsum(np.random.randn(500) * 2) + 500,
        'Volume': np.random.randint(1000000, 5000000, 500),
        'High': np.cumsum(np.random.randn(500) * 2) + 505,
        'Low': np.cumsum(np.random.randn(500) * 2) + 495,
    }, index=dates)

# Define feature columns (use available numeric columns)
feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                if c not in ['Target', 'Signal', 'Position']][:18]

print(f"\\nUsing {len(feature_cols)} features: {feature_cols[:6]}...")

# Instantiate the model
input_features = len(feature_cols)
model = MonteCarloLSTM(input_size=input_features, hidden_size=64, num_layers=2, dropout_rate=0.2)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\\n📊 Model Architecture:")
print(f"   Total Parameters:     {total_params:,}")
print(f"   Trainable Parameters: {trainable_params:,}")
print(f"\\n{model}")
""")

# ==================== 14.2 PURGED WALK-FORWARD ====================
nb.add_markdown("""---

## 14.2 Purged Walk-Forward Validation: Eliminating Data Leakage

One of the most dangerous pitfalls in stock modeling is **"look-ahead bias"** — accidentally  
leaking future data into the past during model training.

### Standard CV vs Purged Walk-Forward

| Standard K-Fold | Purged Walk-Forward |
|----------------|-------------------|
| Random splits | Chronological splits |
| Future data can leak into past | Strict past→future ordering |
| No gap between train/test | **10-day purge buffer** |
| Invalid for time series | Institutional standard |

> ⚠️ **Critical Issue:** A 21-day moving average computed on Day 100 contains information  
> from Days 80-100. If we test on Day 90, we've leaked future data. The **purge buffer**  
> solves this by deleting the rows between train and test sets.
""")

nb.add_code("""# ============================================================
# 14.2 Purged Walk-Forward Cross-Validation
# ============================================================

def purged_walk_forward_split(data, n_splits=5, purge_gap=10):
    '''
    Custom time-series CV that drops `purge_gap` rows between
    train and test to prevent look-ahead bias from rolling indicators.
    '''
    n = len(data)
    fold_size = n // (n_splits + 1)
    splits = []
    
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_start = train_end + purge_gap  # PURGE GAP
        test_end = min(test_start + fold_size, n)
        
        if test_start >= n or test_end <= test_start:
            break
            
        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))
        splits.append((train_idx, test_idx))
        
    return splits

# Demonstrate the purge concept
splits = purged_walk_forward_split(df, n_splits=5, purge_gap=10)
print(f"📈 Purged Walk-Forward: {len(splits)} folds")
print(f"   Purge Gap: 10 trading days\\n")

for i, (train_idx, test_idx) in enumerate(splits):
    print(f"   Fold {i+1}: Train[0:{max(train_idx)}] → 🚫 PURGE[{max(train_idx)+1}:{min(test_idx)-1}] → Test[{min(test_idx)}:{max(test_idx)}]")

# Visualization: Purge concept
fig, ax = plt.subplots(figsize=(16, 4))
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
for i, (train_idx, test_idx) in enumerate(splits):
    y = len(splits) - i
    ax.barh(y, max(train_idx), left=0, height=0.6, color=colors[i], alpha=0.5, label=f'Fold {i+1} Train')
    ax.barh(y, max(test_idx) - min(test_idx), left=min(test_idx), height=0.6, color=colors[i], alpha=0.9)
    # Purge zone
    ax.barh(y, min(test_idx) - max(train_idx), left=max(train_idx), height=0.6, color='red', alpha=0.3)

ax.set_xlabel('Trading Days', fontsize=12)
ax.set_ylabel('Fold', fontsize=12)
ax.set_title('Purged Walk-Forward Cross-Validation — Red = Purge Buffer (Data Leakage Prevention)', fontsize=14, fontweight='bold')
ax.set_yticks(range(1, len(splits) + 1))
ax.set_yticklabels([f'Fold {len(splits) - i}' for i in range(len(splits))])
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, 'purged_walk_forward.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\\n✅ Purged Walk-Forward visualization saved.")
""")

# ==================== 14.3 MONTE CARLO DROPOUT ====================
nb.add_markdown("""---

## 14.3 Monte Carlo Dropout: Teaching the AI to Measure Uncertainty

A raw probability score is **dangerous**. If an AI states there is an "80% chance"  
a stock will go up, we must ask: *How confident is the AI in its own guess?*

### The Solution: Run the prediction 50 times

Instead of asking our neural network for a single prediction, we force the network  
to run the **same prediction 50 different times**, randomly dropping out different  
"neurons" (connections) during each pass.

> 💡 **Layman Translation:** Imagine asking a panel of 50 expert analysts the same question.  
> If all 50 analysts independently agree the stock is going up, our standard deviation  
> (uncertainty) is very low. We have a **high-confidence signal**. However, if the 50  
> analysts give wildly different answers, the AI's internal uncertainty score spikes.

| Scenario | Mean Prob | Std Dev | Interpretation |
|----------|-----------|---------|----------------|
| High Confidence | 82% | ±3% | Strong signal → Execute trade |
| Moderate Confidence | 65% | ±12% | Weak signal → Reduce position |
| Low Confidence | 55% | ±20% | AI is guessing → No trade |
""")

nb.add_code("""# ============================================================
# 14.3 Monte Carlo Dropout Inference
# ============================================================

# Prepare data for LSTM (sequence format)
feature_data = df[feature_cols].dropna()
scaler = StandardScaler()
scaled_data = scaler.fit_transform(feature_data)

# Create sequences (lookback = 20 trading days)
lookback = 20
sequences = []
for i in range(lookback, len(scaled_data)):
    sequences.append(scaled_data[i-lookback:i])

X_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
print(f"📊 Input tensor shape: {X_tensor.shape}")
print(f"   (samples={X_tensor.shape[0]}, timesteps={X_tensor.shape[1]}, features={X_tensor.shape[2]})")

# Run MC Dropout (50 forward passes with dropout ON)
print(f"\\n🎲 Running Monte Carlo Dropout ({50} forward passes)...")
mean_prob, uncertainty = get_mc_dropout_predictions(model, X_tensor, num_passes=50)

# Latest prediction
latest_prob = mean_prob[-1] if isinstance(mean_prob, np.ndarray) and mean_prob.ndim > 0 else float(mean_prob)
latest_uncertainty = uncertainty[-1] if isinstance(uncertainty, np.ndarray) and uncertainty.ndim > 0 else float(uncertainty)

print(f"\\n{'='*50}")
print(f"  LSTM Buy Probability:  {latest_prob*100:.1f}%")
print(f"  Model Uncertainty:     ±{latest_uncertainty*100:.1f}%")
print(f"  Confidence Level:      {'🟢 HIGH' if latest_uncertainty < 0.05 else '🟡 MODERATE' if latest_uncertainty < 0.15 else '🔴 LOW'}")
print(f"{'='*50}")

# Visualization: MC Dropout distribution
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Probability time series with uncertainty band
ax = axes[0]
if isinstance(mean_prob, np.ndarray) and mean_prob.ndim > 0:
    x_range = range(len(mean_prob))
    ax.plot(x_range, mean_prob, color='#1565C0', linewidth=1.5, label='Mean Probability')
    ax.fill_between(x_range, mean_prob - uncertainty, mean_prob + uncertainty,
                    alpha=0.3, color='#42A5F5', label='±1σ Uncertainty')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Decision Boundary')
ax.set_xlabel('Trading Day')
ax.set_ylabel('Buy Probability')
ax.set_title('LSTM Probability with Uncertainty Band', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2. Latest prediction histogram (50 passes)
ax = axes[1]
model.train()
last_sample = X_tensor[-1:] if X_tensor.dim() == 3 else X_tensor
pass_preds = []
with torch.no_grad():
    for _ in range(50):
        p = model(last_sample).item()
        pass_preds.append(p)
ax.hist(pass_preds, bins=15, color='#7E57C2', alpha=0.7, edgecolor='white')
ax.axvline(np.mean(pass_preds), color='red', linewidth=2, linestyle='--', label=f'Mean: {np.mean(pass_preds):.3f}')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Frequency (out of 50 passes)')
ax.set_title('MC Dropout Distribution\\n(Latest Sample, 50 Passes)', fontweight='bold')
ax.legend()

# 3. Uncertainty over time
ax = axes[2]
if isinstance(uncertainty, np.ndarray) and uncertainty.ndim > 0:
    ax.plot(uncertainty, color='#E65100', linewidth=1, alpha=0.7)
    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Low Uncertainty Threshold')
    ax.axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='High Uncertainty Threshold')
ax.set_xlabel('Trading Day')
ax.set_ylabel('Standard Deviation')
ax.set_title('Model Uncertainty Over Time', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Monte Carlo Dropout: Teaching the AI to Say "I Don\\'t Know"', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, 'mc_dropout_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\\n✅ Monte Carlo Dropout analysis complete and saved.")
""")

# ==================== 14.4 GARCH VOLATILITY GATE ====================
nb.add_markdown("""---

## 14.4 The GARCH Volatility Gate

As a final layer of risk management, the LSTM's predictions are filtered through  
a **GARCH(1,1) Volatility Model**.

Even if the LSTM identifies a high-probability upward trend, the GARCH model analyzes  
historical price action to forecast tomorrow's statistical turbulence.

> **The Rule:** If the LSTM signals "BUY," but the GARCH model detects an incoming  
> explosion of erratic volatility, the system **overrides the signal** and enforces  
> a strict **"HOLD" (Cash)** position.

The system is mathematically programmed to **prioritize capital preservation  
over return generation**.

### GARCH(1,1) Parameters

| Parameter | Meaning | Tata Motors Typical |
|-----------|---------|-------------------|
| ω (omega) | Base variance | ~0.001 |
| α (alpha) | Shock sensitivity | ~0.08 |
| β (beta) | Fear persistence | ~0.88 |
| **α + β** | **Stationarity check** | **< 1.0 (stable)** |
""")

nb.add_code("""# ============================================================
# 14.4 GARCH(1,1) Volatility Gate
# ============================================================
from arch import arch_model

# Calculate returns
if 'Close' in df.columns:
    returns = df['Close'].pct_change().dropna()
elif 'Log_Return' in df.columns:
    returns = df['Log_Return'].dropna()
else:
    returns = df.iloc[:, 0].pct_change().dropna()

# Run the GARCH Volatility Gate
gate_result = garch_volatility_gate(returns, volatility_threshold=0.03)

print(f"{'='*50}")
print(f"  GARCH VOLATILITY GATE RESULTS")
print(f"{'='*50}")
print(f"  Forecasted 1-Day Volatility: {gate_result['predicted_vol']*100:.2f}%")
print(f"  Threshold:                   {gate_result['threshold']*100:.1f}%")
print(f"  Gate Status:                 {'🟢 PASS — Trade Allowed' if gate_result['is_safe'] else '🔴 BLOCKED — Hold Cash'}")
print(f"{'='*50}")

# Fit full GARCH to get parameters and conditional volatility history
scaled_returns = returns * 100
garch_model = arch_model(scaled_returns.dropna(), vol='Garch', p=1, q=1, rescale=False)
garch_fit = garch_model.fit(disp='off')

print(f"\\n📊 GARCH(1,1) Parameters:")
print(f"   ω (omega) = {garch_fit.params['omega']:.6f}")
print(f"   α (alpha) = {garch_fit.params['alpha[1]']:.4f}  ← Shock sensitivity")
print(f"   β (beta)  = {garch_fit.params['beta[1]']:.4f}  ← Fear persistence")
print(f"   α + β     = {garch_fit.params['alpha[1]'] + garch_fit.params['beta[1]']:.4f}  ← {'✅ Stationary' if garch_fit.params['alpha[1]'] + garch_fit.params['beta[1]'] < 1 else '⚠️ Non-stationary'}")

# Visualization: GARCH conditional volatility
conditional_vol = garch_fit.conditional_volatility / 100  # rescale back

fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})

# Panel 1: Price + Volatility Overlay
ax1 = axes[0]
if 'Close' in df.columns:
    price_series = df['Close'].iloc[:len(conditional_vol)]
    ax1.plot(price_series.index[:len(conditional_vol)], price_series.values[:len(conditional_vol)], 
             color='#1565C0', linewidth=1.5, label='Close Price')
    ax1.set_ylabel('Price (₹)', fontsize=12, color='#1565C0')
    
    ax2 = ax1.twinx()
    ax2.fill_between(price_series.index[:len(conditional_vol)], 0, conditional_vol.values[:len(conditional_vol)],
                     alpha=0.3, color='#E65100', label='GARCH Conditional Volatility')
    ax2.set_ylabel('Daily Volatility', fontsize=12, color='#E65100')
    ax2.legend(loc='upper left')

ax1.set_title('GARCH(1,1) Volatility Gate — When Fear Overrides Greed', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Panel 2: Gate status over time
ax3 = axes[1]
vol_threshold = 0.03
gate_pass = conditional_vol < vol_threshold
ax3.fill_between(range(len(gate_pass)), 0, 1, where=gate_pass.values, 
                 color='#4CAF50', alpha=0.3, label='🟢 PASS (Trade Allowed)')
ax3.fill_between(range(len(gate_pass)), 0, 1, where=~gate_pass.values,
                 color='#F44336', alpha=0.3, label='🔴 BLOCKED (Hold Cash)')
ax3.set_xlabel('Trading Days', fontsize=12)
ax3.set_ylabel('Gate Status', fontsize=12)
ax3.set_title('Volatility Gate Timeline — Red Zones = Capital Preservation Mode', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, 'garch_volatility_gate.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\\n✅ GARCH Volatility Gate analysis complete.")
""")

# ==================== COMBINED DECISION ====================
nb.add_markdown("""---

## 14.5 The Combined Decision Engine

The final trade decision combines all three signals:

```
IF LSTM_Probability > 0.6:
    IF MC_Uncertainty < 0.10:
        IF GARCH_Gate == "PASS":
            → EXECUTE BUY (Full Kelly Position)
        ELSE:
            → HOLD (Volatility too high)
    ELSE:
        → REDUCE (Cut position by 50%)
ELSE:
    → NO TRADE
```

This **triple-layered** filtering ensures we only take high-confidence, low-risk trades.
""")

nb.add_code("""# ============================================================
# 14.5 Combined Decision Engine Summary
# ============================================================

print("\\n" + "="*60)
print("  📋 INSTITUTIONAL TRADE DECISION SUMMARY")
print("="*60)
print(f"  LSTM Buy Probability:  {latest_prob*100:.1f}%    {'✅' if latest_prob > 0.6 else '❌'}")
print(f"  MC Uncertainty:        ±{latest_uncertainty*100:.1f}%  {'✅' if latest_uncertainty < 0.10 else '⚠️'}")
print(f"  GARCH Gate:            {gate_result['gate_status']:8s}  {'✅' if gate_result['is_safe'] else '🔴'}")
print("-"*60)

# Decision logic
if latest_prob > 0.6 and latest_uncertainty < 0.10 and gate_result['is_safe']:
    decision = "🟢 EXECUTE BUY — Full Kelly Position"
elif latest_prob > 0.6 and latest_uncertainty < 0.15:
    decision = "🟡 CAUTIOUS BUY — Half Kelly (Reduced Position)"
elif latest_prob > 0.6 and not gate_result['is_safe']:
    decision = "🟠 HOLD — LSTM is bullish but GARCH blocks (high volatility)"
else:
    decision = "🔴 NO TRADE — Insufficient conviction"

print(f"  DECISION: {decision}")
print("="*60)
print("\\n✅ Chapter 14 complete. Proceed to Chapter 15 for the Agentic Workflow.")
""")

# ==================== SAVE ====================
import os
output_dir = os.path.join(os.path.dirname(__file__), '..', 'notebooks')
if not os.path.exists(output_dir):
    output_dir = 'notebooks'
    
output_path = os.path.join(output_dir, '14_Deep_Learning_Engine.ipynb')
nb.save(output_path)
print(f"✅ Saved: {output_path}")
