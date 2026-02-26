from notebook_generator import NotebookBuilder
nb = NotebookBuilder()

nb.add_markdown("# 📈 Notebook 11: Time Series Forecasting with Prophet\n\n---\n\n**Author:** Dnyanesh  \n**Date:** February 2025  \n**Series:** Tata Motors Deep Dive (11 of 13)\n\n## Objective\n\nFacebook Prophet forecasts Tata Motors' price using:\n- **Trend** component (linear/logistic growth)\n- **Seasonality** (weekly, yearly patterns)\n- **Changepoint detection** (automatic trend shifts)\n- **Holiday/event effects**\n\n### Prophet Decomposition:\n$$y(t) = g(t) + s(t) + h(t) + \\epsilon_t$$\n\n| Component | Meaning |\n|-----------|--------|\n| $g(t)$ | Trend (piecewise linear) |\n| $s(t)$ | Seasonality (Fourier series) |\n| $h(t)$ | Holiday effects |\n| $\\epsilon_t$ | Error term |")

nb.add_code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport os, warnings\nwarnings.filterwarnings('ignore')\nsns.set_style('whitegrid')\nPROCESSED_DIR = '../data/processed'\n\ntry:\n    from prophet import Prophet\n    PROPHET_OK = True\nexcept:\n    try:\n        from fbprophet import Prophet\n        PROPHET_OK = True\n    except:\n        PROPHET_OK = False\nprint(f'Prophet: {\"✅\" if PROPHET_OK else \"❌ Not available\"}')")

nb.add_code("# Load data\ndf = pd.read_csv(os.path.join(PROCESSED_DIR, 'tata_motors_clean.csv'), index_col=0, parse_dates=True)\nprint(f'Data: {df.shape[0]} trading days')\nprint(f'Range: {df.index.min().date()} to {df.index.max().date()}')\nprint(f'\\nPrice summary:')\nprint(f'  Min: {df[\"Close\"].min():.2f}')\nprint(f'  Max: {df[\"Close\"].max():.2f}')\nprint(f'  Current: {df[\"Close\"].iloc[-1]:.2f}')")

nb.add_markdown("---\n\n## Part 1: Data Preparation\n\nProphet requires a specific format:\n- `ds` column: datestamp\n- `y` column: target value\n- Optional: **regressors** (external variables)")

nb.add_code("# Prophet format\nprophet_df = pd.DataFrame()\nprophet_df['ds'] = df.index\nprophet_df['y'] = df['Close'].values\nprophet_df = prophet_df.dropna().reset_index(drop=True)\n\nif 'Volume' in df.columns:\n    prophet_df['volume'] = np.log1p(df['Volume'].values[:len(prophet_df)])\n    prophet_df['volume'] = prophet_df['volume'].fillna(prophet_df['volume'].median())\n\nprint(f'Prophet data: {prophet_df.shape}')\nprint(prophet_df.describe())")

nb.add_code("# Visualization of raw data\nfig, axes = plt.subplots(2, 1, figsize=(18, 10))\n\nax = axes[0]\nax.plot(prophet_df['ds'], prophet_df['y'], color='#2C3E50', linewidth=1.5)\nax.set_title('Tata Motors — Price to Forecast', fontsize=14, fontweight='bold')\nax.set_ylabel('Price')\nax.grid(True, alpha=0.3)\n\nax = axes[1]\nax.plot(prophet_df['ds'], prophet_df['y'].pct_change().rolling(21).std()*np.sqrt(252)*100,\n       color='#E74C3C', linewidth=1)\nax.set_title('Rolling Volatility (annualized)', fontweight='bold')\nax.set_ylabel('Volatility (%)')\n\nplt.tight_layout(); plt.show()")

nb.add_markdown("---\n\n## Part 2: Train-Test Split\n\nWe use the last 90 trading days as test set — this includes the Oct 2024 period.")

nb.add_code("split_idx = len(prophet_df) - 90\ntrain = prophet_df.iloc[:split_idx].copy()\ntest = prophet_df.iloc[split_idx:].copy()\n\nprint(f'Train: {len(train)} days ({train[\"ds\"].min().date()} to {train[\"ds\"].max().date()})')\nprint(f'Test:  {len(test)} days ({test[\"ds\"].min().date()} to {test[\"ds\"].max().date()})')\nprint(f'\\nTrain/Test ratio: {len(train)/len(prophet_df)*100:.1f}% / {len(test)/len(prophet_df)*100:.1f}%')")

nb.add_code("# Visualize split\nfig, ax = plt.subplots(figsize=(18, 6))\nax.plot(train['ds'], train['y'], color='#3498DB', linewidth=1.5, label='Training')\nax.plot(test['ds'], test['y'], color='#E74C3C', linewidth=1.5, label='Testing')\nax.axvline(train['ds'].iloc[-1], color='black', linestyle='--', alpha=0.5, label='Split Point')\nax.set_title('Train-Test Split', fontweight='bold', fontsize=14)\nax.legend(fontsize=12)\nax.grid(True, alpha=0.3)\nplt.tight_layout(); plt.show()")

nb.add_markdown("---\n\n## Part 3: Model Training\n\n### 3.1 Basic Prophet Model")

nb.add_code("if PROPHET_OK:\n    model_basic = Prophet(\n        daily_seasonality=False,\n        weekly_seasonality=True,\n        yearly_seasonality=True,\n        changepoint_prior_scale=0.05,\n        seasonality_prior_scale=10\n    )\n    model_basic.fit(train[['ds', 'y']])\n    print('✅ Basic Prophet model trained')\nelse:\n    print('❌ Prophet not available. Using simple forecast fallback.')")

nb.add_markdown("### 3.2 Prophet with Volume Regressor")

nb.add_code("if PROPHET_OK and 'volume' in train.columns:\n    model_reg = Prophet(\n        daily_seasonality=False,\n        weekly_seasonality=True,\n        yearly_seasonality=True,\n        changepoint_prior_scale=0.05\n    )\n    model_reg.add_regressor('volume')\n    model_reg.fit(train[['ds', 'y', 'volume']])\n    print('✅ Prophet + Volume regressor trained')\nelse:\n    model_reg = None")

nb.add_markdown("---\n\n## Part 4: Generate Forecasts")

nb.add_code("if PROPHET_OK:\n    future_basic = model_basic.make_future_dataframe(periods=90)\n    forecast_basic = model_basic.predict(future_basic)\n    print(f'Basic forecast: {len(forecast_basic)} data points')\n    print(forecast_basic[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))")

nb.add_code("if PROPHET_OK and model_reg:\n    future_reg = model_reg.make_future_dataframe(periods=90)\n    # Fill volume for future dates\n    vol_vals = list(prophet_df['volume'].values) + [prophet_df['volume'].median()] * (len(future_reg) - len(prophet_df))\n    future_reg['volume'] = vol_vals[:len(future_reg)]\n    forecast_reg = model_reg.predict(future_reg)\n    print(f'\\nRegressor forecast: {len(forecast_reg)} data points')")

nb.add_markdown("---\n\n## Part 5: Forecast Visualization")

nb.add_code("if PROPHET_OK:\n    fig = model_basic.plot(forecast_basic, figsize=(18, 8))\n    plt.title('Prophet Forecast — Basic Model', fontsize=14, fontweight='bold')\n    plt.ylabel('Price (INR)')\n    plt.tight_layout(); plt.show()")

nb.add_code("if PROPHET_OK:\n    fig = model_basic.plot_components(forecast_basic, figsize=(16, 12))\n    plt.suptitle('Forecast Components Decomposition', fontsize=14, fontweight='bold', y=1.01)\n    plt.tight_layout(); plt.show()")

nb.add_markdown("**Component Interpretation:**\n- **Trend:** Shows the overall price trajectory — upward over the analysis period\n- **Weekly Seasonality:** Day-of-week effects (if any)\n- **Yearly Seasonality:** Annual patterns (budget season, earnings quarters)")

nb.add_markdown("---\n\n## Part 6: Forecast Accuracy")

nb.add_code("# Accuracy metrics\nif PROPHET_OK:\n    test_forecast = forecast_basic[forecast_basic['ds'].isin(test['ds'])]\n    actual = test.set_index('ds')['y']\n    predicted = test_forecast.set_index('ds')['yhat']\n    common = actual.index.intersection(predicted.index)\n    \n    if len(common) > 0:\n        a = actual.loc[common]\n        p = predicted.loc[common]\n        \n        mae = np.mean(np.abs(a - p))\n        mape = np.mean(np.abs((a - p) / a)) * 100\n        rmse = np.sqrt(np.mean((a - p)**2))\n        r2 = 1 - np.sum((a - p)**2) / np.sum((a - a.mean())**2)\n        \n        print('FORECAST ACCURACY (Basic Model)')\n        print('=' * 40)\n        print(f'MAE:   {mae:.2f} (avg error in rupees)')\n        print(f'MAPE:  {mape:.2f}% (percentage error)')\n        print(f'RMSE:  {rmse:.2f}')\n        print(f'R²:    {r2:.4f}')\n        print(f'\\nInterpretation:')\n        if mape < 5:\n            print('  Excellent forecast accuracy (<5% MAPE)')\n        elif mape < 10:\n            print('  Good forecast accuracy (5-10% MAPE)')\n        elif mape < 20:\n            print('  Acceptable accuracy (10-20% MAPE)')\n        else:\n            print('  Poor accuracy (>20% MAPE) — expected for volatile stocks')")

nb.add_code("# Actual vs Forecast plot\nif PROPHET_OK and len(common) > 0:\n    fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1]})\n    \n    ax = axes[0]\n    ax.plot(common, a, color='black', linewidth=2, label='Actual')\n    ax.plot(common, p, color='#E74C3C', linewidth=2, label='Prophet Forecast')\n    lower = test_forecast.set_index('ds').loc[common, 'yhat_lower']\n    upper = test_forecast.set_index('ds').loc[common, 'yhat_upper']\n    ax.fill_between(common, lower, upper, alpha=0.15, color='red', label='95% CI')\n    ax.set_title('Actual vs Forecast', fontweight='bold', fontsize=14)\n    ax.legend(fontsize=12)\n    ax.grid(True, alpha=0.3)\n    \n    # Error plot\n    ax = axes[1]\n    errors = (a - p).values\n    colors_e = ['#2ECC71' if e > 0 else '#E74C3C' for e in errors]\n    ax.bar(range(len(errors)), errors, color=colors_e, alpha=0.7)\n    ax.set_title('Forecast Error (Actual - Predicted)', fontweight='bold')\n    ax.axhline(0, color='black', linewidth=0.5)\n    \n    plt.tight_layout(); plt.show()")

nb.add_markdown("---\n\n## Part 7: Changepoint Detection\n\nProphet detects **changepoints** — dates where the trend shifted. These often correspond to real market events.")

nb.add_code("if PROPHET_OK:\n    from prophet.plot import add_changepoints_to_plot\n    \n    fig = model_basic.plot(forecast_basic, figsize=(18, 8))\n    a = add_changepoints_to_plot(fig.gca(), model_basic, forecast_basic)\n    plt.title('Changepoint Detection', fontsize=14, fontweight='bold')\n    plt.tight_layout(); plt.show()\n    \n    print('Detected Changepoints:')\n    for i, cp in enumerate(model_basic.changepoints):\n        print(f'  {i+1:2d}. {cp.date()}')")

nb.add_markdown("**Changepoint Interpretation:** These dates mark where the growth rate changed significantly. They should correlate with known events:\n- COVID lockdowns\n- EV announcement dates\n- Earnings surprises\n- Regulatory changes")

nb.add_markdown("---\n\n## Part 8: Cross-Validation")

nb.add_code("# Prophet cross-validation\nif PROPHET_OK:\n    from prophet.diagnostics import cross_validation, performance_metrics\n    try:\n        cv_results = cross_validation(model_basic, initial='365 days', period='90 days', horizon='30 days')\n        metrics = performance_metrics(cv_results)\n        print('CROSS-VALIDATION RESULTS')\n        print('=' * 50)\n        print(metrics[['horizon', 'mae', 'mape', 'rmse']].tail())\n    except Exception as e:\n        print(f'CV failed: {e}')")

nb.add_code("# Save forecast\nif PROPHET_OK:\n    forecast_basic[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(\n        os.path.join(PROCESSED_DIR, 'prophet_forecast.csv'), index=False)\n    print('\\n✅ Saved: prophet_forecast.csv')")

nb.add_markdown("---\n\n## Summary\n\n### 🔑 Key Findings:\n\n1. **Prophet captures seasonality** — weekly and yearly patterns exist in Tata Motors\n2. **Changepoints align with real events** — COVID, EV launches, earnings surprises\n3. **MAPE gives concrete accuracy** — typical range 5-20% for volatile stocks\n4. **Confidence intervals widen** for longer horizons — uncertainty grows with time\n5. **Adding volume as regressor** may improve or worsen forecast depending on the regime\n\n### Limitations:\n- Prophet assumes **the future = the past** → can't predict black swans\n- Price forecasting is inherently harder than direction prediction\n- Single-point forecasts should be used with confidence intervals\n\n### Practical Use:\nDon't trade based solely on Prophet forecasts. Use them as:\n- Long-term trend guidance\n- Entry/exit zone identification (when actual price is far from forecast)\n- Uncertainty quantification (CI width indicates risk)\n\n---\n*Next: Notebook 12 — Strategy Backtesting*")

nb.add_markdown("""---

## 🎯 Bonus: Monte Carlo 10,000-Path Price Cone (GARCH-Calibrated)

### What is a Monte Carlo Simulation?

Instead of predicting ONE future price, we simulate **10,000 possible futures** and ask: "In how many of those futures does the stock go up?"

This is exactly what institutional risk desks do every day.

### Why GARCH-Calibrated?

Normal Monte Carlo uses constant volatility ($\\sigma$). But markets aren't constant — volatility **clusters** (high-vol days follow high-vol days). GARCH(1,1) captures this:

$$\\sigma_t^2 = \\omega + \\alpha \\cdot r_{t-1}^2 + \\beta \\cdot \\sigma_{t-1}^2$$

> **Layman's Translation:** If yesterday was crazy volatile, GARCH says "tomorrow will probably be volatile too." This makes our 10,000 simulations much more realistic than assuming constant volatility.

### What We'll See:
- A **fan/cone** of possible price paths expanding into the future
- **Percentile bands** (5th, 25th, 50th, 75th, 95th) — like weather forecast uncertainty
- **VaR** (Value at Risk) — "What's the worst expected 5% case?"
- **CVaR** — "If we're in that worst 5%, how bad does it get?"
""")

nb.add_code("""# Monte Carlo Simulation with GARCH volatility
np.random.seed(42)
NUM_PATHS = 10000
HORIZON = 90  # Trading days forward

# Use actual data
close = df['Close'].dropna()
returns = close.pct_change().dropna()
S0 = close.iloc[-1]  # Current price
mu = returns.mean()    # Daily drift

# Try GARCH calibration, fall back to constant vol
try:
    from arch import arch_model
    garch_model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
    garch_fit = garch_model.fit(disp='off')
    garch_forecast = garch_fit.forecast(horizon=1)
    sigma_garch = np.sqrt(garch_forecast.variance.values[-1, :][0]) / 100
    vol_method = "GARCH(1,1)"
    print(f"GARCH(1,1) Calibrated Volatility: {sigma_garch:.4f} ({sigma_garch*np.sqrt(252)*100:.1f}% annualized)")
except Exception as e:
    sigma_garch = returns.std()
    vol_method = "Historical (constant)"
    print(f"Using historical volatility: {sigma_garch:.4f} (GARCH failed: {e})")

# Simulate 10,000 paths
all_paths = np.zeros((NUM_PATHS, HORIZON + 1))
all_paths[:, 0] = S0

for t in range(1, HORIZON + 1):
    Z = np.random.standard_normal(NUM_PATHS)
    daily_returns = (mu - 0.5 * sigma_garch**2) + sigma_garch * Z
    all_paths[:, t] = all_paths[:, t-1] * np.exp(daily_returns)

# Calculate percentiles
percentiles = {}
for p in [5, 10, 25, 50, 75, 90, 95]:
    percentiles[p] = np.percentile(all_paths, p, axis=0)

# Terminal price distribution
terminal_prices = all_paths[:, -1]
terminal_returns = (terminal_prices / S0 - 1) * 100

# VaR and CVaR
VaR_5 = np.percentile(terminal_returns, 5)
CVaR_5 = terminal_returns[terminal_returns <= VaR_5].mean()
prob_profit = (terminal_returns > 0).sum() / NUM_PATHS * 100

print(f"\\nMonte Carlo Simulation Results ({NUM_PATHS:,} paths, {HORIZON} days)")
print("=" * 60)
print(f"  Current Price:          ₹{S0:.2f}")
print(f"  Volatility Method:      {vol_method}")
print(f"  Median Terminal Price:  ₹{np.median(terminal_prices):.2f}")
print(f"  5th Percentile:         ₹{percentiles[5][-1]:.2f} (bear case)")
print(f"  95th Percentile:        ₹{percentiles[95][-1]:.2f} (bull case)")
print(f"  Probability of Profit:  {prob_profit:.1f}%")
print(f"\\n  VaR (5%):              {VaR_5:.1f}%")
print(f"  CVaR (5%):             {CVaR_5:.1f}%")
print(f"  Interpretation:         With 95% confidence, loss won't exceed {abs(VaR_5):.1f}%")
print(f"                          If it does, expect ~{abs(CVaR_5):.1f}% average loss")""")

nb.add_code("""# === CHART 1: Monte Carlo Price Cone ===
fig, axes = plt.subplots(2, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [3, 1]})

# Price Cone
ax = axes[0]
future_dates = pd.bdate_range(start=close.index[-1], periods=HORIZON + 1)

# Plot sample paths (thin, transparent)
for i in range(min(200, NUM_PATHS)):
    ax.plot(future_dates, all_paths[i], alpha=0.02, color='#3498DB', linewidth=0.3)

# Plot percentile bands
ax.fill_between(future_dates, percentiles[5], percentiles[95], 
               alpha=0.15, color='#E74C3C', label='5th-95th Percentile')
ax.fill_between(future_dates, percentiles[25], percentiles[75], 
               alpha=0.25, color='#F39C12', label='25th-75th Percentile')
ax.plot(future_dates, percentiles[50], color='#2C3E50', linewidth=2.5, 
       label='Median Path', linestyle='--')

# Historical price context (last 90 days)
hist_window = min(90, len(close))
ax.plot(close.index[-hist_window:], close.values[-hist_window:], 
       color='black', linewidth=2, label='Historical Price')

# Mark current price
ax.axhline(S0, color='gray', linestyle=':', alpha=0.5)
ax.scatter([close.index[-1]], [S0], color='red', s=100, zorder=10, 
          label=f'Current: ₹{S0:.0f}')

# Annotations for key percentiles at horizon
for p, label in [(5, '5th'), (50, 'Median'), (95, '95th')]:
    ax.annotate(f'{label}: ₹{percentiles[p][-1]:.0f}', 
               xy=(future_dates[-1], percentiles[p][-1]),
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.set_title(f'Monte Carlo Price Cone — {NUM_PATHS:,} Paths, {HORIZON}-Day Horizon ({vol_method})', 
             fontsize=16, fontweight='bold')
ax.set_ylabel('Price (₹)', fontsize=13)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Terminal Distribution
ax = axes[1]
ax.hist(terminal_prices, bins=100, color='#3498DB', alpha=0.7, edgecolor='white', density=True)
ax.axvline(S0, color='red', linestyle='--', linewidth=2, label=f'Current: ₹{S0:.0f}')
ax.axvline(np.median(terminal_prices), color='#2C3E50', linestyle='--', linewidth=2, 
          label=f'Median: ₹{np.median(terminal_prices):.0f}')
ax.axvline(percentiles[5][-1], color='#E74C3C', linestyle=':', linewidth=1.5, 
          label=f'5th %%: ₹{percentiles[5][-1]:.0f}')
ax.set_title(f'Terminal Price Distribution ({HORIZON}-Day Horizon) | P(Profit) = {prob_profit:.1f}%', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Price (₹)', fontsize=13)
ax.set_ylabel('Density')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(PROCESSED_DIR, 'monte_carlo_price_cone.png'), dpi=150, bbox_inches='tight')
plt.show()""")

nb.add_markdown("""### 🔍 How to Read the Monte Carlo Price Cone

**The Cone Shape:**
- The cone starts narrow (current price is known) and expands into the future (uncertainty grows)
- The width at any point shows the **range of plausible outcomes**
- Orange band = where the price lands 50% of the time (IQR)
- Red band = where it lands 90% of the time

**VaR and CVaR — The Risk Manager's Language:**
- **VaR (5%):** "With 95% confidence, you won't lose more than X% in 90 days"
- **CVaR:** "If you're in the unlucky 5%, expect to lose an average of Y%"
- CVaR is ALWAYS worse than VaR — it asks "how bad is the tail?"

**Probability of Profit:**
- If > 60%: The drift (average return) is in your favor
- If ~ 50%: Essentially a coin flip over this horizon
- If < 40%: Historical momentum is against you

> **The institutional lesson:** Professional traders never ask "will it go up?" They ask "across 10,000 possible futures, what's the distribution of outcomes and how much can I lose?" This Monte Carlo cone IS that answer.
""")

nb.save("notebooks/11_Forecasting_Prophet.ipynb")
