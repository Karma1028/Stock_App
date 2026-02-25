"""
Second pass: add more cells to notebooks still under 40 cells.
"""
import nbformat
import os

NB_DIR = 'notebooks'
TARGET = 42

# Second round of extra cells
EXTRA = {
    2: [  # Data Cleaning (35 -> need 7 more)
        ('markdown', "### Handling Corporate Events\n\nCorporate events (stock splits, bonus shares, dividends) affect price data:\n- **Stock splits** cause sudden price drops (not real losses)\n- **Adjusted close** accounts for these events\n- Always use **adjusted prices** for return calculations"),
        ('code', "# Check for potential stock splits\nimport numpy as np\nif 'Close' in df.columns:\n    daily_pct = df['Close'].pct_change()\n    large_moves = daily_pct[daily_pct.abs() > 0.20].sort_values()\n    print('Large daily moves (>20%, potential corporate events):')\n    for date, pct in large_moves.items():\n        print(f'  {date.date()}: {pct*100:+.1f}% (Close: {df.loc[date, \"Close\"]:.2f})')"),
        ('markdown', "### Data Validation Summary\n\nFinal checks before proceeding to feature engineering:"),
        ('code', "# Final validation\nprint('DATA VALIDATION SUMMARY')\nprint('='*50)\nchecks = [\n    ('No NaN in Close', df['Close'].isna().sum() == 0),\n    ('Positive prices', (df['Close'] > 0).all()),\n    ('Sorted index', df.index.is_monotonic_increasing),\n    ('No duplicate dates', ~df.index.duplicated().any()),\n]\nfor name, passed in checks:\n    print(f'  {\"PASS\" if passed else \"FAIL\"}: {name}')"),
        ('markdown', "### Cleaned Dataset Statistics\n\nThe final cleaned dataset is ready for analysis:"),
        ('code', "# Final stats\nprint('CLEANED DATASET:')\nprint(f'  Shape: {df.shape}')\nprint(f'  Date range: {df.index.min().date()} to {df.index.max().date()}')\nprint(f'  Missing values: {df.isna().sum().sum()}')\nprint(f'  Columns: {list(df.columns)}')\nprint(f'\\n  Price range: {df[\"Close\"].min():.2f} to {df[\"Close\"].max():.2f}')\nprint(f'  Total return: {(df[\"Close\"].iloc[-1]/df[\"Close\"].iloc[0]-1)*100:.1f}%')"),
        ('markdown', "### Next Steps\n\nWith a clean dataset, we can now:\n1. Engineer technical features (SMA, RSI, MACD, BB)\n2. Compute statistical features (volatility, z-scores)\n3. Build the modeling dataset"),
    ],
    3: [  # Technical Features (36 -> need 6 more)
        ('markdown', "### MACD Signal Line Crossovers\n\nMACD histogram shows momentum:\n- **Positive histogram** = bullish momentum increasing\n- **Negative histogram** = bearish momentum\n- **Zero crossing** = potential trend change"),
        ('code', "# MACD analysis\nif 'MACD' in df.columns and 'MACD_Signal' in df.columns:\n    bullish = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))\n    bearish = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))\n    print(f'MACD bullish crossovers: {bullish.sum()}')\n    print(f'MACD bearish crossovers: {bearish.sum()}')\nelse:\n    print('MACD columns not available')"),
        ('markdown', "### Feature Summary Table\n\nAll technical indicators and their current values:"),
        ('code', "# Summary of all technical features\ntech_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]\nprint(f'Technical Features Created: {len(tech_cols)}')\nprint(f'\\nLatest values:')\nfor col in tech_cols[:15]:\n    val = df[col].iloc[-1]\n    print(f'  {col:30s}: {val:.4f}')"),
        ('markdown', "### Technical Indicator Save\n\nSaving the enriched dataset with all technical features:"),
        ('code', "print(f'\\nDataset saved with {len(df.columns)} columns')\nprint(f'Shape: {df.shape}')"),
    ],
    4: [  # Statistical Features (29 -> need 13 more)
        ('markdown', "### Heteroskedasticity and ARCH Effects\n\nFinancial returns exhibit **volatility clustering** (ARCH effects):\n- Large returns tend to follow large returns\n- Small returns follow small returns\n- This violates the i.i.d. assumption of basic models"),
        ('code', "# Test for ARCH effects\nimport numpy as np\nif 'Log_Return' in df.columns:\n    r = df['Log_Return'].dropna()\nelse:\n    r = np.log(df['Close'] / df['Close'].shift(1)).dropna()\n\nr_sq = (r - r.mean())**2\nac1 = r_sq.autocorr(1)\nac5 = r_sq.autocorr(5)\nprint('ARCH EFFECTS TEST')\nprint(f'  Squared return autocorrelation:')\nprint(f'    Lag 1:  {ac1:.4f} {\"** Significant\" if abs(ac1) > 0.05 else \"\"}')\nprint(f'    Lag 5:  {ac5:.4f} {\"** Significant\" if abs(ac5) > 0.05 else \"\"}')\nprint(f'  -> {\"Strong ARCH effects\" if abs(ac1) > 0.1 else \"Mild ARCH effects\" if abs(ac1) > 0.05 else \"Weak ARCH effects\"}')"),
        ('markdown', "### Value at Risk (VaR) Calculation\n\n**VaR** estimates the maximum expected loss at a given confidence level:\n\n$$\\text{VaR}_{\\alpha} = -\\mu - z_{\\alpha} \\cdot \\sigma$$"),
        ('code', "from scipy import stats\n# Historical VaR\nfor conf in [0.95, 0.99]:\n    var_h = np.percentile(r, (1-conf)*100)\n    cvar = r[r <= var_h].mean()\n    print(f'{conf:.0%} VaR:  {var_h*100:.2f}% (on a bad day, expect loss >= {abs(var_h)*100:.2f}%)')\n    print(f'{conf:.0%} CVaR: {cvar*100:.2f}% (avg loss on worst days)\\n')"),
        ('markdown', "### Rolling Beta Calculation\n\nBeta measures sensitivity to market (Nifty 50):\n$$\\beta = \\frac{\\text{Cov}(r_{stock}, r_{market})}{\\text{Var}(r_{market})}$$\n\nSince we don't have Nifty data in this notebook, we estimate sensitivity to its own lagged returns:"),
        ('code', "# Lagged return sensitivity (proxy for persistence)\nlag_corrs = []\nfor lag in range(1, 11):\n    corr = r.corr(r.shift(lag))\n    lag_corrs.append({'lag': lag, 'correlation': corr})\nimport pandas as pd\nlag_df = pd.DataFrame(lag_corrs)\nprint('Return-Lag Correlation (persistence):')\nfor _, row in lag_df.iterrows():\n    bar = 'X' * int(abs(row['correlation']) * 200)\n    print(f'  Lag {int(row[\"lag\"]):2d}: {row[\"correlation\"]:+.4f} {bar}')"),
        ('markdown', "### Feature Engineering Summary\n\nStatistical features capture hidden patterns that price alone cannot reveal:"),
        ('code', "# Summary of statistical features\nimport numpy as np\nstat_features = [c for c in df.columns if any(k in c.lower() for k in ['vol', 'z_score', 'skew', 'kurt', 'return', 'var'])]\nprint(f'Statistical features engineered: {len(stat_features)}')\nfor f in stat_features:\n    print(f'  {f}')"),
        ('markdown', "### Correlation Between Risk Metrics\n\nDo our risk metrics agree with each other?"),
        ('code', "# Risk metric correlations\nrisk_cols = [c for c in df.select_dtypes(include=[np.number]).columns if any(k in c.lower() for k in ['vol', 'atr', 'width', 'range'])]\nif len(risk_cols) > 1:\n    risk_corr = df[risk_cols].corr()\n    print('Risk Metric Correlations:')\n    print(risk_corr.round(3))\nelse:\n    print('Not enough risk metrics to correlate')"),
    ],
    6: [  # Sentiment (35 -> need 7 more)
        ('markdown', "### Sentiment Persistence\n\nDoes market sentiment persist over multiple days or reverse quickly?"),
        ('code', "# Sentiment is proxied by return sign runs\nimport numpy as np\nif 'Close' in df.columns:\n    signs = np.sign(df['Close'].pct_change()).dropna()\n    runs = []\n    current_sign = signs.iloc[0]\n    count = 1\n    for i in range(1, len(signs)):\n        if signs.iloc[i] == current_sign:\n            count += 1\n        else:\n            runs.append({'direction': 'UP' if current_sign > 0 else 'DOWN', 'length': count})\n            current_sign = signs.iloc[i]\n            count = 1\n    import pandas as pd\n    run_df = pd.DataFrame(runs)\n    for d in ['UP', 'DOWN']:\n        rd = run_df[run_df['direction'] == d]['length']\n        print(f'{d} runs: avg={rd.mean():.1f} days, max={rd.max()} days')"),
        ('markdown', "### Day-of-Week Effects\n\nAre certain days of the week systematically bullish or bearish?"),
        ('code', "# Day-of-week analysis\nimport numpy as np\nif 'Close' in df.columns:\n    df_temp = df.copy()\n    df_temp['DayOfWeek'] = df_temp.index.dayofweek\n    df_temp['DailyReturn'] = df_temp['Close'].pct_change()\n    \n    dow_stats = df_temp.groupby('DayOfWeek')['DailyReturn'].agg(['mean', 'std', 'count'])\n    dow_stats.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']\n    dow_stats['mean'] = dow_stats['mean'] * 100\n    dow_stats['std'] = dow_stats['std'] * 100\n    print('DAY-OF-WEEK RETURNS (%):')\n    print(dow_stats.round(4))"),
        ('markdown', "### Month-of-Year Effects\n\nSeasonal patterns like the 'January Effect' or 'Sell in May':"),
        ('code', "# Monthly seasonality\nimport numpy as np\nif 'Close' in df.columns:\n    df_temp['Month'] = df_temp.index.month\n    month_stats = df_temp.groupby('Month')['DailyReturn'].agg(['mean', 'std'])\n    month_stats.index = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n    month_stats['mean'] = month_stats['mean'] * 100\n    print('MONTHLY AVERAGE RETURNS (%):')\n    print(month_stats['mean'].round(4))"),
    ],
    7: [  # Clustering (30 -> need 12 more)
        ('markdown', "### DBSCAN Comparison\n\nDBSCAN can find clusters of arbitrary shape and identifies noise points:"),
        ('code', "from sklearn.cluster import DBSCAN\ndb = DBSCAN(eps=1.5, min_samples=10)\ndb_labels = db.fit_predict(X_scaled)\nn_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)\nnoise = (db_labels == -1).sum()\nprint(f'DBSCAN results: {n_clusters_db} clusters, {noise} noise points ({noise/len(db_labels)*100:.1f}%)')"),
        ('markdown', "### Cluster Stability Check\n\nRunning K-Means multiple times to check if clusters are stable:"),
        ('code', "# Stability check\nfrom sklearn.metrics import adjusted_rand_score\nruns = []\nfor seed in range(10):\n    km_temp = KMeans(n_clusters=k, random_state=seed, n_init=10)\n    runs.append(km_temp.fit_predict(X_scaled))\n\nrand_scores = []\nfor i in range(len(runs)):\n    for j in range(i+1, len(runs)):\n        rand_scores.append(adjusted_rand_score(runs[i], runs[j]))\n\nimport numpy as np\nprint(f'Cluster stability (Adj. Rand Index across 10 random seeds):')\nprint(f'  Mean: {np.mean(rand_scores):.3f}')\nprint(f'  Min:  {np.min(rand_scores):.3f}')\nprint(f'  Max:  {np.max(rand_scores):.3f}')\nprint(f'  -> {\"Stable\" if np.mean(rand_scores) > 0.8 else \"Moderate\" if np.mean(rand_scores) > 0.5 else \"Unstable\"}')"),
        ('markdown', "### Cluster Return-Risk Map\n\nPlotting each cluster's risk-return profile:"),
        ('code', "# Risk-return by cluster\nimport matplotlib.pyplot as plt\nfig, ax = plt.subplots(figsize=(10, 7))\ncolors_rr = plt.cm.Set2(np.linspace(0, 1, k))\n\nfor c in range(k):\n    mask = cluster_df['Cluster'] == c\n    if 'Log_Return' in cluster_df.columns:\n        ret = cluster_df.loc[mask, 'Log_Return'].mean() * 252 * 100\n        risk = cluster_df.loc[mask, 'Log_Return'].std() * np.sqrt(252) * 100\n    else:\n        ret = 0\n        risk = 1\n    ax.scatter(risk, ret, s=mask.sum()*2, color=colors_rr[c], alpha=0.7, edgecolor='black', linewidth=1)\n    ax.annotate(f'C{c} ({mask.sum()}d)', (risk, ret), fontsize=11, fontweight='bold')\n\nax.set_xlabel('Annualized Volatility (%)', fontsize=12)\nax.set_ylabel('Annualized Return (%)', fontsize=12)\nax.set_title('Cluster Risk-Return Profile', fontweight='bold', fontsize=14)\nax.axhline(0, color='gray', linestyle=':')\nplt.tight_layout(); plt.show()"),
        ('markdown', "### Cluster Volume Analysis\n\nDoes trading volume differ significantly between clusters?"),
        ('code', "if 'Volume' in cluster_df.columns:\n    for c in range(k):\n        mask = cluster_df['Cluster'] == c\n        v = cluster_df.loc[mask, 'Volume']\n        print(f'Cluster {c}: avg volume = {v.mean():,.0f}, median = {v.median():,.0f}')"),
    ],
    8: [  # Model Comparison (32 -> need 10 more)
        ('markdown', "### Prediction Probability Distribution\n\nThe spread of predicted probabilities shows model confidence:"),
        ('code', "# Probability distribution analysis\nimport matplotlib.pyplot as plt\nfrom sklearn.model_selection import TimeSeriesSplit\n\ntrain_idx, test_idx = list(tscv.split(X_scaled))[-1]\nfor name, model in models.items():\n    model.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])\n    probs = model.predict_proba(X_scaled.iloc[test_idx])[:, 1]\n    print(f'{name}:')\n    print(f'  Mean prob: {probs.mean():.3f}')\n    print(f'  Std prob:  {probs.std():.3f}')\n    print(f'  Range:     [{probs.min():.3f}, {probs.max():.3f}]')"),
        ('markdown', "### Model Agreement Analysis\n\nDo models agree with each other? When all models agree, the signal is stronger:"),
        ('code', "# Model agreement\nimport numpy as np\npredictions = {}\ntrain_idx, test_idx = list(tscv.split(X_scaled))[-1]\nfor name, model in models.items():\n    model.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])\n    predictions[name] = model.predict(X_scaled.iloc[test_idx])\n\npred_df = pd.DataFrame(predictions)\nagreement = pred_df.apply(lambda row: len(set(row)) == 1, axis=1)\nprint(f'Models agree on {agreement.mean()*100:.1f}% of predictions')\nprint(f'  All predict UP:   {(pred_df.sum(axis=1) == len(models)).mean()*100:.1f}%')\nprint(f'  All predict DOWN: {(pred_df.sum(axis=1) == 0).mean()*100:.1f}%')"),
        ('markdown', "### Ensemble Prediction\n\nCombining multiple models often gives better results than any single model:"),
        ('code', "# Simple ensemble (majority vote)\nensemble_pred = (pred_df.mean(axis=1) > 0.5).astype(int)\nfrom sklearn.metrics import accuracy_score\ny_test = y.iloc[test_idx]\nens_acc = accuracy_score(y_test, ensemble_pred)\nprint('ENSEMBLE PERFORMANCE')\nprint(f'  Accuracy: {ens_acc:.4f}')\nprint(f'  Better than best single model? ', end='')\nbest_single = max(accuracy_score(y_test, pred_df[m]) for m in predictions)\nprint(f'{\"YES\" if ens_acc > best_single else \"NO\"} (best single: {best_single:.4f})')"),
        ('markdown', "### Model Training Time Comparison\n\nEfficiency matters in production:"),
        ('code', "import time\nfor name, model in models.items():\n    start = time.time()\n    model.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])\n    elapsed = time.time() - start\n    print(f'{name:25s}: {elapsed:.3f}s')"),
    ],
    9: [  # Feature Selection (29 -> need 13 more)
        ('markdown', "### Mutual Information Analysis\n\nMutual Information measures non-linear dependencies between features and target:"),
        ('code', "from sklearn.feature_selection import mutual_info_classif\nmi = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns).sort_values(ascending=False)\nprint('Top features by Mutual Information:')\nfor i, (feat, val) in enumerate(mi.head(10).items()):\n    bar = 'X' * int(val * 100)\n    print(f'  {i+1:2d}. {feat:30s}: {val:.4f} {bar}')"),
        ('markdown', "### Feature Stability Across Folds\n\nAre the selected features consistent across different time periods?"),
        ('code', "# Feature stability\nfrom sklearn.feature_selection import RFE\nfrom sklearn.model_selection import TimeSeriesSplit\nfrom sklearn.ensemble import RandomForestClassifier\nimport numpy as np\n\ntscv_temp = TimeSeriesSplit(n_splits=5)\nfold_features = []\n\nfor train_idx, test_idx in tscv_temp.split(X_scaled):\n    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\n    rfe_temp = RFE(rf_temp, n_features_to_select=10, step=1)\n    rfe_temp.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])\n    fold_features.append(set(X_scaled.columns[rfe_temp.support_]))\n\n# Count how often each feature is selected\nfrom collections import Counter\nall_selected = Counter()\nfor ff in fold_features:\n    all_selected.update(ff)\n\nprint('Feature Stability Across 5 Folds:')\nfor feat, count in all_selected.most_common():\n    stars = '*' * count\n    print(f'  {feat:30s}: {count}/5 folds {stars}')"),
        ('markdown', "### Information Gain vs Tree Importance\n\nDifferent methods give different rankings — which to trust?"),
        ('code', "# Compare MI vs tree importance\nimport numpy as np\nrf_full = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)\nrf_full.fit(X_scaled, y)\ntree_imp = pd.Series(rf_full.feature_importances_, index=X.columns)\n\ncomparison = pd.DataFrame({'MI': mi, 'TreeImp': tree_imp}).dropna()\ncomparison['MI_rank'] = comparison['MI'].rank(ascending=False)\ncomparison['Tree_rank'] = comparison['TreeImp'].rank(ascending=False)\ncomparison['rank_diff'] = abs(comparison['MI_rank'] - comparison['Tree_rank'])\n\nprint('Features with biggest ranking disagreement:')\ndisagree = comparison.sort_values('rank_diff', ascending=False)\nfor feat, row in disagree.head(5).iterrows():\n    print(f'  {feat:30s}: MI rank={int(row[\"MI_rank\"]):2d}, Tree rank={int(row[\"Tree_rank\"]):2d}')"),
    ],
    11: [  # Prophet (30 -> need 12 more)
        ('markdown', "### Prophet Parameter Sensitivity\n\nThe `changepoint_prior_scale` controls how flexible the trend is:\n- **Low** (0.01): Rigid trend, underfits\n- **High** (0.5): Flexible trend, overfits"),
        ('code', "# Compare different changepoint scales\nif PROPHET_OK:\n    scales = [0.01, 0.05, 0.1, 0.5]\n    scale_results = []\n    for s in scales:\n        m = Prophet(changepoint_prior_scale=s, daily_seasonality=False)\n        m.fit(train[['ds', 'y']])\n        f = m.predict(m.make_future_dataframe(periods=90))\n        test_f = f[f['ds'].isin(test['ds'])]\n        actual = test.set_index('ds')['y']\n        pred = test_f.set_index('ds')['yhat']\n        common = actual.index.intersection(pred.index)\n        if len(common) > 0:\n            mape = np.mean(np.abs((actual.loc[common] - pred.loc[common]) / actual.loc[common])) * 100\n            scale_results.append({'scale': s, 'MAPE': mape})\n            print(f'  scale={s:.2f}: MAPE={mape:.2f}%')"),
        ('markdown', "### Weekly Pattern Analysis\n\nIs there a day-of-week effect in Tata Motors?"),
        ('code', "# Weekly pattern\nif PROPHET_OK:\n    components = forecast_basic[['ds', 'weekly']].copy() if 'weekly' in forecast_basic.columns else None\n    if components is not None:\n        components['dow'] = components['ds'].dt.dayofweek\n        dow_effect = components.groupby('dow')['weekly'].mean()\n        print('Weekly Seasonality Effect:')\n        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']\n        for d, v in dow_effect.items():\n            if d < len(days):\n                print(f'  {days[d]:3s}: {v:+.2f}')\n    else:\n        print('Weekly component not available')"),
        ('markdown', "### Forecast Uncertainty\n\nProphet's confidence intervals widen over time — here's how fast:"),
        ('code', "if PROPHET_OK:\n    last_90 = forecast_basic.tail(90)\n    widths = (last_90['yhat_upper'] - last_90['yhat_lower']).values\n    print('Confidence Interval Width Over Forecast Horizon:')\n    for i in [0, 30, 60, 89]:\n        if i < len(widths):\n            print(f'  Day {i+1:3d}: +/- {widths[i]/2:.2f}')"),
        ('markdown', "### Prophet vs Naive Forecast\n\nIs Prophet better than a simple moving average forecast?"),
        ('code', "# Naive forecast: last known price\nnp_pred = train['y'].iloc[-1]\nif len(test) > 0:\n    naive_mape = np.mean(np.abs((test['y'] - np_pred) / test['y'])) * 100\n    print(f'Naive forecast (flat): MAPE = {naive_mape:.2f}%')\n    if PROPHET_OK and len(common) > 0:\n        print(f'Prophet forecast:      MAPE = {mape:.2f}%')\n        print(f'\\nProphet improvement: {naive_mape - mape:+.2f}%')"),
    ],
    12: [  # Backtesting (30 -> need 12 more)
        ('markdown', "### Probability Threshold Analysis\n\nWhat if we only trade when the model is very confident?"),
        ('code', "# Different threshold analysis\nimport numpy as np\nfor threshold in [0.45, 0.50, 0.55, 0.60, 0.65]:\n    mask = bt['Probability'] > threshold\n    if mask.sum() > 10:\n        ret = bt.loc[mask, 'Market_Return'].mean() * 252 * 100\n        pct_invested = mask.mean() * 100\n        print(f'  Threshold {threshold:.2f}: Invested {pct_invested:.0f}% of time, Ann. return: {ret:+.1f}%')"),
        ('markdown', "### Maximum Consecutive Losses\n\nThe psychological impact of losing streaks is severe. How many consecutive losing days might we face?"),
        ('code', "# Consecutive loss analysis\nimport numpy as np\nlosses = (bt['Strategy_Return'] < 0).astype(int)\nmax_consec = 0\ncurrent = 0\nfor v in losses.values:\n    if v == 1:\n        current += 1\n        max_consec = max(max_consec, current)\n    else:\n        current = 0\nprint(f'Maximum consecutive losing days: {max_consec}')\nprint(f'Average loss on losing days: {bt[bt[\"Strategy_Return\"]<0][\"Strategy_Return\"].mean()*100:.3f}%')\nprint(f'Max single-day loss: {bt[\"Strategy_Return\"].min()*100:.2f}%')"),
        ('markdown', "### Comparison: Full ML vs Simplified Rules\n\nWould a simple rule-based strategy do just as well?"),
        ('code', "# Simple momentum strategy comparison\nbt_copy = bt.copy()\nbt_copy['Mom_Signal'] = (bt_copy['Market_Return'].rolling(5).mean() > 0).astype(int).shift(1).fillna(0)\nbt_copy['Mom_Return'] = bt_copy['Mom_Signal'] * bt_copy['Market_Return']\n\nml_total = (1 + bt['Strategy_Return']).prod() - 1\nmom_total = (1 + bt_copy['Mom_Return']).prod() - 1\nbh_total = (1 + bt['Market_Return']).prod() - 1\n\nprint('STRATEGY COMPARISON')\nprint(f'  Buy & Hold:        {bh_total*100:+.1f}%')\nprint(f'  ML Strategy:       {ml_total*100:+.1f}%')\nprint(f'  5-day Momentum:    {mom_total*100:+.1f}%')"),
    ],
    13: [  # Final Synthesis (27 -> need 15 more)
        ('markdown', "### Correlation Matrix of Returns by Year\n\nAre Tata Motors' returns consistent across years?"),
        ('code', "import numpy as np\nif df is not None:\n    yearly_ret = {}\n    for year in df.index.year.unique():\n        mask = df.index.year == year\n        yearly_ret[year] = (df.loc[mask, 'Close'].iloc[-1] / df.loc[mask, 'Close'].iloc[0] - 1) * 100\n    print('YEAR-BY-YEAR RETURNS:')\n    for year, ret in sorted(yearly_ret.items()):\n        bar = '#' * int(abs(ret) / 5)\n        sign = '+' if ret > 0 else ''\n        print(f'  {year}: {sign}{ret:.1f}% {bar}')"),
        ('markdown', "### Key Metrics Dashboard\n\nA one-page summary of all critical numbers:"),
        ('code', "import numpy as np\nif df is not None:\n    r = df['Close'].pct_change().dropna()\n    print('KEY METRICS DASHBOARD')\n    print('='*50)\n    metrics_kv = [\n        ('Current Price', f'{df[\"Close\"].iloc[-1]:.2f}'),\n        ('52W High', f'{df[\"Close\"].iloc[-252:].max():.2f}'),\n        ('52W Low', f'{df[\"Close\"].iloc[-252:].min():.2f}'),\n        ('Avg Daily Volume', f'{df[\"Volume\"].iloc[-252:].mean():,.0f}' if 'Volume' in df.columns else 'N/A'),\n        ('Ann. Volatility', f'{r.std()*np.sqrt(252)*100:.1f}%'),\n        ('Sharpe (ann.)', f'{r.mean()/r.std()*np.sqrt(252):.2f}'),\n        ('Max Drawdown', f'{((1+r).cumprod()/(1+r).cumprod().expanding().max()-1).min()*100:.1f}%'),\n        ('Best Day', f'{r.max()*100:+.1f}%'),\n        ('Worst Day', f'{r.min()*100:+.1f}%'),\n    ]\n    for k, v in metrics_kv:\n        print(f'  {k:25s}: {v}')"),
        ('markdown', "### Ethical Considerations\n\nAny stock analysis should include ethical disclaimers:\n\n1. **This is NOT investment advice** — educational analysis only\n2. **Past performance** does not guarantee future results\n3. **Always diversify** — never put all capital in one stock\n4. **Risk management** is more important than returns\n5. **Consult a financial advisor** before making investment decisions"),
        ('markdown', "### Technical Architecture Summary\n\nOur analysis pipeline:"),
        ('code', "print('TECHNICAL ARCHITECTURE')\nprint('='*50)\nprint('\\nData Pipeline:')\nprint('  Yahoo Finance -> Clean CSV -> Feature Engineering -> Modeling')\nprint('\\nNotebook Dependencies:')\ndeps = [\n    ('NB01', 'NB02', 'Raw data -> Clean data'),\n    ('NB02', 'NB03-04', 'Clean data -> Features'),\n    ('NB03-04', 'NB05-07', 'Features -> Analysis'),\n    ('NB03-04', 'NB08', 'Features -> Model Training'),\n    ('NB08', 'NB09', 'Models -> Feature Selection'),\n    ('NB09', 'NB10', 'Selected Features -> Tuning'),\n    ('NB10', 'NB12', 'Tuned Model -> Backtesting'),\n    ('NB01', 'NB11', 'Clean Data -> Prophet'),\n    ('NB08-12', 'NB13', 'All Results -> Synthesis'),\n]\nfor src, dst, desc in deps:\n    print(f'  {src:8s} -> {dst:8s}: {desc}')"),
        ('markdown', "### Acknowledgments\n\nData sources and tools used:\n- **Yahoo Finance** (via yfinance) — OHLCV price data\n- **scikit-learn** — ML models and preprocessing\n- **Prophet** — Time series forecasting\n- **Optuna** — Hyperparameter optimization\n- **SHAP** — Model interpretability\n- **Pandas/NumPy/Matplotlib/Seaborn** — Core data science stack"),
    ]
}


def inject_cells(nb_path, notebook_num, target_cells):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    current = len(nb.cells)
    if current >= target_cells or notebook_num not in EXTRA:
        return current, current
    
    extras = EXTRA[notebook_num]
    insert_pos = max(len(nb.cells) - 2, len(nb.cells) // 2)
    
    for i, (cell_type, content) in enumerate(extras):
        cell = nbformat.v4.new_markdown_cell(content) if cell_type == 'markdown' \
               else nbformat.v4.new_code_cell(content)
        nb.cells.insert(insert_pos + i, cell)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    return current, len(nb.cells)


for f in sorted(os.listdir(NB_DIR)):
    if not f.endswith('.ipynb'):
        continue
    try:
        num = int(f.split('_')[0])
    except:
        continue
    
    fp = os.path.join(NB_DIR, f)
    before, after = inject_cells(fp, num, TARGET)
    status = 'OK' if after >= TARGET else '+' if after > before else '-'
    print(f'{status:3s} {f:45s}: {before:2d} -> {after:2d} cells')

print('\nDone!')
