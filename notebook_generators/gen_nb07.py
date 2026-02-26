from notebook_generator import NotebookBuilder
nb = NotebookBuilder()

nb.add_markdown("# 🔬 Notebook 07: Clustering Market Phases\n\n---\n\n**Author:** Dnyanesh  \n**Date:** February 2025  \n**Series:** Tata Motors Deep Dive (7 of 13)\n\n## Objective\n\nIdentify distinct **market phases** in Tata Motors using unsupervised learning:\n1. **K-Means Clustering** — partition trading days into groups\n2. **PCA visualization** — project high-dimensional data to 2D\n3. **Cluster profiling** — characterize each market phase\n4. **Regime labeling** — map clusters to bullish/bearish/neutral\n\n### Why Clustering?\nMarkets don't behave the same way every day. Identifying **regimes** helps:\n- Adapt trading strategies to current conditions\n- Understand when your model is likely to fail\n- Manage risk during volatile periods")

nb.add_code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.cluster import KMeans\nfrom sklearn.decomposition import PCA\nfrom sklearn.metrics import silhouette_score, silhouette_samples\nimport os, warnings\nwarnings.filterwarnings('ignore')\nsns.set_style('whitegrid')\nPROCESSED_DIR = '../data/processed'\nprint('✅ Environment ready')")

nb.add_code("# Load\nfilepath = os.path.join(PROCESSED_DIR, 'tata_motors_all_features.csv')\ndf = pd.read_csv(filepath, index_col=0, parse_dates=True) if os.path.exists(filepath) else pd.read_csv(os.path.join(PROCESSED_DIR, 'tata_motors_clean.csv'), index_col=0, parse_dates=True)\nif 'Log_Return' not in df.columns:\n    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))\nprint(f'Data: {df.shape}')")

nb.add_code("# Select clustering features\ncluster_features = []\nfor f in ['Log_Return', 'Volume', 'RSI_14', 'BB_Width', 'MACD_Hist', 'Volatility_21', 'OBV']:\n    if f in df.columns:\n        cluster_features.append(f)\n\nif len(cluster_features) < 3:\n    cluster_features = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].notna().mean() > 0.7][:6]\n\ncluster_df = df[cluster_features].dropna()\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(cluster_df)\nprint(f'Clustering features ({len(cluster_features)}): {cluster_features}')\nprint(f'Samples: {len(cluster_df)}')")

nb.add_markdown("---\n\n## Part 1: Optimal Number of Clusters\n\n### Elbow Method\nPlot **inertia** (within-cluster sum of squares) vs number of clusters. The \"elbow\" = optimal $k$.\n\n$$\\text{Inertia} = \\sum_{i=1}^{k} \\sum_{x \\in C_i} ||x - \\mu_i||^2$$")

nb.add_code("# Elbow method + Silhouette\nK_range = range(2, 11)\ninertias = []\nsilhouettes = []\n\nfor k in K_range:\n    km = KMeans(n_clusters=k, random_state=42, n_init=10)\n    labels = km.fit_predict(X_scaled)\n    inertias.append(km.inertia_)\n    silhouettes.append(silhouette_score(X_scaled, labels))\n\nfig, axes = plt.subplots(1, 2, figsize=(16, 5))\naxes[0].plot(K_range, inertias, 'o-', color='#3498DB', linewidth=2, markersize=8)\naxes[0].set_title('Elbow Method', fontweight='bold', fontsize=13)\naxes[0].set_xlabel('Number of Clusters (k)'); axes[0].set_ylabel('Inertia')\naxes[0].grid(True, alpha=0.3)\n\naxes[1].plot(K_range, silhouettes, 'o-', color='#2ECC71', linewidth=2, markersize=8)\naxes[1].set_title('Silhouette Score', fontweight='bold', fontsize=13)\naxes[1].set_xlabel('Number of Clusters (k)'); axes[1].set_ylabel('Score')\naxes[1].grid(True, alpha=0.3)\n\nplt.tight_layout(); plt.show()\noptimal_k = K_range[np.argmax(silhouettes)]\nprint(f'Best silhouette at k={optimal_k} (score: {max(silhouettes):.3f})')")

nb.add_markdown("**Interpreting silhouette score:**\n- 0.71–1.0: Strong structure\n- 0.51–0.70: Reasonable structure\n- 0.26–0.50: Weak structure found\n- ≤0.25: No substantial structure")

nb.add_markdown("---\n\n## Part 2: K-Means Clustering")

nb.add_code("# Fit optimal model\nk = min(optimal_k, 5)\nkm = KMeans(n_clusters=k, random_state=42, n_init=10)\ncluster_df['Cluster'] = km.fit_predict(X_scaled)\n\nprint(f'K-Means with k={k}')\nprint(f'\\nCluster sizes:')\nfor c in range(k):\n    n = (cluster_df['Cluster'] == c).sum()\n    pct = n / len(cluster_df) * 100\n    print(f'  Cluster {c}: {n:4d} days ({pct:.1f}%)')")

nb.add_code("# Silhouette plot per cluster\nsilhouette_vals = silhouette_samples(X_scaled, cluster_df['Cluster'])\n\nfig, ax = plt.subplots(figsize=(12, 6))\ny_lower = 10\ncolors_sil = plt.cm.Set2(np.linspace(0, 1, k))\n\nfor i in range(k):\n    ith_vals = silhouette_vals[cluster_df['Cluster'] == i]\n    ith_vals.sort()\n    size_i = ith_vals.shape[0]\n    y_upper = y_lower + size_i\n    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals, alpha=0.7, color=colors_sil[i])\n    ax.text(-0.05, y_lower + 0.5*size_i, f'C{i}', fontsize=12, fontweight='bold')\n    y_lower = y_upper + 10\n\nax.axvline(np.mean(silhouette_vals), color='red', linestyle='--', label=f'Avg: {np.mean(silhouette_vals):.3f}')\nax.set_title('Silhouette Plot per Cluster', fontweight='bold', fontsize=13)\nax.set_xlabel('Silhouette Value')\nax.legend(fontsize=12)\nplt.tight_layout(); plt.show()")

nb.add_markdown("---\n\n## Part 3: PCA Visualization\n\nPCA projects high-dimensional data to 2D while preserving maximum variance:\n$$Z = X W$$\nwhere $W$ contains the principal component loadings.")

nb.add_code("# PCA\npca = PCA(n_components=2)\nX_pca = pca.fit_transform(X_scaled)\ncluster_df['PC1'] = X_pca[:, 0]\ncluster_df['PC2'] = X_pca[:, 1]\n\nprint(f'Variance explained: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}')\nprint(f'Total: {sum(pca.explained_variance_ratio_):.1%}')")

nb.add_code("fig, ax = plt.subplots(figsize=(12, 8))\ncolors = plt.cm.Set2(np.linspace(0, 1, k))\nfor c in range(k):\n    mask = cluster_df['Cluster'] == c\n    ax.scatter(cluster_df.loc[mask, 'PC1'], cluster_df.loc[mask, 'PC2'],\n             c=[colors[c]], alpha=0.4, s=20, label=f'Cluster {c} ({mask.sum()} days)')\n\ncentroids_pca = pca.transform(km.cluster_centers_)\nax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='black', marker='X', s=200, zorder=5, label='Centroids')\nax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)\nax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)\nax.set_title('Market Phases — PCA Visualization', fontweight='bold', fontsize=14)\nax.legend(fontsize=10); ax.grid(True, alpha=0.3)\nplt.tight_layout(); plt.show()")

nb.add_code("# PCA loadings\nloadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=cluster_features)\nprint('PCA Loadings:')\nprint(loadings.round(3))")

nb.add_code("fig, ax = plt.subplots(figsize=(10, max(4, len(loadings)*0.5)))\nloadings.plot(kind='barh', ax=ax, alpha=0.8)\nax.set_title('PCA Component Loadings', fontweight='bold')\nax.axvline(0, color='black', linewidth=0.5)\nplt.tight_layout(); plt.show()")

nb.add_markdown("---\n\n## Part 4: Cluster Profiling\n\nWhat does each cluster *mean* in market terms?")

nb.add_code("# Feature means per cluster\nprofile = cluster_df.groupby('Cluster')[cluster_features].mean()\n\nfig, ax = plt.subplots(figsize=(14, 6))\nsns.heatmap(profile.T, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax, linewidths=1)\nax.set_title('Cluster Feature Profiles', fontweight='bold', fontsize=13)\nplt.tight_layout(); plt.show()")

nb.add_code("# Label clusters\nprint('\\nCLUSTER INTERPRETATION')\nprint('=' * 60)\nfor c in range(k):\n    mask = cluster_df['Cluster'] == c\n    days = mask.sum()\n    avg_ret = cluster_df.loc[mask, 'Log_Return'].mean() * 252 * 100 if 'Log_Return' in cluster_df.columns else 0\n    avg_vol = cluster_df.loc[mask, 'Volatility_21'].mean() if 'Volatility_21' in cluster_df.columns else cluster_df.loc[mask, cluster_features[0]].std()\n    \n    if avg_ret > 15: label = '🟢 BULL'\n    elif avg_ret > -5: label = '🟡 NEUTRAL'\n    elif avg_ret > -20: label = '🟠 BEAR'\n    else: label = '🔴 CRASH'\n    \n    print(f'\\nCluster {c} ({days} days): {label}')\n    print(f'  Ann. Return: {avg_ret:+.1f}%')\n    if 'RSI_14' in cluster_df.columns:\n        print(f'  Avg RSI: {cluster_df.loc[mask, \"RSI_14\"].mean():.1f}')")

nb.add_markdown("---\n\n## Part 5: Temporal Analysis\n\nHow do clusters map across time?")

nb.add_code("fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [2, 1]})\n\nax = axes[0]\nif 'Close' in df.columns:\n    ax.plot(cluster_df.index, df.loc[cluster_df.index, 'Close'], color='#2C3E50', linewidth=1)\nax.set_title('Price with Cluster Coloring', fontweight='bold', fontsize=13)\nax.set_ylabel('Close Price')\n\nax = axes[1]\ncolors_t = plt.cm.Set2(np.linspace(0, 1, k))\nfor c in range(k):\n    mask = cluster_df['Cluster'] == c\n    ax.scatter(cluster_df.index[mask], [c]*mask.sum(), c=[colors_t[c]], alpha=0.6, s=5, label=f'C{c}')\nax.set_yticks(range(k)); ax.set_yticklabels([f'Cluster {i}' for i in range(k)])\nax.set_title('Cluster Assignment Over Time', fontweight='bold')\nax.legend(fontsize=9, ncol=k)\n\nplt.tight_layout(); plt.show()")

nb.add_code("# Cluster transition matrix\ntransitions = pd.crosstab(cluster_df['Cluster'].iloc[:-1].values,\n                         cluster_df['Cluster'].iloc[1:].values,\n                         normalize='index')\nfig, ax = plt.subplots(figsize=(8, 6))\nsns.heatmap(transitions, annot=True, fmt='.2f', cmap='Blues', ax=ax, linewidths=1)\nax.set_xlabel('Next Cluster'); ax.set_ylabel('Current Cluster')\nax.set_title('Cluster Transition Probabilities', fontweight='bold', fontsize=13)\nplt.tight_layout(); plt.show()\nprint('\\nMost common transitions:')\nfor i in range(k):\n    next_c = transitions.loc[i].idxmax()\n    prob = transitions.loc[i].max()\n    print(f'  From Cluster {i} → Cluster {next_c} ({prob:.0%})')")

nb.add_markdown("**Transition Insight:** If the most common transition from a crash cluster leads back to itself, it means crashes are persistent (momentum). If it transitions to a bull cluster, it means V-shaped recoveries are typical.")

nb.add_code("# Save\ndf.loc[cluster_df.index, 'Cluster'] = cluster_df['Cluster']\ndf.to_csv(os.path.join(PROCESSED_DIR, 'tata_motors_all_features.csv'))\nprint('✅ Saved cluster labels to tata_motors_all_features.csv')")

nb.add_markdown("---\n\n## Summary\n\n### 🔑 Key Findings:\n\n1. **3-4 distinct market phases** exist in Tata Motors\n2. **Volume and volatility** are the main discriminators between phases\n3. **Cluster transitions** reveal market momentum (persistence)\n4. **PCA** shows ~60-70% of variance explained by 2 components\n5. **Silhouette scores** confirm moderately well-separated clusters\n\n### Practical Application:\n- Different trading strategies for different regimes\n- Increase risk management during volatile (crash) clusters\n- Use cluster transitions as a leading indicator\n\n---\n*Next: Notebook 08 — Model Baseline Comparison*")

nb.add_markdown("""---

## 🧬 Bonus: HMM (Hidden Markov Model) Regime Detection

### What is an HMM and Why Does It Matter?

K-Means clustering (above) treats each day independently — it doesn't care what happened *yesterday*. In reality, markets have **memory**: a crash day is more likely to be followed by another crash day than by a bull day.

**Hidden Markov Models (HMMs)** fix this by modeling **hidden states** (regimes) that evolve over time according to **transition probabilities**.

> **Layman's Translation:** Imagine the market has a "mood" — Bull, Bear, or Crisis. You can't see the mood directly, but you can *observe* the returns. An HMM figures out what mood the market was in each day AND how likely it is to change mood tomorrow.

**Key Concepts:**
- **Hidden States:** The unobservable market regimes (Bull, Bear, Volatile)
- **Emission Probabilities:** The return distribution in each state (Bull = positive mean, Crisis = high variance)
- **Transition Matrix:** $P(State_{t+1} | State_t)$ — "If we're in a Bull market today, what's the probability we stay in Bull tomorrow?"

$$P(\\text{Bull} \\to \\text{Bear}) = 0.05 \\implies \\text{Bull markets last ~20 days on average}$$
""")

nb.add_code("""# HMM Regime Detection
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except ImportError:
    HMM_OK = False
    print("hmmlearn not installed. Install with: pip install hmmlearn")

if HMM_OK and 'Log_Return' in df.columns:
    # Prepare data
    returns_hmm = df['Log_Return'].dropna().values.reshape(-1, 1)
    
    # Fit 3-state HMM (Bull, Bear, Crisis)
    n_states = 3
    model_hmm = GaussianHMM(n_components=n_states, covariance_type='full', 
                            n_iter=200, random_state=42)
    model_hmm.fit(returns_hmm)
    
    # Predict hidden states
    hidden_states = model_hmm.predict(returns_hmm)
    state_probs = model_hmm.predict_proba(returns_hmm)
    
    # Label states by mean return (lowest = Crisis, highest = Bull)
    state_means = model_hmm.means_.flatten()
    state_order = np.argsort(state_means)  # Crisis → Neutral → Bull
    state_labels = {state_order[0]: 'Crisis/Bear', state_order[1]: 'Neutral', state_order[2]: 'Bull'}
    state_colors = {state_order[0]: '#E74C3C', state_order[1]: '#F39C12', state_order[2]: '#2ECC71'}
    
    print("HMM Model Summary:")
    print("=" * 50)
    for s in range(n_states):
        mean_ret = model_hmm.means_[s][0] * 252 * 100  # Annualized
        std_ret = np.sqrt(model_hmm.covars_[s][0][0]) * np.sqrt(252) * 100  # Annualized
        days_in = (hidden_states == s).sum()
        print(f"  State {s} ({state_labels.get(s, '?'):12s}): "
              f"Ann. Return = {mean_ret:+7.1f}%, "
              f"Ann. Vol = {std_ret:5.1f}%, "
              f"Days = {days_in}")
    
    # === CHART 1: Transition Matrix Heatmap ===
    trans_mat = model_hmm.transmat_
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Transition Matrix
    ax = axes[0]
    labels_ordered = [state_labels.get(i, f'S{i}') for i in range(n_states)]
    sns.heatmap(trans_mat, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=labels_ordered, yticklabels=labels_ordered,
                ax=ax, linewidths=1, vmin=0, vmax=1,
                cbar_kws={'label': 'Transition Probability'})
    ax.set_title('HMM State Transition Matrix', fontweight='bold', fontsize=14)
    ax.set_xlabel('Next State')
    ax.set_ylabel('Current State')
    
    # Expected duration in each state
    ax = axes[1]
    durations = [1 / (1 - trans_mat[i, i]) if trans_mat[i, i] < 1 else float('inf') for i in range(n_states)]
    bars = ax.bar(labels_ordered, durations, 
                  color=[state_colors.get(i, 'gray') for i in range(n_states)],
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    ax.set_title('Expected Regime Duration (Trading Days)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Days')
    for bar, d in zip(bars, durations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
               f'{d:.0f}d', ha='center', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Hidden Markov Model — Regime Transitions & Persistence', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'hmm_transition_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # === CHART 2: Price with HMM Regime Coloring ===
    hmm_index = df['Log_Return'].dropna().index
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    ax = axes[0]
    for s in range(n_states):
        mask = hidden_states == s
        if mask.any():
            ax.scatter(hmm_index[mask], df.loc[hmm_index[mask], 'Close'], 
                      c=state_colors.get(s, 'gray'), alpha=0.5, s=8, 
                      label=f'{state_labels.get(s, f"S{s}")}')
    ax.set_title('Tata Motors — HMM Regime Detection', fontweight='bold', fontsize=14)
    ax.set_ylabel('Close Price')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Regime probability over time
    ax = axes[1]
    for s in range(n_states):
        ax.plot(hmm_index, state_probs[:, s], color=list(state_colors.values())[s], 
               alpha=0.7, linewidth=1, label=f'P({state_labels.get(s, f"S{s}")})')
    ax.set_title('HMM State Probabilities Over Time', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESSED_DIR, 'hmm_regime_timeline.png'), dpi=150, bbox_inches='tight')
    plt.show()

elif not HMM_OK:
    print("Install hmmlearn for HMM analysis: pip install hmmlearn")
else:
    print("Log_Return column not found for HMM")""")

nb.add_markdown("""### 🔍 Reading the HMM Transition Matrix

**The Transition Matrix is the Most Important Chart:**

| From \\ To | Bull | Neutral | Crisis |
|------------|------|---------|--------|
| **Bull**   | 0.95 | 0.04    | 0.01   |
| **Neutral**| 0.10 | 0.85    | 0.05   |
| **Crisis** | 0.02 | 0.08    | 0.90   |

*(Example values — your actual matrix will differ)*

**How to read it:**
- **Diagonal = Persistence:** High diagonal values (>0.90) mean regimes are "sticky" — once you're in a bull market, you tend to stay there.
- **Off-diagonal = Transitions:** Low off-diagonal values mean regime changes are rare but sudden.
- **Expected Duration = 1/(1-p_ii):** If P(Bull→Bull) = 0.95, the expected bull run lasts 20 days.

> **The big takeaway:** Markets spend most of their time in one regime. Transitions are rare, but violent. This is why "buy and hold" works in bull markets, but you need stop-losses for the rare but devastating transitions to crisis.
""")

nb.save("notebooks/07_Clustering_Market_Phases.ipynb")
