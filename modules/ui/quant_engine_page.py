
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from modules.data.manager import StockDataManager

try:
    from modules.ml.quant_strategy import QuantEngine
except ImportError:
    QuantEngine = None


# ────────────────────────────────────────────────────────────────
# STRATEGY EXPLANATIONS
# ────────────────────────────────────────────────────────────────
STRATEGY_INFO = {
    "Momentum": {
        "icon": "🚀",
        "tagline": "Ride the trend — buy strength, sell weakness.",
        "how": (
            "**How it works:** The strategy combines two signals:\n"
            "1. **RSI (Relative Strength Index)** — Measures price momentum on a 0-100 scale. "
            "When RSI crosses *above* your threshold, it indicates bullish momentum.\n"
            "2. **SMA (Simple Moving Average)** — Price trading *above* the SMA confirms an uptrend.\n\n"
            "**BUY** when RSI > threshold AND price > SMA.\n"
            "**SELL** when RSI < threshold OR price < SMA."
        ),
        "when_works": "Trending markets with strong directional moves. Works best for large-cap stocks with institutional flows.",
        "when_fails": "Sideways/choppy markets — RSI triggers false breakouts, SMA whipsaws.",
        "risk": "Medium — follows trends but can be late to exit.",
    },
    "Mean Reversion": {
        "icon": "🔄",
        "tagline": "Buy the dip — prices revert to the mean.",
        "how": (
            "**How it works:** Uses Bollinger Bands to detect extremes:\n"
            "1. **Lower Band** = SMA − (StdDev × multiplier). Price below → *oversold*.\n"
            "2. **Upper Band** = SMA + (StdDev × multiplier). Price above → *overbought*.\n\n"
            "**BUY** when price drops below the Lower Band (oversold bounce expected).\n"
            "**SELL** when price rises above the Upper Band or reverts to mean."
        ),
        "when_works": "Range-bound markets with stable volatility. Works well for banking/FMCG stocks.",
        "when_fails": "Strong trending markets — price can stay 'oversold' for weeks during a crash.",
        "risk": "High — catching falling knives if the trend is genuinely breaking down.",
    },
    "Volatility Breakout": {
        "icon": "⚡",
        "tagline": "Bet on explosive moves after quiet periods.",
        "how": (
            "**How it works:** Uses ATR (Average True Range) to detect volatility squeezes:\n"
            "1. **ATR** measures daily price range. Low ATR = quiet market (compression).\n"
            "2. When price breaks out of the ATR envelope (close > SMA + 1.5×ATR), a big move is starting.\n\n"
            "**BUY** on upside breakout. **SELL** when ATR contracts again (move exhausted)."
        ),
        "when_works": "Before earnings, sector rotations, or macro events. Works for mid-cap stocks with event catalysts.",
        "when_fails": "False breakouts in low-liquidity stocks. Also fails in persistent low-vol regimes.",
        "risk": "Medium-High — breakouts can reverse quickly (bull traps).",
    },
    "AI Composite": {
        "icon": "🤖",
        "tagline": "All signals combined — ML-weighted ensemble.",
        "how": (
            "**How it works:** Combines Momentum + Mean Reversion + Volatility signals "
            "and weights them by ML model confidence. Uses the pre-trained LightGBM model's "
            "predicted return direction to filter trades.\n\n"
            "**BUY** when majority of signals agree AND ML probability > 55%.\n"
            "**SELL** when signals conflict OR ML flips bearish."
        ),
        "when_works": "Most market conditions — adapts by weighting the best-performing sub-strategy.",
        "when_fails": "When all sub-strategies fail simultaneously (e.g., a black swan).",
        "risk": "Low-Medium — diversified across signal types.",
    },
}


# ────────────────────────────────────────────────────────────────
# HELPER: Per-stock metrics
# ────────────────────────────────────────────────────────────────
def _per_stock_metrics(equity_series, capital):
    """Compute risk-return metrics for a single equity curve."""
    if equity_series.empty or len(equity_series) < 10:
        return {}
    start = equity_series.iloc[0]
    end = equity_series.iloc[-1]
    total_ret = (end - start) / start * 100
    daily = equity_series.pct_change().dropna()
    sharpe = np.sqrt(252) * (daily.mean() / daily.std()) if daily.std() > 0 else 0
    rolling_max = equity_series.cummax()
    dd = ((equity_series - rolling_max) / rolling_max)
    max_dd = dd.min() * 100
    # Sortino (downside deviation)
    downside = daily[daily < 0]
    sortino = np.sqrt(252) * (daily.mean() / downside.std()) if len(downside) > 0 and downside.std() > 0 else 0
    return {
        "Total Return": f"{total_ret:+.1f}%",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "Max Drawdown": f"{max_dd:.1f}%",
        "Final Value": f"₹{end:,.0f}",
    }


# ────────────────────────────────────────────────────────────────
# MAIN PAGE
# ────────────────────────────────────────────────────────────────
def render_quant_engine():
    st.title("🧠 Quant Engine & Backtester")
    st.caption("Institutional-grade strategy backtesting on Indian equities")
    st.markdown("---")

    if QuantEngine is None:
        st.error("Quant Engine module not found. Ensure `modules/ml/quant_strategy.py` exists.")
        return

    # ──────── SIDEBAR CONFIG ────────
    col_cfg, col_explain = st.columns([1, 2])

    dm = StockDataManager()

    with col_cfg:
        st.markdown("### ⚙️ Strategy Configuration")

        strategy_type = st.selectbox(
            "Strategy Type",
            list(STRATEGY_INFO.keys()),
            format_func=lambda x: f"{STRATEGY_INFO[x]['icon']} {x}",
        )

        st.markdown("#### Stock Universe")
        universe_mode = st.radio("Select stocks", ["Top 5 Liquid", "Custom"], horizontal=True)
        if universe_mode == "Custom":
            # Pre-fill with globally selected ticker if available
            default_tickers = st.session_state.get('global_ticker', 'RELIANCE.NS')
            custom_tickers = st.text_input(
                "Tickers (comma-separated)",
                value=default_tickers,
            )
            tickers = [t.strip() for t in custom_tickers.split(",") if t.strip()]
        else:
            tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

        lookback = st.slider("Lookback Period (Days)", 20, 365, 100)

        if strategy_type == "Momentum":
            param1 = st.slider("RSI Buy Threshold", 30, 70, 50, help="Buy when RSI rises above this level")
            param2 = st.slider("SMA Window (days)", 5, 50, 14, help="Trend confirmation window")
        elif strategy_type == "Mean Reversion":
            param1 = st.slider("Bollinger Band Width (σ)", 1.0, 3.0, 2.0, help="Higher = wider bands = fewer signals")
            param2 = st.slider("Moving Average Window", 10, 100, 20)
        else:
            param1 = 0
            param2 = 0

        initial_capital = st.number_input("Initial Capital (₹)", value=100000, step=10000)

        run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)
        if run_btn:
            st.session_state.run_backtest = True
            st.session_state.strategy_params = {
                "type": strategy_type,
                "lookback": lookback,
                "p1": param1,
                "p2": param2,
                "capital": initial_capital,
                "tickers": tickers,
            }

    # ──────── STRATEGY EXPLANATION ────────
    with col_explain:
        info = STRATEGY_INFO[strategy_type]
        st.markdown(f"### {info['icon']} {strategy_type} Strategy")
        st.markdown(f"*{info['tagline']}*")
        st.markdown("---")
        st.markdown(info["how"])
        c1, c2 = st.columns(2)
        with c1:
            st.success(f"**✅ Works when:** {info['when_works']}")
        with c2:
            st.warning(f"**⚠️ Fails when:** {info['when_fails']}")
        st.info(f"**Risk Profile:** {info['risk']}")

    # ──────── RESULTS ────────
    if not st.session_state.get("run_backtest"):
        st.markdown("---")
        st.info("👆 Configure your strategy above and click **Run Backtest** to see results.")
        return

    st.markdown("---")
    params = st.session_state.strategy_params
    sel_tickers = params.get("tickers", ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"])

    with st.spinner(f"Running **{params['type']}** strategy on {len(sel_tickers)} stocks..."):
        qe = QuantEngine()
        # We need per-stock data, so run the backtest manually
        all_trades = []
        per_stock_equity = {}
        per_stock_bnh = {}  # buy & hold
        capital_per_stock = params["capital"] / len(sel_tickers)

        for ticker in sel_tickers:
            try:
                df = dm.get_historical_data(ticker, period="2y")
                if df.empty or len(df) < 50:
                    continue
                signals = qe._apply_strategy(df, params["type"], params)
                daily_rets, trade_log, equity_curve = qe._simulate_trading(df, signals, capital_per_stock)

                per_stock_equity[ticker] = equity_curve
                # Buy & hold baseline
                bnh = (df["Close"] / df["Close"].iloc[0]) * capital_per_stock
                per_stock_bnh[ticker] = bnh

                for t in trade_log:
                    all_trades.append({
                        "Date": t["date"],
                        "Symbol": ticker.replace(".NS", ""),
                        "Action": t["action"],
                        "Price": t["price"],
                        "PnL": t["pnl"],
                    })
            except Exception as e:
                st.warning(f"Skipped {ticker}: {e}")

    if not per_stock_equity:
        st.error("No stocks had sufficient data. Try different tickers.")
        return

    # ──────────────────────────────────
    # SECTION 1: AGGREGATE PORTFOLIO
    # ──────────────────────────────────
    st.markdown("## 📊 Portfolio Performance")

    port_equity = pd.DataFrame(per_stock_equity).dropna()
    port_bnh = pd.DataFrame(per_stock_bnh).dropna()
    total_equity = port_equity.sum(axis=1)
    total_bnh = port_bnh.sum(axis=1)

    # Top metrics
    port_metrics = _per_stock_metrics(total_equity, params["capital"])
    bnh_metrics = _per_stock_metrics(total_bnh, params["capital"])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Strategy Return", port_metrics.get("Total Return", "N/A"))
    m2.metric("Buy & Hold Return", bnh_metrics.get("Total Return", "N/A"))
    m3.metric("Sharpe Ratio", port_metrics.get("Sharpe", "N/A"))
    m4.metric("Max Drawdown", port_metrics.get("Max Drawdown", "N/A"))
    m5.metric("Final Portfolio", port_metrics.get("Final Value", "N/A"))

    # Equity curve chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        subplot_titles=["Equity Curve: Strategy vs Buy & Hold", "Drawdown"])
    fig.add_trace(go.Scatter(x=total_equity.index, y=total_equity.values, name="Strategy",
                             line=dict(color="#818cf8", width=2), fill="tozeroy",
                             fillcolor="rgba(129,140,248,0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=total_bnh.index, y=total_bnh.values, name="Buy & Hold",
                             line=dict(color="#94a3b8", width=1, dash="dot")), row=1, col=1)
    # Drawdown
    rolling_max = total_equity.cummax()
    dd = (total_equity - rolling_max) / rolling_max * 100
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown %",
                             fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
                             line=dict(color="#ef4444", width=1)), row=2, col=1)
    fig.update_layout(height=500, template="plotly_dark",
                      margin=dict(l=0, r=0, t=40, b=0), showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text="₹", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────
    # SECTION 2: PER-STOCK BREAKDOWN
    # ──────────────────────────────────
    st.markdown("## 📋 Per-Stock Breakdown")
    st.caption("How each stock contributed to the portfolio performance")

    rows = []
    for ticker in per_stock_equity:
        m = _per_stock_metrics(per_stock_equity[ticker], capital_per_stock)
        n_buys = len([t for t in all_trades if t["Symbol"] == ticker.replace(".NS", "") and t["Action"] == "BUY"])
        n_sells = len([t for t in all_trades if t["Symbol"] == ticker.replace(".NS", "") and t["Action"] == "SELL"])
        wins = len([t for t in all_trades if t["Symbol"] == ticker.replace(".NS", "") and t["Action"] == "SELL" and t["PnL"] > 0])
        wr = f"{wins / n_sells * 100:.0f}%" if n_sells > 0 else "—"
        total_pnl = sum(t["PnL"] for t in all_trades if t["Symbol"] == ticker.replace(".NS", "") and t["Action"] == "SELL")
        rows.append({
            "Stock": ticker.replace(".NS", ""),
            "Return": m.get("Total Return", "—"),
            "Sharpe": m.get("Sharpe", "—"),
            "Sortino": m.get("Sortino", "—"),
            "Max DD": m.get("Max Drawdown", "—"),
            "Trades": n_buys,
            "Win Rate": wr,
            "Net PnL": f"₹{total_pnl:+,.0f}",
            "Final": m.get("Final Value", "—"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Per-stock equity mini-charts
    with st.expander("📈 Individual Stock Equity Curves", expanded=False):
        n_stocks = len(per_stock_equity)
        cols = st.columns(min(n_stocks, 3))
        for i, (ticker, eq) in enumerate(per_stock_equity.items()):
            with cols[i % 3]:
                fig_mini = go.Figure()
                fig_mini.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy",
                                             line=dict(color="#818cf8", width=1.5), fill="tozeroy",
                                             fillcolor="rgba(129,140,248,0.1)"))
                if ticker in per_stock_bnh:
                    bnh_s = per_stock_bnh[ticker]
                    fig_mini.add_trace(go.Scatter(x=bnh_s.index, y=bnh_s.values, name="B&H",
                                                 line=dict(color="#94a3b8", width=1, dash="dot")))
                fig_mini.update_layout(height=200, template="plotly_dark", title=ticker.replace(".NS", ""),
                                       margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
                st.plotly_chart(fig_mini, use_container_width=True)

    # ──────────────────────────────────
    # SECTION 3: RISK ANALYSIS
    # ──────────────────────────────────
    st.markdown("## 📉 Risk Analysis")

    daily_rets = total_equity.pct_change().dropna()
    var_95 = np.percentile(daily_rets, 5) * 100
    cvar_95 = daily_rets[daily_rets <= np.percentile(daily_rets, 5)].mean() * 100
    calmar = float(port_metrics.get("Total Return", "0").replace("%", "").replace("+", "")) / abs(float(port_metrics.get("Max Drawdown", "-1").replace("%", ""))) if float(port_metrics.get("Max Drawdown", "-1").replace("%", "")) != 0 else 0

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("VaR (95%)", f"{var_95:.2f}%", help="Worst daily loss with 95% confidence")
    r2.metric("CVaR / ES", f"{cvar_95:.2f}%", help="Expected loss in worst 5% of days")
    r3.metric("Calmar Ratio", f"{calmar:.2f}", help="Return / Max Drawdown — higher is better")
    r4.metric("Sortino", port_metrics.get("Sortino", "—"), help="Return / Downside deviation")

    # Returns distribution
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(daily_rets * 100, nbins=50, title="Daily Returns Distribution (%)",
                                template="plotly_dark", color_discrete_sequence=["#818cf8"])
        fig_hist.add_vline(x=var_95, line_dash="dash", line_color="#ef4444",
                          annotation_text=f"VaR 95%: {var_95:.2f}%")
        fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        # Monthly returns heatmap
        monthly = daily_rets.resample("ME").sum() * 100
        mdf = pd.DataFrame({"Year": monthly.index.year, "Month": monthly.index.month, "Return": monthly.values})
        piv = mdf.pivot_table(index="Year", columns="Month", values="Return")
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        piv.columns = [month_names[c - 1] for c in piv.columns]
        fig_heat = px.imshow(piv, color_continuous_scale="RdYlGn", text_auto=".1f",
                             title="Monthly Returns Heatmap (%)", template="plotly_dark",
                             aspect="auto")
        fig_heat.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

    # ──────────────────────────────────
    # SECTION 4: TRADE LOG
    # ──────────────────────────────────
    st.markdown("## 📝 Trade Log")
    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("Date", ascending=False)
        # Color PnL
        st.dataframe(trades_df.head(30), use_container_width=True, hide_index=True)
        st.caption(f"Showing latest 30 of {len(trades_df)} total trade events.")
    else:
        st.info("No trades were generated. Try adjusting parameters (lower RSI threshold or wider lookback).")

    # ──────────────────────────────────
    # SECTION 5: INTERPRETATION
    # ──────────────────────────────────
    st.markdown("## 🧠 Strategy Interpretation")

    total_ret_val = float(port_metrics.get("Total Return", "0").replace("%", "").replace("+", ""))
    bnh_ret_val = float(bnh_metrics.get("Total Return", "0").replace("%", "").replace("+", ""))
    sharpe_val = float(port_metrics.get("Sharpe", "0"))
    max_dd_val = float(port_metrics.get("Max Drawdown", "0").replace("%", ""))

    # Build interpretation
    verdicts = []
    if total_ret_val > bnh_ret_val:
        verdicts.append(f"✅ **Outperformed Buy & Hold** by {total_ret_val - bnh_ret_val:.1f}pp — the strategy added alpha.")
    else:
        verdicts.append(f"❌ **Underperformed Buy & Hold** by {bnh_ret_val - total_ret_val:.1f}pp — the strategy destroyed value.")

    if sharpe_val > 1.0:
        verdicts.append(f"✅ **Sharpe Ratio ({sharpe_val:.2f})** exceeds 1.0 — risk-adjusted returns are institutional grade.")
    elif sharpe_val > 0.5:
        verdicts.append(f"🟡 **Sharpe Ratio ({sharpe_val:.2f})** is moderate — acceptable but not strong.")
    else:
        verdicts.append(f"❌ **Sharpe Ratio ({sharpe_val:.2f})** is below 0.5 — risk-adjusted returns are poor.")

    if abs(max_dd_val) < 15:
        verdicts.append(f"✅ **Max Drawdown ({max_dd_val:.1f}%)** is contained — strategy manages risk well.")
    elif abs(max_dd_val) < 30:
        verdicts.append(f"🟡 **Max Drawdown ({max_dd_val:.1f}%)** is significant — consider adding stop-losses.")
    else:
        verdicts.append(f"❌ **Max Drawdown ({max_dd_val:.1f}%)** is severe — strategy has catastrophic risk.")

    n_total_trades = len([t for t in all_trades if t["Action"] == "SELL"])
    if n_total_trades == 0:
        verdicts.append("⚠️ **Zero completed trades** — the signal thresholds may be too strict. Try lowering RSI threshold.")
    elif n_total_trades < 5:
        verdicts.append(f"🟡 **Only {n_total_trades} trades** — insufficient sample size for statistical significance.")

    for v in verdicts:
        st.markdown(v)

    # Overall verdict
    score = 0
    score += 2 if total_ret_val > bnh_ret_val else -1
    score += 2 if sharpe_val > 1.0 else (1 if sharpe_val > 0.5 else -1)
    score += 1 if abs(max_dd_val) < 15 else (-1 if abs(max_dd_val) > 30 else 0)

    if score >= 4:
        st.success("### 🟢 VERDICT: Strong Strategy — suitable for paper trading.")
    elif score >= 1:
        st.warning("### 🟡 VERDICT: Moderate Strategy — needs refinement before deployment.")
    else:
        st.error("### 🔴 VERDICT: Weak Strategy — do NOT deploy. Revisit parameters or strategy type.")
