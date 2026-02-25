
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf

from modules.data.manager import StockDataManager
from modules.ml.features import FeatureEngineer
from modules.ml.engine import MLEngine
from modules.ui.sidebar import show_sidebar
from modules.utils.helpers import format_currency, format_large_number
from modules.ui.chart_config import COLORS, TEMPLATE, format_market_cap, clean_ticker_label
from modules.data.scrapers.news_analysis import fetch_top_news, analyze_news_with_ai, render_news_tiles

try:
    from agentic_backend import run_garch_volatility_forecast, query_deepseek_reasoner
except ImportError:
    run_garch_volatility_forecast = None
    query_deepseek_reasoner = None


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════
def _fmt_metric(val, fmt=".2f", prefix="", suffix=""):
    """Safely format a numeric value."""
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    try:
        return f"{prefix}{val:{fmt}}{suffix}"
    except (ValueError, TypeError):
        return "N/A"


def _safe_get(info, key, default="N/A"):
    v = info.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


def _fetch_news_for_analysis(ticker, company_name=""):
    """Fetch news with yfinance → Google RSS fallback."""
    news_items = []

    # Primary: yfinance .news
    try:
        yt = yf.Ticker(ticker)
        raw = yt.news or []
        for article in raw[:8]:
            title = article.get('title', '')
            if not title:
                title = article.get('content', {}).get('title', 'No title')
            link = article.get('link', '')
            if not link:
                link = article.get('content', {}).get('clickThroughUrl', {}).get('url', '#')
            publisher = article.get('publisher', '')
            if not publisher:
                publisher = article.get('content', {}).get('provider', {}).get('displayName', 'News')
            if title:
                news_items.append({"title": title, "link": link, "publisher": publisher})
    except Exception:
        pass

    # Fallback: Google News RSS
    if not news_items:
        try:
            import urllib.request
            import xml.etree.ElementTree as ET
            search_term = company_name or ticker.replace(".NS", "").replace(".BO", "")
            url = f"https://news.google.com/rss/search?q={search_term}+stock&hl=en-IN&gl=IN"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for item in root.findall('.//item')[:8]:
                t = item.find('title')
                l = item.find('link')
                s = item.find('source')
                if t is not None:
                    news_items.append({
                        "title": t.text or "No title",
                        "link": l.text if l is not None else "#",
                        "publisher": s.text if s is not None else "Google News"
                    })
        except Exception:
            pass

    if not news_items:
        news_items.append({
            "title": f"No recent news found for {ticker}. Search on Moneycontrol or Economic Times.",
            "link": f"https://www.moneycontrol.com/india/stockpricequote/search.php?search={ticker.replace('.NS','')}",
            "publisher": "System"
        })

    return news_items


def calculate_risk_metrics(df):
    """Comprehensive risk metrics from price data."""
    if df.empty or 'Close' not in df.columns:
        return {}

    returns = df['Close'].pct_change().dropna()
    ann_vol = returns.std() * np.sqrt(252)

    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1.0
    wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    kelly_f = max(0, win_rate - (1 - win_rate) / wl_ratio) if wl_ratio > 0 else 0

    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() * 252) / downside if downside > 0 else 0

    rolling_max = df['Close'].cummax()
    drawdown = (df['Close'] - rolling_max) / rolling_max
    max_dd = drawdown.min()

    regime = "🔴 High" if ann_vol > 0.35 else ("🟢 Low" if ann_vol < 0.20 else "🟡 Normal")

    return {
        "VaR_95": var_95, "CVaR_95": cvar_95,
        "Kelly_Full": kelly_f, "Kelly_Half": kelly_f / 2,
        "Win_Rate": win_rate, "Win_Loss_Ratio": wl_ratio,
        "Regime": regime, "Annual_Vol": ann_vol,
        "Sharpe": sharpe, "Sortino": sortino,
        "Max_Drawdown": max_dd,
    }


# ════════════════════════════════════════════════════════════════
# MAIN PAGE
# ════════════════════════════════════════════════════════════════
def render_stock_analysis():
    # Sidebar picker
    selected_company, period, chart_type, show_technicals, ai_model = show_sidebar()

    dm = StockDataManager()
    fe = FeatureEngineer()
    ml = MLEngine()

    # ── Check session_state cache first ──
    cache_hit = (
        st.session_state.get('cache_ticker') == selected_company
        and st.session_state.get('cached_df') is not None
        and st.session_state.get('cached_info') is not None
    )

    if cache_hit:
        df = st.session_state.cached_df
        info = st.session_state.cached_info
        df_feat = st.session_state.get('cached_df_feat', df.copy())
        live_data = st.session_state.get('cached_live_data', {})
        yf_ticker = st.session_state.get('cached_yf_ticker', None)
    else:
        # Fetch fresh data
        with st.spinner(f"Loading data for **{selected_company}**..."):
            live_data = dm.get_live_data(selected_company)
            df = dm.get_historical_data(selected_company, period=period)

        if not live_data or df.empty:
            st.error(f"Could not fetch data for **{selected_company}**. Check the ticker symbol.")
            return

        yf_ticker = None
        try:
            yf_ticker = yf.Ticker(selected_company)
            info = yf_ticker.info or {}
        except Exception:
            info = {}

        try:
            df_feat = fe._compute_single_ticker_features(df.copy())
        except Exception:
            df_feat = df.copy()

        st.session_state.cached_df = df
        st.session_state.cached_info = info
        st.session_state.cached_df_feat = df_feat
        st.session_state.cached_live_data = live_data
        st.session_state.cached_yf_ticker = yf_ticker
        st.session_state.cache_ticker = selected_company

    # Ensure sentiment_score column exists for ML model (required feature)
    if 'sentiment_score' not in df_feat.columns:
        df_feat['sentiment_score'] = 0.0

    # ────────────────────────────────────────────────────────────
    # HERO SECTION
    # ────────────────────────────────────────────────────────────
    symbol = live_data.get('symbol') or selected_company
    company_name = live_data.get('long_name', symbol)
    price = live_data.get('current_price') or df['Close'].iloc[-1]
    prev_close = live_data.get('previous_close') or df['Close'].iloc[-2] if len(df) > 1 else price
    change = price - prev_close
    pct = (change / prev_close * 100) if prev_close != 0 else 0
    color = "#22c55e" if change >= 0 else "#ef4444"
    arrow = "▲" if change >= 0 else "▼"
    sector = live_data.get('sector', _safe_get(info, 'sector', 'Unknown'))
    industry = _safe_get(info, 'industry', 'Unknown')

    st.markdown(f"""
    <div style="padding:24px; border-radius:14px; background:linear-gradient(135deg,#1e293b,#0f172a);
         border:1px solid rgba(255,255,255,0.1); margin-bottom:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <h1 style="margin:0; font-size:2rem; background:linear-gradient(to right,#fff,#94a3b8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">{company_name}</h1>
                <div style="margin-top:6px;">
                    <span style="background:rgba(129,140,248,0.15); color:#818cf8; padding:4px 10px;
                          border-radius:6px; font-size:0.85rem; font-weight:600;">{clean_ticker_label(symbol)}</span>
                    <span style="color:#94a3b8; margin-left:8px; font-size:0.9rem;">{sector} • {industry}</span>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:2.5rem; font-weight:700; color:white;">₹{price:,.2f}</div>
                <div style="color:{color}; font-size:1.2rem; font-weight:600;">{arrow} ₹{abs(change):.2f} ({pct:+.2f}%)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats — custom HTML cards (no truncation)
    mcap = _safe_get(info, 'marketCap', None)
    pe = _fmt_metric(_safe_get(info, 'trailingPE', None), ".1f")
    pb = _fmt_metric(_safe_get(info, 'priceToBook', None), ".2f")
    high52 = _fmt_metric(_safe_get(info, 'fiftyTwoWeekHigh', None), ",.0f", "₹")
    low52 = _fmt_metric(_safe_get(info, 'fiftyTwoWeekLow', None), ",.0f", "₹")
    dv = _safe_get(info, 'dividendYield', None)
    if isinstance(dv, (int, float)):
        dv_pct = dv if dv > 0.10 else dv * 100
        div_str = f"{dv_pct:.2f}%"
    else:
        div_str = "N/A"
    
    def _stat_card(label, value):
        return (f'<div style="flex:1;background:rgba(30,41,59,0.7);border:1px solid rgba(148,163,184,0.2);'
                f'border-radius:8px;padding:10px 12px;text-align:center;min-width:0;">'
                f'<div style="color:#94a3b8;font-size:0.7rem;letter-spacing:0.5px;margin-bottom:4px;">{label}</div>'
                f'<div style="color:#e2e8f0;font-size:1.05rem;font-weight:700;white-space:nowrap;">{value}</div>'
                f'</div>')
    
    stats_html = ('<div style="display:flex;gap:8px;margin:8px 0 16px 0;">'
                  + _stat_card("Market Cap", format_market_cap(mcap))
                  + _stat_card("P/E Ratio", pe)
                  + _stat_card("P/B Ratio", pb)
                  + _stat_card("52W High", high52)
                  + _stat_card("52W Low", low52)
                  + _stat_card("Div. Yield", div_str)
                  + '</div>')
    st.markdown(stats_html, unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────
    # TABS
    # ────────────────────────────────────────────────────────────
    tab_tech, tab_risk, tab_fin, tab_ml, tab_news = st.tabs([
        "📈 Technical Analysis", "📉 Risk & Quant", "💰 Financials", "🔮 ML Predictions", "📰 News & AI"
    ])

    # ============================================================
    # TAB 1: TECHNICAL ANALYSIS
    # ============================================================
    with tab_tech:
        st.markdown("### 📈 Price Action & Technical Signals")
        st.markdown(f"""
        > *Below is the price history for **{company_name}** over the past **{period}**, overlaid
        > with Simple Moving Averages (50 & 200 day) and Bollinger Bands. The signal table
        > translates each indicator into an actionable directional bias.*
        """)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                            subplot_titles=["Price with Overlays", "Volume"])

        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                          low=df['Low'], close=df['Close'], name="OHLC"), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close",
                                     line=dict(color=COLORS['price_line'], width=2)), row=1, col=1)

        if show_technicals and 'SMA_50' in df_feat.columns:
            fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['SMA_50'], name="SMA 50",
                                     line=dict(color=COLORS['sma_50'], width=1.5)), row=1, col=1)
        if show_technicals and 'SMA_200' in df_feat.columns:
            fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['SMA_200'], name="SMA 200",
                                     line=dict(color=COLORS['sma_200'], width=1.5)), row=1, col=1)
        if show_technicals and 'BB_High' in df_feat.columns:
            fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['BB_High'], name="BB Upper",
                                     line=dict(color="#94a3b8", width=0.5, dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_feat.index, y=df_feat['BB_Low'], name="BB Lower",
                                     line=dict(color="#94a3b8", width=0.5, dash="dot"),
                                     fill="tonexty", fillcolor="rgba(148,163,184,0.05)"), row=1, col=1)

        if 'Volume' in df.columns:
            colors = ['#22c55e' if c >= o else '#ef4444' for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume",
                                 marker_color=colors, opacity=0.5), row=2, col=1)

        fig.update_layout(height=520, template="plotly_dark", xaxis_rangeslider_visible=False,
                          margin=dict(l=0, r=0, t=40, b=0), showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        # Signal interpretation table
        st.markdown("### 🎯 Signal Interpretation")
        st.markdown("*Each indicator is categorized as Bullish 🟢, Neutral 🟡, or Bearish 🔴 based on standard thresholds.*")
        latest = df_feat.iloc[-1] if not df_feat.empty else pd.Series()
        signals = []

        rsi = latest.get('RSI', np.nan)
        if not pd.isna(rsi):
            if rsi > 70: lbl = "🔴 Overbought"
            elif rsi > 60: lbl = "🟡 Bullish"
            elif rsi > 40: lbl = "🟢 Neutral"
            elif rsi > 30: lbl = "🟡 Bearish"
            else: lbl = "🔴 Oversold"
            signals.append({"Indicator": "RSI (14)", "Value": f"{rsi:.1f}", "Signal": lbl, "Weight": "High"})

        macd_diff = latest.get('MACD_Diff', np.nan)
        if not pd.isna(macd_diff):
            lbl = "🟢 Bullish Crossover" if macd_diff > 0 else "🔴 Bearish Crossover"
            signals.append({"Indicator": "MACD Histogram", "Value": f"{macd_diff:.4f}", "Signal": lbl, "Weight": "High"})

        bb_h, bb_l = latest.get('BB_High', np.nan), latest.get('BB_Low', np.nan)
        if not pd.isna(bb_h) and not pd.isna(bb_l):
            if price > bb_h: lbl = "🔴 Above Upper Band"
            elif price < bb_l: lbl = "🟢 Below Lower Band"
            else: lbl = "🟡 Within Bands"
            signals.append({"Indicator": "Bollinger Bands", "Value": f"[{bb_l:.0f} – {bb_h:.0f}]", "Signal": lbl, "Weight": "Medium"})

        sma50 = latest.get('SMA_50', np.nan)
        sma200 = latest.get('SMA_200', np.nan)
        if not pd.isna(sma50) and not pd.isna(sma200) and sma200 > 0:
            lbl = "🟢 Golden Cross" if sma50 > sma200 else "🔴 Death Cross"
            signals.append({"Indicator": "SMA 50/200", "Value": f"{sma50:.0f} / {sma200:.0f}", "Signal": lbl, "Weight": "High"})

        vol_z = latest.get('Vol_Z', np.nan)
        if not pd.isna(vol_z):
            if abs(vol_z) > 2: lbl = "🔴 Volume Spike"
            elif abs(vol_z) > 1: lbl = "🟡 Elevated"
            else: lbl = "🟢 Normal"
            signals.append({"Indicator": "Volume Z-Score", "Value": f"{vol_z:.2f}", "Signal": lbl, "Weight": "Medium"})

        atr = latest.get('ATR', np.nan)
        if not pd.isna(atr):
            signals.append({"Indicator": "ATR (14)", "Value": f"₹{atr:.2f}", "Signal": f"Daily range ±₹{atr:.0f}", "Weight": "Info"})

        vol21 = latest.get('Vol_21d', np.nan)
        if not pd.isna(vol21):
            ann_vol_pct = vol21 * np.sqrt(252) * 100
            regime = "🔴 High" if ann_vol_pct > 35 else ("🟢 Low" if ann_vol_pct < 20 else "🟡 Normal")
            signals.append({"Indicator": "Annualized Volatility", "Value": f"{ann_vol_pct:.1f}%", "Signal": f"{regime} Regime", "Weight": "High"})

        if signals:
            st.dataframe(pd.DataFrame(signals), use_container_width=True, hide_index=True)
            bull = sum(1 for s in signals if "🟢" in s["Signal"])
            bear = sum(1 for s in signals if "🔴" in s["Signal"])
            if bull > bear + 1:
                st.success(f"**Overall Technical Bias: BULLISH** — {bull} bullish vs {bear} bearish signals. Momentum favors upside continuation.")
            elif bear > bull + 1:
                st.error(f"**Overall Technical Bias: BEARISH** — {bear} bearish vs {bull} bullish signals. Consider defensive positioning.")
            else:
                st.warning(f"**Overall Technical Bias: NEUTRAL** — {bull} bullish, {bear} bearish. Wait for a clearer directional break.")
        else:
            st.info("Insufficient data for signal computation. Try a longer time period.")

    # ============================================================
    # TAB 2: RISK & QUANT
    # ============================================================
    with tab_risk:
        st.markdown("### 📉 Risk Assessment & Position Sizing")
        st.markdown(f"""
        > *Risk metrics help you understand the downside profile and determine how much capital to allocate.
        > VaR = worst daily loss at 95% confidence; GARCH = volatility forecast; Kelly = optimal bet size.*
        """)

        risk = calculate_risk_metrics(df)
        returns = df['Close'].pct_change().dropna()

        # GARCH
        garch_vol = 0.0
        if run_garch_volatility_forecast:
            try:
                garch_vol = run_garch_volatility_forecast(returns) * 100
            except Exception:
                pass

        # Beta
        beta = float('nan')
        try:
            nifty = yf.download("^NSEI", period=period, progress=False)
            if hasattr(nifty.columns, 'levels') and nifty.columns.nlevels > 1:
                nifty.columns = nifty.columns.get_level_values(0)
            nifty_ret = nifty['Close'].pct_change().dropna()
            common = returns.index.intersection(nifty_ret.index)
            if len(common) > 30:
                cov = np.cov(returns.loc[common], nifty_ret.loc[common])
                beta = cov[0, 1] / cov[1, 1]
        except Exception:
            pass

        # SAFE beta formatting (prevents the f-string bug)
        beta_str = f"{beta:.2f}" if not np.isnan(beta) else "N/A"

        r1, r2, r3, r4 = st.columns(4)
        var_val = risk.get('VaR_95',0)
        cvar_val = risk.get('CVaR_95',0)
        sharpe_val = risk.get('Sharpe',0)
        r1.metric("VaR (95%)", f"{var_val*100:.2f}%", help="Worst daily loss, 95% confidence")
        r2.metric("CVaR / ES", f"{cvar_val*100:.2f}%", help="Avg loss in worst 5% of days")
        r3.metric("Sharpe Ratio", f"{sharpe_val:.2f}")
        r4.metric("Beta (vs NIFTY)", beta_str)

        r5, r6, r7, r8 = st.columns(4)
        sortino_val = risk.get('Sortino',0)
        max_dd_val = risk.get('Max_Drawdown',0)
        kelly_val = risk.get('Kelly_Half',0)
        r5.metric("Sortino Ratio", f"{sortino_val:.2f}")
        r6.metric("Max Drawdown", f"{max_dd_val*100:.1f}%")
        r7.metric("GARCH 1d Vol", f"{garch_vol:.2f}%")
        r8.metric("Half-Kelly", f"{kelly_val*100:.1f}%", help="Optimal position size")

        # Regime
        regime = risk.get("Regime", "Unknown")
        ann_vol_val = risk.get('Annual_Vol', 0) * 100
        if "High" in regime:
            st.error(f"**{regime} Volatility Regime** ({ann_vol_val:.0f}% annualized) — Reduce position size and widen stop-losses.")
        elif "Low" in regime:
            st.success(f"**{regime} Volatility Regime** ({ann_vol_val:.0f}% annualized) — Favorable for trend-following strategies.")
        else:
            st.warning(f"**{regime} Volatility Regime** ({ann_vol_val:.0f}% annualized) — Standard position sizing applies.")

        # ── Layman-friendly Ratio Explanations ──
        with st.expander(f"📖 What do these risk metrics mean for {company_name}?", expanded=False):
            st.markdown("#### 📉 Downside Risk Metrics")

            st.markdown(f"""
**VaR (Value at Risk) = {var_val*100:.2f}%**
> *"On 19 out of 20 trading days, you won't lose more than {abs(var_val*100):.2f}% of your investment in a single day."*
> On a ₹1,00,000 investment → worst expected daily loss: **₹{abs(var_val)*100000:.0f}**
""")

            st.markdown(f"""
**CVaR / Expected Shortfall = {cvar_val*100:.2f}%**
> *"On the rare bad days (worst 5%), the average loss is {abs(cvar_val*100):.2f}%."*
> Think of VaR as the guardrail; CVaR tells you how far you fall when you crash through it.
> On ₹1,00,000 → avg loss on worst days: **₹{abs(cvar_val)*100000:.0f}**
""")

            st.markdown(f"""
**Max Drawdown = {max_dd_val*100:.1f}%**
> *"The worst peak-to-trough decline in the stock's history over this period."*
> {'⚠️ Severe drawdown — this stock has fallen more than 25% from its peak. Needs strong conviction to hold.' if abs(max_dd_val) > 0.25 else '✅ Moderate drawdown — within normal range for equity investments.' if abs(max_dd_val) > 0.10 else '🟢 Shallow drawdown — relatively stable stock.'}
""")

            st.markdown("#### 📊 Risk-Adjusted Returns")

            st.markdown(f"""
**Sharpe Ratio = {sharpe_val:.2f}**
> *"How much extra return you earn per unit of total risk (volatility)."*
> Formula: (Return − Risk-Free Rate) ÷ Volatility
> {'🟢 Excellent — above 1.0 means good risk-adjusted returns.' if sharpe_val > 1.0 else '🟡 Acceptable — between 0.5 and 1.0, adequate compensation for risk.' if sharpe_val > 0.5 else '🔴 Poor — below 0.5, the risk is not being adequately compensated.'}
""")

            st.markdown(f"""
**Sortino Ratio = {sortino_val:.2f}**
> *"Like Sharpe, but only penalizes downside volatility (drops), not upside swings."*
> A Sortino of {sortino_val:.2f} means the stock earns {sortino_val:.2f}× more return than its downside risk.
> {'🟢 Strong — upside volatility is significantly more than downside.' if sortino_val > 1.5 else '🟡 Decent — reasonable downside management.' if sortino_val > 0.8 else '🔴 Weak — too much downside relative to returns.'}
""")

            st.markdown("#### 📈 Market Sensitivity")

            st.markdown(f"""
**Beta (vs NIFTY) = {beta_str}**
> *"How much the stock moves relative to the NIFTY 50 index."*
> Beta = 1.0 → moves exactly like the market. Beta > 1 → more volatile. Beta < 1 → less volatile.
> {'📈 High Beta — this stock amplifies market moves. +10% NIFTY ≈ +' + f'{beta*10:.0f}% on this stock.' if not np.isnan(beta) and beta > 1.2 else '⚖️ Market-neutral — moves roughly in line with NIFTY.' if not np.isnan(beta) and beta > 0.8 else '🛡️ Defensive — less volatile than the broad market.' if not np.isnan(beta) else 'Beta not calculated.'}
""")

            st.markdown("#### 🔮 Forecasted Volatility")

            st.markdown(f"""
**GARCH 1-Day Volatility = {garch_vol:.2f}%**
> *"Tomorrow's expected price swing, predicted by a GARCH(1,1) statistical model."*
> A 1-day vol of {garch_vol:.2f}% on a ₹{price:,.0f} stock → expected daily range: **±₹{price*garch_vol/100:.1f}**
> {'🔴 High forecasted volatility — consider reducing position or hedging.' if garch_vol > 2.0 else '🟡 Normal volatility expected.' if garch_vol > 1.0 else '🟢 Low volatility forecast — calm trading expected.'}
""")

            st.markdown("#### 🎯 Position Sizing")

            st.markdown(f"""
**Half-Kelly = {kelly_val*100:.1f}%**
> *"The mathematically optimal portion of your portfolio to invest, halved for safety."*
> Based on your win rate ({risk.get('Win_Rate',0)*100:.0f}%) and win/loss ratio ({risk.get('Win_Loss_Ratio',0):.2f}×).
> On a ₹10,00,000 portfolio → recommended allocation: **₹{kelly_val*1000000:,.0f}**
> We use Half-Kelly (50% of full Kelly) because full Kelly assumes perfect information and is too aggressive for real markets.
""")

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(returns * 100, nbins=50, title="Daily Returns Distribution (%)",
                                    template="plotly_dark", color_discrete_sequence=["#818cf8"])
            var_line = risk.get('VaR_95', 0) * 100
            fig_hist.add_vline(x=var_line, line_dash="dash", line_color="#ef4444",
                               annotation_text=f"VaR: {var_line:.2f}%")
            fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Stock-specific chart description
            avg_ret = returns.mean() * 100
            std_ret = returns.std() * 100
            neg_pct = (returns < 0).sum() / len(returns) * 100
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.4); border-radius:8px; padding:12px 16px; margin-top:-10px; font-size:0.85rem; color:#CBD5E1; line-height:1.6;">
                📊 <b>What this chart tells you:</b> The histogram shows how {company_name}'s daily returns are distributed.
                The bell shape reveals that most days see small moves (avg: {avg_ret:+.3f}%), but the tails show rare extreme days.
                The <span style="color:#ef4444;">red dashed line</span> marks VaR — {abs(neg_pct):.0f}% of trading days had negative returns.
                A wider/flatter bell = more volatility and uncertainty.
            </div>
            """, unsafe_allow_html=True)

        with c2:
            rolling_max = df['Close'].cummax()
            dd = (df['Close'] - rolling_max) / rolling_max * 100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                                        fillcolor="rgba(239,68,68,0.2)",
                                        line=dict(color="#ef4444", width=1), name="Drawdown"))
            fig_dd.update_layout(title="Underwater (Drawdown) Chart", template="plotly_dark",
                                 height=300, margin=dict(l=0, r=0, t=40, b=0), yaxis_title="%")
            st.plotly_chart(fig_dd, use_container_width=True)

            # Stock-specific chart description
            max_dd_pct = max_dd_val * 100
            current_dd = dd.iloc[-1] if len(dd) > 0 else 0
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.4); border-radius:8px; padding:12px 16px; margin-top:-10px; font-size:0.85rem; color:#CBD5E1; line-height:1.6;">
                📉 <b>What this chart tells you:</b> The underwater chart shows how far {company_name} has fallen from its all-time peak at any point.
                Max drawdown was <b>{max_dd_pct:.1f}%</b> — meaning at its worst, ₹1,00,000 invested would have become ₹{100000*(1+max_dd_val):,.0f}.
                Current drawdown: <b>{current_dd:.1f}%</b> {'— the stock is near its peak!' if abs(current_dd) < 5 else '— still recovering from a dip.' if abs(current_dd) < 15 else '— significant distance from peak. Watch for recovery signals.'}
            </div>
            """, unsafe_allow_html=True)

        # Position sizing
        st.markdown("### 🎯 Computed Position Sizing")
        kelly = risk.get('Kelly_Half', 0)
        atr_val = latest.get('ATR', 0) if not df_feat.empty else 0
        if pd.isna(atr_val): atr_val = 0

        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown(f"""
**Half-Kelly recommends:** `{kelly*100:.1f}%` of portfolio
- On a ₹10L portfolio: **₹{kelly*1000000:,.0f}**
- Win Rate: {risk.get('Win_Rate',0)*100:.0f}% | W/L Ratio: {risk.get('Win_Loss_Ratio',0):.2f}
            """)
        with pc2:
            st.markdown(f"""
**Stop-Loss Levels:**
- VaR-based (statistical): **₹{price * (1 + risk.get('VaR_95',0)):.0f}** ({risk.get('VaR_95',0)*100:.1f}%)
- ATR-based (1.5×): **₹{price - 1.5 * atr_val:.0f}** (−{1.5*atr_val/price*100:.1f}%)
            """)

    # ============================================================
    # TAB 3: FINANCIALS
    # ============================================================
    with tab_fin:
        st.markdown("### 💰 Financial Statement Analysis")
        st.markdown(f"""
        > *Analyze **{company_name}'s** revenue trajectory, profitability margins, and key financial ratios.
        > Consistent growth + expanding margins = strong fundamental health.*
        """)

        try:
            if yf_ticker is None:
                raise ValueError("yf_ticker not available")
            income_stmt = yf_ticker.income_stmt
            balance_sheet = yf_ticker.balance_sheet
            cashflow = yf_ticker.cashflow

            if income_stmt is not None and not income_stmt.empty:
                c1, c2 = st.columns(2)
                with c1:
                    rev_row = income_stmt.loc['Total Revenue'] if 'Total Revenue' in income_stmt.index else None
                    ni_row = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None
                    if rev_row is not None:
                        fig_rev = go.Figure()
                        years = [str(d.year) for d in rev_row.index]
                        fig_rev.add_trace(go.Bar(x=years, y=rev_row.values / 1e7, name="Revenue (Cr)",
                                                 marker_color=COLORS['revenue'],
                                                 text=[f"{v/1e7:,.0f}" for v in rev_row.values],
                                                 textposition='outside'))
                        if ni_row is not None:
                            fig_rev.add_trace(go.Bar(x=years, y=ni_row.values / 1e7, name="Net Income (Cr)",
                                                     marker_color=COLORS['net_income'],
                                                     text=[f"{v/1e7:,.0f}" for v in ni_row.values],
                                                     textposition='outside'))
                        fig_rev.update_layout(title="Revenue & Net Income (₹ Crores)", template="plotly_dark",
                                              height=350, margin=dict(l=0, r=0, t=40, b=0), barmode="group")
                        st.plotly_chart(fig_rev, use_container_width=True)

                with c2:
                    if rev_row is not None and ni_row is not None:
                        years = [str(d.year) for d in rev_row.index]
                        fig_margin = go.Figure()
                        net_margin = (ni_row.values / rev_row.values * 100)
                        fig_margin.add_trace(go.Scatter(x=years, y=net_margin, name="Net Margin %",
                                                        line=dict(color="#22c55e", width=2), mode="lines+markers"))
                        op_income = income_stmt.loc['Operating Income'] if 'Operating Income' in income_stmt.index else None
                        if op_income is not None:
                            op_margin = (op_income.values / rev_row.values * 100)
                            fig_margin.add_trace(go.Scatter(x=years, y=op_margin, name="Op Margin %",
                                                            line=dict(color="#818cf8", width=2), mode="lines+markers"))
                        fig_margin.update_layout(title="Profit Margins (%)", template="plotly_dark",
                                                 height=350, margin=dict(l=0, r=0, t=40, b=0))
                        st.plotly_chart(fig_margin, use_container_width=True)

                # Ratios
                st.markdown("### 📊 Key Financial Ratios")
                ratio_groups = {
                    "Valuation": [("P/E (Trailing)", 'trailingPE'), ("P/E (Forward)", 'forwardPE'),
                                  ("P/B", 'priceToBook'), ("EV/EBITDA", 'enterpriseToEbitda'), ("PEG", 'pegRatio')],
                    "Profitability": [("ROE", 'returnOnEquity'), ("ROA", 'returnOnAssets'),
                                      ("Profit Margin", 'profitMargins'), ("Op Margin", 'operatingMargins')],
                    "Leverage": [("Debt/Equity", 'debtToEquity'), ("Current Ratio", 'currentRatio'),
                                 ("Quick Ratio", 'quickRatio')],
                }
                rcols = st.columns(len(ratio_groups))
                for i, (cat, items) in enumerate(ratio_groups.items()):
                    with rcols[i]:
                        st.markdown(f"**{cat}**")
                        for name, key in items:
                            val = _safe_get(info, key, None)
                            if val is not None and val != "N/A" and isinstance(val, (int, float)):
                                if any(x in name.lower() for x in ['margin', 'roe', 'roa']):
                                    st.markdown(f"- **{name}**: `{val*100:.2f}%`")
                                elif 'debt' in name.lower():
                                    st.markdown(f"- **{name}**: `{val:.0f}%`")
                                else:
                                    st.markdown(f"- **{name}**: `{val:.2f}`")

                # ── Ratio Explanations with Formulas ──
                with st.expander(f"📖 What do these ratios mean for {company_name}?", expanded=False):
                    pe_val = _safe_get(info, 'trailingPE', None)
                    pb_val = _safe_get(info, 'priceToBook', None)
                    roe_val = _safe_get(info, 'returnOnEquity', None)
                    de_val = _safe_get(info, 'debtToEquity', None)
                    cr_val = _safe_get(info, 'currentRatio', None)
                    pm_val = _safe_get(info, 'profitMargins', None)

                    st.markdown("#### 📐 Valuation Ratios")
                    st.latex(r"P/E = \frac{\text{Market Price per Share}}{\text{Earnings per Share (EPS)}}")
                    if pe_val and isinstance(pe_val, (int, float)):
                        if pe_val < 15:
                            st.markdown(f"📊 **{company_name} P/E = {pe_val:.1f}** → Undervalued relative to the broader market. Could signal a value opportunity or declining earnings expectations.")
                        elif pe_val < 25:
                            st.markdown(f"📊 **{company_name} P/E = {pe_val:.1f}** → Fairly valued. Moderate growth expectations are priced in.")
                        else:
                            st.markdown(f"📊 **{company_name} P/E = {pe_val:.1f}** → Premium valuation. The market expects strong future earnings growth from this company.")

                    st.latex(r"P/B = \frac{\text{Market Price per Share}}{\text{Book Value per Share}}")
                    if pb_val and isinstance(pb_val, (int, float)):
                        if pb_val < 1:
                            st.markdown(f"📊 **{company_name} P/B = {pb_val:.2f}** → Trading below book value. Could be deep value or the market sees asset quality concerns.")
                        elif pb_val < 3:
                            st.markdown(f"📊 **{company_name} P/B = {pb_val:.2f}** → Reasonable premium to book value for a company of this quality.")
                        else:
                            st.markdown(f"📊 **{company_name} P/B = {pb_val:.2f}** → High premium — justified only if ROE is consistently above 15-20%.")

                    st.markdown("#### 💰 Profitability Ratios")
                    st.latex(r"ROE = \frac{\text{Net Income}}{\text{Shareholders' Equity}} \times 100")
                    if roe_val and isinstance(roe_val, (int, float)):
                        roe_pct = roe_val * 100
                        if roe_pct > 20:
                            st.markdown(f"📊 **{company_name} ROE = {roe_pct:.1f}%** → Excellent. The company generates strong returns on equity capital — a sign of competitive advantage.")
                        elif roe_pct > 10:
                            st.markdown(f"📊 **{company_name} ROE = {roe_pct:.1f}%** → Healthy. Above the typical cost of equity (~10-12%).")
                        else:
                            st.markdown(f"📊 **{company_name} ROE = {roe_pct:.1f}%** → Below average. May struggle to create shareholder value over time.")

                    st.latex(r"\text{Net Margin} = \frac{\text{Net Income}}{\text{Revenue}} \times 100")
                    if pm_val and isinstance(pm_val, (int, float)):
                        pm_pct = pm_val * 100
                        verdict = 'Strong pricing power & operational efficiency' if pm_pct > 15 else 'Moderate margins — watch for compression' if pm_pct > 5 else 'Thin margins — vulnerable to cost pressures'
                        st.markdown(f"📊 **{company_name} Net Margin = {pm_pct:.1f}%** → {verdict}.")

                    st.markdown("#### 🏗️ Leverage & Liquidity")
                    st.latex(r"D/E = \frac{\text{Total Debt}}{\text{Shareholders' Equity}}")
                    if de_val and isinstance(de_val, (int, float)):
                        if de_val < 50:
                            st.markdown(f"📊 **{company_name} D/E = {de_val:.0f}%** → Conservative leverage. Low default risk, strong borrowing capacity.")
                        elif de_val < 150:
                            st.markdown(f"📊 **{company_name} D/E = {de_val:.0f}%** → Moderate leverage. Monitor interest coverage and debt maturity profile.")
                        else:
                            st.markdown(f"📊 **{company_name} D/E = {de_val:.0f}%** → High leverage. Earnings sensitivity to interest rates is significant.")

                    st.latex(r"\text{Current Ratio} = \frac{\text{Current Assets}}{\text{Current Liabilities}}")
                    if cr_val and isinstance(cr_val, (int, float)):
                        if cr_val > 2:
                            st.markdown(f"📊 **{company_name} Current Ratio = {cr_val:.2f}** → Very liquid. Can comfortably meet all short-term obligations, though may be underutilizing capital.")
                        elif cr_val > 1:
                            st.markdown(f"📊 **{company_name} Current Ratio = {cr_val:.2f}** → Healthy liquidity position. Short-term solvency is not a concern.")
                        else:
                            st.markdown(f"📊 **{company_name} Current Ratio = {cr_val:.2f}** → ⚠️ Below 1.0 — potential liquidity risk. May need to raise capital or restructure debt.")

                with st.expander("📋 Balance Sheet", expanded=False):
                    key_bs = ['Total Assets', 'Total Debt', 'Stockholders Equity', 'Cash And Cash Equivalents']
                    bs_data = {k: balance_sheet.loc[k] for k in key_bs if k in balance_sheet.index}
                    if bs_data:
                        bs_df = pd.DataFrame(bs_data).T / 1e7
                        bs_df.columns = [str(d.year) for d in bs_df.columns]
                        st.dataframe(bs_df.style.format("{:,.0f} Cr"), use_container_width=True)

                with st.expander("💵 Cash Flow", expanded=False):
                    key_cf = ['Operating Cash Flow', 'Free Cash Flow', 'Capital Expenditure']
                    cf_data = {k: cashflow.loc[k] for k in key_cf if k in cashflow.index}
                    if cf_data:
                        cf_df = pd.DataFrame(cf_data).T / 1e7
                        cf_df.columns = [str(d.year) for d in cf_df.columns]
                        st.dataframe(cf_df.style.format("{:,.0f} Cr"), use_container_width=True)
            else:
                st.info("Financial statements not available for this ticker.")
        except Exception as e:
            st.warning(f"Could not load financial statements: {e}")

    # ============================================================
    # TAB 4: ML PREDICTIONS
    # ============================================================
    with tab_ml:
        st.markdown("### 🔮 ML Model Insights")
        st.markdown("""
        > *The ML engine uses a LightGBM model trained on historical technical features to predict
        > 5-day returns. The combined score blends prediction confidence (40%), technical signals (40%),
        > and sentiment (20%) into a single 0–100 score.*
        """)

        try:
            predictions = ml.predict(df_feat)
            if predictions is not None and not predictions.empty:
                scores = ml.calculate_combined_score(df_feat, predictions)

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Combined Score", f"{scores['combined_score']:.0f}/100")
                sc2.metric("Prediction Score", f"{scores['prediction_score']:.0f}/100")
                sc3.metric("Technical Score", f"{scores['technical_score']:.0f}/100")
                sc4.metric("Predicted 5d Return", f"{scores['predicted_return_pct']:+.2f}%")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=scores['combined_score'],
                    title={'text': "AI Combined Score", 'font': {'size': 16}},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#818cf8"},
                           'steps': [{'range': [0, 30], 'color': "rgba(239,68,68,0.2)"},
                                     {'range': [30, 60], 'color': "rgba(234,179,8,0.2)"},
                                     {'range': [60, 100], 'color': "rgba(34,197,94,0.2)"}]}
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark")
                st.plotly_chart(fig_gauge, use_container_width=True)

                cs = scores['combined_score']
                if cs >= 70:
                    st.success(f"🟢 **BULLISH** — Score **{cs:.0f}/100** indicates strong positive momentum across all signals.")
                elif cs >= 40:
                    st.warning(f"🟡 **NEUTRAL** — Score **{cs:.0f}/100** reflects mixed conditions. Wait for clarity.")
                else:
                    st.error(f"🔴 **BEARISH** — Score **{cs:.0f}/100** suggests caution. Consider reducing exposure.")

                with st.expander("📊 Score Breakdown"):
                    st.dataframe(pd.DataFrame({
                        "Component": ["Prediction (40%)", "Technical (40%)", "Sentiment (20%)"],
                        "Score": [scores['prediction_score'], scores['technical_score'],
                                  scores.get('sentiment_score', 50)],
                        "Weight": ["40%", "40%", "20%"],
                    }), use_container_width=True, hide_index=True)
            else:
                st.info("ML model not available. Train it with `train_models.py`.")
        except Exception as e:
            st.warning(f"ML prediction unavailable: {e}")

    # ============================================================
    # TAB 5: NEWS & AI VERDICT
    # ============================================================
    with tab_news:
        st.markdown("### 📰 Top News — AI Impact Analysis")
        st.markdown(f"""
        > *Top engagement news for **{company_name}**, ranked by source authority and recency.
        > Each article is analyzed by AI for market impact and directional bias.*
        """)

        with st.spinner("Fetching and analyzing news..."):
            try:
                articles = fetch_top_news(selected_company, max_articles=5)
                if articles:
                    # Enrich with full article text via crawl4ai
                    try:
                        from modules.data.scrapers.news_analysis import enrich_articles_with_crawl4ai
                        articles = enrich_articles_with_crawl4ai(articles, max_articles=3)
                    except Exception:
                        pass  # crawl4ai optional — continue with title-only
                    articles = analyze_news_with_ai(articles, selected_company)
                render_news_tiles(articles)
            except Exception as e:
                st.warning(f"News analysis unavailable: {e}")

        # AI Verdict — AUTO-TRIGGER on stock selection (no button needed)
        st.markdown("---")
        st.markdown("### 🧠 AI Verdict")

        if query_deepseek_reasoner:
            # Build stock summary
            risk = calculate_risk_metrics(df)
            summary = (
                f"Stock: {company_name} ({symbol}) | Sector: {sector}\n"
                f"Price: ₹{price:,.2f} | Change: {pct:+.2f}%\n"
                f"P/E: {_safe_get(info, 'trailingPE', 'N/A')} | P/B: {_safe_get(info, 'priceToBook', 'N/A')}\n"
                f"RSI: {_fmt_metric(rsi, '.1f')} | MACD: {'Bullish' if not pd.isna(macd_diff) and macd_diff > 0 else 'Bearish'}\n"
                f"VaR 95%: {risk.get('VaR_95',0)*100:.2f}% | Sharpe: {risk.get('Sharpe',0):.2f}\n"
                f"GARCH 1d: {garch_vol:.2f}% | Kelly: {risk.get('Kelly_Half',0)*100:.1f}%"
            )

            analysis = None
            try:
                from modules.ai_prefetch import get_cached_analysis, prefetch_stock_analysis
                analysis = get_cached_analysis(selected_company)
                if not analysis:
                    with st.spinner("🧠 AI analyzing all dimensions..."):
                        analysis = prefetch_stock_analysis(selected_company, summary)
            except Exception:
                pass

            if analysis and isinstance(analysis, dict):
                # ── Quick Verdict ──
                v = analysis.get('quick_verdict', {})
                action = v.get('action', 'N/A')
                conviction = v.get('conviction', 'N/A')
                action_colors = {"BUY": "#4ade80", "SELL": "#f87171", "HOLD": "#fbbf24"}
                ac = action_colors.get(str(action).upper(), "#94a3b8")
                
                st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin:8px 0;">'
                            f'<span style="background:{ac};color:#0f172a;padding:6px 18px;border-radius:6px;'
                            f'font-weight:800;font-size:1.1rem;">{action}</span>'
                            f'<span style="color:#94a3b8;font-size:0.9rem;">Conviction: <b style="color:#e2e8f0;">'
                            f'{conviction}/10</b></span></div>', unsafe_allow_html=True)
                
                thesis = v.get('thesis', '')
                if thesis:
                    st.markdown(f"*{thesis}*")
                
                verdict_summary = v.get('summary', '')
                if verdict_summary:
                    st.markdown(verdict_summary)
                
                # Risks, targets, position
                col_a, col_b = st.columns(2)
                with col_a:
                    risks = v.get('risks', [])
                    if risks:
                        st.markdown("**⚠️ Key Risks:**")
                        for r in risks:
                            st.markdown(f"- {r}")
                with col_b:
                    target = v.get('target_3m', '')
                    sl = v.get('stop_loss', '')
                    pos = v.get('position_size', '')
                    if target or sl or pos:
                        st.markdown("**📊 Position Guidance:**")
                        if target: st.markdown(f"- Target (3M): {target}")
                        if sl: st.markdown(f"- Stop Loss: {sl}")
                        if pos: st.markdown(f"- Position Size: {pos}")
                
                # ── Deep Report ──
                dr = analysis.get('deep_report', {})
                if any(dr.values()):
                    st.markdown("---")
                    st.markdown("#### 📋 Deep Report")
                    for section_key, section_title in [
                        ('executive_summary', '📌 Executive Summary'),
                        ('technical_analysis', '📈 Technical Analysis'),
                        ('fundamental_analysis', '💰 Fundamental Analysis'),
                        ('risk_assessment', '🛡️ Risk Assessment'),
                        ('sector_outlook', '🏭 Sector Outlook')
                    ]:
                        content = dr.get(section_key, '')
                        if content:
                            with st.expander(section_title, expanded=(section_key == 'executive_summary')):
                                st.markdown(content)
                
                # ── News Sentiment ──
                ns = analysis.get('news_sentiment', {})
                if ns and ns.get('overall'):
                    st.markdown("---")
                    overall = ns.get('overall', 'Neutral')
                    score = ns.get('score', 50)
                    sent_colors = {"Bullish": "#4ade80", "Bearish": "#f87171", "Neutral": "#fbbf24"}
                    sc = sent_colors.get(overall, "#94a3b8")
                    st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin:8px 0;">'
                                f'<span style="font-size:0.85rem;color:#94a3b8;">📰 News Sentiment:</span>'
                                f'<span style="color:{sc};font-weight:700;font-size:1rem;">{overall}</span>'
                                f'<span style="color:#64748b;font-size:0.82rem;">({score}/100)</span></div>',
                                unsafe_allow_html=True)
                    themes = ns.get('key_themes', [])
                    if themes:
                        st.markdown("**Key Themes:** " + " · ".join(themes))
                    impact = ns.get('impact_summary', '')
                    if impact:
                        st.markdown(impact)
            else:
                # Fallback — direct single call
                with st.spinner("🧠 AI analyzing..."):
                    try:
                        verdict = query_deepseek_reasoner(
                            "You are a CIO. Give a concise BUY/HOLD/SELL verdict with conviction (1-10), "
                            "2-line thesis, 3 risks, position size, stop-loss, and 3-month target.", summary)
                        st.markdown(verdict)
                    except Exception as e2:
                        st.error(f"AI analysis unavailable: {e2}")
        else:
            st.info("AI verdict requires `agentic_backend.py` with OpenRouter API key.")

    # Footer
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** Automated analysis for educational purposes only. Not financial advice.")
