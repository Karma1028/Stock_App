
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import ta

from modules.data.manager import StockDataManager
from modules.ml.features import FeatureEngineer
from modules.ml.engine import MLEngine
from modules.ui.chart_config import COLORS, TEMPLATE, format_market_cap, clean_ticker_label
from modules.data.scrapers.news_analysis import fetch_top_news, analyze_news_with_ai, render_news_tiles

try:
    from agentic_backend import run_garch_volatility_forecast, query_deepseek_reasoner
except ImportError:
    run_garch_volatility_forecast = None
    query_deepseek_reasoner = None


# ════════════════════════════════════════════════════════════════
# SECTOR → PEER MAPPING
# ════════════════════════════════════════════════════════════════
SECTOR_PEERS = {
    "Auto": ["TATAMOTORS.NS", "MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "ASHOKLEY.NS", "EICHERMOT.NS"],
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIM.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS", "INDUSINDBK.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "ADANIGREEN.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
    "Metals": ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS"],
    "Telecom": ["BHARTIARTL.NS", "IDEA.NS"],
    "Infra": ["LT.NS", "ADANIENT.NS", "ULTRACEMCO.NS", "GRASIM.NS"],
}


def _get_sector_peers(ticker, sector_hint=""):
    """Find sector peers for the given ticker."""
    ticker_upper = ticker.upper()
    for sector, peers in SECTOR_PEERS.items():
        if ticker_upper in peers or sector.lower() in sector_hint.lower():
            return sector, [p for p in peers if p != ticker_upper][:5]
    return "Unknown", ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]


def _safe(val, fmt=".2f"):
    """Safely format a numeric value."""
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    try:
        return f"{val:{fmt}}"
    except (ValueError, TypeError):
        return str(val)


def _safe_get(info, key, default="N/A"):
    """Safely get from yfinance info dict."""
    v = info.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


def _fmt_metric(val, fmt=".2f", prefix="", suffix=""):
    """Format a metric with prefix/suffix, handling N/A gracefully."""
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    try:
        return f"{prefix}{val:{fmt}}{suffix}"
    except (ValueError, TypeError):
        return "N/A"


def _search_ticker(query):
    """Search for a ticker by company name using yfinance."""
    try:
        import re
        # First try direct ticker
        test = yf.Ticker(query)
        if test.info and test.info.get('regularMarketPrice'):
            return query

        # Try with .NS suffix
        ns_query = query.upper() + ".NS" if not query.upper().endswith(".NS") else query.upper()
        test = yf.Ticker(ns_query)
        if test.info and test.info.get('regularMarketPrice'):
            return ns_query

        # Try yfinance search
        results = yf.Search(query)
        if hasattr(results, 'quotes') and results.quotes:
            for q in results.quotes:
                symbol = q.get('symbol', '')
                if symbol.endswith('.NS') or symbol.endswith('.BO'):
                    return symbol
            return results.quotes[0].get('symbol', query)

        return query
    except Exception:
        return query.upper() + ".NS" if not query.upper().endswith(".NS") else query.upper()


def _fetch_news_fallback(ticker, company_name=""):
    """Fetch news with multiple fallback methods."""
    news_items = []

    # Method 1: yfinance .news
    try:
        yt = yf.Ticker(ticker)
        raw_news = yt.news or []
        for article in raw_news[:8]:
            title = article.get('title', '')
            if not title:
                content = article.get('content', {})
                title = content.get('title', 'No title')
            link = article.get('link', '')
            if not link:
                content = article.get('content', {})
                link = content.get('clickThroughUrl', {}).get('url', '#')
            publisher = article.get('publisher', '')
            if not publisher:
                content = article.get('content', {})
                publisher = content.get('provider', {}).get('displayName', 'News')
            if title:
                news_items.append({"title": title, "link": link, "publisher": publisher})
    except Exception:
        pass

    # Method 2: If yfinance returned nothing, try Google News RSS
    if not news_items:
        try:
            import urllib.request
            import xml.etree.ElementTree as ET
            search_term = company_name if company_name else ticker.replace(".NS", "").replace(".BO", "")
            url = f"https://news.google.com/rss/search?q={search_term}+stock&hl=en-IN&gl=IN"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            for item in root.findall('.//item')[:8]:
                title = item.find('title')
                link = item.find('link')
                source = item.find('source')
                if title is not None:
                    news_items.append({
                        "title": title.text or "No title",
                        "link": link.text if link is not None else "#",
                        "publisher": source.text if source is not None else "Google News"
                    })
        except Exception:
            pass

    # Method 3: If still nothing, provide a "no news" placeholder
    if not news_items:
        news_items.append({
            "title": f"No recent news found for {ticker}. Try searching manually on Moneycontrol or Economic Times.",
            "link": f"https://www.moneycontrol.com/india/stockpricequote/search.php?search={ticker.replace('.NS','')}",
            "publisher": "System"
        })

    return news_items


# ════════════════════════════════════════════════════════════════
# MAIN PAGE
# ════════════════════════════════════════════════════════════════
def render_robo_advisor():
    st.title("🤖 AI Stock Analyzer")
    st.markdown("""
    <p style="font-size:1.1rem; color:#94a3b8; margin-top:-10px; margin-bottom:20px;">
        Comprehensive deep-dive: Technicals • Financials • Peer Comparison • Risk Quantification • ML Predictions
    </p>
    """, unsafe_allow_html=True)

    # ── USE GLOBAL SIDEBAR TICKER ──
    from modules.ui.sidebar import show_sidebar
    selected_company, period, chart_type, show_technicals, ai_model = show_sidebar()
    ticker = selected_company

    dm = StockDataManager()
    fe = FeatureEngineer()
    ml = MLEngine()

    # ── Check session_state cache first ──
    cache_hit = (
        st.session_state.get('cache_ticker') == ticker
        and st.session_state.get('cached_df') is not None
        and st.session_state.get('cached_info') is not None
    )

    if cache_hit:
        df = st.session_state.cached_df
        info = st.session_state.cached_info
        df_feat = st.session_state.get('cached_df_feat', df.copy())
        yf_ticker = st.session_state.get('cached_yf_ticker', None)
    else:
        # Fetch fresh data
        with st.spinner(f"⏳ Building comprehensive analysis for **{ticker}** — fetching price data, financials, and computing features..."):
            df = dm.get_historical_data(ticker, period=period)
            if df.empty:
                st.error(f"❌ No data found for **{ticker}**. Check the symbol or try a different search term.")
                return

            yf_ticker = None
            try:
                yf_ticker = yf.Ticker(ticker)
                info = yf_ticker.info or {}
            except Exception:
                info = {}

            df_feat = fe._compute_single_ticker_features(df.copy())
            # Ensure sentiment_score exists for ML model
            if 'sentiment_score' not in df_feat.columns:
                df_feat['sentiment_score'] = 0.0

            st.session_state.cached_df = df
            st.session_state.cached_info = info
            st.session_state.cached_df_feat = df_feat
            st.session_state.cached_yf_ticker = yf_ticker
            st.session_state.cache_ticker = ticker

    company_name = _safe_get(info, 'longName', ticker.replace('.NS', ''))
    sector = _safe_get(info, 'sector', 'Unknown')
    industry = _safe_get(info, 'industry', 'Unknown')

    st.success(f"✅ Loaded **{len(df)} trading days** for **{company_name}** ({ticker})")

    # ════════════════════════════════════════════════════════════
    # 1. COMPANY OVERVIEW
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 1. 📊 Company Overview")
    st.markdown(f"""
    > *This section provides a bird's-eye view of **{company_name}** — its market position,
    > current valuation multiples, and where it stands relative to its 52-week range.
    > Use this as a quick health-check before diving deeper.*
    """)

    price = df['Close'].iloc[-1]
    prev = df['Close'].iloc[-2] if len(df) > 1 else price
    change_pct = (price - prev) / prev * 100
    color = "#22c55e" if change_pct >= 0 else "#ef4444"
    arrow = "▲" if change_pct >= 0 else "▼"

    st.markdown(f"""
    <div style="padding:24px; border-radius:14px; background:linear-gradient(135deg,#1e293b,#0f172a);
         border:1px solid rgba(255,255,255,0.1); margin-bottom:20px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
            <div>
                <h2 style="margin:0; color:white; font-size:1.8rem;">{company_name}</h2>
                <div style="margin-top:6px;">
                    <span style="background:rgba(129,140,248,0.15); color:#818cf8; padding:4px 10px;
                          border-radius:6px; font-size:0.85rem; font-weight:600;">{clean_ticker_label(ticker)}</span>
                    <span style="color:#94a3b8; margin-left:8px; font-size:0.9rem;">{sector} • {industry}</span>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:2.8rem; font-weight:700; color:white;">₹{price:,.2f}</div>
                <div style="color:{color}; font-size:1.3rem; font-weight:600;">{arrow} {change_pct:+.2f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row — custom HTML cards (no truncation)
    mcap = _safe_get(info, 'marketCap', None)
    pe = _fmt_metric(_safe_get(info, 'trailingPE', None), ".1f")
    pb = _fmt_metric(_safe_get(info, 'priceToBook', None), ".2f")
    high52 = _fmt_metric(_safe_get(info, 'fiftyTwoWeekHigh', None), ",.0f", "₹")
    low52 = _fmt_metric(_safe_get(info, 'fiftyTwoWeekLow', None), ",.0f", "₹")
    dv = _safe_get(info, 'dividendYield', None)
    div_str = f"{dv*100:.2f}%" if isinstance(dv, (int, float)) else "N/A"
    
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

    desc = _safe_get(info, 'longBusinessSummary', '')
    if desc and desc != "N/A":
        with st.expander("📖 About the Company", expanded=False):
            st.write(desc)

    # ════════════════════════════════════════════════════════════
    # 2. TECHNICAL ANALYSIS
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 2. 📈 Technical Analysis Dashboard")
    st.markdown(f"""
    > *Technical indicators reveal the stock's momentum, trend strength, and volatility.
    > The chart below shows price action overlaid with key moving averages and Bollinger Bands.
    > The signal table translates raw numbers into actionable buy/sell/hold signals.*
    """)

    fig_price = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                              subplot_titles=["Price Action with Moving Averages", "Volume"])
    fig_price.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                        low=df['Low'], close=df['Close'], name="OHLC"), row=1, col=1)
    if 'SMA_50' in df_feat.columns:
        fig_price.add_trace(go.Scatter(x=df_feat.index, y=df_feat['SMA_50'], name="SMA 50",
                                       line=dict(color=COLORS['sma_50'], width=1.5)), row=1, col=1)
    if 'SMA_200' in df_feat.columns:
        fig_price.add_trace(go.Scatter(x=df_feat.index, y=df_feat['SMA_200'], name="SMA 200",
                                       line=dict(color=COLORS['sma_200'], width=1.5)), row=1, col=1)
    if 'BB_High' in df_feat.columns:
        fig_price.add_trace(go.Scatter(x=df_feat.index, y=df_feat['BB_High'], name="BB Upper",
                                       line=dict(color="#94a3b8", width=0.5, dash="dot")), row=1, col=1)
        fig_price.add_trace(go.Scatter(x=df_feat.index, y=df_feat['BB_Low'], name="BB Lower",
                                       line=dict(color="#94a3b8", width=0.5, dash="dot"),
                                       fill="tonexty", fillcolor="rgba(148,163,184,0.05)"), row=1, col=1)
    if 'Volume' in df.columns:
        colors_vol = [COLORS['vol_up'] if c >= o else COLORS['vol_down'] for c, o in zip(df['Close'], df['Open'])]
        fig_price.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume",
                                   marker_color=colors_vol, opacity=0.5), row=2, col=1)
    fig_price.update_layout(height=520, template="plotly_dark", xaxis_rangeslider_visible=False,
                            margin=dict(l=0, r=0, t=40, b=0), showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_price, use_container_width=True)

    # Signal table
    st.markdown("### 🎯 Signal Interpretation")
    st.markdown("*Each indicator is categorized as Bullish 🟢, Neutral 🟡, or Bearish 🔴 to give you an at-a-glance directional bias.*")
    latest = df_feat.iloc[-1]

    signals = []
    rsi = latest.get('RSI', np.nan)
    if not pd.isna(rsi):
        if rsi > 70: sig = "🔴 Overbought"
        elif rsi > 60: sig = "🟡 Bullish"
        elif rsi > 40: sig = "🟢 Neutral"
        elif rsi > 30: sig = "🟡 Bearish"
        else: sig = "🔴 Oversold"
        signals.append({"Indicator": "RSI (14)", "Value": f"{rsi:.1f}", "Signal": sig, "Weight": "High"})

    macd_diff = latest.get('MACD_Diff', np.nan)
    if not pd.isna(macd_diff):
        sig = "🟢 Bullish Crossover" if macd_diff > 0 else "🔴 Bearish Crossover"
        signals.append({"Indicator": "MACD Histogram", "Value": f"{macd_diff:.4f}", "Signal": sig, "Weight": "High"})

    bb_h, bb_l = latest.get('BB_High', np.nan), latest.get('BB_Low', np.nan)
    if not pd.isna(bb_h) and not pd.isna(bb_l):
        if price > bb_h: sig = "🔴 Above Upper Band"
        elif price < bb_l: sig = "🟢 Below Lower Band"
        else: sig = "🟡 Within Bands"
        signals.append({"Indicator": "Bollinger Bands", "Value": f"[{bb_l:.0f} – {bb_h:.0f}]", "Signal": sig, "Weight": "Medium"})

    sma50 = latest.get('SMA_50', np.nan)
    sma200 = latest.get('SMA_200', np.nan)
    if not pd.isna(sma50) and not pd.isna(sma200) and sma200 > 0:
        sig = "🟢 Golden Cross (Bullish)" if sma50 > sma200 else "🔴 Death Cross (Bearish)"
        signals.append({"Indicator": "SMA 50/200", "Value": f"{sma50:.0f} / {sma200:.0f}", "Signal": sig, "Weight": "High"})

    vol_z = latest.get('Vol_Z', np.nan)
    if not pd.isna(vol_z):
        if abs(vol_z) > 2: sig = "🔴 Volume Spike"
        elif abs(vol_z) > 1: sig = "🟡 Elevated"
        else: sig = "🟢 Normal"
        signals.append({"Indicator": "Volume Z-Score", "Value": f"{vol_z:.2f}", "Signal": sig, "Weight": "Medium"})

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
            st.success(f"**Overall Technical Bias: BULLISH** — {bull} bullish signals vs {bear} bearish")
        elif bear > bull + 1:
            st.error(f"**Overall Technical Bias: BEARISH** — {bear} bearish signals vs {bull} bullish")
        else:
            st.warning(f"**Overall Technical Bias: NEUTRAL** — mixed ({bull} bullish, {bear} bearish)")

    # ════════════════════════════════════════════════════════════
    # 3. NEWS
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 3. 📰 News & Sentiment")
    st.markdown(f"""
    > *Latest headlines related to **{company_name}**. News sentiment drives short-term price action —
    > scan for earnings surprises, regulatory changes, or sector-wide catalysts.*
    """)

    news_items = _fetch_news_fallback(ticker, company_name)
    cols = st.columns(min(len(news_items), 3))
    for i, article in enumerate(news_items[:6]):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:rgba(30,41,59,0.7); padding:14px; border-radius:10px;
                 border:1px solid rgba(255,255,255,0.08); margin-bottom:10px; min-height:130px;">
                <div style="font-size:0.72rem; color:#64748b; margin-bottom:5px; text-transform:uppercase;
                     letter-spacing:0.5px;">{article['publisher']}</div>
                <div style="font-weight:600; font-size:0.92rem; line-height:1.45; margin-bottom:8px;
                     color:#e2e8f0;">{article['title'][:120]}</div>
                <a href="{article['link']}" target="_blank"
                   style="font-size:0.78rem; color:#38bdf8; text-decoration:none;">Read full article →</a>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════
    # 4. FINANCIAL STATEMENTS
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 4. 💰 Financial Statement Analysis")
    st.markdown(f"""
    > *Revenue growth, profitability, and balance sheet strength reveal the fundamental health of the business.
    > Look for consistent revenue growth, expanding margins, and manageable debt levels.*
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
                                                    line=dict(color=COLORS['net_margin'], width=2), mode="lines+markers"))
                    op_income = income_stmt.loc['Operating Income'] if 'Operating Income' in income_stmt.index else None
                    if op_income is not None:
                        op_margin = (op_income.values / rev_row.values * 100)
                        fig_margin.add_trace(go.Scatter(x=years, y=op_margin, name="Op Margin %",
                                                        line=dict(color=COLORS['op_margin'], width=2), mode="lines+markers"))
                    fig_margin.update_layout(title="Profit Margins (%)", template="plotly_dark",
                                            height=350, margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig_margin, use_container_width=True)

            with st.expander("📋 Balance Sheet Details", expanded=False):
                key_bs = ['Total Assets', 'Total Debt', 'Stockholders Equity', 'Cash And Cash Equivalents']
                bs_data = {k: balance_sheet.loc[k] for k in key_bs if k in balance_sheet.index}
                if bs_data:
                    bs_df = pd.DataFrame(bs_data).T / 1e7
                    bs_df.columns = [str(d.year) for d in bs_df.columns]
                    st.dataframe(bs_df.style.format("{:,.0f} Cr"), use_container_width=True)

            with st.expander("💵 Cash Flow Details", expanded=False):
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

    # ════════════════════════════════════════════════════════════
    # 5. KEY RATIOS
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 5. 📊 Key Financial Ratios")
    st.markdown("""
    > *Ratios condense financial statements into comparable metrics. Compare these against industry
    > averages and the peer table in Section 6 to gauge relative value.*
    """)

    ratio_map = {
        "Valuation": [
            ("P/E (Trailing)", 'trailingPE'), ("P/E (Forward)", 'forwardPE'),
            ("P/B Ratio", 'priceToBook'), ("EV/EBITDA", 'enterpriseToEbitda'), ("PEG", 'pegRatio')],
        "Profitability": [
            ("ROE", 'returnOnEquity'), ("ROA", 'returnOnAssets'),
            ("Profit Margin", 'profitMargins'), ("Op Margin", 'operatingMargins'), ("Gross Margin", 'grossMargins')],
        "Leverage": [
            ("Debt/Equity", 'debtToEquity'), ("Current Ratio", 'currentRatio'), ("Quick Ratio", 'quickRatio')],
    }
    rcols = st.columns(len(ratio_map))
    for i, (cat, items) in enumerate(ratio_map.items()):
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

    # --- Ratio explanations with formulas ---
    with st.expander("📖 What do these ratios mean?", expanded=False):
        pe_val = _safe_get(info, 'trailingPE', None)
        pb_val = _safe_get(info, 'priceToBook', None)
        roe_val = _safe_get(info, 'returnOnEquity', None)
        de_val = _safe_get(info, 'debtToEquity', None)
        cr_val = _safe_get(info, 'currentRatio', None)
        pm_val = _safe_get(info, 'profitMargins', None)

        st.markdown("#### Valuation Ratios")
        st.latex(r"P/E = \frac{\text{Market Price per Share}}{\text{Earnings per Share (EPS)}}")
        if pe_val and isinstance(pe_val, (int, float)):
            if pe_val < 15:
                st.markdown(f"📊 **P/E = {pe_val:.1f}** → Undervalued relative to market. May indicate value opportunity or declining earnings.")
            elif pe_val < 25:
                st.markdown(f"📊 **P/E = {pe_val:.1f}** → Fairly valued. Moderate growth expectations priced in.")
            else:
                st.markdown(f"📊 **P/E = {pe_val:.1f}** → Premium valuation. Market expects strong future earnings growth.")

        st.latex(r"P/B = \frac{\text{Market Price per Share}}{\text{Book Value per Share}}")
        if pb_val and isinstance(pb_val, (int, float)):
            if pb_val < 1:
                st.markdown(f"📊 **P/B = {pb_val:.2f}** → Trading below book value. Could be deep value or distressed.")
            elif pb_val < 3:
                st.markdown(f"📊 **P/B = {pb_val:.2f}** → Reasonable premium to book value.")
            else:
                st.markdown(f"📊 **P/B = {pb_val:.2f}** → High premium. Justified if ROE is also high.")

        st.markdown("#### Profitability Ratios")
        st.latex(r"ROE = \frac{\text{Net Income}}{\text{Shareholders' Equity}} \times 100")
        if roe_val and isinstance(roe_val, (int, float)):
            roe_pct = roe_val * 100
            if roe_pct > 20:
                st.markdown(f"📊 **ROE = {roe_pct:.1f}%** → Excellent. The company generates strong returns on equity capital.")
            elif roe_pct > 10:
                st.markdown(f"📊 **ROE = {roe_pct:.1f}%** → Healthy. Above the typical cost of equity.")
            else:
                st.markdown(f"📊 **ROE = {roe_pct:.1f}%** → Below average. May struggle to create shareholder value.")

        st.latex(r"\text{Net Margin} = \frac{\text{Net Income}}{\text{Revenue}} \times 100")
        if pm_val and isinstance(pm_val, (int, float)):
            pm_pct = pm_val * 100
            st.markdown(f"📊 **Net Margin = {pm_pct:.1f}%** → {'Strong pricing power' if pm_pct > 15 else 'Moderate margins' if pm_pct > 5 else 'Thin margins — vulnerable to cost pressures'}")

        st.markdown("#### Leverage Ratios")
        st.latex(r"D/E = \frac{\text{Total Debt}}{\text{Shareholders' Equity}}")
        if de_val and isinstance(de_val, (int, float)):
            if de_val < 50:
                st.markdown(f"📊 **D/E = {de_val:.0f}%** → Conservative leverage. Low financial risk.")
            elif de_val < 150:
                st.markdown(f"📊 **D/E = {de_val:.0f}%** → Moderate leverage. Watch interest coverage.")
            else:
                st.markdown(f"📊 **D/E = {de_val:.0f}%** → High leverage. Earnings could be significantly impacted by interest rates.")

        st.latex(r"\text{Current Ratio} = \frac{\text{Current Assets}}{\text{Current Liabilities}}")
        if cr_val and isinstance(cr_val, (int, float)):
            if cr_val > 2:
                st.markdown(f"📊 **Current Ratio = {cr_val:.2f}** → Very liquid. May be underutilizing assets.")
            elif cr_val > 1:
                st.markdown(f"📊 **Current Ratio = {cr_val:.2f}** → Healthy liquidity. Can meet short-term obligations.")
            else:
                st.markdown(f"📊 **Current Ratio = {cr_val:.2f}** → ⚠️ Below 1.0 — potential liquidity risk.")

    # ════════════════════════════════════════════════════════════
    # 6. PEER COMPARISON
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 6. 🏢 Peer Comparison")
    st.markdown(f"""
    > *How does **{company_name}** stack up against its sector rivals?
    > The table below compares key valuation and profitability metrics. The radar chart
    > visualizes relative positioning across multiple dimensions.*
    """)

    peer_sector, peer_list = _get_sector_peers(ticker, sector)
    st.caption(f"Sector: **{peer_sector}** — comparing with {len(peer_list)} peers")

    with st.spinner("Fetching peer data (this may take a moment)..."):
        peer_data = []
        compare_tickers = [ticker] + peer_list[:4]
        for t in compare_tickers:
            try:
                p_info = yf.Ticker(t).info or {}
                peer_data.append({
                    "Stock": t.replace(".NS", ""),
                    "Price": _fmt_metric(_safe_get(p_info, 'currentPrice', None), ",.0f", "₹"),
                    "Mkt Cap (Cr)": _fmt_metric(_safe_get(p_info, 'marketCap', None), ",.0f", prefix="", suffix="") if isinstance(_safe_get(p_info, 'marketCap', None), (int,float)) else "N/A",
                    "P/E": _fmt_metric(_safe_get(p_info, 'trailingPE', None), ".1f"),
                    "P/B": _fmt_metric(_safe_get(p_info, 'priceToBook', None), ".2f"),
                    "ROE": f"{_safe_get(p_info, 'returnOnEquity', 0)*100:.1f}%" if isinstance(_safe_get(p_info, 'returnOnEquity', None), (int,float)) else "N/A",
                    "D/E": _fmt_metric(_safe_get(p_info, 'debtToEquity', None), ".0f"),
                })
            except Exception:
                peer_data.append({"Stock": t.replace(".NS", ""), "Price": "—", "Mkt Cap (Cr)": "—",
                                  "P/E": "—", "P/B": "—", "ROE": "—", "D/E": "—"})

    if peer_data:
        st.dataframe(pd.DataFrame(peer_data), use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════
    # 7. RISK ASSESSMENT
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 7. 📉 Risk Assessment")
    st.markdown(f"""
    > *Risk metrics quantify the downside potential and help determine appropriate position sizing.
    > VaR tells you the worst-case daily loss; GARCH forecasts tomorrow's volatility;
    > Kelly Criterion calculates the mathematically optimal bet size.*
    """)

    returns = df['Close'].pct_change().dropna()
    var_95 = np.percentile(returns, 5) * 100
    cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
    ann_vol = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    sortino_down = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() * 252) / sortino_down if sortino_down > 0 else 0
    rolling_max = df['Close'].cummax()
    drawdown = (df['Close'] - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    # Beta
    try:
        nifty = yf.download("^NSEI", period=period, progress=False)
        if hasattr(nifty.columns, 'levels') and nifty.columns.nlevels > 1:
            nifty.columns = nifty.columns.get_level_values(0)
        nifty_ret = nifty['Close'].pct_change().dropna()
        common = returns.index.intersection(nifty_ret.index)
        if len(common) > 30:
            cov = np.cov(returns.loc[common], nifty_ret.loc[common])
            beta = cov[0, 1] / cov[1, 1]
        else:
            beta = float('nan')
    except Exception:
        beta = float('nan')

    # GARCH
    garch_vol = 0.0
    if run_garch_volatility_forecast:
        try:
            garch_vol = run_garch_volatility_forecast(returns) * 100
        except Exception:
            pass

    # Kelly
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    wl_ratio = wins.mean() / abs(losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0
    kelly = max(0, win_rate - (1 - win_rate) / wl_ratio) if wl_ratio > 0 else 0

    # Use _safe for beta — THIS FIXES THE BUG
    beta_str = f"{beta:.2f}" if not np.isnan(beta) else "N/A"

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("VaR (95%)", f"{var_95:.2f}%", help="Worst daily loss with 95% confidence")
    r2.metric("Max Drawdown", f"{max_dd:.1f}%")
    r3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    r4.metric("Beta (vs NIFTY)", beta_str)

    r5, r6, r7, r8 = st.columns(4)
    r5.metric("CVaR / ES", f"{cvar_95:.2f}%")
    r6.metric("Sortino Ratio", f"{sortino:.2f}")
    r7.metric("GARCH 1d Vol", f"{garch_vol:.2f}%")
    r8.metric("Half-Kelly", f"{kelly/2*100:.1f}%", help="Optimal position size")

    if ann_vol > 35:
        st.error(f"🔴 **HIGH VOLATILITY REGIME** ({ann_vol:.0f}% annualized) — Reduce position size, widen stops.")
    elif ann_vol > 20:
        st.warning(f"🟡 **NORMAL VOLATILITY** ({ann_vol:.0f}% annualized) — Standard position sizing applies.")
    else:
        st.success(f"🟢 **LOW VOLATILITY** ({ann_vol:.0f}% annualized) — Favorable for trend-following strategies.")

    # ════════════════════════════════════════════════════════════
    # 8. ML MODEL INSIGHTS
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 8. 🔮 ML Model Insights")
    st.markdown("""
    > *The ML engine combines a LightGBM model (trained on historical features) with technical signals
    > to produce a combined score from 0–100. Scores above 60 suggest bullish conditions; below 40 suggest caution.*
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
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': COLORS['gauge_bar']},
                       'steps': [{'range': [0, 30], 'color': COLORS['gauge_low']},
                                 {'range': [30, 60], 'color': COLORS['gauge_mid']},
                                 {'range': [60, 100], 'color': COLORS['gauge_high']}]}
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark")
            st.plotly_chart(fig_gauge, use_container_width=True)

            cs = scores['combined_score']
            if cs >= 70:
                st.success(f"🟢 **BULLISH** — Combined score of **{cs:.0f}/100** indicates strong positive momentum across technical and ML signals.")
            elif cs >= 40:
                st.warning(f"🟡 **NEUTRAL** — Combined score of **{cs:.0f}/100** reflects mixed signals. Consider waiting for a clearer directional move.")
            else:
                st.error(f"🔴 **BEARISH** — Combined score of **{cs:.0f}/100** suggests downside risk. Exercise caution with new positions.")
        else:
            st.info("ML model not available. Train the model first using `train_models.py`.")
    except Exception as e:
        st.warning(f"ML prediction unavailable: {e}")

    # ════════════════════════════════════════════════════════════
    # 9. NEWS — AI IMPACT ANALYSIS
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 9. 📰 News — AI Impact Analysis")
    st.markdown(f"""
    > *Top engagement news for **{company_name}**, ranked by source authority and recency.
    > Each article is analyzed by AI for market impact and directional bias.*
    """)

    with st.spinner("Fetching and analyzing news..."):
        try:
            news_articles = fetch_top_news(ticker, max_articles=5)
            if news_articles:
                news_articles = analyze_news_with_ai(news_articles, ticker)
            render_news_tiles(news_articles)
        except Exception as e:
            st.warning(f"News analysis unavailable: {e}")

    # ════════════════════════════════════════════════════════════
    # 10. EXECUTIVE VERDICT
    # ════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("## 10. 📋 Executive Verdict & Action Plan")
    st.markdown(f"""
    > *This section synthesizes all prior analysis into concrete, actionable recommendations.
    > Position sizing is derived from Kelly Criterion; stop-loss levels are computed from VaR and ATR.*
    """)

    # Safe value for ATR
    atr_val = atr if not pd.isna(atr) else 0.0

    # Build summary for AI (with safe beta formatting)
    summary_data = (
        f"Stock: {company_name} ({ticker})\n"
        f"Sector: {sector} | Industry: {industry}\n"
        f"Current Price: ₹{price:,.2f} | Change: {change_pct:+.2f}%\n"
        f"Market Cap: {_safe_get(info, 'marketCap', 'N/A')}\n"
        f"P/E: {_safe_get(info, 'trailingPE', 'N/A')} | P/B: {_safe_get(info, 'priceToBook', 'N/A')}\n"
        f"RSI: {_safe(rsi, '.1f')} | MACD: {'Bullish' if not pd.isna(macd_diff) and macd_diff > 0 else 'Bearish'}\n"
        f"SMA 50/200: {'Golden Cross' if not pd.isna(sma50) and not pd.isna(sma200) and sma50 > sma200 else 'Death Cross'}\n"
        f"Annualized Volatility: {ann_vol:.1f}% | GARCH 1d: {garch_vol:.2f}%\n"
        f"VaR 95%: {var_95:.2f}% | Max Drawdown: {max_dd:.1f}%\n"
        f"Sharpe: {sharpe:.2f} | Beta: {beta_str}\n"
        f"Half-Kelly Position: {kelly/2*100:.1f}%"
    )

    # Auto-Trigger AI Verdict
    if query_deepseek_reasoner:
        # Try prefetch cache first
        try:
            from modules.ai_prefetch import get_cached_analysis, prefetch_stock_analysis, display_with_animation
            cached = get_cached_analysis(st.session_state.get('global_ticker', ticker))
            if cached:
                dr = cached.get('deep_report', {})
                v = cached.get('quick_verdict', {})
                # Show deep report sections
                verdict_parts = []
                if v.get('summary'):
                    verdict_parts.append(f"### Verdict: **{v.get('action', 'N/A')}** (Conviction: {v.get('conviction', 'N/A')}/10)\n{v['summary']}")
                for section, title in [('executive_summary', '📋 Executive Summary'),
                                       ('technical_analysis', '📊 Technical Analysis'),
                                       ('risk_assessment', '⚠️ Risk Assessment'),
                                       ('sector_outlook', '🏭 Sector Outlook')]:
                    text = dr.get(section, '')
                    if text:
                        verdict_parts.append(f"### {title}\n{text}")
                full_verdict = "\n\n".join(verdict_parts) if verdict_parts else "No analysis available."
                display_with_animation(full_verdict, delay=0.4)
            else:
                with st.spinner("🧠 AI analyzing all dimensions (single call)..."):
                    analysis = prefetch_stock_analysis(st.session_state.get('global_ticker', ticker), summary_data)
                if analysis:
                    v = analysis.get('quick_verdict', {})
                    verdict_text = v.get('summary', f"**{v.get('action', 'N/A')}** — {v.get('thesis', '')}")
                    st.markdown(verdict_text)
                    
                    dr = analysis.get('deep_report', {})
                    if any(dr.values()):
                        for section, title in [('executive_summary', '📋 Executive Summary'),
                                               ('technical_analysis', '📊 Technical Analysis'),
                                               ('risk_assessment', '⚠️ Risk Assessment'),
                                               ('sector_outlook', '🏭 Sector Outlook')]:
                            text = dr.get(section, '')
                            if text:
                                st.markdown(f"### {title}\n{text}")
                else:
                    with st.spinner("DeepSeek synthesizing CIO verdict..."):
                        system = """You are a senior CIO at an institutional fund. Based on the quantitative data provided,
                        give a concise verdict: BUY/HOLD/SELL with conviction level (1-10).
                        Include: (1) 2-line thesis, (2) 3 key risks, (3) position sizing recommendation,
                        (4) stop-loss level, (5) 3-month price target range. Be specific with numbers."""
                        verdict = query_deepseek_reasoner(system, summary_data)
                        st.markdown(verdict)
        except Exception:
            with st.spinner("DeepSeek synthesizing CIO verdict..."):
                try:
                    system = """You are a senior CIO at an institutional fund. Based on the quantitative data provided,
                    give a concise verdict: BUY/HOLD/SELL with conviction level (1-10).
                    Include: (1) 2-line thesis, (2) 3 key risks, (3) position sizing recommendation,
                    (4) stop-loss level, (5) 3-month price target range. Be specific with numbers."""
                    verdict = query_deepseek_reasoner(system, summary_data)
                    st.markdown(verdict)
                except Exception as e:
                    st.error(f"AI Verdict generation failed: {e}")
    else:
        st.info("AI verdict requires `agentic_backend.py` with OpenRouter API key configured.")

    # Computed action plan
    st.markdown("### 🎯 Computed Action Plan")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
**Position Sizing:**
- Half-Kelly recommends: **{kelly/2*100:.1f}%** of portfolio
- On a ₹10L portfolio: **₹{kelly/2*1000000:,.0f}**
- {"⚠️ Reduce to 2.5% in high-vol regime" if ann_vol > 35 else "✅ Standard sizing applies"}

**Stop-Loss Levels:**
- VaR-based (statistical): **₹{price * (1 + var_95/100):.0f}** ({var_95:.1f}%)
- ATR-based (1.5×): **₹{price - 1.5 * atr_val:.0f}** (−{1.5*atr_val/price*100:.1f}%)
        """)
    with col_b:
        st.markdown(f"""
**Volatility Assessment:**
- Current regime: **{"High 🔴" if ann_vol > 35 else ("Normal 🟡" if ann_vol > 20 else "Low 🟢")}** ({ann_vol:.0f}% annualized)
- GARCH 1-day forecast: **{garch_vol:.2f}%**
- {"Trade with caution — volatility is elevated" if garch_vol > 2 else "Calm market — standard execution"}

**Review Triggers:**
- Re-analyze if price moves >₹{2*atr_val:.0f} in a day (2× ATR)
- Re-analyze on earnings dates or macro events
        """)

    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This is an automated analysis for educational purposes only. Not financial advice. Always do your own due diligence before making investment decisions.")
