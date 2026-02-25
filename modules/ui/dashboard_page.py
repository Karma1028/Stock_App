import streamlit as st
import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from modules.data.manager import StockDataManager
from modules.data.scrapers.news_scraper import NewsScraper
from modules.ui.chart_config import format_market_cap, clean_ticker_label

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None


def _get_market_status():
    """Returns market status based on Indian Standard Time (UTC+5:30)."""
    import pytz
    try:
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.datetime.now(ist)
    except Exception:
        # fallback if pytz not available
        now = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
    
    weekday = now.weekday()  # 0=Mon, 6=Sun
    t = now.time()
    
    if weekday >= 5:  # Saturday/Sunday
        return "🔴 Market Closed", "Weekend", False
    
    market_open = datetime.time(9, 15)
    market_close = datetime.time(15, 30)
    pre_market = datetime.time(9, 0)
    
    if pre_market <= t < market_open:
        return "🟡 Pre-Market", f"{now.strftime('%H:%M')} IST", False
    elif market_open <= t <= market_close:
        return "🟢 Market Open", f"{now.strftime('%H:%M')} IST", True
    else:
        return "🔴 Market Closed", f"{now.strftime('%H:%M')} IST", False


def _fetch_intraday(ticker, interval="5m"):
    """Fetch today's intraday data from yfinance."""
    try:
        df = yf.download(ticker, period="1d", interval=interval, progress=False)
        if df.empty:
            # Try 5d for non-trading day (shows last trading day)
            df = yf.download(ticker, period="5d", interval=interval, progress=False)
        # Flatten multi-index if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def render_dashboard():
    st.title("📊 Market Dashboard")
    st.markdown("> *Live market pulse — sentiment score, top performers, and curated headlines across Indian equities.*")

    # ════════════════════════════════════════════════════════════
    # LIVE INTRADAY CHART
    # ════════════════════════════════════════════════════════════
    status_label, time_label, is_market_open = _get_market_status()
    
    # Auto-refresh during market hours (every 60 seconds)
    if st_autorefresh and is_market_open:
        st_autorefresh(interval=60_000, key="live_chart_refresh")

    # Live chart ticker — use sidebar selection if available, else NIFTY 50
    live_ticker = st.session_state.get('global_ticker', '^NSEI')
    # For dashboard default, prefer NIFTY 50 if no stock was explicitly selected
    if live_ticker == 'RELIANCE.NS' and 'cache_ticker' not in st.session_state:
        live_ticker = '^NSEI'
    
    display_name = "NIFTY 50" if live_ticker == "^NSEI" else clean_ticker_label(live_ticker)
    
    # Header row: Market Status + Controls
    hdr1, hdr2, hdr3, hdr4 = st.columns([2, 0.8, 0.8, 1])
    with hdr1:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:1.4rem; font-weight:700;">{status_label}</span>
            <span style="color:#94a3b8; font-size:0.9rem;">{time_label}</span>
            <span style="color:#c4b5fd; font-weight:600; font-size:1rem; margin-left:8px;">{display_name}</span>
        </div>
        """, unsafe_allow_html=True)
    with hdr2:
        interval = st.selectbox("Interval", ["1m", "5m", "15m"], index=1, label_visibility="collapsed")
    with hdr3:
        chart_mode = st.selectbox("Chart", ["Candlestick", "Line"], index=0, label_visibility="collapsed")
    with hdr4:
        show_sma = st.checkbox("SMA₂₀", value=True, key="live_sma")
        show_ema = st.checkbox("EMA₉", value=False, key="live_ema")
        show_vwap = st.checkbox("VWAP", value=False, key="live_vwap")
    
    # Fetch intraday data
    intra_df = _fetch_intraday(live_ticker, interval)
    
    if not intra_df.empty and 'Close' in intra_df.columns:
        # Compute indicators
        if show_sma:
            intra_df['SMA20'] = intra_df['Close'].rolling(window=20).mean()
        if show_ema:
            intra_df['EMA9'] = intra_df['Close'].ewm(span=9, adjust=False).mean()
        if show_vwap and 'Volume' in intra_df.columns:
            cum_vol = intra_df['Volume'].cumsum()
            cum_tp_vol = (intra_df['Close'] * intra_df['Volume']).cumsum()
            intra_df['VWAP'] = cum_tp_vol / cum_vol

        # Build chart
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                           row_heights=[0.78, 0.22])
        
        if chart_mode == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=intra_df.index,
                open=intra_df['Open'], high=intra_df['High'],
                low=intra_df['Low'], close=intra_df['Close'],
                increasing_line_color='#22c55e', decreasing_line_color='#ef4444',
                increasing_fillcolor='#22c55e', decreasing_fillcolor='#ef4444',
                name='Price'
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=intra_df.index, y=intra_df['Close'],
                mode='lines', name='Price',
                line=dict(color='#818cf8', width=2),
                fill='tozeroy', fillcolor='rgba(129,140,248,0.08)'
            ), row=1, col=1)

        # Overlay indicators
        if show_sma and 'SMA20' in intra_df.columns:
            fig.add_trace(go.Scatter(
                x=intra_df.index, y=intra_df['SMA20'],
                mode='lines', name='SMA 20',
                line=dict(color='#f59e0b', width=1.5, dash='dot')
            ), row=1, col=1)
        if show_ema and 'EMA9' in intra_df.columns:
            fig.add_trace(go.Scatter(
                x=intra_df.index, y=intra_df['EMA9'],
                mode='lines', name='EMA 9',
                line=dict(color='#06b6d4', width=1.5)
            ), row=1, col=1)
        if show_vwap and 'VWAP' in intra_df.columns:
            fig.add_trace(go.Scatter(
                x=intra_df.index, y=intra_df['VWAP'],
                mode='lines', name='VWAP',
                line=dict(color='#a78bfa', width=1.5, dash='dash')
            ), row=1, col=1)
        
        # Volume bars
        if 'Volume' in intra_df.columns:
            colors = ['#22c55e' if c >= o else '#ef4444' 
                      for c, o in zip(intra_df['Close'], intra_df['Open'])]
            fig.add_trace(go.Bar(
                x=intra_df.index, y=intra_df['Volume'],
                marker_color=colors, opacity=0.5, name='Volume',
                showlegend=False
            ), row=2, col=1)
        
        fig.update_layout(
            height=440,
            margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            font=dict(color='#e2e8f0', size=11),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                       font=dict(size=10)),
            xaxis2=dict(showticklabels=True),
            yaxis=dict(gridcolor='rgba(148,163,184,0.1)'),
            yaxis2=dict(gridcolor='rgba(148,163,184,0.1)'),
        )
        fig.update_xaxes(gridcolor='rgba(148,163,184,0.05)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # OHLC stats bar — custom HTML (no truncation)
        last_close = float(intra_df['Close'].iloc[-1])
        day_open = float(intra_df['Open'].iloc[0])
        day_high = float(intra_df['High'].max())
        day_low = float(intra_df['Low'].min())
        day_vol = int(intra_df['Volume'].sum()) if 'Volume' in intra_df.columns else 0
        change_pct = ((last_close - day_open) / day_open * 100) if day_open != 0 else 0
        
        def _fmt_vol(v):
            if v >= 1e7: return f"{v/1e7:.2f} Cr"
            elif v >= 1e5: return f"{v/1e5:.1f} L"
            elif v >= 1e3: return f"{v/1e3:.1f} K"
            elif v == 0: return "—"
            return str(v)
        
        def _fmt_price(v):
            if v >= 10000: return f"₹{v:,.0f}"
            return f"₹{v:,.2f}"
        
        chg_col = "#4ade80" if change_pct >= 0 else "#f87171"
        chg_arrow = "↑" if change_pct >= 0 else "↓"
        
        def _card(label, value, extra=""):
            return (f'<div style="flex:1;background:rgba(30,41,59,0.7);border:1px solid rgba(148,163,184,0.2);'
                    f'border-radius:8px;padding:10px 14px;text-align:center;min-width:0;">'
                    f'<div style="color:#94a3b8;font-size:0.72rem;letter-spacing:0.5px;margin-bottom:4px;">{label}</div>'
                    f'<div style="color:#e2e8f0;font-size:1.15rem;font-weight:700;white-space:nowrap;">{value}</div>'
                    f'{extra}</div>')
        
        chg_html = f'<div style="color:{chg_col};font-size:0.75rem;margin-top:2px;">{chg_arrow} {change_pct:+.2f}%</div>'
        
        ohlc_html = ('<div style="display:flex;gap:10px;margin:8px 0 12px 0;">'
                     + _card("Open", _fmt_price(day_open))
                     + _card("High", _fmt_price(day_high))
                     + _card("Low", _fmt_price(day_low))
                     + _card("Close", _fmt_price(last_close), chg_html)
                     + _card("Volume", _fmt_vol(day_vol))
                     + '</div>')
        st.markdown(ohlc_html, unsafe_allow_html=True)
    else:
        st.info(f"📊 No intraday data available for **{display_name}**. Market may be closed or data delayed.")

    st.markdown("---")

    # Initialize Data Manager
    dm = StockDataManager()
    
    # Optimized/Cached Data Fetching
    @st.cache_data(ttl=3600) # Cache for 1 hour
    def get_dashboard_data():
        return {
            "sentiment": dm.get_market_sentiment(),
            "gainers": dm.get_top_gainers(limit=5),
            "stock_count": len(dm.get_stock_list())
        }

    # Load Data
    with st.spinner("Analyzing market data..."):
        dash_data = get_dashboard_data()

    # --- Important Metrics Section ---
    col1, col2, col3, col4 = st.columns(4)
    
    market_sentiment = dash_data['sentiment']
    sentiment_color = market_sentiment.get('color', 'white')
    stock_list_len = dash_data['stock_count']
    
    with col1:
        st.markdown(f"""
        <div class="st-card">
            <div class="metric-value">{stock_list_len}+</div>
            <div class="metric-label">Stocks Tracked</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="st-card">
            <div class="metric-value" style="background: {sentiment_color}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{market_sentiment.get('score', 50):.0f}/100</div>
            <div class="metric-label">Market Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        # Dynamic GARCH volatility metric
        try:
            # We can try to get Volatility from specific major index or just placeholder
            # Reusing logic from original file
            import yfinance as yf
            _nifty = yf.download('^NSEI', period='3mo', progress=False)
            if hasattr(_nifty.columns, 'levels') and _nifty.columns.nlevels > 1:
                _nifty.columns = _nifty.columns.get_level_values(0)
            
            if not _nifty.empty:
                _nifty_vol = float(_nifty['Close'].pct_change().std() * (252**0.5) * 100)
                st.markdown(f"""
                <div class="st-card">
                    <div class="metric-value">{_nifty_vol:.1f}%</div>
                    <div class="metric-label">NIFTY Volatility</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                 raise Exception("Empty Nifty Data")
        except Exception:
            st.markdown(f"""
            <div class="st-card">
                <div class="metric-value">—</div>
                <div class="metric-label">NIFTY Volatility</div>
            </div>
            """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="st-card">
            <div class="metric-value">{datetime.datetime.now().strftime('%H:%M')}</div>
            <div class="metric-label">Last Updated</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Market Sentiment & Top Gainers ---
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("<h3>📊 Market Sentiment</h3>", unsafe_allow_html=True)
        
        # Display Sentiment Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = market_sentiment.get('score', 50),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"<b>{market_sentiment.get('status', 'Neutral')}</b>", 'font': {'size': 24, 'color': 'white'}},
            gauge = {
                'axis': {'range': [0, 100], 'tickcolor': "white"},
                'bar': {'color': sentiment_color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': "rgba(239, 68, 68, 0.4)"},  # Red
                    {'range': [40, 60], 'color': "rgba(245, 158, 11, 0.4)"}, # Yellow
                    {'range': [60, 100], 'color': "rgba(16, 185, 129, 0.4)"}], # Green
            }
        ))
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Insight**: {market_sentiment.get('summary', 'Data unavailable')}")

    with c_right:
        st.subheader("📈 Top 5 Performing Stocks")
        gainer_data = dash_data['gainers']
        if not gainer_data:
            st.warning("Could not fetch top gainers at this moment.")
        else:
            st.markdown('<div class="st-card" style="padding: 10px 20px;">', unsafe_allow_html=True)
            for stock in gainer_data:
                sym = clean_ticker_label(stock['symbol'])
                pct = stock['change_pct']
                price = stock['price']
                st.markdown(f"""
                <div class="stock-row">
                    <span style="font-weight: 700; font-size: 1.1em; color: #f8fafc;">{sym}</span>
                    <div style="text-align: right;">
                        <div style="font-weight: 700; color: #10b981;">+{pct:.2f}%</div>
                        <div style="font-size: 0.85em; color: #94a3b8;">₹{price:,.2f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
                    
    st.markdown("---")

    # --- Top News Section ---
    st.subheader("📰 Top Market News")
    
    with st.spinner("Curating latest headlines..."):
        @st.cache_data(ttl=1800)
        def get_cached_news():
            _ns = NewsScraper()
            n1_df = _ns.fetch_news_history("RELIANCE.NS", days=2)
            n2_df = _ns.fetch_news_history("HDFCBANK.NS", days=2)
            combined_news = pd.concat([n1_df, n2_df], ignore_index=True)
            if not combined_news.empty:
                combined_news = combined_news.sort_values('date', ascending=False).head(6)
                return combined_news.to_dict('records')
            return []

        raw_news = get_cached_news()
        
        if not raw_news:
            st.info("No major market news found right now.")
        else:
            row1 = st.columns(3)
            row2 = st.columns(3)
            for i, article in enumerate(raw_news):
                col = row1[i] if i < 3 else row2[i-3] if i < 6 else None
                if col:
                    with col:
                        pub_str = article['date'].strftime('%d %b %H:%M') if isinstance(article['date'], datetime.datetime) else str(article.get('date'))
                        st.markdown(f"""
                        <div class="news-card">
                            <div style="font-size: 0.8em; color: #94a3b8; margin-bottom: 5px;">{pub_str} | {article.get('source', 'News')}</div>
                            <div style="font-weight: 600; font-size: 1em; line-height: 1.4; margin-bottom: 10px;">{article['title'][:80]}...</div>
                            <a href="{article['link']}" target="_blank" style="font-size: 0.85em;">Read Full Article →</a>
                        </div>
                        """, unsafe_allow_html=True)

    # --- Stock Screener / Ticker Lookup ---
    st.markdown("---")
    st.subheader("🔍 Stock Screener — Ticker Lookup")
    st.markdown("> *Search by ticker or company name. Copy-paste any ticker into the analysis pages.*")

    try:
        import os
        equity_path = os.path.join("data", "EQUITY.csv")
        if os.path.exists(equity_path):
            equity_df = pd.read_csv(equity_path)
            equity_df.columns = equity_df.columns.str.strip()  # Fix leading spaces in column names
            
            # Search filter
            search_query = st.text_input("🔎 Search (ticker or company name)", placeholder="e.g., RELIANCE, Tata Motors, HDFC...")
            
            if search_query:
                mask = (
                    equity_df['SYMBOL'].str.contains(search_query.upper(), na=False) |
                    equity_df['NAME OF COMPANY'].str.contains(search_query, case=False, na=False)
                )
                filtered = equity_df[mask].head(20)
            else:
                # Show curated top stocks (commonly analyzed)
                top_tickers = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'HINDUNILVR',
                              'ITC', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'LT', 'AXISBANK',
                              'BAJFINANCE', 'MARUTI', 'TITAN', 'WIPRO', 'HCLTECH', 'NTPC',
                              'POWERGRID', 'TATASTEEL']
                filtered = equity_df[equity_df['SYMBOL'].isin(top_tickers)]
            
            if not filtered.empty:
                display_df = filtered[['SYMBOL', 'NAME OF COMPANY', 'SERIES', 'ISIN NUMBER']].copy()
                display_df.columns = ['Ticker', 'Company Name', 'Series', 'ISIN']
                display_df['NSE Ticker'] = display_df['Ticker'] + '.NS'
                display_df = display_df.reset_index(drop=True)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                        "Company Name": st.column_config.TextColumn("Company Name", width="large"),
                        "Series": st.column_config.TextColumn("Series", width="small"),
                        "ISIN": st.column_config.TextColumn("ISIN", width="medium"),
                        "NSE Ticker": st.column_config.TextColumn("Use in App", width="medium"),
                    }
                )
                st.caption(f"Showing {len(display_df)} of {len(equity_df)} stocks. Use the NSE Ticker (e.g., RELIANCE.NS) in the analysis pages.")
            else:
                st.info(f"No stocks found matching '{search_query}'. Try a different search term.")
        else:
            st.warning("EQUITY.csv not found in data directory. Ticker lookup unavailable.")
    except Exception as e:
        st.warning(f"Screener unavailable: {e}")

    st.markdown("---")
    st.info("👉 Use the sidebar to navigate to **Stock Analysis** for deep dives or **Portfolio Analysis** to manage your investments.")
    st.markdown("""
    <div class="footer">
        © 2026 Institutional Quant Terminal • PyTorch + LangGraph + Streamlit
    </div>
    """, unsafe_allow_html=True)
