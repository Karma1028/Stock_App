import streamlit as st

def render_landing_page():
    # ── HERO SECTION ──
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Institutional Quant Terminal</div>
        <div class="hero-subtitle">
            Harness the power of institutional-grade analytics, AI predictions, and
            quantitative strategies — all in one platform built for Indian equities.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── FEATURE CARDS ──
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Real-Time Dashboard</h3>
            <p style="color:#94a3b8;">Track market pulse, top gainers, and sector performance with live data.</p>
            <div style="margin-top:12px; font-size:0.8rem; color:#64748b;">→ Navigate to <b>Dashboard</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3>AI Deep Reports</h3>
            <p style="color:#94a3b8;">9-section institutional analysis: technicals, financials, peer comparison, ML scoring, and AI verdict.</p>
            <div style="margin-top:12px; font-size:0.8rem; color:#64748b;">→ Navigate to <b>Deep Report</b></div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3>Quant Backtester</h3>
            <p style="color:#94a3b8;">Backtest Momentum, Mean Reversion, and Volatility Breakout strategies across any stock universe.</p>
            <div style="margin-top:12px; font-size:0.8rem; color:#64748b;">→ Navigate to <b>Quant Engine</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── HOW IT WORKS — Pipeline Steps ──
    st.markdown("### ⚙️ How It Works")
    st.markdown("> *From raw market data to actionable insights — the pipeline behind every analysis.*")

    s1, s2, s3, s4 = st.columns(4)
    steps = [
        ("1", "📡 Data Ingest", "Live prices, fundamentals, and news via yfinance + Google RSS"),
        ("2", "🧮 Feature Engineering", "50+ technical indicators, volatility regimes, GARCH models"),
        ("3", "🤖 ML Scoring", "LightGBM + ensemble signals produce a 0–100 conviction score"),
        ("4", "📋 Verdict", "AI synthesizes everything into a BUY / HOLD / SELL recommendation"),
    ]
    for col, (num, title, desc) in zip([s1, s2, s3, s4], steps):
        with col:
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-number">{num}</div>
                <h4 style="color:white; margin:0 0 6px;">{title}</h4>
                <p style="font-size:0.82rem; color:#94a3b8; margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── WHY CHOOSE ──
    col_info, col_stats = st.columns([3, 2])
    with col_info:
        st.markdown("### 🏛️ Why Choose This Terminal?")
        st.markdown("""
- **Institutional Grade** — Built with Python, PyTorch, Plotly — the same stack used by hedge funds.
- **Data-Driven** — All insights backed by rigorous quantitative analysis, not opinion.
- **Transparent** — We show the probability, risk, and logic behind every AI prediction.
- **Holistic** — Combines technicals, fundamentals, news sentiment, and macro factors.
        """)

    with col_stats:
        st.markdown("### 📈 Platform Stats")
        ps1, ps2 = st.columns(2)
        ps1.metric("Stocks Covered", "1,973+", help="NSE EQUITY.csv universe")
        ps2.metric("AI Models", "3", help="LightGBM, GARCH, DeepSeek-R1")
        ps3, ps4 = st.columns(2)
        ps3.metric("Strategies", "4", help="Momentum, Mean Reversion, Vol Breakout, AI Composite")
        ps4.metric("Indicators", "50+", help="RSI, MACD, Bollinger, ATR, Kelly, VaR, etc.")

    # ── FOOTER ──
    st.markdown("""
    <div class="footer">
        Institutional Quant Terminal v3.0 • Built with Streamlit + Plotly + PyTorch<br>
        © 2026 • For educational and research purposes only
    </div>
    """, unsafe_allow_html=True)
