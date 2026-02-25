
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import yfinance as yf
from modules.ui.chart_config import COLORS, clean_ticker_label

def load_data_dynamic(ticker):
    """
    Loads data for a given ticker.
    1. Checks 'data/processed' for a clean CSV.
    2. If not found, downloads from yfinance.
    """
    # 1. Try Local Map (Faster, Pre-processed)
    file_map = {
        "TATAMOTORS": "tata_motors_clean.csv",
        "MARUTI": "maruti_suzuki_clean.csv",
        "M&M": "mahindra_and_mahindra_clean.csv",
        "NIFTY50": "nifty_50_clean.csv"
    }
    
    filename = file_map.get(ticker.upper())
    if filename:
        path = os.path.join("data", "processed", filename)
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=['Date'], index_col='Date')

    # 2. Fallback to Yahoo Finance
    try:
        # Add .NS if not present and it looks like an Indian stock (simple heuristic or user responsibility)
        # For now, we assume user might type 'RELIANCE' => 'RELIANCE.NS'
        # But let's try as is first.
        yf_ticker = ticker if "." in ticker else f"{ticker}.NS"
        df = yf.download(yf_ticker, period="5y", progress=False)
        
        if df.empty:
             # Try without .NS (maybe US stock)
             df = yf.download(ticker, period="5y", progress=False)
             
        if not df.empty:
             # Cleaning for consistency
             if isinstance(df.columns, pd.MultiIndex):
                # Flatten or extract Close
                # yfinance often returns (Price, Ticker)
                try:
                    df = df.xs(df.columns.get_level_values(1)[0], axis=1, level=1)
                except:
                    pass
             return df
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        
    return None

def render_deep_dive_page():
    st.title("📐 EDA Lab")
    st.markdown("> *Exploratory Data Analysis, peer correlation, and AI-driven narrative for any asset on NSE.*")

    # ── USE GLOBAL SIDEBAR TICKER ──
    from modules.ui.sidebar import show_sidebar
    selected_company, period, chart_type, show_technicals, ai_model = show_sidebar()
    user_ticker = selected_company

    # ── Check session_state cache first ──
    cache_hit = (
        st.session_state.get('cache_ticker') == user_ticker
        and st.session_state.get('cached_df') is not None
    )

    if cache_hit:
        df = st.session_state.cached_df
    else:
        df = load_data_dynamic(user_ticker)
        if df is not None and not df.empty:
            # Cache for other pages
            st.session_state.cached_df = df
            st.session_state.cache_ticker = user_ticker
    
    if df is None or df.empty:
        st.error(f"Could not load data for **{user_ticker}**. Please check the symbol.")
        return

    display_name = clean_ticker_label(user_ticker)
    st.success(f"Loaded **{len(df)}** trading days for **{display_name}**.")

    # Store computed values at module scope for cross-tab access
    computed = {'kurt': None, 'skew': None, 'corr_df': None, 'peers_list': []}

    tab1, tab2, tab3 = st.tabs(["📊 Distribution & EDA", "⚖️ Peer Comparison", "🧠 AI Narrative"])

    with tab1:
        st.subheader(f"Statistical Distributions: {display_name}")
        st.markdown("> *Returns distribution reveals tail risk. High kurtosis means more extreme price swings than a normal distribution predicts.*")
        
        # Returns Distribution
        if 'Close' in df.columns:
            df['Returns'] = df['Close'].pct_change()
            
            fig = px.histogram(df, x='Returns', nbins=100, title=f"{display_name} Daily Returns Distribution",
                               marginal="box", opacity=0.7, color_discrete_sequence=[COLORS['histogram']])
            st.plotly_chart(fig, use_container_width=True)
            
            # Kurtosis & Skew
            kurt = df['Returns'].kurt()
            skew = df['Returns'].skew()
            computed['kurt'] = kurt
            computed['skew'] = skew
            
            c1, c2 = st.columns(2)
            c1.metric("Kurtosis (Fat Tails)", f"{kurt:.2f}", help=">3 indicates fat tails")
            c2.metric("Skewness (Direction)", f"{skew:.2f}", help="Negative = left tail risk")
            
            if kurt > 3:
                st.info("💡 High Kurtosis indicates frequent extreme price moves (fat tails). Risk management is crucial.")
            
            # Rolling Volatility
            df['Vol_30'] = df['Returns'].rolling(30).std() * (252**0.5)
            fig_vol = px.line(df, y='Vol_30', title="Annualized Rolling Volatility (30-Day)")
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # ── AI Statistical Interpretation ──
            st.markdown("---")
            st.markdown("### 🤖 AI Statistical Interpretation")
            
            # Compute summary metrics for AI prompt
            avg_ret = df['Returns'].mean() * 252 * 100  # annualized %
            vol_now = df['Vol_30'].iloc[-1] * 100 if not pd.isna(df['Vol_30'].iloc[-1]) else 0
            max_daily_loss = df['Returns'].min() * 100
            max_daily_gain = df['Returns'].max() * 100
            
            # Auto-generate with AI (token-optimized)
            ai_success = False
            try:
                from agentic_backend import query_deepseek_reasoner
                
                compact_data = (
                    f"{display_name}: Kurt={kurt:.2f}, Skew={skew:.2f}, "
                    f"AnnRet={avg_ret:.1f}%, CurrVol={vol_now:.1f}%, "
                    f"MaxLoss={max_daily_loss:.1f}%, MaxGain={max_daily_gain:.1f}%"
                )
                
                ai_prompt = (
                    "You are a quant analyst. In exactly 4 bullet points (max 20 words each), "
                    "interpret these EDA statistics for a retail investor. "
                    "Cover: tail risk, trend bias, volatility regime, and actionable insight."
                )
                
                from agentic_backend import stream_deepseek_reasoner
                
                # Streaming UI containers
                box_html_head = f"""
                <div style="background:rgba(30,41,59,0.6); border-left:3px solid #818cf8; 
                     padding:16px 20px; border-radius:8px; margin:10px 0;">
                    <div style="color:#a5b4fc; font-size:0.78rem; margin-bottom:8px;">
                        🤖 AI INTERPRETATION — {display_name}
                    </div>
                    <div style="color:#e2e8f0; font-size:0.88rem; line-height:1.7;">
                """
                box_html_tail = """
                    </div>
                </div>
                """
                
                st.markdown("#### 🧠 Live AI Analysis")
                
                try:
                    # 1. Provide an expander container for the live thought process
                    status_container = st.status("🧠 **AI is thinking...**", expanded=True)
                    thought_placeholder = status_container.empty()
                    
                    # 2. Provide a main container for the final text
                    main_placeholder = st.empty()
                    
                    thinking_buf = ""
                    content_buf = ""
                    
                    for chunk in stream_deepseek_reasoner(ai_prompt, compact_data):
                        ctype = chunk.get("type")
                        cdelta = chunk.get("delta", "")
                        
                        if ctype == "reasoning":
                            thinking_buf += cdelta
                            thought_placeholder.markdown(f"*{thinking_buf}*")
                        elif ctype == "content":
                            # Once content starts arriving, collapse the thinking box
                            if thinking_buf and status_container.state == "running":
                                status_container.update(label="🧠 **Thought Process Complete**", state="complete", expanded=False)
                            
                            content_buf += cdelta
                            main_placeholder.markdown(box_html_head + content_buf + box_html_tail, unsafe_allow_html=True)
                    
                    # Final cleanup if the stream ended while still "running"
                    if status_container.state == "running":
                        if thinking_buf:
                            status_container.update(label="🧠 **Thought Process Complete**", state="complete", expanded=False)
                        else:
                            # If it never thought at all, just hide it
                            status_container.empty()

                    if content_buf and not content_buf.startswith("[AI Error]"):
                        ai_success = True
                    else:
                        st.caption("⚠️ AI models temporarily unavailable — showing rule-based analysis below.")
                except Exception as e:
                    st.caption(f"⚠️ AI backend stream failed: {e}")
            except Exception as e:
                st.caption(f"⚠️ AI backend unavailable — showing rule-based analysis below.")

            # Always show rule-based insights as baseline
            st.markdown("#### 📋 Statistical Summary")
            insights = []
            if kurt > 5:
                insights.append("⚠️ **Extreme fat tails** — this stock experiences outsized price swings far beyond normal expectations. Use wider stop-losses.")
            elif kurt > 3:
                insights.append("📊 **Moderate fat tails** — occasional large moves are expected; factor this into position sizing.")
            else:
                insights.append("✅ **Near-normal distribution** — price movements are relatively well-behaved and predictable.")
            
            if skew < -0.5:
                insights.append("📉 **Negative skew** — the stock has a tendency toward sharp drops (left-tail risk). Consider protective puts.")
            elif skew > 0.5:
                insights.append("📈 **Positive skew** — right-tail events (rallies) are more frequent than deep crashes.")
            else:
                insights.append("⚖️ **Symmetric returns** — no significant directional bias in the distribution.")
            
            if vol_now > 35:
                insights.append(f"🔴 **High volatility regime** ({vol_now:.0f}%) — reduce position size to ≤2% of portfolio per trade.")
            elif vol_now > 20:
                insights.append(f"🟡 **Normal volatility** ({vol_now:.0f}%) — standard risk management applies; 3-5% position sizing.")
            else:
                insights.append(f"🟢 **Low volatility** ({vol_now:.0f}%) — favorable for trend-following and momentum strategies.")
            
            insights.append(f"📋 **Annualized return: {avg_ret:+.1f}%** | Worst day: {max_daily_loss:.1f}% | Best day: {max_daily_gain:+.1f}%")
            
            st.markdown("\n\n".join(insights))
            
        else:
            st.error("Close price column missing.")

    with tab2:
        st.subheader("Sector Peer Comparison")
        
        # Automated Peer Discovery
        suggested_peers = []
        try:
            # 1. Try to get sector from live_data
            if 'Sector' not in df.columns:
                 # Fetch info if not already available
                 t = yf.Ticker(user_ticker)
                 info = t.info
                 sector = info.get('sector')
                 industry = info.get('industry')
            
            # Simple Indian Market Sector Map (Fallback)
            sector_map = {
                "Automobile": ["MARUTI.NS", "M&M.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
                "Technology": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
                "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
                "Finance": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "CHOLAFIN.NS"],
                "Consumer": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "TITAN.NS"],
                "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS"],
                "Metals": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS"]
            }
            
            # Heuristic match
            if sector:
                st.info(f"Identified Sector: **{sector}**")
                for k, v in sector_map.items():
                    if k.lower() in sector.lower():
                        suggested_peers = [p for p in v if p != user_ticker and p.replace('.NS','') != user_ticker]
            
            # Fallback if no sector match but specific tickers
            if not suggested_peers:
                if 'TATA' in user_ticker: suggested_peers = ["MARUTI.NS", "M&M.NS", "ASHOKLEY.NS"]
                elif 'HDFC' in user_ticker: suggested_peers = ["ICICIBANK.NS", "SBIN.NS"]
                elif 'INFY' in user_ticker or 'TCS' in user_ticker: suggested_peers = ["HCLTECH.NS", "WIPRO.NS"]
                
        except Exception as e:
            print(f"Peer automation error: {e}")

        # Default string for input
        default_peers = ",".join(suggested_peers) if suggested_peers else "MARUTI.NS,M&M.NS"
        
        st.write("Compare with (Auto-detected Peers):")
        
        # Dynamic Peer Selection
        peers_input = st.text_input("Peer Tickers (comma separated)", value=default_peers)
        peers_list = [p.strip().upper() for p in peers_input.split(",") if p.strip()]
        
        if peers_list:
            peer_dfs = {user_ticker: df}
            valid_peers = []
            
            progress_bar = st.progress(0)
            for i, p in enumerate(peers_list):
                pdf = load_data_dynamic(p)
                if pdf is not None:
                    # Ensure same currency/scale if possible, but for now just normalize
                    peer_dfs[p] = pdf
                    valid_peers.append(p)
                progress_bar.progress((i + 1) / len(peers_list))
            
            if len(peer_dfs) > 1:
                # Align dates
                common_idx = df.index
                for p in valid_peers:
                    common_idx = common_idx.intersection(peer_dfs[p].index)
                
                if len(common_idx) > 0:
                    fig_peer = go.Figure()
                    
                    # Normalize
                    for name, pdf in peer_dfs.items():
                        series = pdf.loc[common_idx]['Close']
                        # Handle potential zero division
                        start_val = series.iloc[0]
                        if start_val > 0:
                            norm = series / start_val * 100
                            fig_peer.add_trace(go.Scatter(x=common_idx, y=norm, name=name))
                    
                    fig_peer.update_layout(
                        title="Relative Performance (Rebased to 100)", 
                        yaxis_title="Normalized Return (%)",
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_peer, use_container_width=True)
                    
                    # Correlation
                    st.write("### Correlation Matrix")
                    corr_data = {name: pdf.loc[common_idx]['Close'] for name, pdf in peer_dfs.items()}
                    corr_df = pd.DataFrame(corr_data).corr()
                    computed['corr_df'] = corr_df
                    
                    fig_corr = px.imshow(corr_df, text_auto=True, color_continuous_scale='RdBu_r', title="Price Correlation")
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.warning("No overlapping dates found between selected assets.")
            else:
                st.warning("Could not load peer data.")

    with tab3:
        st.subheader("🧠 The AI Narrative")
        st.markdown(f"> *Ask **DeepSeek-R1** to synthesize a compelling market story for **{display_name}** and its peers.*")
        
        if st.button("Generate Analytical Narrative", type="primary"):
            with st.spinner("Synthesizing data into a story..."):
                try:
                    from agentic_backend import query_deepseek_reasoner
                    
                    peers_str = ', '.join(computed.get('peers_list', []) or ['N/A'])
                    prompt = f"""
                    You are a financial historian and quant analyst. 
                    Tell a compelling "Story of the Market" for {user_ticker} and its peers ({peers_str}). 
                    Explain why they moved together or diverged. Mention volatility and risk.
                    """
                    
                    # Build context from computed values (safe access)
                    kurt_val = computed.get('kurt')
                    corr_data = computed.get('corr_df')
                    context = f"""
                    {user_ticker} Kurtosis: {f'{kurt_val:.2f}' if kurt_val is not None else 'N/A'}
                    CORRELATION MATRIX:
                    {corr_data.to_string() if corr_data is not None else 'Not computed yet'}
                    """
                    
                    story = query_deepseek_reasoner(prompt, context)
                    st.markdown(story)
                except ImportError:
                    st.error("AI backend not available. Install `agentic_backend` to enable narrative generation.")
                except Exception as e:
                    st.error(f"AI Narrative failed: {e}")
