
import streamlit as st
import plotly.graph_objects as go

def show_sidebar():
    """Stock Analysis page controls — rendered inside sidebar.
    Persists selection in session_state so all pages share the same ticker."""

    # Dynamic Ticker Selection
    search_mode = st.sidebar.checkbox("Free Search Mode", value=False, help="Type any ticker symbol directly.")

    if search_mode:
        default_val = st.session_state.get('global_ticker', 'RELIANCE.NS')
        selected_company = st.sidebar.text_input("Enter Ticker", value=default_val).upper()
    else:
        try:
            import pandas as pd
            import os

            @st.cache_data
            def load_tickers():
                csv_path = os.path.join(os.path.dirname(__file__), '../../data/EQUITY.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df['Ticker'] = df['SYMBOL'].apply(lambda x: f"{x}.NS" if not str(x).endswith('.NS') else x)
                    df['Display'] = df['Ticker'] + " - " + df['NAME OF COMPANY']
                    return df
                return None

            df_tickers = load_tickers()
            if df_tickers is not None:
                all_tickers = df_tickers['Display'].tolist()
                prev_ticker = st.session_state.get('global_ticker', 'RELIANCE.NS')
                default_idx = next((i for i, s in enumerate(all_tickers) if prev_ticker in s), 0)
                selected_display = st.sidebar.selectbox("Select Asset", all_tickers, index=default_idx)
                selected_company = selected_display.split(" - ")[0]
            else:
                st.warning("Ticker list not found.")
                companies = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "TATAMOTORS.NS"]
                selected_company = st.sidebar.selectbox("Select Asset", companies)
        except Exception as e:
             st.error(f"Error loading tickers: {e}")
             selected_company = st.sidebar.text_input("Enter Ticker", value="RELIANCE.NS").upper()

    # Time Period
    period = st.sidebar.select_slider("Time Horizon", options=["1mo", "3mo", "6mo", "1y", "5y", "max"], value="1y")

    # Chart Settings
    st.sidebar.markdown("#### ⚙️ Chart Settings")
    chart_type = st.sidebar.radio("Type", ["Line", "Candlestick"], horizontal=True)
    show_technicals = st.sidebar.checkbox("Show Technicals (SMA/EMA)", value=True)

    # AI Engine & Tier Selection
    st.sidebar.markdown("#### 🧠 AI Engine")
    
    # Tier selector — controls which AI backend to use
    tier_options = ["⚡ Auto (Smart Cascade)", "🌐 OpenRouter Only", "🚀 Groq Only", "💻 LM Studio (Local)"]
    tier_choice = st.sidebar.selectbox(
        "AI Tier",
        tier_options,
        index=3,  # Local LLM as default main source
        help="Auto = tries OpenRouter → Groq → LM Studio. Or force one tier."
    )
    # Map to a simple string for the backend
    tier_map = {
        tier_options[0]: "auto",
        tier_options[1]: "openrouter",
        tier_options[2]: "groq",
        tier_options[3]: "lmstudio",
    }
    st.session_state["ai_tier"] = tier_map[tier_choice]

    from config import Config
    model_list = Config.AI_MODEL_FALLBACKS
    model_names = [m.split("/")[-1].replace(":free", "") for m in model_list]
    default_model_idx = next((i for i, m in enumerate(model_list) if "deepseek" in m.lower()), 0)
    ai_model = st.sidebar.selectbox("Default Model", model_names, index=default_model_idx)

    if tier_choice == "💻 LM Studio (Local)":
        default_lm_url = st.session_state.get("lm_studio_url", "http://localhost:1234")
        custom_lm_url = st.sidebar.text_input("Local LLM Base URL", value=default_lm_url)
        st.session_state["lm_studio_url"] = custom_lm_url

    # ── Custom API Key ──
    with st.sidebar.expander("🔑 Add Your Own API Key", expanded=False):
        PROVIDERS = {
            "OpenRouter": "https://openrouter.ai/api/v1",
            "OpenAI": "https://api.openai.com/v1",
            "DeepSeek": "https://api.deepseek.com/v1",
            "Groq": "https://api.groq.com/openai/v1",
            "Anthropic": "https://api.anthropic.com/v1",
            "xAI (Grok)": "https://api.x.ai/v1",
        }
        provider = st.selectbox("Provider", list(PROVIDERS.keys()), index=0)
        custom_key = st.text_input("API Key", type="password", placeholder="sk-...")
        if custom_key:
            st.session_state.custom_api_key = custom_key
            st.session_state.custom_api_provider = provider
            st.session_state.custom_api_base_url = PROVIDERS[provider]
            st.success(f"✅ {provider} key active")
        elif 'custom_api_key' in st.session_state and not custom_key:
            for k in ['custom_api_key', 'custom_api_provider', 'custom_api_base_url']:
                st.session_state.pop(k, None)

    # ════════════════════════════════════════════════════════
    # API USAGE MONITOR v2 — powered by global file store
    # ════════════════════════════════════════════════════════
    st.sidebar.markdown("#### 📊 API Monitor")
    
    # Load global stats
    try:
        from modules.api_stats_store import load_stats
        stats = load_stats()
    except Exception:
        stats = {"total_calls": 0, "successful_calls": 0, "failed_calls": 0,
                 "input_tokens": 0, "output_tokens": 0, "keys_exhausted": 0,
                 "last_model": "None", "call_history": [], "model_usage": {}}

    total_keys = len(getattr(Config, 'OPENROUTER_API_KEYS', []))
    keys_alive = total_keys - stats.get('keys_exhausted', 0)
    
    # ── Sparkline: token usage over recent calls ──
    call_history = stats.get('call_history', [])
    if len(call_history) >= 2:
        tokens_series = [c.get('tokens', 0) for c in call_history]
        spark = go.Figure(go.Scatter(
            y=tokens_series, mode='lines+markers',
            line=dict(color='#818cf8', width=2),
            marker=dict(size=3, color='#a5b4fc'),
            fill='tozeroy', fillcolor='rgba(129,140,248,0.15)',
            hovertemplate='%{y:,} tokens<extra></extra>'
        ))
        spark.update_layout(
            height=70, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            showlegend=False,
        )
        st.sidebar.plotly_chart(spark, use_container_width=True, key="api_spark")

    # ── Model usage donut ──
    model_usage = stats.get('model_usage', {})
    if model_usage:
        labels = list(model_usage.keys())[:5]  # top 5
        values = [model_usage[l] for l in labels]
        donut = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.55,
            textinfo='label+percent', textposition='outside',
            textfont=dict(size=9, color='#e2e8f0'),
            marker=dict(colors=['#818cf8', '#22c55e', '#f59e0b', '#06b6d4', '#f87171']),
            hovertemplate='%{label}: %{value} calls<extra></extra>'
        ))
        donut.update_layout(
            height=130, margin=dict(l=5, r=5, t=5, b=5),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            annotations=[dict(text=f"{stats['total_calls']}", x=0.5, y=0.5,
                            font_size=16, font_color='#e2e8f0', showarrow=False)]
        )
        st.sidebar.plotly_chart(donut, use_container_width=True, key="api_donut")
    
    # ── Stats text (compact) ──
    st.sidebar.markdown(f"""
    <div style="background:rgba(30,41,59,0.95); border:1px solid rgba(129,140,248,0.4);
         border-radius:10px; padding:12px; font-size:0.8rem; line-height:1.8;">
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#e2e8f0;">🔗 Calls</span>
            <span style="color:#fff; font-weight:700;">{stats['total_calls']}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#e2e8f0;">✅ / ❌</span>
            <span><span style="color:#4ade80; font-weight:700;">{stats['successful_calls']}</span> / <span style="color:#f87171; font-weight:700;">{stats['failed_calls']}</span></span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#e2e8f0;">📥📤 Tokens</span>
            <span style="color:#a5b4fc; font-weight:600;">{stats['input_tokens']:,} / {stats['output_tokens']:,}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#e2e8f0;">🔑 Keys</span>
            <span style="color:{'#4ade80' if keys_alive > 1 else '#f87171'}; font-weight:700;">{keys_alive}/{total_keys}</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
            <span style="color:#e2e8f0;">🤖 Model</span>
            <span style="color:#c4b5fd; font-weight:600;">{stats.get('last_model', 'None')[:22]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Quota Reset Countdown (OpenRouter resets at midnight UTC) ──
    import datetime
    now_utc = datetime.datetime.utcnow()
    midnight_utc = (now_utc + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    remaining = midnight_utc - now_utc
    hours_left = int(remaining.total_seconds() // 3600)
    mins_left = int((remaining.total_seconds() % 3600) // 60)
    pct_elapsed = ((24 * 3600 - remaining.total_seconds()) / (24 * 3600)) * 100
    
    st.sidebar.markdown(f"""
    <div style="background:rgba(30,41,59,0.95); border:1px solid rgba(99,102,241,0.3);
         border-radius:8px; padding:10px; margin-top:8px; font-size:0.78rem;">
        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
            <span style="color:#e2e8f0;">⏱ Quota Reset</span>
            <span style="color:#fbbf24; font-weight:700;">{hours_left}h {mins_left}m</span>
        </div>
        <div style="background:rgba(148,163,184,0.2); border-radius:4px; height:6px; overflow:hidden;">
            <div style="background:linear-gradient(90deg, #818cf8, #6366f1);
                 width:{pct_elapsed:.1f}%; height:100%; border-radius:4px;
                 transition:width 0.3s;"></div>
        </div>
        <div style="color:#64748b; font-size:0.68rem; margin-top:4px; text-align:right;">
            resets at 00:00 UTC ({(now_utc + datetime.timedelta(hours=5, minutes=30)).replace(hour=5, minute=30).strftime('05:30')} IST)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Persist in session_state for cross-page sharing ──
    st.session_state.global_ticker = selected_company
    st.session_state.global_period = period

    # Invalidate cache if ticker changed
    if st.session_state.get('cache_ticker') != selected_company:
        st.session_state.cached_df = None
        st.session_state.cached_info = None
        st.session_state.cached_df_feat = None
        st.session_state.cache_ticker = None

    # ════════════════════════════════════════════════════════
    # 📋 API CALL LOG — real-time, today only, auto-reset midnight
    # ════════════════════════════════════════════════════════
    try:
        from modules.api_call_log import read_today_logs, get_log_summary
        
        # Determine filter based on selected tier
        current_tier = st.session_state.get("ai_tier", "auto")
        provider_filter = "all"
        log_title = "📋 API Call Log"
        if current_tier == "lmstudio":
            provider_filter = "lmstudio"
            log_title = "💻 Local LLM Log"
        elif current_tier == "groq":
            provider_filter = "groq"
            log_title = "🚀 Groq API Log"
        elif current_tier == "openrouter":
            provider_filter = "openrouter"
            log_title = "🌐 OpenRouter Log"
            
        log_summary = get_log_summary(provider=provider_filter)
        total_logged = log_summary["total"]

        with st.sidebar.expander(f"{log_title}  [{total_logged}]", expanded=False):
            if total_logged == 0:
                st.caption("No API calls logged today yet.")
            else:
                # Summary strip — no indentation to avoid markdown code-block
                summary_html = (
                    '<div style="background:rgba(30,41,59,0.9);border-radius:8px;padding:8px 12px;'
                    'font-size:0.75rem;margin-bottom:8px;display:flex;gap:10px;flex-wrap:wrap;">'
                    f'<span style="color:#4ade80;">✅ {log_summary["ok"]}</span>'
                    f'<span style="color:#fbbf24;">⏳ {log_summary["rate_limited"]} rate-ltd</span>'
                    f'<span style="color:#f87171;">🔐 {log_summary["auth_errors"]} auth</span>'
                    f'<span style="color:#a5b4fc;">🪙 {log_summary["input_tokens"]:,}→{log_summary["output_tokens"]:,} tok</span>'
                    '</div>'
                )
                st.markdown(summary_html, unsafe_allow_html=True)

                logs = read_today_logs(limit=50, provider=provider_filter)
                STATUS_ICONS = {
                    "success": "✅", "rate_limited": "⏳",
                    "auth_error": "🔐", "error": "❌", "empty": "⚪",
                }
                STATUS_COLORS = {
                    "success": "#4ade80", "rate_limited": "#fbbf24",
                    "auth_error": "#f87171", "error": "#f87171", "empty": "#94a3b8",
                }

                # Build table rows with NO leading whitespace
                rows = []
                for entry in logs:
                    s = entry.get("status", "error")
                    icon = STATUS_ICONS.get(s, "❓")
                    sc = STATUS_COLORS.get(s, "#e2e8f0")
                    tok = entry.get("total_tokens", 0)
                    lat = entry.get("latency_ms", 0)
                    lat_s = f"{lat/1000:.1f}s" if lat >= 1000 else f"{lat}ms"
                    tok_s = f"{tok:,}" if tok > 0 else "—"
                    err = entry.get("error", "")
                    row = (
                        f'<tr style="border-bottom:1px solid rgba(255,255,255,0.04);">'
                        f'<td style="padding:3px 5px;color:#94a3b8;font-size:.68rem;white-space:nowrap;">{entry.get("time","")}</td>'
                        f'<td style="padding:3px 4px;color:#a5b4fc;font-size:.67rem;">{entry.get("key","—")}</td>'
                        f'<td style="padding:3px 4px;font-size:.67rem;max-width:90px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{entry.get("model_full","")}">'
                        f'<span style="color:#c4b5fd;">{entry.get("model","—")[:16]}</span></td>'
                        f'<td style="padding:3px 4px;color:{sc};font-weight:700;font-size:.72rem;">{icon}</td>'
                        f'<td style="padding:3px 4px;color:#818cf8;font-size:.67rem;text-align:right;">{tok_s}</td>'
                        f'<td style="padding:3px 4px;color:#64748b;font-size:.66rem;text-align:right;">{lat_s}</td>'
                        f'</tr>'
                    )
                    if err:
                        row += f'<tr><td colspan="6" style="padding:0 5px 4px 5px;color:#94a3b8;font-size:.65rem;">{err[:50]}</td></tr>'
                    rows.append(row)

                table_html = (
                    '<div style="max-height:300px;overflow-y:auto;border-radius:8px;'
                    'background:rgba(15,23,42,0.9);border:1px solid rgba(129,140,248,0.2);">'
                    '<table style="width:100%;border-collapse:collapse;">'
                    '<thead><tr style="border-bottom:1px solid rgba(129,140,248,0.3);position:sticky;top:0;background:rgba(15,23,42,0.98);">'
                    '<th style="padding:5px;color:#475569;font-size:.63rem;text-align:left;">Time</th>'
                    '<th style="padding:5px 4px;color:#475569;font-size:.63rem;text-align:left;">Key</th>'
                    '<th style="padding:5px 4px;color:#475569;font-size:.63rem;text-align:left;">Model</th>'
                    '<th style="padding:5px 4px;color:#475569;font-size:.63rem;">St.</th>'
                    '<th style="padding:5px 4px;color:#475569;font-size:.63rem;text-align:right;">Tok</th>'
                    '<th style="padding:5px 4px;color:#475569;font-size:.63rem;text-align:right;">Lat</th>'
                    '</tr></thead>'
                    '<tbody>' + ''.join(rows) + '</tbody>'
                    '</table></div>'
                    '<div style="color:#334155;font-size:.63rem;margin-top:5px;text-align:right;">'
                    'Today\'s log · auto-resets at midnight UTC</div>'
                )
                st.markdown(table_html, unsafe_allow_html=True)
    except Exception:
        pass

    return selected_company, period, chart_type, show_technicals, ai_model

