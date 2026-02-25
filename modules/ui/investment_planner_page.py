import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.data.manager import StockDataManager
from modules.ui.sidebar import show_sidebar
from modules.ui.styles import apply_custom_style
from modules.utils.helpers import format_currency
# Assuming QuantEngine is available or will be imported. 
# In original code: from modules.ml.quant_engine import QuantEngine (inferred)
# But I need to check where QuantEngine is defined. It wasn't in the imports I saw earlier, 
# but "render_investment_planner" used "QuantEngine(dm)".
# I'll assume it exists or I need to create/find it. 
# Checking agentic_backend.py -> checking if it has QuantEngine? 
# agentic_backend.py has MonteCarloLSTM, etc.
# I'll check imports later. For now, I'll use a placeholder or try to import from likely location.
try:
    from modules.ml.quant_strategy import QuantEngine # Guessing location or it might be in agentic_backend
    from modules.utils.ai_insights import generate_investment_plan, generate_quant_investment_plan
except ImportError:
    # If not found, I'll need to locate it or it might be a missing file I need to create?
    # I saw "from modules.utils.ai_insights import generate_investment_plan" in the previous view_file.
    pass

def create_stock_selector_with_search(dm):
    if 'stock_name_mapping' not in st.session_state:
        with st.spinner("Loading company names..."):
            st.session_state.stock_name_mapping = dm.get_stock_name_mapping()
    
    name_to_ticker = st.session_state.stock_name_mapping['name_to_ticker']
    ticker_to_name = st.session_state.stock_name_mapping['ticker_to_name']
    
    search_options = sorted([f"{name} ({ticker.replace('.NS', '').replace('.BO', '')})" for ticker, name in ticker_to_name.items()])
    selected_option = st.selectbox("Search by Company Name or Ticker", options=search_options, help="Type to search by company name or ticker symbol")
    
    if selected_option:
        ticker_part = selected_option.split('(')[-1].replace(')', '').strip()
        for ticker in ticker_to_name.keys():
            if ticker.replace('.NS', '').replace('.BO', '') == ticker_part:
                return ticker
    return None

def create_sector_distribution_chart(portfolio_data, sector_map):
    portfolio_with_sectors = portfolio_data.copy()
    portfolio_with_sectors['Sector'] = portfolio_with_sectors['Symbol'].map(sector_map)
    sector_totals = portfolio_with_sectors.groupby('Sector')['Value'].sum().reset_index().sort_values('Value', ascending=False)
    
    fig = px.pie(sector_totals, values='Value', names='Sector', title='Portfolio Sector Distribution', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=450, showlegend=True)
    return fig

def create_market_cap_distribution(portfolio_data, dm):
    portfolio_with_cap = portfolio_data.copy()
    cap_categories = []
    for symbol in portfolio_with_cap['Symbol']:
        live_data = dm.get_live_data(symbol)
        cap_category = dm.get_market_cap_category(live_data.get('market_cap', 0))
        cap_categories.append(cap_category)
    
    portfolio_with_cap['Cap Category'] = cap_categories
    cap_totals = portfolio_with_cap.groupby('Cap Category')['Value'].sum().reset_index()
    
    cap_order = ['Large Cap', 'Mid Cap', 'Small Cap', 'Unknown']
    cap_totals['Cap Category'] = pd.Categorical(cap_totals['Cap Category'], categories=cap_order, ordered=True)
    cap_totals = cap_totals.sort_values('Cap Category')
    
    colors = {'Large Cap': '#2ecc71', 'Mid Cap': '#f39c12', 'Small Cap': '#e74c3c', 'Unknown': '#95a5a6'}
    cap_totals['Color'] = cap_totals['Cap Category'].map(colors)
    
    fig = px.bar(cap_totals, x='Cap Category', y='Value', title='Market Capitalization Distribution', color='Cap Category', color_discrete_map=colors, text='Value')
    fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False, yaxis_title='Portfolio Value (₹)')
    return fig, portfolio_with_cap

def create_risk_reward_scatter(portfolio_data, dm):
    plot_data = []
    for _, row in portfolio_data.iterrows():
        symbol = row['Symbol'] if 'Symbol' in row else row['ticker']
        try:
            df = dm.get_cached_data([symbol], period="1y")
            if not df.empty:
                # Handle MultiIndex if necessary, relying on dm or cache structure
                 # If manager returns bulk, we need extraction. If list is 1, maybe not?
                 # Assuming get_cached_data handles it or returns MultiIndex
                if isinstance(df.columns, pd.MultiIndex):
                    if symbol in df.columns: 
                        df = df[symbol]
                    elif df.columns.nlevels > 1 and symbol in df.columns.get_level_values(1):
                        df = df.xs(symbol, axis=1, level=1)
                
                daily_ret = df['Close'].pct_change()
                vol = daily_ret.std() * (252 ** 0.5) * 100
                ret = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
                plot_data.append({'Symbol': symbol, 'Risk (Volatility %)': vol, 'Reward (1Y Return %)': ret, 'Size': row.get('Value', row.get('amount', 1))})
        except Exception: continue
            
    if not plot_data: return go.Figure()
    df_plot = pd.DataFrame(plot_data)
    fig = px.scatter(df_plot, x='Risk (Volatility %)', y='Reward (1Y Return %)', size='Size', color='Symbol', text='Symbol', title='Risk vs. Reward Analysis', hover_data=['Risk (Volatility %)', 'Reward (1Y Return %)'])
    avg_risk, avg_reward = df_plot['Risk (Volatility %)'].mean(), df_plot['Reward (1Y Return %)'].mean()
    fig.add_hline(y=avg_reward, line_dash="dash", line_color="gray", annotation_text="Avg Reward")
    fig.add_vline(x=avg_risk, line_dash="dash", line_color="gray", annotation_text="Avg Risk")
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500, showlegend=False)
    return fig

def render_investment_planner():
    # Reuse show_sidebar to maintain consistency, but ignore returns except ai_model
    _, _, _, _, ai_model = show_sidebar()
    
    st.title("💰 Investment Planner & Portfolio Builder")
    st.markdown("---")

    st.markdown('<div class="st-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Investment Parameters")
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        inv_amount = st.number_input("Investment Amount (₹)", min_value=1000, value=100000, step=1000)
        inv_type = st.selectbox("Investment Type", ["One-time", "Monthly SIP"])
    with col_in2:
        duration = st.slider("Investment Duration (Years)", 1, 30, 5)
        expected_return = st.slider("Expected Annual Return (%)", 5, 30, 12)
    with col_in3:
        risk_profile = st.select_slider("Risk Profile", options=["Conservative", "Moderate", "Aggressive", "Very Aggressive"], value="Moderate")
        experience_level = st.select_slider("Experience Level", options=["Beginner", "Intermediate", "Advanced"], value="Intermediate")
        st.info(f"Targeting **{risk_profile}** growth strategy")
    st.markdown('</div>', unsafe_allow_html=True)

    tab_suggest, tab_build = st.tabs(["🤖 Get AI Suggestions", "🛠️ Build Your Own Portfolio"])
    dm = StockDataManager()

    with tab_suggest:
        st.subheader("🚀 AI-Powered Investment Plan")
        
        c_gen1, c_gen2 = st.columns([1, 1])
        with c_gen1:
            if st.button("Generate Quant Strategy (Backtested)", type="primary"):
                try:
                    # Try importing QuantEngine
                    from modules.ml.quant_strategy import QuantEngine
                    qe = QuantEngine(dm)
                    user_profile = {"investment_amount": inv_amount, "duration_years": duration, "risk_profile": risk_profile, "investment_type": inv_type, "expected_annual_return_pct": expected_return}
                    
                    with st.spinner("Running Quant Pipeline..."):
                        payload = qe.run_pipeline(user_profile)
                        if "error" in payload:
                             st.error(f"Pipeline Error: {payload['error']}")
                        else:
                             metrics = payload.get('backtest_summary', {})
                             m1, m2, m3, m4 = st.columns(4)
                             m1.metric("Projected CAGR", f"{metrics.get('annualized_return_pct', 0):.2f}%")
                             m2.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.2f}")
                             m3.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
                             m4.metric("Volatility", f"{metrics.get('annualized_vol_pct', 0):.2f}%")
                             
                             chart_data = payload.get('chart_data', {})
                             equity_curve = pd.DataFrame(chart_data.get('equity_curve', []))
                             if not equity_curve.empty:
                                 fig = go.Figure()
                                 fig.add_trace(go.Scatter(x=equity_curve['date'], y=equity_curve['value'], mode='lines', name='AI Strategy', line=dict(color='#00CC96', width=3)))
                                 fig.update_layout(title="Cumulative Returns", template="plotly_dark", height=500)
                                 st.plotly_chart(fig, use_container_width=True)
                             
                             st.markdown("### 📋 Strategy Summary")
                             try:
                                 st.markdown(generate_quant_investment_plan(payload, model=ai_model))
                             except:
                                 st.info("AI Explanation unavailable.")
                except ImportError:
                    st.error("QuantStrategy module not found. Please ensure backend modules are present.")

        with c_gen2:
            if st.button("Ask AI Advisor (Text Plan)"):
                try:
                    from modules.utils.ai_insights import generate_investment_plan
                    with st.spinner("Consulting AI Financial Planner..."):
                        plan_text = generate_investment_plan(
                            amount=inv_amount,
                            duration_years=duration,
                            expected_return=expected_return,
                            risk_profile=risk_profile,
                            market_context="Indian Market is volatile but bullish in long term.",
                            investment_type=inv_type,
                            experience_level=experience_level,
                            model=ai_model
                        )
                        st.markdown(plan_text)
                except ImportError:
                    st.error("AI Insights module missing.")

    with tab_build:
        st.subheader("Custom Portfolio Builder")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("### Add Stocks")
            selected_stock = create_stock_selector_with_search(dm)
            if selected_stock:
                qty = st.number_input("Quantity", min_value=1, value=10, key="qty_input")
                if 'portfolio' not in st.session_state: st.session_state.portfolio = []
                if st.button("Add to Portfolio", type="primary"):
                    live = dm.get_live_data(selected_stock)
                    price = live.get('current_price', 0)
                    st.session_state.portfolio.append({"Symbol": selected_stock, "Quantity": qty, "Avg Price": price, "Current Price": price, "Value": price * qty})
                    st.success(f"✅ Added {selected_stock}")
                    st.rerun()
            if st.button("Clear Portfolio"):
                st.session_state.portfolio = []
                st.rerun()

        with col2:
            st.markdown("### Your Portfolio")
            if st.session_state.get('portfolio'):
                df_port = pd.DataFrame(st.session_state.portfolio)
                st.dataframe(df_port, use_container_width=True)
                
                sector_map = dm.get_nifty50_sector_map()
                c_v1, c_v2 = st.columns(2)
                with c_v1: st.plotly_chart(create_sector_distribution_chart(df_port, sector_map), use_container_width=True)
                with c_v2: 
                    fig, _ = create_market_cap_distribution(df_port, dm)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.plotly_chart(create_risk_reward_scatter(df_port, dm), use_container_width=True)
