import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from modules.data.manager import StockDataManager
from modules.utils.helpers import format_currency, format_large_number
from modules.ui.sidebar import show_sidebar
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Investment Planner", layout="wide")
from modules.ui.styles import apply_custom_style
apply_custom_style()

def create_sector_distribution_chart(portfolio_data, sector_map):
    """Creates a pie chart showing sector distribution of the portfolio."""
    # Add sector information to portfolio
    portfolio_with_sectors = portfolio_data.copy()
    portfolio_with_sectors['Sector'] = portfolio_with_sectors['Symbol'].map(sector_map)
    
    # Aggregate by sector
    sector_totals = portfolio_with_sectors.groupby('Sector')['Value'].sum().reset_index()
    sector_totals = sector_totals.sort_values('Value', ascending=False)
    
    fig = px.pie(
        sector_totals,
        values='Value',
        names='Sector',
        title='Portfolio Sector Distribution',
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=450, showlegend=True)
    
    return fig

def create_market_cap_distribution(portfolio_data, dm):
    """Creates a bar chart showing market cap distribution."""
    portfolio_with_cap = portfolio_data.copy()
    
    # Get market cap for each stock
    cap_categories = []
    for symbol in portfolio_with_cap['Symbol']:
        live_data = dm.get_live_data(symbol)
        market_cap = live_data.get('market_cap', 0)
        cap_category = dm.get_market_cap_category(market_cap)
        cap_categories.append(cap_category)
    
    portfolio_with_cap['Cap Category'] = cap_categories
    
    # Aggregate by cap category
    cap_totals = portfolio_with_cap.groupby('Cap Category')['Value'].sum().reset_index()
    
    # Define order
    cap_order = ['Large Cap', 'Mid Cap', 'Small Cap', 'Unknown']
    cap_totals['Cap Category'] = pd.Categorical(
        cap_totals['Cap Category'],
        categories=cap_order,
        ordered=True
    )
    cap_totals = cap_totals.sort_values('Cap Category')
    
    colors = {'Large Cap': '#2ecc71', 'Mid Cap': '#f39c12', 'Small Cap': '#e74c3c', 'Unknown': '#95a5a6'}
    cap_totals['Color'] = cap_totals['Cap Category'].map(colors)
    
    fig = px.bar(
        cap_totals,
        x='Cap Category',
        y='Value',
        title='Market Capitalization Distribution',
        color='Cap Category',
        color_discrete_map=colors,
        text='Value'
    )
    fig.update_traces(texttemplate='₹%{text:,.0f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False, yaxis_title='Portfolio Value (₹)')
    
    return fig, portfolio_with_cap

def create_projection_chart(inv_amount, duration, expected_return, inv_type):
    """Creates a multi-scenario projection chart."""
    years = list(range(duration + 1))
    
    # Calculate different scenarios
    conservative_rate = expected_return * 0.7  # 70% of expected
    expected_rate = expected_return
    optimistic_rate = expected_return * 1.3  # 130% of expected
    
    if inv_type == "One-time":
        conservative_vals = [inv_amount * ((1 + conservative_rate/100) ** y) for y in years]
        expected_vals = [inv_amount * ((1 + expected_rate/100) ** y) for y in years]
        optimistic_vals = [inv_amount * ((1 + optimistic_rate/100) ** y) for y in years]
    else:  # SIP
        # Simplified SIP calculation: FV of annuity
        monthly_inv = inv_amount
        conservative_vals = []
        expected_vals = []
        optimistic_vals = []
        
        for y in years:
            months = y * 12
            if months == 0:
                conservative_vals.append(0)
                expected_vals.append(0)
                optimistic_vals.append(0)
            else:
                # FV of annuity formula
                cons_fv = monthly_inv * (((1 + conservative_rate/1200) ** months - 1) / (conservative_rate/1200)) * (1 + conservative_rate/1200)
                exp_fv = monthly_inv * (((1 + expected_rate/1200) ** months - 1) / (expected_rate/1200)) * (1 + expected_rate/1200)
                opt_fv = monthly_inv * (((1 + optimistic_rate/1200) ** months - 1) / (optimistic_rate/1200)) * (1 + optimistic_rate/1200)
                conservative_vals.append(cons_fv)
                expected_vals.append(exp_fv)
                optimistic_vals.append(opt_fv)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years, y=conservative_vals,
        mode='lines',
        name=f'Conservative ({conservative_rate:.1f}%)',
        line=dict(color='#e74c3c', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=years, y=expected_vals,
        mode='lines+markers',
        name=f'Expected ({expected_rate:.1f}%)',
        line=dict(color='#3498db', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=years, y=optimistic_vals,
        mode='lines',
        name=f'Optimistic ({optimistic_rate:.1f}%)',
        line=dict(color='#2ecc71', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Portfolio Growth Projection (Multiple Scenarios)',
        xaxis_title='Years',
        yaxis_title='Portfolio Value (₹)',
        height=450,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig, expected_vals[-1]

def create_treemap_chart(portfolio_data, sector_map):
    """Creates a treemap showing stock weightage in portfolio."""
    portfolio_with_sectors = portfolio_data.copy()
    portfolio_with_sectors['Sector'] = portfolio_with_sectors['Symbol'].map(sector_map)
    portfolio_with_sectors['Display'] = portfolio_with_sectors['Symbol'].str.replace('.NS', '').str.replace('.BO', '')
    
    fig = px.treemap(
        portfolio_with_sectors,
        path=['Sector', 'Display'],
        values='Value',
        title='Portfolio Stock Weightage by Sector',
        color='Value',
        color_continuous_scale='Viridis',
        hover_data=['Quantity', 'Current Price']
    )
    fig.update_layout(height=500)
    
    fig.update_layout(height=500)
    
    return fig

def create_risk_reward_scatter(portfolio_data, dm):
    """
    Creates a Risk vs. Reward scatter plot for the portfolio holdings.
    Risk = Volatility (21d), Reward = 1Y Return (Momentum) or Expected Return.
    """
    # We need to fetch metrics for each stock
    plot_data = []
    
    for _, row in portfolio_data.iterrows():
        symbol = row['Symbol'] if 'Symbol' in row else row['ticker']
        # Handle ticker key difference between custom and AI portfolio
        
        # Fetch metrics
        # Ideally we use cached bulk data, but for now fetching live/cached one by one or using what we have
        # Let's use dm.calculate_kpis or fetch history
        try:
            df = dm.get_cached_data([symbol], period="1y")
            if not df.empty and isinstance(df.columns, pd.MultiIndex):
                df = df[symbol]
            
            if not df.empty:
                # Calculate Volatility (Annualized)
                daily_ret = df['Close'].pct_change()
                vol = daily_ret.std() * (252 ** 0.5) * 100
                
                # Calculate Return (1Y)
                ret = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
                
                plot_data.append({
                    'Symbol': symbol,
                    'Risk (Volatility %)': vol,
                    'Reward (1Y Return %)': ret,
                    'Size': row.get('Value', row.get('amount', 1)) # Size bubble by value
                })
        except Exception as e:
            print(f"Error metrics for {symbol}: {e}")
            continue
            
    if not plot_data:
        return go.Figure()
        
    df_plot = pd.DataFrame(plot_data)
    
    fig = px.scatter(
        df_plot,
        x='Risk (Volatility %)',
        y='Reward (1Y Return %)',
        size='Size',
        color='Symbol',
        text='Symbol',
        title='Risk vs. Reward Analysis',
        hover_data=['Risk (Volatility %)', 'Reward (1Y Return %)']
    )
    
    # Add quadrants
    avg_risk = df_plot['Risk (Volatility %)'].mean()
    avg_reward = df_plot['Reward (1Y Return %)'].mean()
    
    fig.add_hline(y=avg_reward, line_dash="dash", line_color="gray", annotation_text="Avg Reward")
    fig.add_vline(x=avg_risk, line_dash="dash", line_color="gray", annotation_text="Avg Risk")
    
    fig.update_traces(textposition='top center')
    fig.update_layout(height=500, showlegend=False)
    
    return fig

def create_stock_selector_with_search(dm):
    """Creates an enhanced stock selector with company name search."""
    # Get stock names if not cached
    if 'stock_name_mapping' not in st.session_state:
        with st.spinner("Loading company names..."):
            st.session_state.stock_name_mapping = dm.get_stock_name_mapping()
    
    name_to_ticker = st.session_state.stock_name_mapping['name_to_ticker']
    ticker_to_name = st.session_state.stock_name_mapping['ticker_to_name']
    
    # Create searchable options: "Company Name (TICKER)"
    search_options = []
    for ticker, name in ticker_to_name.items():
        base_ticker = ticker.replace('.NS', '').replace('.BO', '')
        search_options.append(f"{name} ({base_ticker})")
    
    search_options = sorted(search_options)
    
    # Search box
    selected_option = st.selectbox(
        "Search by Company Name or Ticker",
        options=search_options,
        help="Type to search by company name or ticker symbol"
    )
    
    # Extract ticker from selection
    if selected_option:
        # Extract the ticker from the format "Name (TICKER)"
        ticker_part = selected_option.split('(')[-1].replace(')', '').strip()
        # Find the matching ticker
        for ticker in ticker_to_name.keys():
            if ticker.replace('.NS', '').replace('.BO', '') == ticker_part:
                return ticker
    
    return None

def main():
    # Sidebar
    _, _, _, _, ai_model = show_sidebar()
    
    st.title("💰 Investment Planner & Portfolio Builder")
    st.markdown("---")

    # --- Input Section (Moved to Main Page) ---
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
        st.info(f"Targeting **{risk_profile}** growth strategy")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Tabs ---
    tab_suggest, tab_build = st.tabs(["🤖 Get AI Suggestions", "🛠️ Build Your Own Portfolio"])

    dm = StockDataManager()

    # --- Tab 1: AI Suggestions ---
    with tab_suggest:
        st.subheader("🚀 AI-Powered Investment Plan")
        st.markdown("Generate a sophisticated, data-driven portfolio strategy tailored to your profile.")
        
        if st.button("Generate Premium Plan", type="primary"):
            from modules.utils.quant import QuantEngine
            from modules.utils.ai_insights import generate_quant_investment_plan
            import time
            
            qe = QuantEngine(dm)
            
            user_profile = {
                "investment_amount": inv_amount,
                "duration_years": duration,
                "risk_profile": risk_profile,
                "investment_type": inv_type,
                "expected_annual_return_pct": expected_return
            }
            
            start_time = time.time()
            status_container = st.status("Initializing Quant Pipeline...", expanded=True)
            
            try:
                status_container.write("🔍 Scanning Market Universe & Fetching Data...")
                # We can't easily stream progress from the blocking call, but we can show stages
                
                payload = qe.run_pipeline(user_profile)
                
                elapsed_time = time.time() - start_time
                status_container.update(label=f"✅ Pipeline Completed in {elapsed_time:.2f}s", state="complete", expanded=False)
                
                if "error" in payload:
                    st.error(f"Pipeline Error: {payload['error']}")
                else:
                    # --- Premium Dashboard ---
                    
                    # 1. Executive Summary Metrics
                    st.markdown("### 📊 Performance Summary")
                    metrics = payload.get('backtest_summary', {})
                    bench_metrics = metrics.get('benchmark', {})
                    
                    m1, m2, m3, m4 = st.columns(4)
                    
                    cagr = metrics.get('annualized_return_pct', 0)
                    bench_cagr = bench_metrics.get('annualized_return_pct', 0)
                    cagr_delta = cagr - bench_cagr
                    
                    sharpe = metrics.get('sharpe', 0)
                    bench_sharpe = bench_metrics.get('sharpe', 0)
                    sharpe_delta = sharpe - bench_sharpe
                    
                    m1.metric("Projected CAGR", f"{cagr:.2f}%", delta=f"{cagr_delta:.2f}% vs Nifty 50")
                    m2.metric("Sharpe Ratio", f"{sharpe:.2f}", delta=f"{sharpe_delta:.2f} vs Nifty 50")
                    m3.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%", delta_color="inverse")
                    m4.metric("Volatility", f"{metrics.get('annualized_vol_pct', 0):.2f}%", delta_color="inverse")
                    
                    st.markdown("---")
                    
                    # 2. Interactive Charts (Plotly)
                    st.markdown("### 📈 Strategy Performance")
                    
                    chart_data = payload.get('chart_data', {})
                    equity_curve = pd.DataFrame(chart_data.get('equity_curve', []))
                    benchmark_curve = pd.DataFrame(chart_data.get('benchmark_curve', []))
                    
                    if not equity_curve.empty:
                        fig = go.Figure()
                        
                        # Strategy Trace
                        fig.add_trace(go.Scatter(
                            x=equity_curve['date'], 
                            y=equity_curve['value'],
                            mode='lines',
                            name='AI Strategy',
                            line=dict(color='#00CC96', width=3)
                        ))
                        
                        # Benchmark Trace
                        if not benchmark_curve.empty:
                            fig.add_trace(go.Scatter(
                                x=benchmark_curve['date'], 
                                y=benchmark_curve['value'],
                                mode='lines',
                                name='Nifty 50 Benchmark',
                                line=dict(color='#EF553B', width=2, dash='dot')
                            ))
                            
                        fig.update_layout(
                            title="Cumulative Returns Comparison",
                            xaxis_title="Date",
                            yaxis_title="Growth of ₹1",
                            hovermode="x unified",
                            height=500,
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 3. Strategy Summary
                    st.markdown("### 📋 Investment Strategy Summary")
                    with st.spinner("Generating Summary..."):
                        insight = generate_quant_investment_plan(payload, model=ai_model)
                        st.markdown(insight)
                    
                    st.markdown("---")
                    
                    # 4. Detailed Allocation
                    st.markdown("### 💼 Portfolio Allocation")
                    col_alloc1, col_alloc2 = st.columns([2, 1])
                    
                    with col_alloc1:
                        alloc_data = payload.get('allocation', [])
                        if alloc_data:
                            df_alloc = pd.DataFrame(alloc_data)
                            # Format columns
                            df_alloc['amount'] = df_alloc['amount'].apply(lambda x: f"₹{x:,.2f}")
                            df_alloc['weight_pct'] = df_alloc['weight_pct'].apply(lambda x: f"{x*100:.1f}%")
                            
                            st.dataframe(
                                df_alloc[['ticker', 'weight_pct', 'amount']],
                                column_config={
                                    "ticker": "Ticker",
                                    "weight_pct": "Weight",
                                    "amount": "Amount (₹)"
                                },
                                use_container_width=True,
                                hide_index=True
                            )
                            
                    with col_alloc2:
                        # Pie Chart using Plotly
                        if alloc_data:
                            df_alloc = pd.DataFrame(alloc_data)
                            fig_pie = px.pie(
                                df_alloc, 
                                values='weight_pct', 
                                names='ticker', 
                                title='Asset Allocation',
                                hole=0.4
                            )
                            fig_pie.update_layout(showlegend=False)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
            except Exception as e:
                status_container.update(label="❌ Pipeline Failed", state="error")
                st.error(f"An error occurred: {e}")
                st.exception(e)

    # --- Tab 2: Build Your Own ---
    with tab_build:
        st.subheader("Custom Portfolio Builder")
        
        # Portfolio Builder Interface
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Add Stocks")
            
            # Enhanced search
            selected_stock = create_stock_selector_with_search(dm)
            
            if selected_stock:
                st.info(f"Selected: {selected_stock}")
                qty = st.number_input("Quantity", min_value=1, value=10, key="qty_input")
                
                if 'portfolio' not in st.session_state:
                    st.session_state.portfolio = []
                
                if st.button("Add to Portfolio", type="primary"):
                    live = dm.get_live_data(selected_stock)
                    price = live.get('current_price', 0)
                    
                    st.session_state.portfolio.append({
                        "Symbol": selected_stock,
                        "Quantity": qty,
                        "Avg Price": price,
                        "Current Price": price,
                        "Value": price * qty
                    })
                    st.success(f"✅ Added {selected_stock}")
                    st.rerun()

            if st.button("Clear Portfolio", type="secondary"):
                st.session_state.portfolio = []
                st.rerun()

        with col2:
            st.markdown("### Your Portfolio")
            if st.session_state.get('portfolio'):
                df_port = pd.DataFrame(st.session_state.portfolio)
                
                # Display portfolio table
                st.dataframe(
                    df_port.style.format({
                        "Avg Price": "₹{:.2f}",
                        "Current Price": "₹{:.2f}",
                        "Value": "₹{:.2f}"
                    }),
                    use_container_width=True
                )
                
                total_value = df_port['Value'].sum()
                st.metric("Total Portfolio Value", format_currency(total_value))
                
                st.markdown("---")
                
                # Visualizations for Custom Portfolio
                st.markdown("### 📊 Portfolio Visualizations")
                
                sector_map = dm.get_nifty50_sector_map()
                
                # Sector Distribution
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    sector_fig = create_sector_distribution_chart(df_port, sector_map)
                    st.plotly_chart(sector_fig, use_container_width=True)
                
                with col_viz2:
                    market_cap_fig, df_port_enhanced = create_market_cap_distribution(df_port, dm)
                    st.plotly_chart(market_cap_fig, use_container_width=True)
                
                # Treemap
                st.markdown("### 🗺️ Portfolio Stock Weightage")
                treemap_fig = create_treemap_chart(df_port, sector_map)
                st.plotly_chart(treemap_fig, use_container_width=True)
                
                # Risk/Reward Scatter
                st.markdown("### 🎯 Risk vs. Reward")
                scatter_fig = create_risk_reward_scatter(df_port, dm)
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("### 🤖 AI Portfolio Check")
                if st.button("Analyze My Portfolio"):
                    if ai_model:
                        from modules.utils.ai_insights import analyze_portfolio
                        with st.spinner("Analyzing your custom portfolio..."):
                            # Mock portfolio dataframe as dict for AI
                            p_data = df_port.to_dict(orient='records')
                            analysis = analyze_portfolio(p_data, model=ai_model)
                            st.markdown(analysis)
                    else:
                        st.warning("Please select an AI Model (and provide API Key) in the Sidebar first.")
                # Ensure portfolio data is saved for PDF
                if 'portfolio_data' not in st.session_state:
                    st.session_state['portfolio_data'] = df_port_enhanced
                
                # PDF Export for Custom Portfolio
                if st.session_state.get('portfolio_analysis'):
                    st.markdown("---")
                    st.markdown("### 📥 Download Portfolio Report")
                    if st.button("Generate Portfolio PDF", type="secondary"):
                        with st.spinner("Creating PDF report..."):
                            pdf_buffer = create_portfolio_pdf(
                                investment_params={
                                    'amount': total_value,
                                    'type': 'Custom Portfolio',
                                    'duration': 'N/A',
                                    'expected_return': 'N/A',
                                    'risk_profile': 'Custom'
                                },
                                ai_generated_content=st.session_state.get('portfolio_analysis', ''),
                                sector_chart_fig=sector_fig,
                                market_cap_fig=market_cap_fig,
                                projection_fig=None,
                                portfolio_data=df_port_enhanced,
                                treemap_fig=treemap_fig
                            )
                            
                            st.download_button(
                                label="📄 Download Portfolio PDF",
                                data=pdf_buffer,
                                file_name="my_portfolio_analysis.pdf",
                                mime="application/pdf"
                            )
                            st.success("PDF ready!")
                            
                    # CSV Download Option
                    csv = df_port_enhanced.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📊 Download Portfolio CSV",
                        data=csv,
                        file_name="my_portfolio.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Add stocks to start building your portfolio.")

if __name__ == "__main__":
    main()
