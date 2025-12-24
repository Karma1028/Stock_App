import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.ui.sidebar import show_sidebar
from modules.data.manager import StockDataManager
from modules.utils.ai_insights import get_ai_insights, generate_company_summary
from modules.ml.prediction import StockPredictor
from modules.utils.helpers import format_currency, get_current_time, format_large_number
import modules.ui.plots as fp
from modules.data.scrapers.news_scraper import NewsScraper
from modules.ml.features import FeatureEngineer
from modules.ml.engine import MLEngine
import pandas as pd

st.set_page_config(page_title="Stock Analysis", layout="wide")
from modules.ui.styles import apply_custom_style
apply_custom_style()

def plot_stock_chart(df, symbol, chart_type, show_technicals, sentiment_df=None):
    # Create subplots: Price, Volume, Sentiment (if available)
    rows = 2
    row_heights = [0.7, 0.3]
    specs = [[{"secondary_y": True}], [{"secondary_y": False}]]
    
    if sentiment_df is not None and not sentiment_df.empty:
        rows = 3
        row_heights = [0.6, 0.2, 0.2]
        specs = [[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{symbol} Price', 'Volume', 'Sentiment Trend'), 
                        row_heights=row_heights, specs=specs)

    # 1. Price Chart
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name='OHLC'), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price'), row=1, col=1)

    # Add Technicals
    if show_technicals:
        if 'SMA_50' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='orange')), row=1, col=1)
        if 'SMA_200' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='green')), row=1, col=1)
        if 'BB_High' in df.columns and 'BB_Low' in df.columns:
             fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name='BB High', line=dict(color='gray', dash='dash')), row=1, col=1)
             fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name='BB Low', line=dict(color='gray', dash='dash')), row=1, col=1)

    # 2. Volume Chart
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)

    # 3. Sentiment Chart
    if sentiment_df is not None and not sentiment_df.empty:
        # Align sentiment dates with price dates
        # sentiment_df has 'date_only', we need to map it to df index
        fig.add_trace(go.Bar(x=sentiment_df['date_only'], y=sentiment_df['sentiment_score'], 
                             name='Sentiment Score', marker_color='purple'), row=3, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, height=800 if rows==3 else 600)
    return fig

def plot_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Combined AI Score"},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'red'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'green'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score}}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def main():
    # Sidebar
    selected_company, period, chart_type, show_technicals, ai_model = show_sidebar()
    
    # Initialize Managers
    dm = StockDataManager()
    ns = NewsScraper()
    fe = FeatureEngineer()
    ml = MLEngine()
    
    # Fetch Live Data
    live_data = dm.get_live_data(selected_company)
    
    if not live_data:
        st.error(f"Could not fetch data for {selected_company}. Please check your internet connection or try another stock.")
        return

    # --- Hero Section ---
    st.title(f"{live_data.get('long_name', selected_company)}")
    
    # Company Badges
    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    c1.info(f"**Symbol:** {live_data.get('symbol')}")
    c2.info(f"**Sector:** {live_data.get('sector', 'N/A')}")
    c3.info(f"**Industry:** {live_data.get('industry', 'N/A')}")
    
    # Live Metrics
    curr_price = live_data.get('current_price') or 0
    prev_close = live_data.get('previous_close') or 0
    change = curr_price - prev_close
    pct_change = (change / prev_close) * 100 if prev_close else 0
    
    # Custom CSS for metrics (Removed - using global styles)

    
    m1, m2, m3, m4, m5 = st.columns(5)
    
    m1.metric("Current Price", format_currency(curr_price), f"{change:.2f} ({pct_change:.2f}%)")
    m2.metric("Day High", format_currency(live_data.get('day_high') or 0))
    m3.metric("Day Low", format_currency(live_data.get('day_low') or 0))
    m4.metric("Volume", format_large_number(live_data.get('volume') or 0))
    m5.metric("Market Cap", format_large_number(live_data.get('market_cap')))

    st.markdown("---")

    # --- 1. Company Overview ---
    st.subheader("🏢 Company Overview")
    
    # Row 1: Valuation
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P/E Ratio", f"{live_data.get('pe_ratio', 'N/A')}")
    c2.metric("P/B Ratio", f"{live_data.get('price_to_book', 'N/A')}")
    c3.metric("Dividend Yield", f"{live_data.get('dividend_yield', 0)*100:.2f}%" if live_data.get('dividend_yield') else "N/A")
    c4.metric("EPS (TTM)", f"₹{live_data.get('trailing_eps', 'N/A')}")
    
    # Row 2: Financials
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", format_large_number(live_data.get('total_revenue')))
    c2.metric("Profit Margins", f"{live_data.get('profit_margins', 0)*100:.2f}%" if live_data.get('profit_margins') else "N/A")
    c3.metric("Return on Equity", f"{live_data.get('return_on_equity', 0)*100:.2f}%" if live_data.get('return_on_equity') else "N/A")
    c4.metric("Debt/Equity", f"{live_data.get('debt_to_equity', 'N/A')}")
    
    st.markdown("### Business Summary")
    st.write(live_data.get('long_business_summary', 'No summary available.'))
    
    st.markdown("---")

    # --- 2. Detailed Market Data ---
    st.subheader("Detailed Market Data")
    
    # CSS for tables (Removed - using global styles)


    def make_table_row(label, value):
        val_str = value
        if isinstance(value, (int, float)):
             if value > 1000000: # Use large number formatting for > 1M
                 val_str = format_large_number(value)
             elif value > 1000:
                 val_str = format_currency(value)
             else:
                 val_str = str(value)
        elif value is None:
            val_str = "N/A"
        
        return f"<tr><td class='label'>{label}</td><td class='value'>{val_str}</td></tr>"

    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown("#### Trade Information")
        rows = ""
        rows += make_table_row("Traded Volume", live_data.get('volume', 0))
        rows += make_table_row("Traded Value", live_data.get('volume', 0) * live_data.get('current_price', 0))
        rows += make_table_row("Market Cap", live_data.get('market_cap', 0))
        rows += make_table_row("Free Float", live_data.get('float_shares', 'N/A'))
        rows += make_table_row("Shares Outstanding", live_data.get('shares_outstanding', 0))
        rows += make_table_row("Face Value", "N/A")
        rows += make_table_row("Deliverable %", "N/A")
        
        st.markdown(f"<table class='market-data-table'>{rows}</table>", unsafe_allow_html=True)

    with t2:
        st.markdown("#### Price Information")
        rows = ""
        rows += make_table_row("52 Week High", live_data.get('fifty_two_week_high', 'N/A'))
        rows += make_table_row("52 Week Low", live_data.get('fifty_two_week_low', 'N/A'))
        rows += make_table_row("Day High", live_data.get('day_high', 'N/A'))
        rows += make_table_row("Day Low", live_data.get('day_low', 'N/A'))
        rows += make_table_row("Tick Size", "0.05")
        rows += make_table_row("Upper Band", "N/A")
        rows += make_table_row("Lower Band", "N/A")
        
        st.markdown(f"<table class='market-data-table'>{rows}</table>", unsafe_allow_html=True)

    with t3:
        st.markdown("#### Securities Information")
        rows = ""
        rows += make_table_row("Status", "Listed")
        rows += make_table_row("Trading Status", "Active")
        rows += make_table_row("Sector", live_data.get('sector', 'N/A'))
        rows += make_table_row("Industry", live_data.get('industry', 'N/A'))
        rows += make_table_row("P/E Ratio", live_data.get('pe_ratio', 'N/A'))
        rows += make_table_row("Book Value", live_data.get('book_value', 'N/A'))
        rows += make_table_row("Currency", live_data.get('currency', 'INR'))
        
        st.markdown(f"<table class='market-data-table'>{rows}</table>", unsafe_allow_html=True)

    st.markdown("---")

    # --- 3. Price & Technical Analysis ---
    st.subheader("Price & Technical Analysis")
    
    # Fetch Data
    with st.spinner("Analyzing Market Data..."):
        # 1. Historical Data
        df = dm.get_cached_data([selected_company], period="2y")
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                if selected_company in df.columns:
                    df = df[selected_company]
                elif df.columns.nlevels > 1 and selected_company in df.columns.get_level_values(1):
                    df = df.xs(selected_company, axis=1, level=1)
                elif 'Close' in df.columns.get_level_values(0):
                     df.columns = df.columns.get_level_values(0)
        
        # 2. News & Sentiment (Optimized)
        news_df = ns.fetch_news_history(selected_company, days=30)
        if not news_df.empty:
            news_df = ns.analyze_sentiment(news_df)
            # Create sentiment_daily from news_df
            df_sent = news_df.copy()
            df_sent['date_only'] = df_sent['date'].dt.date
            sentiment_daily = df_sent.groupby('date_only')['sentiment'].agg(['mean', 'count', 'std']).reset_index()
            sentiment_daily = sentiment_daily.rename(columns={'mean': 'sentiment_score', 'count': 'news_count', 'std': 'sentiment_volatility'})
            sentiment_daily = sentiment_daily.fillna(0)
        else:
            sentiment_daily = pd.DataFrame()
        
        # 3. Features
        df_features = fe.compute_all_features(df, sentiment_daily)
        if not sentiment_daily.empty:
            df_features = fe.merge_sentiment(df_features, sentiment_daily)
        
        # 4. Prediction (Using XGBoost Global Model)
        try:
            ml_global = MLEngine(model_name="xgb_model_global.pkl")
            if ml_global.load_model():
                preds = ml_global.predict(df_features)
                shap_explanation = None
            else:
                st.toast("Global model not found. Training local model...", icon="⚠️")
                model = ml.train_model(df_features, model_type="xgboost")
                preds = ml.predict(df_features)
                shap_explanation = None
        except Exception as e:
            print(f"Prediction error: {e}")
            preds = None
            shap_explanation = None
        
        combined_stats = {}
        if preds is not None:
            combined_stats = ml.calculate_combined_score(df_features, preds)

    # 1. Chart (Full Width)
    if not df.empty:
        st.plotly_chart(plot_stock_chart(df_features, selected_company, chart_type, show_technicals, sentiment_daily), use_container_width=True)
    else:
        st.warning("No data available.")
        
    st.markdown("---")
    
    # 2. KPIs (3 Columns)
    k1, k2, k3 = st.columns(3)
    
    tech_score = combined_stats.get('technical_score', 0)
    sent_score = combined_stats.get('sentiment_score', 0)
    model_score = combined_stats.get('prediction_score', 0)
    
    with k1:
        st.metric("Technical Score", f"{tech_score}/100")
        st.progress(tech_score/100)
        
    with k2:
        st.metric("Sentiment Score", f"{sent_score}/100")
        st.progress(sent_score/100)
        
    with k3:
        st.metric("AI Model Score", f"{model_score}/100")
        st.progress(model_score/100)
        
    st.markdown("---")
    
    # 3. Prediction
    st.subheader(f"AI Prediction (Next 5 Days)")
    
    pred_return = combined_stats.get('predicted_return_pct', 0)
    color = "green" if pred_return > 0 else "red"
    
    # Center the prediction
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown(f"""
        <div class="st-card" style="text-align: center;">
            <h2 style="margin:0; color: #94a3b8; font-size: 1.2rem;">Expected Return</h2>
            <h1 style="font-size: 3.5em; margin:10px 0; color: {color};">{pred_return:.2f}%</h1>
            <p style="color: #64748b; font-size: 0.9rem;">Based on XGBoost Global Model</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")

    # --- 4. Future Price Forecast (N Days) ---
    st.subheader("📈 Future Price Forecast")
    
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        forecast_days = st.slider("Select Forecast Days", min_value=7, max_value=90, value=30, step=1)
        
    with st.spinner(f"Generating {forecast_days}-day forecast..."):
        sp = StockPredictor()
        # Ensure df is clean for Prophet
        df_prophet = df.copy()
        forecast, error_msg = sp.train_and_predict(df_prophet, periods=forecast_days)
        
        if forecast is not None:
            # Plot using Plotly
            fig_forecast = go.Figure()
            
            # Historical (Last 6 months for context)
            hist_start = df.index.max() - pd.Timedelta(days=180)
            df_hist = df[df.index >= hist_start]
            
            # Correctly access Close column whether it's single or multi-level
            if isinstance(df_hist.columns, pd.MultiIndex):
                try:
                   y_data = df_hist.xs('Close', axis=1, level=1).iloc[:, 0]
                except KeyError:
                   # Try level 0
                   if 'Close' in df_hist.columns.get_level_values(0):
                       y_data = df_hist['Close']
                   else:
                       # Fallback to column name matching
                       cols = [c for c in df_hist.columns if 'Close' in str(c)]
                       y_data = df_hist[cols[0]] if cols else df_hist.iloc[:, 0]
            else:
                 y_data = df_hist['Close']

            fig_forecast.add_trace(go.Scatter(x=df_hist.index, y=y_data, name='Historical', line=dict(color='#00B4D8', width=2)))
            
            # Forecast
            # forecast['ds'] is date, forecast['yhat'] is prediction
            # Filter for future dates
            last_date = df.index.max()
            # Ensure last_date is timezone-naive if forecast['ds'] is
            if last_date.tzinfo is not None:
                last_date = last_date.tz_localize(None)
                
            future_forecast = forecast[forecast['ds'] > last_date]
            
            fig_forecast.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], name='Forecast', line=dict(color='#00CC96', width=3)))
            
            # Confidence Interval
            fig_forecast.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
            fig_forecast.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 204, 150, 0.2)', showlegend=False))
            
            fig_forecast.update_layout(
                title=dict(text=f"{selected_company} Price Forecast ({forecast_days} Days)", font=dict(size=20)),
                xaxis_title="Date", 
                yaxis_title="Price", 
                height=500,
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='#333'),
                yaxis=dict(showgrid=True, gridcolor='#333'),
                legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right")
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            with st.expander("View Forecast Data"):
                st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'Date', 'yhat': 'Predicted Price', 'yhat_lower': 'Lower', 'yhat_upper': 'Upper'}), use_container_width=True)
        else:
            st.error(f"Could not generate forecast. Error: {error_msg}")

    st.markdown("---")
    
    # --- AI Insights ---
    st.markdown(f"### 🤖 AI Executive Summary <span style='font-size:0.6em; color:#94a3b8;'>({ai_model.split('/')[1] if ai_model else 'Off'})</span>", unsafe_allow_html=True)
    
    if ai_model:
        with st.spinner(f"Generating insights using {ai_model}..."):
            st.markdown('<div class="st-card">', unsafe_allow_html=True)
            # 1. Company Summary
            summary = generate_company_summary(selected_company, model=ai_model)
            st.markdown(summary)
            
            st.markdown("---")
            
            # 2. Technical Deep Dive
            if combined_stats:
                st.markdown("#### 🧠 Technical Deep Dive")
                insights = get_ai_insights(selected_company, combined_stats, model=ai_model)
                st.markdown(insights)
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("💡 Select an AI Model in the sidebar to enable real-time AI analysis.")
    
    st.markdown("---")
    
    # --- Tabs for Details ---
    tab_news, tab_financials, tab_learn = st.tabs(["📰 News & Sentiment", "💰 Financials", "🎓 Learn"])
    
    with tab_news:
        st.subheader("Recent News Analysis")
        if not news_df.empty:
            for _, row in news_df.iterrows():
                with st.expander(f"{row['date'].strftime('%Y-%m-%d')} - {row['title']}"):
                    st.write(f"**Source:** {row.get('source', 'Unknown')}")
                    st.write(f"**Sentiment:** {row.get('sentiment', 0):.2f}")
                    st.markdown(f"[Read Article]({row['link']})")
        else:
            st.info("No recent news found.")
            
    with tab_financials:
        st.subheader("Financial Health Dashboard")
        
        # Fetch all financial data
        with st.spinner("Fetching financial statements..."):
            bs = dm.get_balance_sheet(selected_company)
            ist = dm.get_income_statement(selected_company)
            cf = dm.get_cash_flow(selected_company)
        
        if bs.empty and ist.empty and cf.empty:
            st.warning("Financial data unavailable for this company.")
        else:
            # Sub-tabs for different statements
            fin_tab1, fin_tab2, fin_tab3, fin_tab4 = st.tabs(["Capital Structure", "Balance Sheet", "Income Statement", "Cash Flow"])
            
            # --- Capital Structure ---
            with fin_tab1:
                st.markdown("### 🏗️ Capital Structure Analysis")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig_cap = fp.plot_capital_structure(bs, selected_company)
                    st.plotly_chart(fig_cap, use_container_width=True)
                
                with col2:
                    st.info("""
                    **Understanding Capital Structure:**
                    
                    This chart shows the proportion of Debt vs. Equity used to finance the company's assets.
                    
                    - **High Debt**: Can increase returns but adds risk.
                    - **High Equity**: More stable but might dilute ownership.
                    """)
                    
                    # Key Solvency Ratios
                    ratios = dm.calculate_liquidity_ratios(bs)
                    if ratios:
                        st.metric("Debt/Equity Ratio", f"{ratios.get('Debt/Equity', 0):.2f}")
                        st.metric("Current Ratio", f"{ratios.get('Current Ratio', 0):.2f}")

            # --- Balance Sheet ---
            with fin_tab2:
                st.markdown("### ⚖️ Balance Sheet Trends")
                st.plotly_chart(fp.plot_balance_sheet_trends(bs, selected_company), use_container_width=True)
                
                with st.expander("View Balance Sheet Data"):
                    # Format numbers
                    bs_display = bs.copy()
                    for col in bs_display.columns:
                        bs_display[col] = bs_display[col].apply(lambda x: format_large_number(x) if isinstance(x, (int, float)) else x)
                    st.dataframe(bs_display, use_container_width=True)

            # --- Income Statement ---
            with fin_tab3:
                st.markdown("### 💰 Profitability Analysis")
                
                # Revenue & Net Income
                st.plotly_chart(fp.plot_income_statement_trends(ist, selected_company), use_container_width=True)
                
                # Margins
                st.markdown("#### Profit Margins")
                st.plotly_chart(fp.plot_margins(ist, selected_company), use_container_width=True)
                
                with st.expander("View Income Statement Data"):
                    ist_display = ist.copy()
                    for col in ist_display.columns:
                        ist_display[col] = ist_display[col].apply(lambda x: format_large_number(x) if isinstance(x, (int, float)) else x)
                    st.dataframe(ist_display, use_container_width=True)

            # --- Cash Flow ---
            with fin_tab4:
                st.markdown("### 💸 Cash Flow Analysis")
                st.plotly_chart(fp.plot_cash_flow_trends(cf, selected_company), use_container_width=True)
                
                with st.expander("View Cash Flow Data"):
                    cf_display = cf.copy()
                    for col in cf_display.columns:
                        cf_display[col] = cf_display[col].apply(lambda x: format_large_number(x) if isinstance(x, (int, float)) else x)
                    st.dataframe(cf_display, use_container_width=True)

    with tab_learn:
         st.subheader("🎓 Stock Market Education")
         
         # Terms to Learn
         terms = {
             "P/E Ratio": "Price-to-Earnings Ratio",
             "RSI": "Relative Strength Index",
             "Beta": "Volatility Measure",
             "Market Cap": "Market Capitalization",
             "Dividend Yield": "Annual Dividend / Price"
         }
         
         # Navigation Buttons
         cols = st.columns(len(terms))
         
         # Initialize Session State for Learn Tab
         if 'selected_learn_term' not in st.session_state:
             st.session_state.selected_learn_term = "P/E Ratio"
             
         for i, (term_key, full_name) in enumerate(terms.items()):
             if cols[i].button(term_key, use_container_width=True, key=f"btn_learn_{i}"):
                 st.session_state.selected_learn_term = term_key
         
         term = st.session_state.selected_learn_term
         st.markdown(f"### {term}: {terms[term]}")
         
         # Content Generation Logic
         content = ""
         example = ""
         
         if term == "P/E Ratio":
             pe = live_data.get('pe_ratio')
             content = "The Price-to-Earnings (P/E) ratio relates a company's share price to its earnings per share. A high P/E ratio could mean that a company's stock is over-valued, or else that investors are expecting high growth rates in the future."
             if pe:
                 example = f"**{selected_company} Example**: The current P/E is **{pe:.2f}**. This means investors are willing to pay ₹{pe:.2f} for every ₹1 of earnings."
             else:
                 example = f"**{selected_company} Example**: P/E data is not available."
                 
         elif term == "RSI":
             current_rsi = "N/A"
             if 'df_features' in locals() and not df_features.empty:
                 current_rsi = f"{df_features['RSI'].iloc[-1]:.2f}"
                 
             content = "The Relative Strength Index (RSI) is a momentum indicator used in technical analysis. It measures the speed and magnitude of a security's recent price changes to evaluate overvalued or undervalued conditions."
             example = f"**{selected_company} Example**: The current RSI is **{current_rsi}**. (Above 70 = Overbought, Below 30 = Oversold)."
             
         elif term == "Beta":
             beta = live_data.get('beta')
             content = "Beta is a measure of a stock's volatility in relation to the overall market. A beta greater than 1.0 suggests that the stock is more volatile than the broader market, and a beta less than 1.0 indicates a stock with lower volatility."
             if beta:
                 example = f"**{selected_company} Example**: The Beta is **{beta:.2f}**. " + ("It represents high volatility." if beta > 1 else "It represents low volatility.")
             else:
                 example = f"**{selected_company} Example**: Beta data is not available."
                 
         elif term == "Market Cap":
             cap = live_data.get('market_cap')
             content = "Market Capitalization refers to the total dollar market value of a company's outstanding shares of stock. It is calculated by multiplying the total number of company's outstanding shares by the current market price of one share."
             if cap:
                 example = f"**{selected_company} Example**: The Market Cap is **{format_large_number(cap)}**. Large Cap > 20k Cr, Mid Cap > 5k Cr."
             else:
                 example = f"**{selected_company} Example**: Data unavailable."

         elif term == "Dividend Yield":
             dy = live_data.get('dividend_yield')
             content = "The dividend yield is the financial ratio that shows how much a company pays out in dividends each year relative to its stock price."
             if dy:
                 example = f"**{selected_company} Example**: The yield is **{dy*100:.2f}%**."
             else:
                 example = f"**{selected_company} Example**: This company does not appear to pay a dividend (or data is missing)."

         st.info(f"{content}\n\n{example}")
         
         st.divider()
         st.info("💡 Tip: Click on different terms above to see examples from the current stock.")

if __name__ == "__main__":
    main()
