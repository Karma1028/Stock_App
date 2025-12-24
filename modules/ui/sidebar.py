import streamlit as st
from modules.data.manager import StockDataManager

def show_sidebar():
    st.sidebar.title("Stock Analysis Tool")
    
    dm = StockDataManager()
    stock_list = dm.get_stock_list()
    
    selected_company = st.sidebar.selectbox("Select Company", stock_list)
    
    period = st.sidebar.selectbox(
        "Select Period", 
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], 
        index=3
    )
    
    chart_type = st.sidebar.radio("Chart Type", ["Line", "Candlestick"])
    
    show_technicals = st.sidebar.checkbox("Show Technical Indicators", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Artificial Intelligence")
    ai_model = st.sidebar.selectbox(
        "Select AI Model",
        [
            "tngtech/deepseek-r1t2-chimera:free",
            "tngtech/deepseek-r1t-chimera:free",
            "z-ai/glm-4.5-air:free",
            "amazon/nova-2-lite-v1:free",
            "google/gemma-3-27b-it:free",
            "openai/gpt-oss-20b:free"
        ],
        index=0
    )
    
    # Optional: Allow user to override API Key in session
    api_key = st.sidebar.text_input("OpenRouter API Key (Optional)", type="password", help="Leave empty to use system environment variable.")
    if api_key:
        st.session_state['OPENROUTER_API_KEY'] = api_key
    
    return selected_company, period, chart_type, show_technicals, ai_model
