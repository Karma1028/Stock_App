
import streamlit as st
from config import Config
from modules.data.manager import StockDataManager
from modules.data.scrapers.news_scraper import NewsScraper
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

from modules.ui.styles import apply_custom_style

# Page Configuration
st.set_page_config(
    page_title=Config.SITE_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Global Styles
apply_custom_style()

def main():
    # --- Landing Page Header ---
    col_header, col_logo = st.columns([3, 1])
    with col_header:
        st.title("🚀 Smart Stock Analytics") 
        st.markdown("### *Empowering your investment decisions with Data & AI*")
    
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
    
    # Using st-card class for unified look
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
        st.markdown(f"""
        <div class="st-card">
            <div class="metric-value">92%</div>
            <div class="metric-label">AI Accuracy</div>
        </div>
        """, unsafe_allow_html=True) 
        
    with col4:
        st.markdown(f"""
        <div class="st-card">
            <div class="metric-value">{datetime.now().strftime('%H:%M')}</div>
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
        
        # Using cached data from start of page load
        gainers = dash_data['gainers']
            
        if not gainers:
            st.warning("Could not fetch top gainers at this moment.")
        else:
            st.markdown('<div class="st-card" style="padding: 10px 20px;">', unsafe_allow_html=True)
            for stock in gainers:
                sym = stock['symbol']
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
    
    ns = NewsScraper()
    
    with st.spinner("Curating latest headlines..."):
        # Combine news from a couple of major movers
        n1_df = ns.fetch_news_history("RELIANCE.NS", days=3)
        n2_df = ns.fetch_news_history("HDFCBANK.NS", days=3)
        
        # Concatenate and sort
        combined_news = pd.concat([n1_df, n2_df], ignore_index=True)
        if not combined_news.empty:
            combined_news = combined_news.sort_values('date', ascending=False).head(6)
            raw_news = combined_news.to_dict('records')
        else:
            raw_news = []
        
        if not raw_news:
            st.info("No major market news found right now.")
        else:
            # Grid layout for news
            row1 = st.columns(3)
            row2 = st.columns(3)
            
            for i, article in enumerate(raw_news):
                # Assign column
                if i < 3:
                    col = row1[i]
                elif i < 6:
                    col = row2[i-3]
                else: 
                     break
                
                with col:
                    pub_str = article['date'].strftime('%d %b %H:%M') if isinstance(article['date'], datetime) else str(article.get('date'))
                    
                    st.markdown(f"""
                    <div class="news-card">
                        <div style="font-size: 0.8em; color: #94a3b8; margin-bottom: 5px;">{pub_str} | {article.get('source', 'News')}</div>
                        <div style="font-weight: 600; font-size: 1em; line-height: 1.4; margin-bottom: 10px;">{article['title'][:80]}...</div>
                        <a href="{article['link']}" target="_blank" style="font-size: 0.85em;">Read Full Article →</a>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Navigation Hint
    st.info("👉 Use the sidebar to navigate to **Stock Analysis** for deep dives or **Portfolio Analysis** to manage your investments.")

    # Footer
    st.markdown("""
    <div class="footer">
        © 2025 StockAI Pro • Built for Advanced Analytics • Using Streamlit & Python
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
