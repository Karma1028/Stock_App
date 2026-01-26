import streamlit as st
from modules.ui.styles import apply_custom_style

def render_landing_page():
    # Hero Section
    st.markdown("""
        <div style="text-align: center; padding: 60px 20px; background: radial-gradient(circle at center, rgba(59, 130, 246, 0.15) 0%, rgba(10, 14, 23, 0) 70%);">
            <h1 style="font-size: 4rem !important; background: linear-gradient(to right, #60A5FA, #A78BFA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 20px;">
                Intelligent Market Research
            </h1>
            <p style="font-size: 1.2rem; color: #94A3B8; max-width: 600px; margin: 0 auto 40px auto;">
                Next-generation stock analysis powered by AI. Get deep financial insights, real-time sentiment, and institutional-grade tools.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Search Logic (Visual only, actual navigation happens via Sidebar for now due to Streamlit limitations)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.info("👈 Use the **Sidebar** to Select a Company or Navigate to Tools")
    
    st.markdown("---")

    # Features Grid
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="st-card" style="height: 100%;">
            <h3 style="color: #60A5FA;">📊 Advanced Charting</h3>
            <p style="color: #94A3B8;">Interactive price charts with 50+ technical indicators and overlay capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="st-card" style="height: 100%;">
            <h3 style="color: #A78BFA;">🤖 AI Analyst</h3>
            <p style="color: #94A3B8;">Get instant executive summaries, visual trend analysis, and conversational insights.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
        <div class="st-card" style="height: 100%;">
            <h3 style="color: #34D399;">💰 Quant Planner</h3>
            <p style="color: #94A3B8;">Build robust portfolios with AI-driven backtesting and risk management.</p>
        </div>
        """, unsafe_allow_html=True)

    # Market Highlights Ticker (Mockup)
    st.markdown("---")
    st.subheader("🔥 Trending Now")
    
    cols = st.columns(4)
    trends = [
        {"sym": "RELIANCE", "change": "+2.4%", "color": "#10B981"},
        {"sym": "HDFCBANK", "change": "-0.8%", "color": "#EF4444"},
        {"sym": "TATASTEEL", "change": "+1.2%", "color": "#10B981"},
        {"sym": "INFY", "change": "+0.5%", "color": "#10B981"},
    ]
    
    for i, trend in enumerate(trends):
        cols[i].markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; text-align: center;">
            <div style="font-weight: 700; color: white;">{trend['sym']}</div>
            <div style="color: {trend['color']}; font-weight: 600;">{trend['change']}</div>
        </div>
        """, unsafe_allow_html=True)
