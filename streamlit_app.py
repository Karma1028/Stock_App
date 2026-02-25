import streamlit as st
import os

# Set layout before imports (Streamlit requirement)
st.set_page_config(
    page_title="Institutional Quant Terminal",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modules
from config import Config
from modules.ui.styles import apply_custom_style
from modules.ui.dashboard_page import render_dashboard
from modules.ui.analysis_page import render_stock_analysis
from modules.ui.quant_engine_page import render_quant_engine
from modules.ui.robo_advisor_page import render_robo_advisor
from modules.ui.landing_page import render_landing_page
from modules.ui.deep_dive_page import render_deep_dive_page

def main():
    # Apply Global Styles (CSS)
    apply_custom_style()

    # ── SIDEBAR NAVIGATION ──
    st.sidebar.markdown("""
    <div style="text-align:center; padding:12px 0 8px;">
        <span style="font-size:1.6rem; font-weight:800;
              background:linear-gradient(135deg,#00C9FF,#92FE9D);
              -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Quant Terminal
        </span>
        <div style="font-size:0.7rem; color:#64748b; margin-top:2px;">Institutional Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # Grouped Navigation
    st.sidebar.markdown('<div class="nav-group-label">📈 Markets</div>', unsafe_allow_html=True)

    pages = {
        "🏠 Home": render_landing_page,
        "📊 Dashboard": render_dashboard,
        "🔬 Quick Analysis": render_stock_analysis,
    }

    st.sidebar.markdown('<div class="nav-group-label">🧠 AI & Strategy</div>', unsafe_allow_html=True)

    pages.update({
        "🤖 Deep Report": render_robo_advisor,
        "⚡ Quant Engine": render_quant_engine,
        "📐 EDA Lab": render_deep_dive_page,
    })

    selection = st.sidebar.radio("Go to", list(pages.keys()), label_visibility="collapsed")

    # Render Selected Page
    page_func = pages[selection]
    try:
        page_func()
    except Exception as e:
        st.error(f"Error loading {selection}: {e}")

    # Global Footer Area in Sidebar for Reports
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Reports")

    # Check for generated reports (e.g. Tata Motors)
    report_path = "Tata_Motors_Complete_Report.pdf"
    if os.path.exists(report_path):
        with open(report_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        st.sidebar.download_button(
            label="📄 Institutional Report (PDF)",
            data=pdf_bytes,
            file_name="Tata_Motors_Complete_Report.pdf",
            mime="application/pdf",
            help="Download the latest institutional-grade analysis for Tata Motors."
        )
    else:
        st.sidebar.info("Run backend to generate report.")

    st.sidebar.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.7rem; padding:16px 0 4px;">
        v3.0 • PyTorch + LangGraph<br>© 2026
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
