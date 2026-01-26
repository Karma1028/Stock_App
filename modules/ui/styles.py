import streamlit as st

def apply_custom_style():
    st.markdown("""
        <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

        /* VARIABLES */
        :root {
            --bg-dark: #0A0E17;
            --bg-card: rgba(30, 41, 59, 0.4);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary-gradient: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
            --accent-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --text-primary: #F1F5F9;
            --text-secondary: #94A3B8;
            --success: #10B981;
            --danger: #EF4444;
        }

        /* RESET & BASE */
        .stApp {
            background-color: var(--bg-dark);
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
        }
        
        h1, h2, h3, h4, h5 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: white;
        }
        
        h1 { font-size: 2.5rem !important; margin-bottom: 1rem !important; }
        
        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid var(--border-color);
        }
        
        /* FORM ELEMENTS */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #1E293B !important;
            border: 1px solid var(--border-color) !important;
            color: white !important;
            border-radius: 8px !important;
        }
        
        .stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within > div {
            border-color: #00C9FF !important;
            box-shadow: 0 0 0 1px #00C9FF !important;
        }
        
        /* BUTTONS */
        /* Primary Button */
        div.stButton > button[kind="primary"] {
            background: var(--primary-gradient) !important;
            color: #0F172A !important;
            font-weight: 600 !important;
            border: none !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 14px 0 rgba(0, 201, 255, 0.3) !important;
        }
        
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(0, 201, 255, 0.5) !important;
        }

        /* Secondary Button */
        div.stButton > button[kind="secondary"] {
            background: transparent !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            color: white !important;
        }
        
        /* CARDS (Glassmorphism) */
        .st-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease;
        }
        
        .st-card:hover {
            border-color: rgba(255,255,255,0.15);
        }
        
        /* TABS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background: transparent;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            font-size: 1rem;
            color: var(--text-secondary);
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            color: #00C9FF !important;
            border-bottom: 2px solid #00C9FF;
            background: transparent !important;
        }

        /* METRICS */
        div[data-testid="metric-container"] {
            background: #1E293B;
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
        }
        
        div[data-testid="metric-container"] label {
            color: var(--text-secondary);
            font-size: 0.85rem;
        }
        
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: white;
            font-weight: 700;
        }

        /* CUSTOM CLASSES */
        .metric-label {
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .footer {
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border-color);
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.8rem;
        }
        
        /* TABLE OVERRIDES */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
        }
        
        thead tr th {
            background-color: #1E293B !important;
            color: white !important;
        }

        /* MARKET DATA TABLE */
        .market-data-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }
        .market-data-table td {
            padding: 12px 0;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-secondary);
        }
        .market-data-table td:last-child {
            text-align: right;
            color: white;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
