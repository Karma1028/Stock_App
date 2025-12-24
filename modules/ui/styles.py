import streamlit as st

def apply_custom_style():
    st.markdown("""
        <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

        /* GLOBAL THEME */
        :root {
            --bg-color: #0e1117;
            --card-bg: #1e2530;
            --text-color: #e0e0e0;
            --accent-color: #3b82f6;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --warning-color: #f59e0b;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--text-color);
        }

        /* HEADERS */
        h1, h2, h3 {
            font-weight: 700;
            letter-spacing: -0.5px;
            color: #ffffff;
        }
        
        h1 { font-size: 2.2rem !important; }
        h2 { font-size: 1.8rem !important; }
        h3 { font-size: 1.4rem !important; }

        /* METRIC CARDS */
        div[data-testid="metric-container"] {
            background-color: var(--card-bg) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            border-color: var(--accent-color);
        }

        /* CUSTOM CARDS */
        .st-card {
            background: linear-gradient(135deg, rgba(30, 37, 48, 0.7), rgba(30, 37, 48, 0.4));
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        }

        .metric-value {
            font-family: 'Inter', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }

        .metric-label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #94a3b8;
            font-weight: 600;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: #0b0f15;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        /* BUTTONS */
        .stButton > button {
            background: linear-gradient(90deg, #2563eb, #1d4ed8);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }
        
        .stButton > button:hover {
            background: linear-gradient(90deg, #1d4ed8, #1e40af);
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.4);
        }

        /* DATAFRAMES */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* INPUTS */
        .stTextInput > div > div > input, .stSelectbox > div > div > div {
            background-color: #1e2530;
            color: white;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .stTextInput > div > div > input:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 1px var(--accent-color);
        }

        /* TABS */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px;
            color: #94a3b8;
            font-weight: 600;
            padding: 0 16px; 
        }

        .stTabs [aria-selected="true"] {
            background-color: rgba(37, 99, 235, 0.1);
            color: #60a5fa;
        }

        /* NEWS & STOCK ROWS */
        .stock-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .stock-row:last-child {
            border-bottom: none;
        }

        .news-card {
            background-color: var(--card-bg);
            border-radius: 10px;
            padding: 15px;
            height: 100%;
            border: 1px solid rgba(255,255,255,0.05);
            transition: all 0.2s;
        }
        
        .news-card:hover {
            background-color: #262e3b;
        }
        
        a {
            color: #60a5fa;
            text-decoration: none;
            transition: color 0.2s;
        }
        
        a:hover {
            color: #93c5fd;
            text-decoration: underline;
        }
        
        /* TABLES */
        .market-data-table {
            width: 100%;
            border-collapse: collapse;
            background-color: var(--card-bg);
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .market-data-table td {
            padding: 12px 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.95rem;
            color: var(--text-color);
        }
        
        .market-data-table tr:last-child td {
            border-bottom: none;
        }
        
        .market-data-table .label {
            color: #94a3b8;
            font-weight: 500;
            width: 40%;
        }
        
        .market-data-table .value {
            text-align: right;
            font-weight: 600;
            color: #f8fafc;
            width: 60%;
            font-family: 'JetBrains Mono', monospace;
        }

        /* FOOTER */
        .footer {
            text-align: center;
            color: #64748b;
            padding: 40px 0;
            font-size: 0.85rem;
            margin-top: 40px;
            border-top: 1px solid rgba(255,255,255,0.05);
        }
        </style>
    """, unsafe_allow_html=True)
