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
            --bg-card: rgba(30, 41, 59, 0.55);
            --bg-card-hover: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.12);
            --border-glow: rgba(0, 201, 255, 0.25);
            --primary-gradient: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
            --accent-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --neon-cyan: #00C9FF;
            --neon-green: #92FE9D;
            --neon-purple: #A78BFA;
            --neon-amber: #FBBF24;
            --text-primary: #F1F5F9;
            --text-secondary: #CBD5E1;
            --success: #10B981;
            --danger: #EF4444;
            --warning: #F59E0B;
        }

        /* KEYFRAME ANIMATIONS */
        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 5px rgba(0, 201, 255, 0.2), 0 0 20px rgba(0, 201, 255, 0.05); }
            50% { box-shadow: 0 0 10px rgba(0, 201, 255, 0.4), 0 0 40px rgba(0, 201, 255, 0.1); }
        }

        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }

        @keyframes borderRotate {
            0% { --angle: 0deg; }
            100% { --angle: 360deg; }
        }

        /* RESET & BASE */
        .stApp {
            background-color: var(--bg-dark);
            font-family: 'Inter', sans-serif;
            color: var(--text-primary);
        }

        /* DARK HEADER BAR — fixes white top nav */
        header[data-testid="stHeader"],
        .stAppHeader,
        .stDecoration {
            background-color: rgba(10, 14, 23, 0.95) !important;
            backdrop-filter: blur(12px) !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
        }

        /* METRIC READABILITY */
        [data-testid="stMetric"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
        }
        [data-testid="stMetric"] label,
        [data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: var(--text-secondary) !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {
            font-weight: 500 !important;
        }

        /* GENERAL TEXT CONTRAST */
        .stMarkdown p, .stMarkdown li {
            color: var(--text-secondary) !important;
        }
        .stMarkdown strong, .stMarkdown b {
            color: var(--text-primary) !important;
        }

        h1, h2, h3, h4, h5 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: white;
        }

        h1 { font-size: 2.8rem !important; margin-bottom: 1rem !important; }
        h2 { font-size: 1.8rem !important; margin-top: 2rem !important; margin-bottom: 0.8rem !important; }
        h3 { font-size: 1.35rem !important; margin-bottom: 0.6rem !important; }
        h4 { font-size: 1.1rem !important; color: var(--text-secondary) !important; }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F172A 0%, #1A1F35 100%);
            border-right: 1px solid var(--border-color);
        }

        section[data-testid="stSidebar"] .stRadio > label {
            transition: all 0.2s ease;
        }

        /* FORM ELEMENTS */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
            background-color: #1E293B !important;
            border: 1px solid var(--border-color) !important;
            color: white !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }

        .stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within > div {
            border-color: var(--neon-cyan) !important;
            box-shadow: 0 0 0 1px var(--neon-cyan), 0 0 15px rgba(0, 201, 255, 0.15) !important;
        }

        /* BUTTONS */
        div.stButton > button[kind="primary"] {
            background: var(--primary-gradient) !important;
            background-size: 200% 200% !important;
            animation: gradientShift 3s ease infinite !important;
            color: #0F172A !important;
            font-weight: 600 !important;
            border: none !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 14px 0 rgba(0, 201, 255, 0.3) !important;
            border-radius: 10px !important;
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 25px 0 rgba(0, 201, 255, 0.5) !important;
        }

        div.stButton > button[kind="secondary"] {
            background: transparent !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            color: white !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
        }

        div.stButton > button[kind="secondary"]:hover {
            border-color: var(--neon-cyan) !important;
            box-shadow: 0 0 15px rgba(0, 201, 255, 0.15) !important;
        }

        /* CARDS (Glassmorphism) */
        .st-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: all 0.3s ease;
            animation: fadeInUp 0.5s ease-out;
        }

        .st-card:hover {
            border-color: rgba(255,255,255,0.15);
            background: var(--bg-card-hover);
            transform: translateY(-2px);
        }

        /* QUANT CARD — neon border + glow */
        .quant-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.7) 100%);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(0, 201, 255, 0.3);
            border-radius: 16px;
            padding: 28px;
            margin-bottom: 24px;
            animation: pulseGlow 3s ease-in-out infinite;
            transition: all 0.3s ease;
        }

        .quant-card:hover {
            border-color: rgba(0, 201, 255, 0.6);
            transform: translateY(-3px);
        }

        .quant-card h3 {
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }

        /* REGIME BADGES */
        .regime-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            animation: fadeInUp 0.4s ease-out;
        }

        .regime-bull {
            background: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .regime-bear {
            background: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        .regime-sideways {
            background: rgba(245, 158, 11, 0.15);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        /* GATE STATUS BADGES */
        .gate-pass {
            background: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 8px 20px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1rem;
            display: inline-block;
        }

        .gate-blocked {
            background: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.4);
            padding: 8px 20px;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1rem;
            display: inline-block;
        }

        /* CONFIDENCE BAR */
        .confidence-bar-container {
            background: rgba(30, 41, 59, 0.6);
            border-radius: 10px;
            overflow: hidden;
            height: 12px;
            margin: 8px 0;
        }

        .confidence-bar-fill {
            height: 100%;
            border-radius: 10px;
            background: var(--primary-gradient);
            background-size: 200% 100%;
            animation: shimmer 2s linear infinite;
            transition: width 1s ease;
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
            transition: all 0.2s ease;
        }

        .stTabs [aria-selected="true"] {
            color: var(--neon-cyan) !important;
            border-bottom: 2px solid var(--neon-cyan);
            background: transparent !important;
        }

        /* METRICS */
        div[data-testid="metric-container"] {
            background: #1E293B;
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            transition: all 0.2s ease;
        }

        div[data-testid="metric-container"]:hover {
            border-color: rgba(0, 201, 255, 0.2);
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

        /* HERO SECTION */
        .hero-section {
            text-align: center;
            padding: 80px 20px 60px;
            background: radial-gradient(circle at 30% 40%, rgba(0, 201, 255, 0.08) 0%, transparent 50%),
                        radial-gradient(circle at 70% 60%, rgba(146, 254, 157, 0.06) 0%, transparent 50%),
                        radial-gradient(circle at 50% 50%, rgba(167, 139, 250, 0.05) 0%, transparent 60%);
            position: relative;
        }

        .hero-title {
            font-size: 4rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #00C9FF 0%, #A78BFA 50%, #92FE9D 100%);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradientShift 4s ease infinite;
            margin-bottom: 16px !important;
            line-height: 1.1 !important;
        }

        .hero-subtitle {
            font-size: 1.25rem;
            color: var(--text-secondary);
            max-width: 650px;
            margin: 0 auto 30px auto;
            line-height: 1.6;
        }

        /* FEATURE CARDS */
        .feature-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 28px 24px;
            height: 100%;
            transition: all 0.3s ease;
            animation: fadeInUp 0.5s ease-out;
        }

        .feature-card:hover {
            border-color: rgba(0, 201, 255, 0.3);
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(0, 201, 255, 0.1);
        }

        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 12px;
            display: block;
        }

        /* NEWS CARD */
        .news-card {
            background: var(--bg-card);
            backdrop-filter: blur(8px);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            transition: all 0.2s ease;
        }

        .news-card:hover {
            border-color: rgba(0, 201, 255, 0.2);
            transform: translateY(-2px);
        }

        /* STOCK ROW */
        .stock-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 0;
            border-bottom: 1px solid var(--border-color);
            transition: all 0.2s ease;
        }

        .stock-row:hover {
            background: rgba(255,255,255,0.03);
            padding-left: 8px;
        }

        .stock-row:last-child {
            border-bottom: none;
        }

        /* TICKER STRIP */
        .ticker-strip {
            background: rgba(0, 201, 255, 0.05);
            border: 1px solid rgba(0, 201, 255, 0.1);
            border-radius: 12px;
            padding: 16px 24px;
            text-align: center;
            transition: all 0.2s ease;
        }

        .ticker-strip:hover {
            border-color: rgba(0, 201, 255, 0.3);
            background: rgba(0, 201, 255, 0.08);
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

        /* HOW IT WORKS SECTION */
        .pipeline-step {
            background: rgba(30, 41, 59, 0.3);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            position: relative;
            transition: all 0.3s ease;
        }

        .pipeline-step:hover {
            border-color: rgba(0, 201, 255, 0.3);
            background: rgba(30, 41, 59, 0.5);
        }

        .pipeline-step .step-number {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: var(--primary-gradient);
            color: #0F172A;
            font-weight: 800;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 12px;
            font-size: 0.9rem;
        }

        /* ENHANCED BLOCKQUOTES — Analyst Notes */
        blockquote, .stMarkdown blockquote {
            background: rgba(30, 41, 59, 0.5) !important;
            border-left: 3px solid transparent !important;
            border-image: linear-gradient(180deg, var(--neon-cyan), var(--neon-green)) 1 !important;
            border-radius: 0 10px 10px 0 !important;
            padding: 14px 18px !important;
            margin: 12px 0 18px 0 !important;
            color: var(--text-secondary) !important;
            font-size: 0.92rem !important;
            line-height: 1.6 !important;
        }

        blockquote em, .stMarkdown blockquote em {
            color: #cbd5e1 !important;
        }

        blockquote strong, .stMarkdown blockquote strong {
            color: var(--neon-cyan) !important;
        }

        /* NAV GROUP LABELS */
        .nav-group-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #475569;
            margin-top: 12px;
            margin-bottom: 4px;
            padding-left: 4px;
        }

        /* LANDING PAGE */
        .hero-section {
            text-align: center;
            padding: 48px 24px 40px;
            border-radius: 16px;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
        }
        .hero-section::before {
            content: '';
            position: absolute;
            top: -50%; left: -50%;
            width: 200%; height: 200%;
            background: radial-gradient(circle at 30% 20%, rgba(56,189,248,0.06) 0%, transparent 50%),
                        radial-gradient(circle at 70% 80%, rgba(129,140,248,0.06) 0%, transparent 50%);
            animation: pulseGlow 8s infinite;
        }
        .hero-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00C9FF, #92FE9D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            position: relative;
            z-index: 1;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #94a3b8;
            margin-top: 12px;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
            position: relative;
            z-index: 1;
        }

        .feature-card {
            background: var(--bg-card);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 28px 20px;
            text-align: center;
            transition: transform 0.3s ease, border-color 0.3s ease;
            min-height: 220px;
        }
        .feature-card:hover {
            transform: translateY(-4px);
            border-color: rgba(56,189,248,0.3);
        }
        .feature-icon {
            font-size: 2.2rem;
            margin-bottom: 12px;
        }

        .pipeline-step {
            background: rgba(30,41,59,0.3);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px 16px;
            text-align: center;
            min-height: 140px;
        }
        .step-number {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--neon-cyan), var(--neon-green));
            color: #0f172a;
            font-weight: 800;
            font-size: 0.9rem;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 10px;
        }

        .footer {
            text-align: center;
            color: #475569;
            font-size: 0.78rem;
            padding: 24px 0 8px;
            margin-top: 32px;
            border-top: 1px solid rgba(255,255,255,0.05);
        }

        /* SCROLLBAR */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: var(--bg-dark);
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.15);
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255,255,255,0.25);
        }
        </style>
    """, unsafe_allow_html=True)
