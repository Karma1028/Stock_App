"""
Shared chart configuration — single source of truth for all Plotly visuals.
"""

# ── BRAND PALETTE ──────────────────────────────────────────────
COLORS = {
    # Primary chart traces
    "price_line": "#818cf8",       # Indigo
    "sma_50": "#38bdf8",           # Sky-blue
    "sma_200": "#f97316",          # Orange
    "bb_band": "#94a3b8",          # Slate (dotted)
    "bb_fill": "rgba(148,163,184,0.05)",

    # Volume
    "vol_up": "#22c55e",           # Green
    "vol_down": "#ef4444",         # Red

    # Bars / Areas
    "revenue": "#38bdf8",
    "net_income": "#22c55e",
    "net_margin": "#22c55e",
    "op_margin": "#818cf8",

    # Strategy / Backtest
    "strategy": "#818cf8",
    "benchmark": "#94a3b8",
    "drawdown": "#ef4444",
    "drawdown_fill": "rgba(239,68,68,0.2)",

    # Gauge / Scores
    "gauge_bar": "#818cf8",
    "gauge_low": "rgba(239,68,68,0.2)",
    "gauge_mid": "rgba(234,179,8,0.2)",
    "gauge_high": "rgba(34,197,94,0.2)",

    # Signal badges
    "bullish": "#22c55e",
    "bearish": "#ef4444",
    "neutral": "#f59e0b",

    # Distribution
    "histogram": "#818cf8",
}

# ── CHART TEMPLATE ─────────────────────────────────────────────
TEMPLATE = "plotly_dark"

# ── SHARED LAYOUT DEFAULTS ─────────────────────────────────────
DEFAULT_LAYOUT = dict(
    template=TEMPLATE,
    margin=dict(l=0, r=0, t=40, b=0),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)


def format_market_cap(mcap):
    """Format market cap into readable Indian format (compact)."""
    if mcap is None or not isinstance(mcap, (int, float)):
        return "N/A"
    if mcap >= 1e12:
        return f"₹{mcap/1e12:.1f}L Cr"
    elif mcap >= 1e9:
        val_in_kcr = mcap / 1e9
        if val_in_kcr >= 100:
            return f"₹{val_in_kcr/100:.1f}L Cr"
        return f"₹{val_in_kcr:,.0f}K Cr"
    elif mcap >= 1e7:
        return f"₹{mcap/1e7:,.0f} Cr"
    else:
        return f"₹{mcap:,.0f}"


def clean_ticker_label(ticker):
    """Strip .NS/.BO suffix for display."""
    return ticker.replace(".NS", "").replace(".BO", "")
