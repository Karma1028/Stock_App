"""
AI Pre-fetch Module — Single-prompt architecture for stock analysis.
Generates ALL AI content for a stock in one API call, caches per session.
Sections are served instantly when users navigate between tabs.
"""
import streamlit as st
import json
import time


def _build_master_prompt(ticker, stock_summary, news_text=""):
    """Build a single comprehensive prompt that generates all AI analysis sections."""
    return f"""You are a senior CIO at an institutional fund analyzing {ticker}.

Based on the data below, generate a COMPLETE analysis in strict JSON format.
Respond with ONLY valid JSON — no markdown, no explanation outside JSON.

=== STOCK DATA ===
{stock_summary}

=== RECENT NEWS ===
{news_text if news_text else "No news available."}

=== REQUIRED JSON OUTPUT ===
{{
  "quick_verdict": {{
    "action": "BUY/HOLD/SELL",
    "conviction": 8,
    "thesis": "2-line investment thesis here",
    "risks": ["risk 1", "risk 2", "risk 3"],
    "position_size": "X% of portfolio",
    "stop_loss": "₹XXXX",
    "target_3m": "₹XXXX - ₹XXXX",
    "summary": "Full 3-paragraph CIO verdict here with data-driven reasoning"
  }},
  "deep_report": {{
    "executive_summary": "3-paragraph institutional-grade executive summary",
    "technical_analysis": "Detailed technical analysis — RSI, MACD, SMA, support/resistance levels",
    "fundamental_analysis": "CRITICAL: You MUST explicitly explain and critique ALL provided ratios (Valuation, Profitability, Leverage), and critically analyze the Balance Sheet and Cash Flow details based on the overall fundamental health of the business.",
    "risk_assessment": "VaR, drawdown, volatility, beta interpretation",
    "sector_outlook": "Industry positioning and competitive landscape"
  }},
  "news_sentiment": {{
    "overall": "Bullish/Bearish/Neutral",
    "score": 65,
    "key_themes": ["theme1", "theme2"],
    "impact_summary": "How news affects the stock outlook"
  }}
}}

CRITICAL: Return ONLY valid JSON. No markdown code blocks. No extra text."""


def get_cached_analysis(ticker):
    """Return cached AI analysis for a ticker, or None if not available."""
    cache_key = f"ai_analysis_{ticker}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    return None


from modules.ui.thought_formatter import parse_and_format_thought

def prefetch_stock_analysis(ticker, stock_summary, news_text="", status_container=None, thought_placeholder=None):
    """
    Generate all AI analysis sections in a SINGLE API call.
    Caches result in session_state for instant retrieval.
    Returns the parsed analysis dict or None on failure.
    Accepts status containers to live-stream the thinking process to the UI.
    """
    # Check cache first
    cache_key = f"ai_analysis_{ticker}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    try:
        from agentic_backend import query_deepseek_reasoner, stream_deepseek_reasoner
    except ImportError:
        return None
    
    if not query_deepseek_reasoner:
        return None
    
    prompt = _build_master_prompt(ticker, stock_summary, news_text)
    
    system = (
        "You are an institutional-grade financial AI. "
        "Respond with ONLY valid JSON — no markdown, no explanation. "
        "Be specific with numbers, prices, and percentages."
    )
    
    try:
        thinking_buf = ""
        content_buf = ""
        
        if thought_placeholder is not None:
            for chunk in stream_deepseek_reasoner(system, prompt):
                ctype = chunk.get("type")
                cdelta = chunk.get("delta", "")
                
                if ctype == "reasoning":
                    thinking_buf += cdelta
                    formatted_thought = parse_and_format_thought(thinking_buf)
                    thought_placeholder.markdown(formatted_thought, unsafe_allow_html=True)
                elif ctype == "content":
                    content_buf += cdelta
                    # Note: We do not display the raw JSON content during streaming because 
                    # it isn't formatted for the user. It will be parsed later.
            raw = content_buf
        else:
            raw = query_deepseek_reasoner(system, prompt)
            
        if not raw or raw.startswith("[AI Error]"):
            return None
        
        # 1. Clean up response — strip all thinking tags
        cleaned = raw.strip()
        import re
        
        # Remove anything inside <think> tags ( DeepSeek-R1 ) or <|!|> ( some other models )
        cleaned = re.sub(r'<(think|\|!\|)>.*?</\1>', '', cleaned, flags=re.DOTALL + re.IGNORECASE)
        # Final safety catch for unclosed tags
        if "<think>" in cleaned.lower():
            cleaned = cleaned.split("</think>")[-1] if "</think>" in cleaned.lower() else cleaned.split("<think>")[-1]
        
        cleaned = cleaned.strip()
        
        # 2. Extract strictly the JSON part between { and }
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            json_text = cleaned[start_idx:end_idx+1]
            try:
                analysis = json.loads(json_text)
                
                # Cache it
                st.session_state[cache_key] = analysis
                return analysis
            except json.JSONDecodeError as e:
                print(f"   [Cache] JSON Decode Error: {e}")
                raise e # Fall through to the handler below
        else:
            raise ValueError("No JSON structure found in AI response")

        
    except json.JSONDecodeError as jde:
        # If JSON parsing fails, try to extract the verdict text as-is
        fallback = {
            "quick_verdict": {
                "action": "N/A",
                "conviction": 0,
                "summary": f"JSON Error: {jde}\nRAW DATA:\n{cleaned[:300]}...",
                "risks": [],
                "position_size": "N/A",
                "stop_loss": "N/A",
                "target_3m": "N/A",
                "thesis": ""
            },
            "deep_report": {
                "executive_summary": "Analysis unavailable due to non-JSON format.",
                "technical_analysis": "",
                "fundamental_analysis": "",
                "risk_assessment": "",
                "sector_outlook": ""
            },
            "news_sentiment": {
                "overall": "Neutral",
                "score": 50,
                "key_themes": [],
                "impact_summary": ""
            }
        }
        st.session_state[cache_key] = fallback
        return fallback
    except Exception:
        return None


def invalidate_cache(ticker):
    """Clear cached analysis for a ticker."""
    cache_key = f"ai_analysis_{ticker}"
    st.session_state.pop(cache_key, None)


def display_with_animation(content, delay=0.5):
    """Display content with a brief animation for UX polish."""
    placeholder = st.empty()
    with placeholder:
        with st.spinner("🧠 Loading AI analysis..."):
            time.sleep(delay)
    placeholder.empty()
    st.markdown(content)
