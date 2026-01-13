import os
import json
import time
import hashlib
import streamlit as st
from config import Config
from modules.utils.cache_manager import get_cache

# Try to import OpenAI, handle absence gracefully
import sys
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

def _get_api_key():
    # Check session state first (user override)
    if 'OPENROUTER_API_KEY' in st.session_state:
        return st.session_state['OPENROUTER_API_KEY']
    return Config.OPENROUTER_API_KEY

def _get_client():
    if OpenAI is None:
        return None
        
    api_key = _get_api_key()
    if not api_key:
        return None
    
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

def _generate_cache_key(messages, model):
    """Generate a cache key from messages and model."""
    content = json.dumps(messages, sort_keys=True) + model
    return hashlib.md5(content.encode()).hexdigest()

def _call_ai_api(messages, model="google/gemini-2.0-flash-exp:free", use_cache=True):
    """Call AI API with caching and fallback mechanisms."""
    if OpenAI is None:
        return "⚠️ OpenAI module not found. Please install it to use AI features."
        
    client = _get_client()
    if not client:
        return "⚠️ AI API Key is missing. Please set OPENROUTER_API_KEY in .env or Sidebar."
    
    # Check cache first
    cache = get_cache()
    cache_key = f"ai_response_{_generate_cache_key(messages, model)}"
    
    if use_cache and Config.API_CACHE_ENABLED:
        cached_response = cache.get(cache_key)
        if cached_response:
            print(f"✓ Using cached AI response for model {model}")
            return cached_response
    
    # Try primary model with fallbacks
    models_to_try = [model] + [m for m in Config.AI_MODEL_FALLBACKS if m != model]
    last_error = None
    
    for attempt, current_model in enumerate(models_to_try):
        try:
            print(f"Attempting AI call with model: {current_model} (attempt {attempt + 1}/{len(models_to_try)})")
            
            completion = client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            response = completion.choices[0].message.content
            
            # Cache successful response
            if Config.API_CACHE_ENABLED:
                cache.set(cache_key, response, ttl=Config.API_CACHE_TTL)
            
            # Notify if we used a fallback model
            if current_model != model:
                response = f"ℹ️ *Using fallback model {current_model.split('/')[-1]} due to rate limits*\n\n{response}"
            
            return response
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            # Check if it's a rate limit error
            if "429" in error_str or "rate" in error_str.lower():
                print(f"Rate limit hit for {current_model}, trying next model...")
                
                # Add exponential backoff for same model retries
                if attempt < len(models_to_try) - 1:
                    wait_time = min(2 ** attempt, 5)  # Max 5 seconds
                    time.sleep(wait_time)
                continue
            else:
                # For non-rate-limit errors, try next model immediately
                print(f"Error with {current_model}: {error_str}")
                continue
    
    # All models failed - return cached response if available (even if expired)
    cache_path = cache._get_cache_path(cache_key)
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                entry = json.load(f)
                return f"⚠️ *All AI models are rate-limited. Showing cached response:*\n\n{entry['value']}"
        except:
            pass
    
    return f"❌ Error calling AI API after trying {len(models_to_try)} models. Last error: {last_error}\n\n💡 Please try again in a few minutes or add your own API key to avoid rate limits."

def generate_company_summary(symbol, model="google/gemini-2.0-flash-exp:free"):
    """
    Generates a professional executive summary for a company using the Magic Formula Prompting.
    """
    prompt = f"""
    **Persona**: You are a Senior Equity Analyst at a top-tier investment bank (e.g., Goldman Sachs).
    
    **Task**: Provide a comprehensive Executive Summary for the company **{symbol}**.
    
    **Context**: The user is a potential investor looking for a high-level overview to decide whether to research further.
    
    **Constraints**:
    - Be concise but insightful.
    - Focus on Business Model, Competitive Advantage (Moat), and Recent Key Developments.
    - Use professional financial terminology.
    - No financial advice disclaimer needed as this is for educational purposes.
    
    **Format**:
    - **Business Overview**: What they do.
    - **Economic Moat**: Why they win.
    - **Key Risks**: What could go wrong.
    - **Outlook**: Short sentence on future prospects.
    """
    
    messages = [{"role": "user", "content": prompt}]
    return _call_ai_api(messages, model)

def get_ai_insights(symbol, data_summary, model="google/gemini-2.0-flash-exp:free"):
    """
    Generates technical and sentimental insights.
    """
    prompt = f"""
    **Persona**: You are a Quantitative Market Strategist associated with a hedge fund.
    
    **Task**: Analyze the provided technical and sentiment data for **{symbol}** and provide an actionable insight.
    
    **Data Provided**:
    {json.dumps(data_summary, indent=2)}
    
    **Constraints**:
    - Interpret the Technical Indicators (RSI, MACD) in conjunction with Sentiment.
    - Identify if there is a divergence (e.g., Price down but Sentiment up).
    - Provide a "Bullish", "Bearish", or "Neutral" bias with confidence level.
    
    **Format**: Markdown. Use emojis for list items.
    """
    messages = [{"role": "user", "content": prompt}]
    return _call_ai_api(messages, model)

def generate_investment_plan(amount, duration_years, expected_return, risk_profile, market_context, model="google/gemini-2.0-flash-exp:free"):
    """
    Generates a personalized investment plan.
    """
    prompt = f"""
    **Persona**: You are a Certified Financial Planner (CFP) and Wealth Manager for High Net Worth Individuals.
    
    **Task**: Create a detailed Investment Plan for a client with the following profile:
    - **Capital**: ₹{amount:,}
    - **Horizon**: {duration_years} Years
    - **Target Return**: {expected_return}% Annually
    - **Risk Profile**: {risk_profile}
    - **Current Market Context**: {market_context}
    
    **Constraints**:
    - Suggest a robust Asset Allocation (Equity vs Debt vs Gold).
    - Recommend sectors based on the current market context.
    - Suggest an investment schedule (Lumpsum vs SIP).
    - Be realistic about returns.
    
    **Format**:
    ## 🎯 Investment Strategy Report
    ### 1. Asset Allocation
    (Table or List)
    
    ### 2. Sectoral Focus
    (List of top 3 sectors)
    
    ### 3. Execution Plan
    (Step-by-step guide)
    
    ### 4. Risk Management
    (Hedging or diversification tips)
    """
    messages = [{"role": "user", "content": prompt}]
    return _call_ai_api(messages, model)

def generate_quant_investment_plan(json_payload, model="google/gemini-2.0-flash-exp:free"):
    """
    Generates a structured Investment Report based on Quant Engine's output.
    Uses AI to enhance the summary if available, otherwise falls back to template.
    """
    user = json_payload.get('meta', {}).get('user', {})
    summary = json_payload.get('backtest_summary', {})
    
    # Try AI Enhancement first
    client = _get_client()
    if client:
        prompt = f"""
        **Persona**: You are a Algo-Trading Portfolio Manager.
        
        **Task**: Review the backtest results of a quantitative strategy and write a commentary.
        
        **Data**:
        {json.dumps(json_payload, indent=2)}
        
        **Constraints**:
        - Analyze the Sharpe Ratio ({summary.get('sharpe', 0)}) and Max Drawdown ({summary.get('max_drawdown_pct', 0)}%).
        - Comment on the asset allocation concentration.
        - Explain why specific stocks might have been picked based on "High Momentum" or "Value" signals inferred from data.
        
        **Format**: Markdown. Keep it under 300 words. Start with "## 🤖 AI Portfolio Analysis".
        """
        messages = [{"role": "user", "content": prompt}]
        ai_commentary = _call_ai_api(messages, model)
    else:
        ai_commentary = "## 🤖 AI Analysis Unavailable (API Key missing)"

    # Static generation fallback/append
    allocation = json_payload.get('allocation', [])
    tickers_info = json_payload.get('tickers', [])
    ticker_stats = {t['ticker']: t for t in tickers_info}
    
    text = f"""
# 📋 Wealth Management Report

{ai_commentary}

---

## 📊 Quantitative Breakdown
**Risk Profile**: {user.get('risk_profile', 'Unknown')} | **Target Return**: {user.get('expected_annual_return_pct', 0)}%

### 🚀 Performance Metrics (Backtested)
| Metric | Strategy |
| :--- | :--- |
| **Annual Return** | **{summary.get('annualized_return_pct', 0):.2f}%** |
| **Volatility** | **{summary.get('annualized_vol_pct', 0):.2f}%** |
| **Sharpe Ratio** | **{summary.get('sharpe', 0):.2f}** |
| **Max Drawdown** | **{summary.get('max_drawdown_pct', 0):.2f}%** |

### 🏗️ Targeted Allocation
| Ticker | Weight | Amount | Signal |
| :--- | :--- | :--- | :--- |
"""
    for item in allocation:
        asset = item.get('ticker', 'Unknown')
        weight = item.get('weight_pct', 0)
        alloc_amt = item.get('amount', 0)
        stats = ticker_stats.get(asset, {})
        mom = stats.get('1y_return_pct', 0)
        
        text += f"| **{asset}** | {weight*100:.1f}% | ₹{alloc_amt:,.0f} | 1Y Return: {mom:.1f}% |\n"

    return text

def analyze_portfolio(portfolio_data, model="google/gemini-2.0-flash-exp:free"):
    """
    Analyzes a user-created portfolio for risk, diversification, and quality.
    """
    prompt = f"""
    **Persona**: You are a Portfolio Risk Manager at a large endowment fund.
    
    **Task**: stress-test the following portfolio constructed by a user.
    
    **Portfolio Data**:
    {json.dumps(portfolio_data, indent=2)}
    
    **Constraints**:
    - Identify concentration risk (sector or single stock).
    - Comment on the beta/volatility if inferable.
    - Provide 3 concrete suggestions to improve the Sharpe ratio.
    - Be critical but constructive.
    
    **Format**:
    ## 🛡️ Portfolio Health Check
    ### 🚨 Critical Risks
    - Point 1...
    
    ### ✅ Strengths
    - Point 1...
    
    ### 💡 Optimization Suggestions
    1. ...
    2. ...
    3. ...
    """
    messages = [{"role": "user", "content": prompt}]
    return _call_ai_api(messages, model)
