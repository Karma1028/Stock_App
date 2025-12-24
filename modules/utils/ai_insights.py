import os
import json
import streamlit as st
from config import Config

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

def _call_ai_api(messages, model="google/gemma-3-27b-it:free"):
    if OpenAI is None:
        return "⚠️ OpenAI module not found. Please install it to use AI features."
        
    client = _get_client()
    if not client:
        return "⚠️ AI API Key is missing. Please set OPENROUTER_API_KEY in .env or Sidebar."
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error calling AI API: {e}"

def generate_company_summary(symbol, model="google/gemma-3-27b-it:free"):
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

def get_ai_insights(symbol, data_summary, model="google/gemma-3-27b-it:free"):
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

def generate_investment_plan(amount, duration_years, expected_return, risk_profile, market_context, model="google/gemma-3-27b-it:free"):
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

def generate_quant_investment_plan(json_payload, model="google/gemma-3-27b-it:free"):
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

def analyze_portfolio(portfolio_data, model="google/gemma-3-27b-it:free"):
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
