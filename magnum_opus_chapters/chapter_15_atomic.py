from reportlab.platypus import Paragraph, Spacer, Image, PageBreak
from .utils import create_custom_styles, add_dataframe_table, add_img

def generate(story, charts):
    """Generate Chapter 15: The Autonomous LangGraph Portfolio Manager"""
    styles = create_custom_styles()
    
    # metrics
    metrics = charts.get('_metrics', {})
    lstm_prob = metrics.get('lstm_prob', 82.0)
    lstm_uncertainty = metrics.get('lstm_uncertainty', 4.0)
    garch_vol = metrics.get('garch_vol', 2.5)
    user_risk = metrics.get('user_risk', 'Conservative')
    user_horizon = metrics.get('user_horizon', '5 Years')

    # Chapter Title
    story.append(Paragraph("Chapter 15: The Autonomous LangGraph Portfolio Manager", styles['ChapterTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("""<i>"The future belongs to those who understand that doing more with less 
    is compassionate, prosperous, and enduring, and thus more intelligent." — Buckminster Fuller</i>""", styles['Caption']))
    story.append(Spacer(1, 20))

    # 15.1 Agentic Architecture
    story.append(Paragraph("15.1 The Agentic Architecture: A Quant Fund in Code", styles['SectionTitle']))
    story.append(Paragraph("""
    Chapters 1-14 built the mathematical foundation: data engineering, technical indicators, regime 
    detection, machine learning models, and risk gating. But a quantitative model operating in isolation 
    is like a Formula 1 engine without a chassis — powerful but directionless.
    <br/><br/>In this chapter, we wrap the entire analytical stack inside a <b>LangGraph Multi-Agent AI 
    System</b> — an agentic workflow that simulates the decision-making process of an institutional 
    investment committee.
    <br/><br/><b>The Three Pillars of Autonomous Portfolio Management:</b>
    <br/><br/><b>Agent 1: The Quant Engine (Quantitative Analyst)</b>
    <br/>This agent runs our pre-trained Monte Carlo LSTM and GARCH models. It takes no shortcuts — it 
    performs 50 forward passes with active dropout, calculates the uncertainty distribution, and checks 
    the Volatility Gate. The output is a structured payload:
    <br/>• LSTM Buy Probability: {lstm_prob:.1f}%
    <br/>• Monte Carlo Uncertainty: ±{lstm_uncertainty:.1f}%
    <br/>• GARCH Predicted Volatility: {garch_vol:.2f}%
    <br/><br/><b>Agent 2: The Research Engine (Fundamental Analyst)</b>
    <br/>This agent deploys <b>Crawl4AI</b> — an async headless browser framework running Playwright — 
    to scrape the live internet for the latest financial news, earnings reports, and macroeconomic data 
    about the target ticker. Unlike a simple API call, Crawl4AI renders full JavaScript pages, bypasses 
    cookie popups and age gates, and converts the result into clean Markdown text.
    <br/><br/>Why does this matter? A model might identify a perfect technical setup for Tata Motors, but 
    if the company just announced a massive earnings miss, the technical signal is <i>invalidated</i>. 
    The Research Agent provides the fundamental "reality check."
    <br/><br/><b>Agent 3: The Reasoning Engine (Chief Investment Officer)</b>
    <br/>This is where <b>DeepSeek-R1</b>, accessed via the OpenRouter API, performs chain-of-thought 
    reasoning. It doesn't just summarize the data — it <i>debates</i> it. The AI acts as a fiduciary: 
    it considers the user's risk tolerance, investment horizon, and capital constraints before issuing 
    a recommendation.
    """, styles['BodyTextCustom']))

    if 'langgraph_architecture' in charts:
        add_img(story, charts['langgraph_architecture'],
                "Figure 15.1: LangGraph Multi-Agent Pipeline — From Data to Decision", styles)

    story.append(Paragraph("""
    <b>Why LangGraph (Not LangChain)?</b> LangChain is a linear chain of LLM calls. LangGraph is a 
    <i>graph</i> — it supports conditional routing, cycles, and parallel execution. In our system, 
    the Quant Engine and Research Agent could theoretically run in parallel (both are independent data 
    gatherers), with their outputs converging at the Reasoning Agent. This graph-based architecture 
    is how real-world agentic systems operate in production.
    """, styles['InsightBox']))

    # 15.2 DeepSeek Chain-of-Thought
    story.append(Paragraph("15.2 DeepSeek Chain-of-Thought: The AI Investment Committee", styles['SectionTitle']))
    story.append(Paragraph(f"""
    The raw mathematical data and live news are fed into <b>DeepSeek-R1</b> — a reasoning-optimized 
    large language model — through a carefully engineered system prompt that instructs the AI to act 
    as a Chief Investment Officer.
    <br/><br/><b>The Synthesis Process:</b>
    <br/><br/>1. <b>The Quantitative View (Paragraph 1):</b> The CIO translates LSTM probabilities, 
    Monte Carlo uncertainty bands, and GARCH volatility forecasts into plain English. No jargon without 
    explanation. The reader — whether a seasoned portfolio manager or a first-time SIP investor — must 
    understand exactly what the numbers mean.
    <br/><br/>2. <b>The Fundamental View (Paragraph 2):</b> The CIO synthesizes the scraped news, 
    identifying the single biggest tailwind and the single biggest risk. For Tata Motors, a tailwind 
    might be "India's infrastructure spending under the National Infrastructure Pipeline drives CV demand," 
    while a risk might be "rising global HRC steel prices (+5% QoQ) compress vehicle margins."
    <br/><br/>3. <b>The Verdict & Sizing (Paragraph 3):</b> This is where the <b>user profile</b> 
    becomes critical. The exact same LSTM and GARCH data will produce different recommendations for 
    different users:
    <br/>• A <b>{user_risk}</b> investor with a <b>{user_horizon}</b> horizon receives a measured allocation 
    with Half-Kelly sizing and a trailing stop-loss.
    <br/>• An <b>Aggressive</b> investor with a <b>1 Year</b> horizon might receive a Full Kelly allocation 
    with a wider stop.
    <br/>• A <b>Defensive</b> investor might receive a "HOLD" recommendation even when the math is bullish, 
    because the fundamental risks outweigh the technical edge for their risk profile.
    <br/><br/><b>The Key Principle:</b> The AI is not making the decision <i>for</i> the user. It is 
    presenting the data transparently and reasoning through the decision <i>with</i> the user. Every 
    recommendation is accompanied by the exact data that produced it. This is the <b>fiduciary 
    transparency</b> standard that separates institutional advice from retail guesswork.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    <b>Why DeepSeek-R1 Specifically?</b> Most LLMs (GPT-4o, Claude) process queries as a single forward 
    pass — they produce an answer immediately. DeepSeek-R1 is a <i>reasoning model</i> — it generates 
    an internal "chain of thought" before producing its final answer. This means it can weigh conflicting 
    evidence (bullish technicals vs. bearish fundamentals), consider edge cases, and arrive at a 
    <i>nuanced</i> conclusion. For financial analysis, where the answer is almost never black-or-white, 
    this reasoning capability is indispensable.
    """, styles['InsightBox']))

    # 15.3 The Final Output
    story.append(Paragraph("15.3 The Final Output: Hyper-Personalized Portfolio Intelligence", styles['SectionTitle']))
    story.append(Paragraph(f"""
    The culmination of the 15-notebook analytical pipeline is a dynamically generated, 
    <b>hyper-personalized portfolio report</b>. This report is not a static template — every word, 
    every number, and every recommendation is computed in real-time from the target ticker and the 
    user's specific constraints.
    <br/><br/><b>What the System Produces:</b>
    <br/>• A probability-weighted trade recommendation with explicit confidence bounds
    <br/>• A risk-adjusted position size using Kelly Criterion (or Half-Kelly for conservative profiles)
    <br/>• A fundamental synthesis of live news that identifies tailwinds and headwinds
    <br/>• A transparent reasoning chain showing exactly <i>how</i> and <i>why</i> the recommendation was made
    <br/><br/><b>Sample Output Summary for {user_risk} Investor ({user_horizon} SIP):</b>
    <br/>• LSTM Signal: {lstm_prob:.1f}% bullish (±{lstm_uncertainty:.1f}% uncertainty)
    <br/>• GARCH Gate: {'OPEN — volatility within threshold' if garch_vol < 3.0 else 'CLOSED — excessive volatility'}
    <br/>• Recommendation: {'Allocate at Half-Kelly (~8% of portfolio) with 3% trailing stop' if garch_vol < 3.0 else 'Hold cash — wait for volatility regime to normalize'}
    <br/><br/><b>The Institutional Edge:</b> This level of transparent, data-driven, risk-aware portfolio 
    construction was previously available <i>only</i> to institutional high-net-worth clients paying 
    1-2% management fees to quantitative hedge funds. Our system democratizes this capability by 
    making the entire analytical chain auditable, reproducible, and personalized.
    """, styles['BodyTextCustom']))

    # 15.4 System Limitations
    story.append(Paragraph("15.4 Honest Limitations & Future Roadmap", styles['SectionTitle']))
    story.append(Paragraph("""
    No system is infallible. Transparency demands we acknowledge the boundaries:
    <br/><br/><b>Current Limitations:</b>
    <br/>• <b>LSTM is not a crystal ball:</b> MC Dropout provides uncertainty, but cannot predict 
    Black Swan events (pandemics, geopolitical shocks, sudden leadership changes).
    <br/>• <b>Crawl4AI depends on website availability:</b> If Google or financial news sites change 
    their HTML structure, the scraper requires maintenance.
    <br/>• <b>DeepSeek-R1 is not financial advice:</b> The reasoning model can hallucinate if given 
    contradictory or sparse input data. All outputs should be treated as analytical support, not 
    fiduciary recommendations.
    <br/>• <b>Latency:</b> The full pipeline (GARCH → Crawl4AI → DeepSeek) takes 30-60 seconds. 
    This is acceptable for daily-horizon strategies but insufficient for high-frequency trading.
    <br/><br/><b>Future Roadmap (V3.0):</b>
    <br/>• <b>Hidden Markov Model (HMM) Regime Router:</b> Classify the market regime <i>first</i> 
    (Bull Quiet, Bear Fast, Chop), then deploy regime-specific sub-agents.
    <br/>• <b>Multi-Asset Correlation Engine:</b> Extend from single-stock analysis to portfolio-level 
    optimization using Markowitz mean-variance with Monte Carlo constraints.
    <br/>• <b>Reinforcement Learning Position Manager:</b> Replace static Kelly Criterion with a learned 
    policy that adapts position sizing based on recent P&L trajectory.
    <br/>• <b>Intraday Micro-Structure Integration:</b> Combine the daily LSTM signal with a 5-minute 
    execution agent that optimizes entry timing using volume profile analysis.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    <b>The Philosophy:</b> This project is not about building a "stock prediction bot." It is about 
    building a <b>disciplined, autonomous, and transparent trading system</b> that respects the 
    uncertainty of financial markets. The system is designed to:
    <br/>1. <b>Quantify conviction</b> (LSTM + Monte Carlo)
    <br/>2. <b>Gate risk</b> (GARCH Volatility Filter)
    <br/>3. <b>Check fundamentals</b> (Crawl4AI live scraping)
    <br/>4. <b>Reason transparently</b> (DeepSeek chain-of-thought)
    <br/>5. <b>Personalize advice</b> (LangGraph user profile routing)
    <br/><br/>Every number is auditable. Every decision is explainable. Every recommendation comes 
    with a confidence interval. This is the institutional standard.
    """, styles['InsightBox']))

    story.append(PageBreak())
