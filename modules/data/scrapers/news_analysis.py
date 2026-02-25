"""
news_analysis.py — AI-Powered News Impact Analysis
====================================================
Fetches top engagement news via Google News RSS (public, legal),
ranks by source authority, and generates per-article AI analysis
using OpenRouter free models with minimal token usage.
Rotates across ALL API keys to maximize free-tier throughput.
"""

import os
import json
import requests
import feedparser
import urllib.parse
from datetime import datetime, timedelta
from textblob import TextBlob
from config import Config


# Tier-1 financial sources get 2x engagement weight
TIER1_SOURCES = {
    'economic times', 'moneycontrol', 'livemint', 'reuters',
    'bloomberg', 'cnbc', 'business standard', 'financial express',
    'ndtv profit', 'mint', 'the hindu businessline', 'business today',
    'yahoo finance', 'investopedia', 'seeking alpha', 'barrons'
}


def _source_weight(source_name: str) -> float:
    """Returns a weight multiplier based on source authority."""
    lower = source_name.lower()
    for tier1 in TIER1_SOURCES:
        if tier1 in lower:
            return 2.0
    return 1.0


def fetch_top_news(ticker: str, max_articles: int = 5) -> list:
    """
    Fetches top engagement news for a stock ticker via Google News RSS.
    Returns list of dicts: {title, source, link, published, sentiment, score}
    Ranked by recency x source authority.
    """
    search_term = ticker.replace('.NS', '').replace('.BO', '')
    
    # Two queries: company-specific + stock market context
    queries = [
        f"{search_term} stock news",
        f"{search_term} share price analysis",
    ]
    
    all_articles = []
    base_url = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    for query in queries:
        try:
            encoded = urllib.parse.quote(query)
            url = base_url.format(query=encoded)
            feed = feedparser.parse(url)
            
            for entry in feed.entries:
                try:
                    dt_struct = entry.published_parsed
                    pub_date = datetime(*dt_struct[:6]) if dt_struct else datetime.now()
                    
                    source = entry.source.title if hasattr(entry, 'source') and hasattr(entry.source, 'title') else 'Google News'
                    title = entry.title
                    
                    # TextBlob sentiment
                    sentiment = TextBlob(title).sentiment.polarity
                    
                    # Engagement score = recency_weight x source_weight x abs(sentiment)
                    days_ago = (datetime.now() - pub_date).days
                    recency_w = max(0.1, 1.0 - (days_ago * 0.1))  # Decays over 10 days
                    source_w = _source_weight(source)
                    engagement_score = recency_w * source_w * (0.5 + abs(sentiment))
                    
                    all_articles.append({
                        'title': title,
                        'source': source,
                        'link': entry.link,
                        'published': pub_date.strftime('%Y-%m-%d %H:%M'),
                        'sentiment': round(sentiment, 3),
                        'engagement_score': round(engagement_score, 3),
                    })
                except Exception:
                    continue
        except Exception:
            continue
    
    # Deduplicate by title similarity (exact match for now)
    seen_titles = set()
    unique = []
    for a in all_articles:
        key = a['title'][:60].lower()
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(a)
    
    # Sort by engagement score, return top N
    unique.sort(key=lambda x: x['engagement_score'], reverse=True)
    return unique[:max_articles]


# ═══════════════════════════════════════════════════
# CRAWL4AI — full article text extraction
# ═══════════════════════════════════════════════════

def _crawl_article_text(url: str, max_chars: int = 500) -> str:
    """Fetch full article text via crawl4ai. Returns clean text excerpt."""
    try:
        import asyncio
        from crawl4ai import AsyncWebCrawler
        
        async def _fetch():
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await crawler.arun(url=url)
                if result and result.markdown:
                    text = result.markdown.strip()
                    if len(text) > max_chars:
                        text = text[:max_chars] + "..."
                    return text
            return ""
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(lambda: asyncio.run(_fetch())).result(timeout=15)
            else:
                return loop.run_until_complete(_fetch())
        except RuntimeError:
            return asyncio.run(_fetch())
    except Exception as e:
        print(f"   [crawl4ai] Error crawling {url}: {e}")
        return ""


def enrich_articles_with_crawl4ai(articles: list, max_articles: int = 3) -> list:
    """Enrich top articles with full text via crawl4ai."""
    for i, article in enumerate(articles[:max_articles]):
        if 'full_text' not in article or not article['full_text']:
            url = article.get('link', '')
            if url:
                print(f"   [crawl4ai] Fetching article {i+1}/{min(len(articles), max_articles)}...")
                text = _crawl_article_text(url)
                article['full_text'] = text
    return articles


def analyze_news_with_ai(articles: list, ticker: str) -> list:
    """
    Sends top news articles to OpenRouter free model for per-article analysis.
    Returns articles enriched with: impact_rating, direction, key_takeaway.
    Rotates across ALL API keys to avoid rate limits.
    """
    if not articles:
        return articles
    
    # Load all API keys
    api_keys = getattr(Config, 'OPENROUTER_API_KEYS', [])
    if not api_keys:
        # Try single key fallback
        single = Config.OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY", "")
        api_keys = [single] if single else []
    api_keys = [k for k in api_keys if k]
    
    if not api_keys:
        # No API keys — use sentiment fallback
        for a in articles:
            s = a['sentiment']
            a['direction'] = 'Bullish' if s > 0.1 else ('Bearish' if s < -0.1 else 'Neutral')
            a['impact_rating'] = min(5, max(1, int(abs(s) * 10) + 1))
            a['key_takeaway'] = f"Sentiment {'positive' if s > 0 else 'negative' if s < 0 else 'neutral'} based on headline analysis."
        return articles
    
    # Build compact prompt
    headlines = "\n".join([f"{i+1}. [{a['source']}] {a['title']}" for i, a in enumerate(articles)])
    
    prompt = f"""Analyze these {len(articles)} news headlines for {ticker.replace('.NS','')} stock.
For EACH headline, respond with ONLY a JSON array:
[{{"id":1,"impact":3,"dir":"Bullish","take":"one-line takeaway"}},...]

impact: 1-5 (5=highest market impact)
dir: Bullish/Bearish/Neutral
take: max 15 words

Headlines:
{headlines}"""
    
    models = Config.AI_MODEL_FALLBACKS
    
    # Rotate across all keys x models (4 keys x 7 models = 28 attempts)
    for current_key in api_keys:
        for model in models:
            try:
                from openai import OpenAI
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=current_key)
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a financial news analyst. Respond ONLY with valid JSON array. No markdown, no explanation."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400,
                    timeout=30
                )
                
                raw = response.choices[0].message.content.strip()
                # Extract JSON from potential markdown wrapping
                if '```' in raw:
                    raw = raw.split('```')[1]
                    if raw.startswith('json'):
                        raw = raw[4:]
                
                # Find the JSON array
                start = raw.find('[')
                end = raw.rfind(']') + 1
                if start >= 0 and end > start:
                    raw = raw[start:end]
                
                analysis = json.loads(raw)
                
                # Merge analysis back into articles
                for item in analysis:
                    idx = item.get('id', 0) - 1
                    if 0 <= idx < len(articles):
                        articles[idx]['impact_rating'] = min(5, max(1, item.get('impact', 3)))
                        articles[idx]['direction'] = item.get('dir', 'Neutral')
                        articles[idx]['key_takeaway'] = item.get('take', 'Analysis unavailable.')
                
                # Fill any gaps
                for a in articles:
                    if 'impact_rating' not in a:
                        s = a['sentiment']
                        a['direction'] = 'Bullish' if s > 0.1 else ('Bearish' if s < -0.1 else 'Neutral')
                        a['impact_rating'] = 3
                        a['key_takeaway'] = 'Analysis pending.'
                
                print(f"   [NewsAI] Analysis complete via {model}")
                return articles
                
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str:
                    print(f"   [NewsAI] Key rate-limited, switching to next key...")
                    break  # break inner model loop, try next key
                print(f"   [NewsAI] Model {model} failed: {e}")
                continue
    
    # All keys/models failed — use sentiment fallback
    for a in articles:
        s = a['sentiment']
        a['direction'] = 'Bullish' if s > 0.1 else ('Bearish' if s < -0.1 else 'Neutral')
        a['impact_rating'] = min(5, max(1, int(abs(s) * 10) + 1))
        a['key_takeaway'] = f"Based on sentiment: {'positive' if s > 0 else 'negative' if s < 0 else 'mixed'} outlook."
    
    return articles


def render_news_tiles(articles: list):
    """Renders news impact tiles in Streamlit with glassmorphic styling."""
    import streamlit as st
    
    if not articles:
        st.info("No recent news found for this stock.")
        return
    
    for a in articles:
        direction = a.get('direction', 'Neutral')
        impact = a.get('impact_rating', 3)
        
        # Direction badge
        if direction == 'Bullish':
            badge_color = "#22c55e"
            badge_icon = "🟢"
        elif direction == 'Bearish':
            badge_color = "#ef4444"
            badge_icon = "🔴"
        else:
            badge_color = "#eab308"
            badge_icon = "🟡"
        
        # Impact stars
        stars = "⭐" * impact + "☆" * (5 - impact)
        
        st.markdown(f"""
        <div style="
            background: rgba(30, 41, 59, 0.6);
            border: 1px solid rgba(255,255,255,0.1);
            border-left: 3px solid {badge_color};
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 10px;
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="color:#CBD5E1; font-size:0.78rem;">{a.get('source', 'News')} · {a.get('published', '')}</span>
                <span style="font-size:0.75rem;">{badge_icon} {direction} | Impact: {stars}</span>
            </div>
            <div style="color:#F1F5F9; font-size:0.92rem; font-weight:500; margin-bottom:6px;">
                <a href="{a.get('link', '#')}" target="_blank" style="color:#F1F5F9; text-decoration:none;">
                    {a.get('title', 'Untitled')}
                </a>
            </div>
            <div style="color:#94A3B8; font-size:0.82rem; font-style:italic;">
                💡 {a.get('key_takeaway', 'Analysis pending.')}
            </div>
        </div>
        """, unsafe_allow_html=True)
