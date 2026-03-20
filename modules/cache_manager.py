"""
cache_manager.py — Centralized Session-Level Cache for Streamlit App.
=====================================================================
All expensive I/O (stock data, news, AI analysis) is cached in 
st.session_state keyed by (ticker, params). Data persists across tab 
switches and is only re-fetched when the user changes stock or parameters.
"""
import streamlit as st
import hashlib


# ────────────────────────────────────────────────────
# CACHE KEYS
# ────────────────────────────────────────────────────
def _key(prefix: str, ticker: str, extra: str = "") -> str:
    """Generate a deterministic cache key."""
    raw = f"{prefix}_{ticker}_{extra}"
    return raw


# ────────────────────────────────────────────────────
# NEWS CACHE
# ────────────────────────────────────────────────────
def get_cached_news(ticker: str):
    """Return cached news articles for a ticker, or None."""
    k = _key("news", ticker)
    return st.session_state.get(k)


def set_cached_news(ticker: str, articles: list):
    """Cache news articles for a ticker."""
    k = _key("news", ticker)
    st.session_state[k] = articles


def get_or_fetch_news(ticker: str, max_articles: int = 5, enrich: bool = True):
    """
    Return cached news if available, else fetch + analyze + cache.
    This is the single entry point for all news across all pages.
    """
    cached = get_cached_news(ticker)
    if cached is not None:
        return cached

    try:
        from modules.data.scrapers.news_analysis import (
            fetch_top_news, analyze_news_with_ai
        )
        articles = fetch_top_news(ticker, max_articles=max_articles)
        if articles and enrich:
            # Optional: enrich with crawl4ai (best-effort)
            try:
                from modules.data.scrapers.news_analysis import enrich_articles_with_crawl4ai
                articles = enrich_articles_with_crawl4ai(articles, max_articles=3)
            except Exception:
                pass
            articles = analyze_news_with_ai(articles, ticker)
        set_cached_news(ticker, articles)
        return articles
    except Exception as e:
        print(f"[CacheManager] News fetch failed: {e}")
        return []


# ────────────────────────────────────────────────────
# SIMPLE NEWS CACHE (yfinance + Google RSS fallback)
# ────────────────────────────────────────────────────
def get_cached_simple_news(ticker: str, company_name: str = ""):
    """Return cached simple news items for a ticker, or None."""
    k = _key("simple_news", ticker)
    return st.session_state.get(k)


def set_cached_simple_news(ticker: str, news_items: list):
    """Cache simple news items for a ticker."""
    k = _key("simple_news", ticker)
    st.session_state[k] = news_items


# ────────────────────────────────────────────────────
# STOCK DATA CACHE
# ────────────────────────────────────────────────────
def get_cached_stock_data(ticker: str, period: str = "1y"):
    """Return cached stock data bundle, or None."""
    k = _key("stock_data", ticker, period)
    return st.session_state.get(k)


def set_cached_stock_data(ticker: str, period: str, bundle: dict):
    """Cache a stock data bundle (df, info, df_feat, live_data, yf_ticker)."""
    k = _key("stock_data", ticker, period)
    st.session_state[k] = bundle


# ────────────────────────────────────────────────────
# INVALIDATION
# ────────────────────────────────────────────────────
def invalidate_ticker(ticker: str):
    """Clear ALL caches for a given ticker (called when params change)."""
    keys_to_remove = [k for k in st.session_state if k.startswith(f"news_{ticker}") or 
                      k.startswith(f"simple_news_{ticker}") or
                      k.startswith(f"stock_data_{ticker}") or
                      k.startswith(f"ai_analysis_{ticker}")]
    for k in keys_to_remove:
        del st.session_state[k]
