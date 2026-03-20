"""
Unified News Scraper v6.0 — PRODUCTION GRADE (NO Google News)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Google News RSS redirect URLs are JS-only and CANNOT be resolved
by any HTTP library. This version uses sources with REAL article URLs.

Link Sources (all return real, scrapeable URLs):
  1. Bing News RSS
  2. DuckDuckGo News RSS
  3. Indian Financial RSS feeds (ET, MoneyControl, LiveMint, etc.)

Content Extraction (5-tier):
  1. trafilatura (precision text extractor)
  2. newspaper3k (NLP article parser)
  3. BeautifulSoup (article-tag aware)
  4. Crawl4AI (JS rendering for dynamic sites)
  5. Playwright (headless browser last resort)

Also: GDELT geopolitical events (with fast timeout)
"""
import asyncio
import os
import pandas as pd
import logging
import random
import re
import requests
import feedparser
import urllib.parse
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from crawl4ai import AsyncWebCrawler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from bs4 import BeautifulSoup
import trafilatura
from newspaper import Article, Config as NewspaperConfig

# Optional engines
try:
    from playwright.async_api import async_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

console = Console()
os.makedirs('Universal_Engine_Workspace/mas_logs', exist_ok=True)
logging.basicConfig(
    level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s',
    filename="Universal_Engine_Workspace/mas_logs/master_scraper.log"
)
logger = logging.getLogger('MasterScraper')

# Suppress noisy logs
os.environ['WDM_LOG'] = '0'
for n in ['WDM', 'selenium', 'urllib3', 'gnews', 'trafilatura']:
    logging.getLogger(n).setLevel(logging.CRITICAL)

# ── Paths ──
DATA_DIR = Path("data")
NEWS_HISTORY_DIR = DATA_DIR / "news_history"
GEOPOLITICAL_DIR = DATA_DIR / "geopolitical"
NEWS_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
GEOPOLITICAL_DIR.mkdir(parents=True, exist_ok=True)
DISCARD_CSV_PATH = NEWS_HISTORY_DIR / "discarded_replaced_news.csv"

# ── Blocked domains ──
BLOCKED_DOMAINS = {
    'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com',
    'linkedin.com', 'pinterest.com', 'reddit.com', 'tiktok.com',
    'play.google.com', 'apps.apple.com', 'apple.news',
    'accounts.google.com', 'support.google.com', 'consent.google.com',
    'news.google.com', 'google.com/sorry', 'web.archive.org',
}

# ── Paywall markers ──
PAYWALL_MARKERS = [
    "subscribe to read", "access denied", "please log in",
    "enable javascript", "sign up to continue", "create account",
    "premium content", "paywall", "member only", "register to continue",
    "you have reached your limit", "free articles remaining",
    "turn off your ad blocker",
]

# ── newspaper3k config ──
NP_CONFIG = NewspaperConfig()
NP_CONFIG.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36'
NP_CONFIG.request_timeout = 7
NP_CONFIG.fetch_images = False
NP_CONFIG.memoize_articles = False
NP_CONFIG.language = 'en'

# ── Indian Financial News RSS Feeds (direct, real URLs!) ──
INDIA_FINANCE_RSS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/results.xml",
    "https://www.livemint.com/rss/markets",
    "https://www.business-standard.com/rss/markets-106.rss",
    "https://www.thehindubusinessline.com/markets/feeder/default.rss",
]


class MasterNewsScraper:
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
        self.session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ═══════════════════════════════════════
    #  TICKER LIST
    # ═══════════════════════════════════════
    def get_all_equity_tickers(self):
        equity_csv = DATA_DIR / 'EQUITY.csv'
        try:
            df = pd.read_csv(equity_csv)
            return [
                {'symbol': str(row['SYMBOL']).strip() + '.NS', 'name': str(row['NAME OF COMPANY']).strip()}
                for _, row in df.iterrows()
            ]
        except Exception as e:
            logger.error(f"Failed to read EQUITY.csv: {e}")
            return []

    # ═══════════════════════════════════════
    #  URL HELPERS
    # ═══════════════════════════════════════
    def is_valid_url(self, url):
        if not url or not url.startswith('http'):
            return False
        url_lower = url.lower()
        return not any(blocked in url_lower for blocked in BLOCKED_DOMAINS)

    def _is_paywall(self, text):
        if not text: return True
        return any(m in text.lower() for m in PAYWALL_MARKERS)

    # ═══════════════════════════════════════
    #  LINK FETCHING — Bing + DuckDuckGo + Indian RSS
    # ═══════════════════════════════════════
    def fetch_links(self, ticker, name):
        """Fetch real article URLs from Bing, DuckDuckGo, and Indian financial RSS."""
        clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
        all_articles = []
        seen_urls = set()

        # Base queries
        base_queries = [
            f'{clean_ticker} stock news India',
            f'{name} quarterly results earnings',
            f'{clean_ticker} dividend bonus split merger',
            f'{name} share price market',
        ]

        # Year-specific queries for 5-year coverage (2021-2026)
        current_year = datetime.now().year
        year_queries = []
        for year in range(current_year - 5, current_year + 1):
            year_queries.append(f'{clean_ticker} stock {year}')
            year_queries.append(f'{name} results {year}')

        all_queries = base_queries + year_queries

        # ── Source 1: Bing News RSS (gives REAL URLs, 5-year coverage) ──
        for query in all_queries:
            try:
                encoded = urllib.parse.quote(query)
                resp = self.session.get(
                    f"https://www.bing.com/news/search?q={encoded}&format=rss",
                    headers={'User-Agent': random.choice(self.user_agents)}, timeout=5,
                )
                if resp.status_code == 200:
                    feed = feedparser.parse(resp.content)
                    for entry in feed.entries:
                        url = entry.link
                        if url in seen_urls or not self.is_valid_url(url):
                            continue
                        seen_urls.add(url)
                        pub = self._parse_pub_date(entry)
                        all_articles.append({
                            'Ticker': ticker, 'Date': pub,
                            'Title': entry.title, 'Source': 'Bing News',
                            'Link': url, 'Full_Title': None, 'Content': None,
                        })
            except Exception:
                pass

        # ── Source 2: DuckDuckGo News RSS (gives REAL URLs) ──
        for query in base_queries[:2]:  # First 2 queries only
            try:
                encoded = urllib.parse.quote(query)
                resp = self.session.get(
                    f"https://duckduckgo.com/?q={encoded}&iar=news&ia=news&format=rss",
                    headers={'User-Agent': random.choice(self.user_agents)}, timeout=5,
                )
                if resp.status_code == 200:
                    feed = feedparser.parse(resp.content)
                    for entry in feed.entries:
                        url = entry.link
                        if url in seen_urls or not self.is_valid_url(url):
                            continue
                        seen_urls.add(url)
                        pub = self._parse_pub_date(entry)
                        all_articles.append({
                            'Ticker': ticker, 'Date': pub,
                            'Title': entry.title, 'Source': 'DuckDuckGo',
                            'Link': url, 'Full_Title': None, 'Content': None,
                        })
            except Exception:
                pass

        # ── Source 3: Indian Financial News RSS (ticker-specific filter) ──
        name_lower = name.lower()
        ticker_lower = clean_ticker.lower()
        for rss_url in INDIA_FINANCE_RSS:
            try:
                resp = self.session.get(rss_url, headers={'User-Agent': random.choice(self.user_agents)}, timeout=5)
                if resp.status_code == 200:
                    feed = feedparser.parse(resp.content)
                    for entry in feed.entries:
                        title_lower = entry.title.lower()
                        # Only include if the entry mentions this ticker/company
                        if ticker_lower in title_lower or name_lower in title_lower or any(
                            word in title_lower for word in name_lower.split()[:2] if len(word) > 3
                        ):
                            url = entry.link
                            if url in seen_urls or not self.is_valid_url(url):
                                continue
                            seen_urls.add(url)
                            pub = self._parse_pub_date(entry)
                            all_articles.append({
                                'Ticker': ticker, 'Date': pub,
                                'Title': entry.title, 'Source': rss_url.split('/')[2],
                                'Link': url, 'Full_Title': None, 'Content': None,
                            })
            except Exception:
                pass

        # ── Source 4: Yahoo Finance RSS ──
        try:
            resp = self.session.get(
                f"https://finance.yahoo.com/rss/headline?s={clean_ticker}.NS",
                headers={'User-Agent': random.choice(self.user_agents)}, timeout=5,
            )
            if resp.status_code == 200:
                feed = feedparser.parse(resp.content)
                for entry in feed.entries:
                    if entry.link not in seen_urls and self.is_valid_url(entry.link):
                        seen_urls.add(entry.link)
                        pub = self._parse_pub_date(entry)
                        all_articles.append({
                            'Ticker': ticker, 'Date': pub,
                            'Title': entry.title, 'Source': 'Yahoo Finance',
                            'Link': entry.link, 'Full_Title': None, 'Content': None,
                        })
        except Exception:
            pass

        return all_articles

    @staticmethod
    def _parse_pub_date(entry):
        if hasattr(entry, 'published_parsed') and entry.published_parsed:
            try:
                return str(datetime(*entry.published_parsed[:6]))
            except Exception:
                pass
        return str(datetime.now())

    # ═══════════════════════════════════════
    #  CONTENT EXTRACTION — 5 ENGINES
    # ═══════════════════════════════════════

    # Engine 1: trafilatura (fastest and most accurate for news)
    def engine_trafilatura(self, url):
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, include_comments=False,
                                            include_tables=False, no_fallback=False,
                                            favor_precision=True)
                if text and len(text) >= 200 and not self._is_paywall(text):
                    title = None
                    try:
                        soup = BeautifulSoup(downloaded, "lxml")
                        title = soup.title.string.strip() if soup.title and soup.title.string else None
                    except Exception:
                        pass
                    return title, text
        except Exception:
            pass
        return None, None

    # Engine 2: newspaper3k (NLP-based)
    def engine_newspaper(self, url):
        try:
            article = Article(url, config=NP_CONFIG)
            article.download()
            article.parse()
            if article.text and len(article.text) >= 200 and not self._is_paywall(article.text):
                return article.title, article.text
        except Exception:
            pass
        return None, None

    # Engine 3: BS4 (article-tag aware)
    def engine_bs4(self, url):
        try:
            resp = self.session.get(url, headers={'User-Agent': random.choice(self.user_agents)},
                                     timeout=7, allow_redirects=True)
            if resp.status_code != 200:
                return None, None
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "aside", "header", "noscript", "form", "iframe", "svg"]):
                tag.decompose()

            title = soup.title.string.strip() if soup.title and soup.title.string else None
            root = (soup.find('article')
                    or soup.find('div', class_=re.compile(r'article[-_]?body|story[-_]?content|post[-_]?content|entry[-_]?content', re.I))
                    or soup.find('div', {'itemprop': 'articleBody'})
                    or soup)

            paragraphs = []
            for p in root.find_all('p'):
                text = p.get_text(separator=' ', strip=True)
                if len(text) > 50 and text.count(' ') > 8 and not self._is_paywall(text):
                    paragraphs.append(text)

            content = "\n\n".join(paragraphs)
            if content and len(content) > 200:
                return title, content
        except Exception:
            pass
        return None, None

    # Engine 4: Crawl4AI (JS rendering)
    async def _crawl4ai_async(self, url):
        try:
            async with AsyncWebCrawler(verbose=False) as crawler:
                result = await asyncio.wait_for(
                    crawler.arun(url=url, magic=True, simulate_user=True, bypass_cache=True,
                                 word_count_threshold=20, page_timeout=8000,
                                 excluded_tags=['nav', 'footer', 'aside', 'header', 'script', 'style']),
                    timeout=10.0
                )
                if result.success:
                    text = getattr(result, 'fit_markdown', None) or result.markdown
                    if text:
                        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
                        text = re.sub(r'[*#>!\-]+', ' ', text)
                        text = re.sub(r'\s+', ' ', text).strip()
                        if len(text) >= 200 and not self._is_paywall(text):
                            title = result.metadata.get('og:title') or result.metadata.get('title')
                            return title, text
        except Exception:
            pass
        return None, None

    def engine_crawl4ai(self, url):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._crawl4ai_async(url))
            finally:
                loop.close()
        except Exception:
            return None, None

    # Engine 5: Playwright (headless browser)
    async def _playwright_async(self, url):
        if not HAS_PLAYWRIGHT:
            return None, None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                ctx = await browser.new_context(user_agent=random.choice(self.user_agents))
                page = await ctx.new_page()
                page.set_default_timeout(8000)
                await page.goto(url, wait_until="domcontentloaded")
                await asyncio.sleep(1)
                html = await page.content()
                await browser.close()
                return self._parse_html(html)
        except Exception:
            return None, None

    def engine_playwright(self, url):
        if not HAS_PLAYWRIGHT:
            return None, None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._playwright_async(url))
            finally:
                loop.close()
        except Exception:
            return None, None

    def _parse_html(self, html):
        try:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "aside", "header", "noscript", "form", "iframe"]):
                tag.decompose()
            title = soup.title.string.strip() if soup.title and soup.title.string else None
            root = (soup.find('article')
                    or soup.find('div', class_=re.compile(r'article[-_]?body|story[-_]?content', re.I))
                    or soup)
            paragraphs = [p.get_text(separator=' ', strip=True) for p in root.find_all('p')
                          if len(p.get_text(strip=True)) > 50 and not self._is_paywall(p.get_text(strip=True))]
            content = "\n\n".join(paragraphs)
            if content and len(content) > 200:
                return title, content
        except Exception:
            pass
        return None, None

    # ═══════════════════════════════════════
    #  MASTER CONTENT EXTRACTION
    # ═══════════════════════════════════════
    def fetch_content(self, url):
        """5-tier extraction. Returns (title, content, engine_name)."""
        if not self.is_valid_url(url):
            return None, None, "blocked"

        engines = [
            ("trafilatura",  self.engine_trafilatura),
            ("newspaper",    self.engine_newspaper),
            ("bs4",          self.engine_bs4),
            ("crawl4ai",     self.engine_crawl4ai),
            ("playwright",   self.engine_playwright),
        ]

        for eng_name, fn in engines:
            try:
                title, content = fn(url)
                if content and len(content) >= 200:
                    clean = re.sub(r'\s+', ' ', content).strip()
                    return title, clean[:6000], eng_name
            except Exception:
                continue

        return None, None, "all_failed"

    # ═══════════════════════════════════════
    #  GEOPOLITICAL NEWS SCRAPING (full content)
    # ═══════════════════════════════════════
    def scrape_geopolitical_news(self, progress_ui):
        """Scrape geopolitical / macro-economic news with FULL article content."""
        save_path = GEOPOLITICAL_DIR / "geopolitical_news_scraped.csv"

        # Load existing
        df = pd.DataFrame()
        existing_links = set()
        if save_path.exists():
            df = pd.read_csv(save_path)
            if 'Link' in df.columns:
                existing_links = set(df['Link'].dropna().tolist())

        GEO_QUERIES = [
            # India macro
            'India GDP growth forecast',
            'RBI interest rate monetary policy',
            'India inflation CPI WPI',
            'India budget fiscal deficit',
            'India rupee dollar exchange rate',
            # Global macro
            'Federal Reserve interest rate decision',
            'US China trade war tariff sanctions',
            'crude oil prices OPEC production',
            'global recession inflation fears',
            'European Central Bank policy',
            # Geopolitical events
            'geopolitical tensions war conflict',
            'Middle East crisis oil supply',
            'Russia Ukraine war sanctions impact',
            'China Taiwan tensions semiconductor',
            'India elections economic reform',
            # Market-moving
            'stock market crash correction rally',
            'FII DII foreign investors India',
            'global supply chain disruption',
        ]

        # Add year-specific queries for 5-year coverage
        current_year = datetime.now().year
        year_geo_queries = []
        for year in range(current_year - 5, current_year + 1):
            year_geo_queries.append(f'India economy GDP {year}')
            year_geo_queries.append(f'RBI policy interest rate {year}')
            year_geo_queries.append(f'crude oil geopolitical {year}')
            year_geo_queries.append(f'US Fed rate decision {year}')

        GEO_QUERIES.extend(year_geo_queries)

        all_articles = []
        seen_urls = set(existing_links)

        console.print("[cyan]📡 Scraping geopolitical news links...[/cyan]")
        geo_task = progress_ui.add_task("[bold cyan]Geopolitical Links[/bold cyan]", total=len(GEO_QUERIES))

        for query in GEO_QUERIES:
            try:
                encoded = urllib.parse.quote(query)
                # Bing News
                resp = self.session.get(
                    f"https://www.bing.com/news/search?q={encoded}&format=rss",
                    headers={'User-Agent': random.choice(self.user_agents)}, timeout=5,
                )
                if resp.status_code == 200:
                    feed = feedparser.parse(resp.content)
                    for entry in feed.entries:
                        url = entry.link
                        if url in seen_urls or not self.is_valid_url(url):
                            continue
                        seen_urls.add(url)
                        all_articles.append({
                            'Category': query,
                            'Date': self._parse_pub_date(entry),
                            'Title': entry.title,
                            'Source': 'Bing News',
                            'Link': url,
                            'Full_Title': None,
                            'Content': None,
                        })
            except Exception:
                pass

            try:
                # DuckDuckGo News
                encoded = urllib.parse.quote(query)
                resp = self.session.get(
                    f"https://duckduckgo.com/?q={encoded}&iar=news&ia=news&format=rss",
                    headers={'User-Agent': random.choice(self.user_agents)}, timeout=5,
                )
                if resp.status_code == 200:
                    feed = feedparser.parse(resp.content)
                    for entry in feed.entries:
                        url = entry.link
                        if url in seen_urls or not self.is_valid_url(url):
                            continue
                        seen_urls.add(url)
                        all_articles.append({
                            'Category': query,
                            'Date': self._parse_pub_date(entry),
                            'Title': entry.title,
                            'Source': 'DuckDuckGo',
                            'Link': url,
                            'Full_Title': None,
                            'Content': None,
                        })
            except Exception:
                pass

            progress_ui.advance(geo_task, 1)

        progress_ui.remove_task(geo_task)

        if not all_articles:
            console.print("[yellow]  ⚠ No geopolitical links found[/yellow]")
            return

        # Merge with existing
        df_new = pd.DataFrame(all_articles)
        df_new = df_new[~df_new['Link'].isin(existing_links)]
        if not df_new.empty:
            df = pd.concat([df, df_new]).drop_duplicates(subset=['Link']).reset_index(drop=True)

        # Content extraction on articles missing content
        for col in ['Full_Title', 'Content']:
            if col not in df.columns:
                df[col] = None

        mask = (df['Content'].isna() | (df['Content'].astype(str).str.len() < 200)) & df['Link'].notna()
        targets = df[mask].index.tolist()[:50]  # Cap at 50 articles per run

        if targets:
            console.print(f"[cyan]  Extracting content from {len(targets)} geopolitical articles...[/cyan]")
            geo_scrape = progress_ui.add_task("[bold cyan]Geo Content[/bold cyan]", total=len(targets))

            discarded_idx = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.process_single_link, df.at[idx, 'Link']): idx for idx in targets}
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        title, content, engine = future.result(timeout=45)
                        if content:
                            df.at[idx, 'Full_Title'] = title
                            df.at[idx, 'Content'] = content
                        else:
                            discarded_idx.append(idx)
                    except Exception:
                        discarded_idx.append(idx)
                    progress_ui.advance(geo_scrape, 1)

            if discarded_idx:
                df = df.drop(discarded_idx).reset_index(drop=True)

            progress_ui.remove_task(geo_scrape)

        df.to_csv(save_path, index=False)
        has_content = int((df['Content'].notna() & (df['Content'].astype(str).str.len() >= 200)).sum())
        console.print(f"[green]  ✓ {has_content} geopolitical articles with content → {save_path}[/green]")

    # ═══════════════════════════════════════
    #  DISCARD LOGGING
    # ═══════════════════════════════════════
    def log_discarded(self, records):
        if not records: return
        try:
            df = pd.DataFrame(records)
            exists = DISCARD_CSV_PATH.exists()
            df.to_csv(DISCARD_CSV_PATH, mode='a', header=not exists, index=False)
        except Exception as e:
            logger.warning(f"log_discarded failed: {e}")

    # ═══════════════════════════════════════
    #  SINGLE LINK PROCESSOR (thread-safe)
    # ═══════════════════════════════════════
    def process_single_link(self, url):
        title, content, engine = self.fetch_content(url)
        return title, content, engine

    # ═══════════════════════════════════════
    #  PER-TICKER ORCHESTRATOR
    # ═══════════════════════════════════════
    def process_ticker(self, company, progress_ui, task_id):
        ticker = company['symbol']
        save_path = NEWS_HISTORY_DIR / f"{ticker}_news.csv"

        try:
            df = pd.DataFrame()
            existing_links = set()
            if save_path.exists():
                df = pd.read_csv(save_path)
                if 'Link' in df.columns:
                    existing_links = set(df['Link'].dropna().tolist())
            for col in ['Full_Title', 'Content']:
                if col not in df.columns:
                    df[col] = None

            readable = 0
            unscraped = 0
            if not df.empty:
                readable = int((df['Content'].notna() & (df['Content'].astype(str).str.len() >= 200)).sum())
                unscraped = int(((df['Content'].isna() | (df['Content'].astype(str).str.len() < 200)) & df['Link'].notna()).sum())

            # ── LINK Phase ──
            if (df.empty or readable < 20) and unscraped < 10:
                progress_ui.update(task_id, description=f"[cyan]Links: {ticker}[/cyan]")
                new_articles = self.fetch_links(ticker, company['name'])
                if new_articles:
                    df_new = pd.DataFrame(new_articles)
                    df_new = df_new[~df_new['Link'].isin(existing_links)]
                    if not df_new.empty:
                        df = pd.concat([df, df_new]).drop_duplicates(subset=['Link']).reset_index(drop=True)

            if df.empty:
                progress_ui.update(task_id, description=f"[dim]No Data: {ticker}[/dim]", completed=100)
                return

            for col in ['Full_Title', 'Content']:
                if col not in df.columns:
                    df[col] = None

            # Find targets
            mask = (df['Content'].isna() | (df['Content'].astype(str).str.len() < 200)) & df['Link'].notna()
            # Also filter out google.com links from old data
            if 'Link' in df.columns:
                google_mask = df['Link'].astype(str).str.contains('news.google.com', case=False, na=False)
                mask = mask & ~google_mask

            targets = df[mask].index.tolist()

            if not targets:
                progress_ui.update(task_id, description=f"[green]✓ {ticker} ({readable} articles)[/green]", completed=100)
                return

            targets = targets[:10]
            progress_ui.update(task_id, description=f"[yellow]Scraping {len(targets)}: {ticker}[/yellow]", total=len(targets))

            success = 0
            discarded_idx = []
            discarded_records = []

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.process_single_link, df.at[idx, 'Link']): idx for idx in targets}

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        title, content, engine = future.result(timeout=45)
                        if content:
                            df.at[idx, 'Full_Title'] = title
                            df.at[idx, 'Content'] = content
                            success += 1
                        else:
                            discarded_idx.append(idx)
                            discarded_records.append({
                                'Ticker': ticker, 'Link': str(df.at[idx, 'Link'])[:200],
                                'Title': str(df.at[idx, 'Title'])[:200],
                                'Discard_Reason': f'No content ({engine})',
                                'Discard_Date': str(datetime.now()),
                            })
                    except Exception as e:
                        discarded_idx.append(idx)
                        discarded_records.append({
                            'Ticker': ticker, 'Link': str(df.at[idx, 'Link'])[:200],
                            'Discard_Reason': f'Error: {str(e)[:60]}',
                            'Discard_Date': str(datetime.now()),
                        })

                    progress_ui.advance(task_id, 1)

            if discarded_idx:
                df = df.drop(discarded_idx).reset_index(drop=True)
                self.log_discarded(discarded_records)

            # Sort by date before saving
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date', ascending=False).reset_index(drop=True)

            df.to_csv(save_path, index=False)
            progress_ui.update(task_id, description=f"[green]✓ {ticker} (+{success}, -{len(discarded_idx)})[/green]")

        except Exception as e:
            logger.error(f"FATAL on {ticker}: {e}")
            progress_ui.update(task_id, description=f"[red]ERR: {ticker}[/red]")

    # ═══════════════════════════════════════
    #  CONSOLIDATION
    # ═══════════════════════════════════════
    def consolidate_all(self):
        """Merge all stock news + geopolitical news into consolidated files."""
        import glob
        console.print("\n[bold cyan]📊 Consolidating all data...[/bold cyan]")

        # ── 1. Stock news consolidated ──
        stock_files = glob.glob(str(NEWS_HISTORY_DIR / '*_news.csv'))
        stock_dfs = []
        for f in stock_files:
            try:
                d = pd.read_csv(f)
                if 'Ticker' not in d.columns:
                    d['Ticker'] = Path(f).stem.replace('_news', '')
                stock_dfs.append(d)
            except Exception:
                pass

        if stock_dfs:
            df_stocks = pd.concat(stock_dfs, ignore_index=True)
            df_stocks['Date'] = pd.to_datetime(df_stocks['Date'], errors='coerce')
            df_stocks = df_stocks.sort_values('Date', ascending=False)
        else:
            df_stocks = pd.DataFrame()

        # ── 2. Geopolitical news ──
        geo_path = GEOPOLITICAL_DIR / 'geopolitical_news_scraped.csv'
        df_geo = pd.DataFrame()
        if geo_path.exists():
            df_geo = pd.read_csv(geo_path)
            df_geo['Date'] = pd.to_datetime(df_geo['Date'], errors='coerce')

        # ── 3. Combined text/review file (all articles with content) ──
        combined_parts = []
        if not df_stocks.empty:
            df_s = df_stocks[df_stocks['Content'].notna() & (df_stocks['Content'].astype(str).str.len() >= 200)].copy()
            df_s['Type'] = 'Stock'
            combined_parts.append(df_s)
        if not df_geo.empty:
            df_g = df_geo[df_geo['Content'].notna() & (df_geo['Content'].astype(str).str.len() >= 200)].copy()
            df_g['Type'] = 'Geopolitical'
            if 'Category' in df_g.columns and 'Ticker' not in df_g.columns:
                df_g['Ticker'] = df_g['Category']
            combined_parts.append(df_g)

        if combined_parts:
            df_combined = pd.concat(combined_parts, ignore_index=True)
            df_combined = df_combined.sort_values('Date', ascending=False)
            combined_path = DATA_DIR / 'all_news_with_content.csv'
            df_combined.to_csv(combined_path, index=False)
            console.print(f"[green]  ✓ All articles with content → {combined_path} ({len(df_combined)} rows)[/green]")

        # ── 4. Numerical summary ──
        summary_rows = []
        if not df_stocks.empty:
            for ticker, grp in df_stocks.groupby('Ticker'):
                has_content = int((grp['Content'].notna() & (grp['Content'].astype(str).str.len() >= 200)).sum())
                summary_rows.append({
                    'Ticker': ticker,
                    'Type': 'Stock',
                    'Total_Articles': len(grp),
                    'Articles_With_Content': has_content,
                    'Content_Rate': round(has_content / len(grp) * 100, 1) if len(grp) > 0 else 0,
                    'Sources': ', '.join(grp['Source'].dropna().unique()[:5]),
                    'Date_Range': f"{grp['Date'].min()} to {grp['Date'].max()}",
                })
        if not df_geo.empty:
            for cat, grp in df_geo.groupby('Category'):
                has_content = int((grp['Content'].notna() & (grp['Content'].astype(str).str.len() >= 200)).sum())
                summary_rows.append({
                    'Ticker': cat,
                    'Type': 'Geopolitical',
                    'Total_Articles': len(grp),
                    'Articles_With_Content': has_content,
                    'Content_Rate': round(has_content / len(grp) * 100, 1) if len(grp) > 0 else 0,
                    'Sources': ', '.join(grp['Source'].dropna().unique()[:5]),
                    'Date_Range': f"{grp['Date'].min()} to {grp['Date'].max()}",
                })

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows).sort_values('Articles_With_Content', ascending=False)
            summary_path = DATA_DIR / 'news_numerical_summary.csv'
            df_summary.to_csv(summary_path, index=False)
            console.print(f"[green]  ✓ Numerical summary → {summary_path} ({len(df_summary)} entries)[/green]")

        # ── 5. All stock news (incl. ones without content) ──
        if not df_stocks.empty:
            all_path = DATA_DIR / 'all_stocks_news_consolidated.csv'
            df_stocks.to_csv(all_path, index=False)
            console.print(f"[green]  ✓ All stock news → {all_path} ({len(df_stocks)} rows)[/green]")

        # ── 6. Day-wise news files ──
        if combined_parts:
            df_all = pd.concat(combined_parts, ignore_index=True)
            df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')
            df_all = df_all.dropna(subset=['Date'])
            df_all['DateOnly'] = df_all['Date'].dt.date

            daywise_dir = DATA_DIR / 'daywise_news'
            daywise_dir.mkdir(parents=True, exist_ok=True)

            day_summary = []
            for day, grp in df_all.groupby('DateOnly'):
                day_str = str(day)
                day_file = daywise_dir / f"{day_str}.csv"
                grp_sorted = grp.sort_values('Date', ascending=False)
                grp_sorted.to_csv(day_file, index=False)

                stock_count = len(grp_sorted[grp_sorted['Type'] == 'Stock']) if 'Type' in grp_sorted.columns else len(grp_sorted)
                geo_count = len(grp_sorted[grp_sorted['Type'] == 'Geopolitical']) if 'Type' in grp_sorted.columns else 0

                day_summary.append({
                    'Date': day_str,
                    'Total_Articles': len(grp_sorted),
                    'Stock_Articles': stock_count,
                    'Geo_Articles': geo_count,
                    'Tickers': ', '.join(grp_sorted['Ticker'].dropna().unique()[:10]),
                    'Sources': ', '.join(grp_sorted['Source'].dropna().unique()[:5]),
                })

            if day_summary:
                df_days = pd.DataFrame(day_summary).sort_values('Date', ascending=False)
                df_days.to_csv(DATA_DIR / 'daywise_news_summary.csv', index=False)
                console.print(f"[green]  ✓ Day-wise news → {daywise_dir}/ ({len(day_summary)} days)[/green]")
                console.print(f"[green]  ✓ Day-wise summary → {DATA_DIR / 'daywise_news_summary.csv'}[/green]")

            # ── 6b. Per-ticker day-wise folders ──
            # Structure: data/daywise_news/{TICKER}/{YYYY-MM-DD}.csv
            if 'Ticker' in df_all.columns:
                ticker_count = 0
                for ticker, t_grp in df_all.groupby('Ticker'):
                    ticker_dir = daywise_dir / str(ticker).replace('.', '_')
                    ticker_dir.mkdir(parents=True, exist_ok=True)
                    for day, d_grp in t_grp.groupby('DateOnly'):
                        d_grp.sort_values('Date', ascending=False).to_csv(
                            ticker_dir / f"{str(day)}.csv", index=False
                        )
                    ticker_count += 1
                console.print(f"[green]  ✓ Per-ticker day-wise → {daywise_dir}/<TICKER>/ ({ticker_count} tickers)[/green]")

        console.print("[bold green]\n✅ Consolidation complete![/bold green]")

    # ═══════════════════════════════════════
    #  MAIN
    # ═══════════════════════════════════════
    def run(self):
        companies = self.get_all_equity_tickers()
        console.print(f"\n[bold magenta]🚀 UNIFIED Scraper v6.1 — {len(companies)} Stocks[/bold magenta]")
        console.print("[dim]Links: Bing + DuckDuckGo + Indian Finance RSS + Yahoo[/dim]")
        console.print("[dim]Content: trafilatura → newspaper3k → BS4 → Crawl4AI → Playwright[/dim]\n")

        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            BarColumn(), TaskProgressColumn(),
            console=console, expand=True,
        ) as progress_ui:

            # Phase 1: Geopolitical news (with content!)
            console.print("[bold cyan]═══ Phase 1: Geopolitical News ═══[/bold cyan]")
            self.scrape_geopolitical_news(progress_ui)
            console.print()

            # Phase 2: Stock news
            console.print("[bold cyan]═══ Phase 2: Stock News ═══[/bold cyan]")
            overall = progress_ui.add_task("[bold yellow]Stock Progress[/bold yellow]", total=len(companies))

            for company in companies:
                tid = progress_ui.add_task(f"[dim]{company['symbol']}[/dim]", total=100)
                try:
                    self.process_ticker(company, progress_ui, tid)
                except Exception as e:
                    logger.error(f"Unhandled: {company['symbol']}: {e}")
                    progress_ui.update(tid, description=f"[red]FAIL: {company['symbol']}[/red]")

                progress_ui.remove_task(tid)
                progress_ui.advance(overall, 1)

        # Phase 3: Consolidation
        console.print("\n[bold cyan]═══ Phase 3: Consolidation ═══[/bold cyan]")
        self.consolidate_all()


if __name__ == "__main__":
    MasterNewsScraper().run()
