from reportlab.platypus import Paragraph, Spacer, Image, PageBreak
from .utils import create_custom_styles, add_dataframe_table

def generate(story):
    """Generate Chapter 2: The Harvest"""
    styles = create_custom_styles()
    
    # Chapter Title
    story.append(Paragraph("Chapter 2: The Harvest", styles['ChapterTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>\"The finest insights are like gold; rarely found on the surface, often buried deep within the bedrock of unstructured chaos.\"</i>", styles['Caption']))
    story.append(Spacer(1, 20))
    
    # 2.1 The Philosophy of Acquisition
    story.append(Paragraph("2.1 The Philosophy of Acquisition", styles['SectionTitle']))
    story.append(Paragraph("""
    The integrity of any predictive model is inextricably linked to the purity of its input data. In financial markets, data exists in two primary states: structured (price, volume) and unstructured (news, sentiment). Our 'Harvest' phase is designed to systematically ingest both, transforming raw digital noise into a coherent, queryable ledger.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    We employ a dual-pipeline strategy:
    1.  <b>Quantitative Pipeline:</b> Sourcing high-fidelity OHLCV (Open, High, Low, Close, Volume) data via the `yfinance` API.
    2.  <b>Qualitative Pipeline:</b> Scraping and parsing real-time news headlines via Google News RSS feeds to capture market sentiment.
    """, styles['BodyTextCustom']))

    # 2.2 The Quantitative Engine (yfinance)
    story.append(Paragraph("2.2 The Quantitative Engine: Price Action", styles['SectionTitle']))
    story.append(Paragraph("""
    Price data is the heartbeat of the market. We utilize the `yfinance` library, a robust wrapper around the Yahoo Finance API, to fetch historical data. The retrieval process is not merely a download; it is a rigorous validation exercise.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("<b>Implementation Details:</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    The `Ticker` object is instantiated for each symbol in the Nifty 500. We request data with `period='max'` to ensure a deep historical context. Crucially, we handle:
    -   <b>Missing Data Adjustment:</b> Forward-filling gaps to maintain time-series continuity.
    -   <b>Split/Dividend Adjustment:</b> Ensuring price continuity across corporate actions.
    -   <b>Timezone Normalization:</b> Converting all timestamps to UTC for consistency.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    python
    # Code Snippet: Robust Data Fetching
    def fetch_price_data(ticker):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="max")
            if df.empty:
                raise ValueError("No data returned")
            # Standardization
            df.index = df.index.tz_convert(None) 
            return df
        except Exception as e:
            log_error(f"Failed to fetch {ticker}: {e}")
            return None
    """, styles['CodeBlock']))

    # 2.3 The Qualitative Engine (News Scraping)
    story.append(Paragraph("2.3 The Qualitative Engine: Mining Sentiment", styles['SectionTitle']))
    story.append(Paragraph("""
    While price tells us *what* happened, news often explains *why*. To capture this, we built a custom scraper targeting Google News RSS feeds. This is a non-trivial engineering challenge due to the dynamic nature of web content and anti-scraping mechanisms.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    <b>The RSS Strategy:</b>
    We construct dynamic search queries using `urllib.parse` to encode time-bounded requests (e.g., "Reliance Industries after:2023-01-01 before:2023-02-01"). The `feedparser` library then parses the XML response, extracting:
    -   <b>Title:</b> The headline, key for sentiment analysis.
    -   <b>Link:</b> The source URL for lineage.
    -   <b>pubDate:</b> The publication timestamp for temporal alignment.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    python
    # Code Snippet: RSS Parsing Logic
    base_url = "https://news.google.com/rss/search?q={query}"
    encoded_query = urllib.parse.quote(f"{company} stock news")
    feed = feedparser.parse(base_url.format(query=encoded_query))
    
    entries = []
    for entry in feed.entries:
        entries.append({
            'Ticker': ticker,
            'Title': entry.title,
            'Link': entry.link,
            'Date': parse_date(entry.published)
        })
    """, styles['CodeBlock']))

    # 2.4 Consolidation & Storage
    story.append(Paragraph("2.4 Consolidation: The Ledger", styles['SectionTitle']))
    story.append(Paragraph("""
    Data ingestion is futile without efficient storage. We consolidate our findings into two primary artifacts:
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    1.  <b>`EQUITY.csv` (The Master Ledger):</b> A consolidated CSV containing metadata for all tracked companies (Symbol, Name, Sector, Industry). This serves as the relational backbone of our database.
    2.  <b>`all_stocks_news_consolidated.csv` (The Narrative Archive):</b> A massive, append-only log of every news headline harvested. This file grows incrementally, ensuring no historical context is lost.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    <b>Data Quality Gates:</b>
    Before persistence, data passes through rigorous quality gates. We drop duplicates based on (Ticker, Date), handle null values in critical columns, and enforce schema consistency. This ensures that downstream processes—Feature Engineering and Modeling—ingest only the highest quality fuel.
    """, styles['InsightBox']))

    story.append(PageBreak())
