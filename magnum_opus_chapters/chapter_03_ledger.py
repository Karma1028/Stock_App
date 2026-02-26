from reportlab.platypus import Paragraph, Spacer, Image, PageBreak
from .utils import create_custom_styles, add_dataframe_table
import pandas as pd

def generate(story):
    """Generate Chapter 3: The Ledger"""
    styles = create_custom_styles()
    
    # Chapter Title
    story.append(Paragraph("Chapter 3: The Ledger", styles['ChapterTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>\"Data is the new oil. It’s valuable, but if unrefined it cannot really be used.\"\n— Clive Humby</i>", styles['Caption']))
    story.append(Spacer(1, 20))
    
    # 3.1 Overview
    story.append(Paragraph("3.1 The Dataset Architecture", styles['SectionTitle']))
    story.append(Paragraph("""
    Our data ecosystem is built upon three primary pillars, each serving a distinct function in the analytical pipeline. This chapter provides a granular audit of these datasets, detailing their schema, source, and quality.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    The three core datasets are:
    1.  <b>EQUITY.csv:</b> The static metadata repository.
    2.  <b>consolidated_sentiments.csv:</b> The dynamic sentiment engine.
    3.  <b>all_stocks_news_consolidated.csv:</b> The raw textual archive.
    """, styles['BodyTextCustom']))

    # 3.2 EQUITY.csv
    story.append(Paragraph("3.2 The Metadata Backbone: EQUITY.csv", styles['SectionTitle']))
    story.append(Paragraph("""
    This file acts as the registry for our universe. It contains fundamental attributes for every company in the Nifty 500. It is the starting point for any ticker-based query.
    """, styles['BodyTextCustom']))
    
    # Load and show sample of EQUITY.csv
    try:
        df_equity = pd.read_csv("data/EQUITY.csv").head(5)
        add_dataframe_table(story, df_equity[["SYMBOL", "NAME OF COMPANY", " DATE OF LISTING"]], "Exhibit 3.1: EQUITY.csv Sample")
    except Exception as e:
        story.append(Paragraph(f"<i>Error loading EQUITY.csv: {e}</i>", styles['BodyTextCustom']))

    story.append(Paragraph("<b>Key Columns & Purpose:</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    -   <b>SYMBOL:</b> The unique identifier (e.g., RELIANCE). Used as the foreign key across all datasets.
    -   <b>NAME OF COMPANY:</b> Full legal entity name. Important for news query construction.
    -   <b>DATE OF LISTING:</b> Helps determine the start date for historical data fetching.
    -   <b>PAID UP VALUE:</b> Fundamental metric for valuation models.
    """, styles['BodyTextCustom']))

    # 3.3 Sentiment Data
    story.append(Paragraph("3.3 The Pulse: consolidated_sentiments.csv", styles['SectionTitle']))
    story.append(Paragraph("""
    This file is the output of our Natural Language Processing (NLP) engine. It aggregates daily sentiment scores derived from thousands of news articles.
    """, styles['BodyTextCustom']))

    try:
        df_sent = pd.read_csv("data/consolidated_sentiments.csv").head(5)
        add_dataframe_table(story, df_sent, "Exhibit 3.2: consolidated_sentiments.csv Sample")
    except:
        pass

    story.append(Paragraph("<b>Fetching & Transformation Methodology:</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    The journey from raw HTML to a float value involves several sophisticated steps:
    1.  <b>Fetching:</b> We use `feedparser` to query Google News RSS specifically for each ticker.
    2.  <b>Cleaning:</b> HTML tags are stripped using `BeautifulSoup`. Non-alphanumeric characters are removed via RegEx.
    3.  <b>Scoring:</b> We employ `TextBlob` to compute polarity (-1 to +1) and subjectivity (0 to 1).
    4.  <b>Aggregation:</b> Individual article scores are grouped by Date and Ticker, computing the mean to reduce noise.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    python
    # Code Snippet: Sentiment Transformation
    def calculate_sentiment(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    # Aggregation Logic
    daily_sentiment = news_df.groupby(['Ticker', 'Date']).agg({
        'sentiment_score': 'mean',
        'title': 'count' # Volume of news
    }).reset_index()
    """, styles['CodeBlock']))

    # 3.4 Raw News Archive
    story.append(Paragraph("3.4 The Raw Feed: all_stocks_news_consolidated.csv", styles['SectionTitle']))
    story.append(Paragraph("""
    This dataset is our detailed audit trail. It stores every single news headline fetched, preserving the original source and timestamp. It is crucial for backtesting and verification 
    """, styles['BodyTextCustom']))

    try:
        df_news = pd.read_csv("data/all_stocks_news_consolidated.csv").head(5)
        # Select few cols for display
        cols = [c for c in df_news.columns if c in ['Ticker', 'Date', 'Title', 'Source']]
        add_dataframe_table(story, df_news[cols], "Exhibit 3.3: Raw News Archive Sample")
    except:
        pass

    story.append(Paragraph("""
    <b>Data Cleaning Techniques Applied:</b>
    -   <b>Deduplication:</b> We enforce strict uniqueness on (Ticker, Title, Date) to prevent double-counting.
    -   <b>Date Parsing:</b> Utilizing `dateutil.parser` to handle various RSS date formats (RFC 822, ISO 8601).
    -   <b>Source Verification:</b> Filtering out low-quality or aggregator domains to maintain signal purity.
    """, styles['InsightBox']))

    story.append(PageBreak())
