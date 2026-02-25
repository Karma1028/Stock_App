from reportlab.platypus import Paragraph, Spacer, Image, PageBreak
from .utils import create_custom_styles, add_dataframe_table

def generate(story):
    """Generate Chapter 4: The Pulse"""
    styles = create_custom_styles()
    
    # Chapter Title
    story.append(Paragraph("Chapter 4: The Pulse", styles['ChapterTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>\"The market is a pendulum that forever swings between untenable optimism and unjustified pessimism.\"</i>", styles['Caption']))
    story.append(Spacer(1, 20))
    
    # 4.1 Introduction
    story.append(Paragraph("4.1 The Heartbeat of the Market", styles['SectionTitle']))
    story.append(Paragraph("""
    Having established our data foundation, we now turn our gaze to the market itself. 'The Pulse' chapter is dedicated to Exploratory Data Analysis (EDA) of price and volume—the raw signals of supply and demand.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    Our objective is threefold:
    1.  <b>Trend Identification:</b> Distinguishing between secular bull markets, cyclical corrections, and noise.
    2.  <b>Distribution Analysis:</b> Understanding the statistical properties of returns (Normal vs. Fat-tailed).
    3.  <b>Sector Rotation:</b> Visualizing capital flow across different industries.
    """, styles['BodyTextCustom']))

    # 4.2 Sector Distribution
    story.append(Paragraph("4.2 Sector Composition", styles['SectionTitle']))
    story.append(Paragraph("""
    The Nifty 500 is not a monolith; it is a conglomerate of diverse sectors. Understanding this composition is crucial for portfolio allocation.
    """, styles['BodyTextCustom']))
    
    try:
        story.append(Image("report/figures/01_sector_distribution.png", width=450, height=300))
        story.append(Paragraph("<i>Figure 4.1: Sectoral Distribution of Companies</i>", styles['Caption']))
    except:
        story.append(Paragraph("[Figure 4.1 missing]", styles['Caption']))

    story.append(Paragraph("""
    <b>Insight:</b> As observed in Figure 4.1, the Financial Services and IT sectors dominate the index by weight. This concentration risk implies that any systemic shock to banking or tech will have a disproportionate impact on the overall market.
    """, styles['InsightBox']))

    # 4.3 Price Dynamics
    story.append(Paragraph("4.3 Price Dynamics & Volatility", styles['SectionTitle']))
    story.append(Paragraph("""
    We analyze the distribution of closing prices to detect skewness and kurtosis. A normal distribution (Bell Curve) is the theoretical ideal for many statistical models (e.g., Black-Scholes), but empirical market data often exhibits 'fat tails'—extreme events occur more frequently than predicted.
    """, styles['BodyTextCustom']))

    try:
        story.append(Image("report/figures/02_price_distribution.png", width=450, height=300))
        story.append(Paragraph("<i>Figure 4.2: Distribution of Closing Prices</i>", styles['Caption']))
    except:
        story.append(Paragraph("[Figure 4.2 missing]", styles['Caption']))

    story.append(Paragraph("""
    <b>Statistical Anomaly:</b> The distribution in Figure 4.2 is right-skewed. This is characteristic of equity markets where prices have a lower bound (0) but no theoretical upper bound. This log-normal behavior validates our decision to use Log Returns for feature engineering in Chapter 7.
    """, styles['BodyTextCustom']))

    # 4.4 Trend Analysis
    story.append(Paragraph("4.4 Trend Analysis: The Moving Average", styles['SectionTitle']))
    story.append(Paragraph("""
    To smooth out daily noise and reveal the underlying trend, we employ the 50-day Simple Moving Average (SMA).
    """, styles['BodyTextCustom']))

    try:
        story.append(Image("report/figures/04_price_trend.png", width=500, height=300))
        story.append(Paragraph("<i>Figure 4.3: Price Trend vs SMA 50</i>", styles['Caption']))
    except:
        story.append(Paragraph("[Figure 4.3 missing]", styles['Caption']))

    story.append(Paragraph("""
    <b>The Crossover Strategy:</b>
    When the price crosses above the SMA 50 (Golden Cross), it often signals the start of a bullish phase. Conversely, a drop below (Death Cross) suggests weakness. Our analysis confirms that this simple indicator remains a potent filter for identifying regime changes.
    """, styles['InsightBox']))

    story.append(PageBreak())
