from reportlab.platypus import Paragraph, Spacer, Image, PageBreak
from .utils import create_custom_styles, add_dataframe_table

def generate(story):
    """Generate Chapter 7: The Alchemy"""
    styles = create_custom_styles()
    
    # Chapter Title
    story.append(Paragraph("Chapter 7: The Alchemy", styles['ChapterTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>\"The world is not driven by greed. It’s driven by envy.\" — Charlie Munger</i>", styles['Caption']))
    story.append(Spacer(1, 20))
    
    # 7.1 Introduction
    story.append(Paragraph("7.1 The Art of Feature Engineering", styles['SectionTitle']))
    story.append(Paragraph("""
    Raw data is rarely predictive. Feature Engineering is the alchemy that transmutes base data (Open, High, Low, Close, Volume) into gold (Predictive Signals). In this chapter, we mathematically define the 28 features that feed our machine learning models.
    """, styles['BodyTextCustom']))
    
    # 7.2 Momentum Indicators
    story.append(Paragraph("7.2 Momentum: The Velocity of Price", styles['SectionTitle']))
    story.append(Paragraph("""
    Momentum measures the speed of price changes. We use the Relative Strength Index (RSI) to identify overbought and oversold conditions.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("<b>Relative Strength Index (RSI):</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    The RSI is a momentum oscillator that measures the magnitude of recent price changes.
    $$RSI = 100 - \\frac{100}{1 + RS}$$
    Where $RS = \\frac{\\text{Average Gain}}{\\text{Average Loss}}$ over a 14-day period.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    <b>Rationale:</b>
    -   <b>RSI > 70:</b> Signals overbought conditions (potential reversal).
    -   <b>RSI < 30:</b> Signals oversold conditions (potential bounce).
    Our model learns that extremes in RSI often precede mean reversion.
    """, styles['InsightBox']))

    # 7.3 Trend Indicators
    story.append(Paragraph("7.3 Trend: The Direction of Flow", styles['SectionTitle']))
    story.append(Paragraph("""
    To capture the prevailing market direction, we use the Moving Average Convergence Divergence (MACD).
    """, styles['BodyTextCustom']))

    story.append(Paragraph("<b>MACD Logic:</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    $$MACD = EMA_{12}(Close) - EMA_{26}(Close)$$
    $$Signal = EMA_{9}(MACD)$$
    $$Histogram = MACD - Signal$$
    
    Where $EMA$ is the Exponential Moving Average, giving more weight to recent prices.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    <b>Interpretation:</b>
    Positive Histogram values indicate bullish momentum (MACD > Signal), while negative values indicate bearish momentum. This feature helps the model distinguish between trending and ranging markets.
    """, styles['BodyTextCustom']))

    # 7.4 Volatility Indicators
    story.append(Paragraph("7.4 Volatility: Expected Range", styles['SectionTitle']))
    story.append(Paragraph("""
    Understanding volatility is key to risk management. We employ Bollinger Bands and Average True Range (ATR).
    """, styles['BodyTextCustom']))

    story.append(Paragraph("<b>Bollinger Bands:</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    $$Upper Band = SMA_{20} + (2 \\times \\sigma_{20})$$
    $$Lower Band = SMA_{20} - (2 \\times \\sigma_{20})$$
    
    Price touching the upper band suggests overextension, while touching the lower band suggests value.
    """, styles['BodyTextCustom']))

    # 7.5 Lag Features
    story.append(Paragraph("7.5 Lag Features: The Memory of Price", styles['SectionTitle']))
    story.append(Paragraph("""
    Time-series data is autocorrelated. Yesterday's return influences today's. To capture this 'memory', we engineer lag features:
    -   <b>Returns_1d:</b> Return over the last 1 day.
    -   <b>Returns_5d:</b> Return over the last 5 days (Weekly momentum).
    -   <b>Returns_21d:</b> Return over the last 21 days (Monthly momentum).
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    python
    # Code Snippet: Feature Generation
    df['Returns_1d'] = df['Close'].pct_change(1)
    df['Returns_5d'] = df['Close'].pct_change(5)
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    """, styles['CodeBlock']))

    story.append(PageBreak())
