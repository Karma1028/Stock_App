from reportlab.platypus import Paragraph, Spacer, Image, PageBreak
from .utils import create_custom_styles, add_dataframe_table

def generate(story):
    """Generate Chapter 6: The Sentinel"""
    styles = create_custom_styles()
    
    # Chapter Title
    story.append(Paragraph("Chapter 6: The Sentinel", styles['ChapterTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>\"Markets are never wrong, but opinions often are.\" — Philip Fisher</i>", styles['Caption']))
    story.append(Spacer(1, 20))
    
    # 6.1 The Qualitative Dimension
    story.append(Paragraph("6.1 Beyond the Ticker", styles['SectionTitle']))
    story.append(Paragraph("""
    Traditional algorithmic trading relies heavily on price and volume. However, the catalyst for price movement often lies outside the charts—in earnings calls, regulatory news, and geopolitical shifts. 'The Sentinel' is our NLP subsystem designed to quantify this qualitative data.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    Our hypothesis is simple: <b>News precedes Price.</b> A sudden surge in negative sentiment often foreshadows a price correction, while sustained positive coverage can fuel a rally.
    """, styles['BodyTextCustom']))

    # 6.2 NLP Methodology
    story.append(Paragraph("6.2 Natural Language Processing Architecture", styles['SectionTitle']))
    story.append(Paragraph("""
    We treat financial news not as text, but as a signal. We employ a lexicon-based approach using `TextBlob`, a library built on NLTK (Natural Language Toolkit). 
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("<b>The Scoring Calculus:</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    For every headline $H$, we compute a polarity score $P(H) \in [-1, +1]$.
    $$P(H) = \\frac{\\sum w_i \\cdot s_i}{\\sum |w_i|}$$
    Where $w_i$ is the sentiment weight of word $i$.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    -   <b>Polarity > 0:</b> Bullish (e.g., "Reliance profit jumps 20%")
    -   <b>Polarity < 0:</b> Bearish (e.g., "Adani faces regulatory probe")
    -   <b>Polarity ≈ 0:</b> Neutral (e.g., "TCS announces board meeting")
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    python
    # Code Snippet: The Sentinel Engine
    from textblob import TextBlob
    
    def get_sentiment(text):
        blob = TextBlob(text)
        # Returns (polarity, subjectivity)
        return blob.sentiment
       
    # We aggregate these scores daily to create a 'Sentiment Index' for each stock.
    """, styles['CodeBlock']))

    # 6.3 Sentiment vs Price
    story.append(Paragraph("6.3 Signal Efficacy: Sentiment vs Price", styles['SectionTitle']))
    story.append(Paragraph("""
    To validate the efficacy of our sentiment engine, we analyze the correlation between the `Sentiment_Score` (t) and `Returns` (t+1).
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    <b>Findings:</b>
    Our analysis reveals a <b>lead-lag relationship</b>. Extreme sentiment spikes (both positive and negative) tend to precede high volatility sessions. Specifically, negative sentiment has a stronger predictive power for downside risk than positive sentiment has for upside potential—a phenomenon known as 'Loss Aversion' in behavioral finance.
    """, styles['InsightBox']))

    # 6.4 Limitations
    story.append(Paragraph("6.4 The Irony of Sarcasm", styles['SectionTitle']))
    story.append(Paragraph("""
    Lexicon-based models struggle with nuance. A headline like "Competitor's failure is good for Reliance" might be scored negatively due to the word "failure," even though it is bullish for Reliance. Future iterations of The Sentinel will employ transformer-based models (e.g., FinBERT) to capture context and financial-specific semantics.
    """, styles['BodyTextCustom']))

    story.append(PageBreak())
