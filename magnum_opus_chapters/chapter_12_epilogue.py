from reportlab.platypus import Paragraph, Spacer, Image, PageBreak
from .utils import create_custom_styles, add_dataframe_table

def generate(story):
    """Generate Chapter 12: The Epilogue"""
    styles = create_custom_styles()
    
    # Chapter Title
    story.append(Paragraph("Chapter 12: The Epilogue", styles['ChapterTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>\"The only true wisdom is in knowing you know nothing.\" — Socrates</i>", styles['Caption']))
    story.append(Spacer(1, 20))
    
    # 12.1 Conclusion
    story.append(Paragraph("12.1 The Verdict", styles['SectionTitle']))
    story.append(Paragraph("""
    Our journey from raw data to actionable intelligence has been an odyssey of discovery. We have built a system that not only analyzes the market but perceives it. Through the fusion of quantitative rigor (XGBoost) and qualitative insight (NLP), we have achieved a comprehensive framework for stock analysis.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    <b>Key Takeaways:</b>
    1.  <b>Trend is Persistent:</b> Simple moving averages remain surprisingly effective filters.
    2.  <b>Sentiment Matters:</b> News sentiment is a leading indicator of volatility.
    3.  <b>Complexity has Diminishing Returns:</b> A well-tuned XGBoost model with domain-specific features outperforms complex deep learning architectures on tabular financial data.
    """, styles['BodyTextCustom']))

    # 12.2 Future Work
    story.append(Paragraph("12.2 The Road Ahead", styles['SectionTitle']))
    story.append(Paragraph("""
    This Magnum Opus is not the end; it is a milestone. Future iterations will explore:
    -   <b>Reinforcement Learning (RL):</b> Training an agent to execute trades autonomously in a simulated environment.
    -   <b>Transformer Models for NLP:</b> Replacing TextBlob with FinBERT for nuanced sentiment analysis.
    -   <b>Alternative Data:</b> Integrating satellite imagery, credit card transaction data, and social media trends.
    """, styles['BodyTextCustom']))

    # 12.3 References
    story.append(Paragraph("12.3 Bibliography", styles['SectionTitle']))
    story.append(Paragraph("""
    1.  Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
    2.  Taylor, S. J., & Letham, B. (2018). Forecasting at Scale (Prophet).
    3.  Loughran, T., & McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks.
    4.  Murphy, J. J. (1999). Technical Analysis of the Financial Markets.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("<b>End of Report</b>", styles['Caption']))
    story.append(PageBreak())
