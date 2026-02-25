from reportlab.platypus import Paragraph, Spacer, Image, PageBreak
from .utils import create_custom_styles, add_dataframe_table

def generate(story):
    """Generate Chapter 5: The Connection"""
    styles = create_custom_styles()
    
    # Chapter Title
    story.append(Paragraph("Chapter 5: The Connection", styles['ChapterTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>\"Correlation does not imply causation, but it sure is a hint.\" — Edward Tufte</i>", styles['Caption']))
    story.append(Spacer(1, 20))
    
    # 5.1 Introduction
    story.append(Paragraph("5.1 Untangling the Web", styles['SectionTitle']))
    story.append(Paragraph("""
    Financial markets are a complex adaptive system where no variable acts in isolation. Price is influenced by Volume; Volume follows News; News drives Sentiment; and Sentiment feeds back into Price. Chapter 5: 'The Connection' is our forensic investigation into these interdependencies.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    We employ Pearson’s Correlation Coefficient (r) to quantify the linear relationship between variables. A value of +1 implies perfect positive correlation, -1 implies perfect negative correlation, and 0 implies no linear relationship.
    """, styles['BodyTextCustom']))

    # 5.2 The Heatmap
    story.append(Paragraph("5.2 visualising Dependencies: The Heatmap", styles['SectionTitle']))
    story.append(Paragraph("""
    To visualize the multidimensional relationships simultaneously, we construct a Correlation Heatmap. This matrix reveals how our engineered features interact with the target variable (Future Returns).
    """, styles['BodyTextCustom']))

    try:
        story.append(Image("report/figures/03_correlation_heatmap.png", width=450, height=400))
        story.append(Paragraph("<i>Figure 5.1: Feature Correlation Matrix</i>", styles['Caption']))
    except:
        story.append(Paragraph("[Figure 5.1 missing]", styles['Caption']))

    story.append(Paragraph("<b>Key Observations from Figure 5.1:</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    1.  <b>Price & Moving Averages (Strong Positive):</b> As expected, `Close` price shows a near-perfect correlation with `SMA_50` and `SMA_200`. This confirms that trend-following indicators lag price but are highly reliable.
    2.  <b>Volume & Volatility (Moderate Positive):</b> High trading volume often accompanies periods of high volatility (`ATR`, `BB_Width`). This validates the market adage: "Volume precedes price."
    3.  <b>RSI & Momentum (Oscillating):</b> `RSI` shows cyclical correlation, stronger at extremes (overbought/oversold) and weaker during consolidation.
    """, styles['InsightBox']))

    # 5.3 Multicollinearity Check
    story.append(Paragraph("5.3 The Curse of Multicollinearity", styles['SectionTitle']))
    story.append(Paragraph("""
    In machine learning, highly correlated features (multicollinearity) can destabilize model coefficients, though tree-based models like XGBoost are generally robust to it.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    <b>Dimensionality Reduction Strategy:</b>
    We observed that `SMA_50` and `SMA_200` carry redundant information regarding the absolute price level. However, their <i>divergence</i> (Golden Cross) is a unique signal. Therefore, we prioritize relative features (e.g., `Price / SMA_50`) over absolute values in our final feature set (Chapter 7).
    """, styles['BodyTextCustom']))

    story.append(PageBreak())
