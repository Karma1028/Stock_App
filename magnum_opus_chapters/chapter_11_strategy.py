from reportlab.platypus import Paragraph, Spacer, Image, PageBreak
from .utils import create_custom_styles, add_dataframe_table

def generate(story):
    """Generate Chapter 11: The Strategy"""
    styles = create_custom_styles()
    
    # Chapter Title
    story.append(Paragraph("Chapter 11: The Strategy", styles['ChapterTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>\"The stock market is a device for transferring money from the impatient to the patient.\" — Warren Buffett</i>", styles['Caption']))
    story.append(Spacer(1, 20))
    
    # 11.1 Introduction
    story.append(Paragraph("11.1 From Prediction to Profit", styles['SectionTitle']))
    story.append(Paragraph("""
    Prediction is merely a step; profit is the goal. 'The Strategy' translates our probabilistic outputs into deterministic actions. We integrate our signals into a cohesive portfolio management framework.
    """, styles['BodyTextCustom']))
    
    # 11.2 Portfolio Construction
    story.append(Paragraph("11.2 Portfolio Construction: Modern Portfolio Theory (MPT)", styles['SectionTitle']))
    story.append(Paragraph("""
    We adhere to the principles of MPT, seeking to maximize expected return for a given level of risk. Our specific implementation uses a <b>Score-Based Allocation Model</b>.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("<b>The Scoring Algorithm:</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    For each asset $i$, we compute a composite score $S_i$:
    $$S_i = \alpha \cdot \text{ModelPred}_i + \beta \cdot \text{Sentiment}_i + \gamma \cdot \text{TrendStrength}_i$$
    \tWhere $\alpha, \beta, \gamma$ are weights determined by backtesting (e.g., 0.6, 0.2, 0.2).
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    <b>Allocation Logic:</b>
    We select the top $N$ assets ranked by $S_i$ and allocate capital using Risk Parity.
    $$w_i = \\frac{1/\sigma_i}{\sum_{j=1}^{N} 1/\sigma_j}$$
    This ensures that more volatile assets receive a smaller allocation, keeping the overall portfolio risk constant.
    """, styles['BodyTextCustom']))

    # 11.3 Risk Management
    story.append(Paragraph("11.3 Risk Management: The Shield", styles['SectionTitle']))
    story.append(Paragraph("""
    In investing, defense is more important than offense. Our system employs three layers of risk control:
    1.  <b>Stop-Loss Mechanism:</b> Hard stop at 2x ATR (Average True Range).
    2.  <b>Position Sizing:</b> No single position exceeds 5% of total equity (Kelly Criterion constraint).
    3.  <b>Correlation Check:</b> Avoiding overexposure to highly correlated assets (Chapter 5).
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    <b>The Kelly Criterion (Modified):</b>
    $$f^* = p - \\frac{q}{b}$$
    We use a 'Fractional Kelly' (e.g., 0.5 * f*) to avoid the ruinous volatility associated with full Kelly betting.
    """, styles['InsightBox']))

    story.append(PageBreak())
