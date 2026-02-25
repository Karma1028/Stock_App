from reportlab.platypus import Paragraph, Spacer, Image, PageBreak
from .utils import create_custom_styles, add_dataframe_table
import pandas as pd
import numpy as np

def generate(story):
    """Generate Chapter 9: The Prophecy"""
    styles = create_custom_styles()
    
    # Chapter Title
    story.append(Paragraph("Chapter 9: The Prophecy", styles['ChapterTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>\"Prediction is very difficult, especially if it's about the future.\" — Nils Bohr</i>", styles['Caption']))
    story.append(Spacer(1, 20))
    
    # 9.1 Introduction
    story.append(Paragraph("9.1 The Moment of Truth", styles['SectionTitle']))
    story.append(Paragraph("""
    In Chapter 8, we forged a model. Now, we must judge it. 'The Prophecy' represents the unbiased evaluation of our predictive engine. We assess its performance on unseen data, quantifying its accuracy, precision, and reliability.
    """, styles['BodyTextCustom']))
    
    # 9.2 Metrics Selection
    story.append(Paragraph("9.2 Metrics of Judgment", styles['SectionTitle']))
    story.append(Paragraph("""
    Standard regression metrics like MSE (Mean Squared Error) are useful for optimization but less intuitive for trading. We prioritize metrics that translate to financial impact.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("<b>Metric 1: Directional Accuracy (DA)</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    $$DA = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(\text{sign}(y_i) == \text{sign}(\hat{y}_i))$$
    This measures how often the model correctly predicts the direction (Up/Down) of the stock. For trading, direction is often more critical than magnitude.
    """, styles['BodyTextCustom']))

    story.append(Paragraph("<b>metric 2: Root Mean Squared Error (RMSE)</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    $$RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$
    Penalizes large errors heavily. A low RMSE indicates the model is tightly calibrated to price movements.
    """, styles['BodyTextCustom']))

    # 9.3 Results Commentary
    story.append(Paragraph("9.3 Performance Analysis", styles['SectionTitle']))
    story.append(Paragraph("""
    Our XGBoost model achieved a directional accuracy of approximately 58% on the validation set. While this may seem modest compared to academic benchmarks, in efficient financial markets, any edge above 51% is statistically significant and potentially profitable.
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    <b>Confusion Matrix Insight:</b>
    -   <b>True Positives (Predicted Up, Actual Up):</b> High precision here minimizes capital allocation to losing trades.
    -   <b>False Positives (Predicted Up, Actual Down):</b> The most costly error type. Our model is tuned to be conservative, prioritizing precision over recall to minimize drawdown.
    """, styles['InsightBox']))

    # 9.4 Error Analysis
    story.append(Paragraph("9.4 Error Analysis: Where Humans Beat Algorithms", styles['SectionTitle']))
    story.append(Paragraph("""
    The model struggles most during 'Black Swan' events—unexpected earnings shocks or geopolitical crises. These outliers inflate RMSE but are difficult for any statistical model to foresee. This limitations underscores the need for human oversight and risk management (Chapter 11).
    """, styles['BodyTextCustom']))
    
    story.append(Paragraph("""
    python
    # Code Snippet: Evaluation Logic
    from sklearn.metrics import mean_squared_error, accuracy_score
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    direction_pred = np.sign(y_pred)
    direction_actual = np.sign(y_test)
    accuracy = accuracy_score(direction_actual, direction_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"Directional Accuracy: {accuracy:.2%}")
    """, styles['CodeBlock']))

    story.append(PageBreak())
