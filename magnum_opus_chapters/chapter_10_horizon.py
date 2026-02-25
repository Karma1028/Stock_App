from reportlab.platypus import Paragraph, Spacer, Image, PageBreak
from .utils import create_custom_styles, add_dataframe_table

def generate(story):
    """Generate Chapter 10: The Horizon"""
    styles = create_custom_styles()
    
    # Chapter Title
    story.append(Paragraph("Chapter 10: The Horizon", styles['ChapterTitle']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<i>\"The future is not a destination, but a direction.\" — Ed Catmull</i>", styles['Caption']))
    story.append(Spacer(1, 20))
    
    # 10.1 Introduction
    story.append(Paragraph("10.1 Seeing Beyond the Now", styles['SectionTitle']))
    story.append(Paragraph("""
    In Chapter 8, we focused on short-term prediction (5-day return) using XGBoost. 'The Horizon' shifts our gaze to the medium term (30-90 days). For this, we require a different class of model—one that understands seasonality, trends, and holidays. We selected <b>Facebook Prophet</b>.
    """, styles['BodyTextCustom']))
    
    # 10.2 Prophet Architecture
    story.append(Paragraph("10.2 Prophet Architecture: Decomposable Time Series", styles['SectionTitle']))
    story.append(Paragraph("""
    Prophet is an additive regression model that decomposes a time series into three main components:
    $$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    1.  <b>Trend $g(t)$:</b> A non-periodic change in the value of the time series. We use a piecewise linear growth model to capture trend shifts.
    2.  <b>Seasonality $s(t)$:</b> Periodic changes (e.g., weekly trading cycles, quarterly earnings effects). Modeled using Fourier Series.
    3.  <b>Holidays $h(t)$:</b> Irregular events (e.g., Budget Day, Diwali) that impact market behavior.
    """, styles['BodyTextCustom']))

    # 10.3 The Forecast
    story.append(Paragraph("10.3 The Forecast: Visualizing the Future", styles['SectionTitle']))
    story.append(Paragraph("""
    The power of Prophet lies in its interpretability and uncertainty intervals. The shaded region in our forecast represents the confidence interval, widening as we project further into the future—a visual representation of increasing entropy.
    """, styles['BodyTextCustom']))
    
    # Placeholder for Prophet plot if we had one, but describing the concept
    story.append(Paragraph("<b>The 'Fan Chart' Concept:</b>", styles['BodyTextCustom']))
    story.append(Paragraph("""
    Our forecast generates a 'Fan Chart', showing the most likely path (median) surrounded by probabilistic bands (80% confidence). This allows traders to manage risk by visualizing worst-case and best-case scenarios.
    """, styles['InsightBox']))

    # 10.4 Limitations
    story.append(Paragraph("10.4 Limitations of extrapolation", styles['SectionTitle']))
    story.append(Paragraph("""
    Prophet assumes that the future will resemble the past. However, structural breaks (e.g., a pandemic or a new regulation) can invalidate historical patterns. Therefore, Prophet forecasts should be used as a baseline trend, overlayed with fundamental analysis (The Sentinel).
    """, styles['BodyTextCustom']))

    story.append(Paragraph("""
    python
    # Code Snippet: Prophet Forecasting
    from prophet import Prophet
    
    m = Prophet(daily_seasonality=True)
    m.add_country_holidays(country_name='IN')
    m.fit(df)
    
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    
    fig = m.plot(forecast)
    """, styles['CodeBlock']))

    story.append(PageBreak())
