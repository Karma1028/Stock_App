import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# Try to import custom styles from magnum_opus, else fallback
try:
    from magnum_opus_chapters.utils import create_custom_styles, create_header_footer
except ImportError:
    # Fallback styles if import fails
    def create_custom_styles():
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='ChapterTitle', parent=styles['Heading1'], fontSize=24, textColor=colors.darkblue))
        styles.add(ParagraphStyle(name='SectionTitle', parent=styles['Heading2'], fontSize=18, spaceBefore=15))
        styles.add(ParagraphStyle(name='BodyTextCustom', parent=styles['Normal'], fontSize=12, leading=16))
        styles.add(ParagraphStyle(name='InsightBox', parent=styles['Normal'], backColor=colors.whitesmoke, borderPadding=10, textColor=colors.darkgreen))
        return styles

    def create_header_footer(canvas, doc):
        pass

class QuantReportGenerator:
    def __init__(self, filename):
        self.filename = filename
        self.styles = create_custom_styles()
        self.story = []

    def generate(self, state, user_input):
        # 1. Title Page
        self._add_title_page(state, user_input)
        
        # 2. Executive Synthesis (DeepSeek Output)
        self._add_executive_synthesis(state)
        
        # 3. Quantitative Engine (Technical Analysis)
        self._add_quant_engine(state)
        
        # 4. Fundamental Reality (News)
        self._add_fundamental_reality(state)
        
        # 5. Build PDF
        doc = SimpleDocTemplate(
            self.filename,
            pagesize=letter,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=72
        )
        doc.build(self.story, onFirstPage=create_header_footer, onLaterPages=create_header_footer)
        return self.filename

    def _add_title_page(self, state, user_input):
        title_style = self.styles['ChapterTitle']
        normal_style = self.styles['BodyTextCustom']
        
        # Title
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(f"Institutional Strategy Report: {state['ticker']}", title_style))
        self.story.append(Spacer(1, 0.5*inch))
        
        # User Metadata Table
        data = [
            ["Client Profile", ""],
            ["Capital", f"Rs. {user_input.get('capital', 0):,}"],
            ["Horizon", user_input.get('horizon', 'N/A')],
            ["Risk Tolerance", user_input.get('risk', 'N/A')],
            ["Date", datetime.now().strftime("%Y-%m-%d")]
        ]
        
        t = Table(data, colWidths=[2*inch, 3*inch])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
            ('TEXTCOLOR', (0,0), (0,-1), colors.grey),
            ('TEXTCOLOR', (1,0), (1,-1), colors.black),
            ('LINEBELOW', (0,0), (-1,-1), 1, colors.lightgrey),
        ]))
        self.story.append(t)
        self.story.append(PageBreak())

    def _add_executive_synthesis(self, state):
        self.story.append(Paragraph("1. The Executive Verdict", self.styles['ChapterTitle']))
        self.story.append(Paragraph("<i>Authored by the reasoning engine (DeepSeek-R1)</i>", self.styles['BodyTextCustom']))
        self.story.append(Spacer(1, 20))
        
        # The prompt output is usually 3 paragraphs. We'll format them nicely.
        report_text = state['final_report']
        
        # Split by double newlines to find paragraphs, simple heuristic
        paragraphs = report_text.split('\n\n')
        
        for p in paragraphs:
            if p.strip():
                # Check for "Verdict" keyword to highlight
                if "Verdict" in p or "Conclusion" in p:
                    self.story.append(Paragraph(p, self.styles['InsightBox']))
                else:
                    self.story.append(Paragraph(p, self.styles['BodyTextCustom']))
                self.story.append(Spacer(1, 10))
                
        self.story.append(PageBreak())

    def _add_quant_engine(self, state):
        self.story.append(Paragraph("2. The Quantitative Engine", self.styles['ChapterTitle']))
        self.story.append(Spacer(1, 10))
        
        # LSTM Section
        self.story.append(Paragraph("2.1 LSTM Trend Analysis", self.styles['SectionTitle']))
        prob = state['lstm_prob'] * 100
        unc = state['lstm_uncertainty'] * 100
        
        lstm_text = f"""
        Our Monte Carlo LSTM neural network has analyzed the price action. 
        It assigns a <b>{prob:.1f}% probability</b> of a bullish trend continuation or breakout. 
        The model uncertainty is ±{unc:.1f}%, indicating the confidence level of this prediction.
        """
        self.story.append(Paragraph(lstm_text, self.styles['BodyTextCustom']))
        self.story.append(Spacer(1, 15))

        # GARCH Section
        self.story.append(Paragraph("2.2 GARCH Volatility Gate", self.styles['SectionTitle']))
        vol = state['garch_volatility'] * 100
        
        garch_text = f"""
        The GARCH(1,1) model forecasts a 1-day volatility of <b>{vol:.2f}%</b>. 
        High volatility regimes often precede trend reversals.
        
        Risk Status: 
        {'<b>🟢 LOW RISK</b>' if vol < 1.5 else '<b>🔴 HIGH RISK</b>' if vol > 3.0 else '<b>🟡 MODERATE RISK</b>'}
        """
        self.story.append(Paragraph(garch_text, self.styles['BodyTextCustom']))
        
        # Suggestion
        self.story.append(Paragraph(f"""
        <b>Quantitative Suggestion:</b> Based on the combination of strong technicals ({prob:.0f}%) and 
        {vol:.2f}% volatility, the algo suggests a position sizing of 
        <b>{min(100, int(prob/vol)) if vol > 0 else 0}%</b> of the target allocation.
        """, self.styles['InsightBox']))
        
        self.story.append(PageBreak())

    def _add_fundamental_reality(self, state):
        self.story.append(Paragraph("3. Fundamental Reality", self.styles['ChapterTitle']))
        self.story.append(Spacer(1, 10))
        self.story.append(Paragraph("Latest market news and sentiment analysis (Scraped Live).", self.styles['BodyTextCustom']))
        self.story.append(Spacer(1, 10))
        
        news_text = state['scraped_news']
        # Clean up markdown format for PDF
        import re
        # Replace bold **text** with <b>text</b>
        clean_news = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', news_text)
        # Remove headers
        clean_news = clean_news.replace('###', '')
        
        # Limit length to avoid overflow
        if len(clean_news) > 3000:
            clean_news = clean_news[:3000] + "... [Truncated]"
            
        # Escape special characters that XML/ReportLab might hate, 
        # but keep our <b> tags
        clean_news = clean_news.replace('&', '&amp;') # Basic escaping
        # Re-apply bold tags because we just escaped them &lt;b&gt;...
        clean_news = clean_news.replace('&amp;lt;b&amp;gt;', '<b>').replace('&amp;lt;/b&amp;gt;', '</b>')
        
        # Actually simplest approach: 
        # 1. Escape everything first
        # 2. THEN apply regex for bold
        
        clean_news = news_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        clean_news = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_news)
        clean_news = clean_news.replace('###', '')

        if len(clean_news) > 3000:
             clean_news = clean_news[:3000] + "... [Truncated]"

        try:
            self.story.append(Paragraph(clean_news, self.styles['BodyTextCustom']))
        except:
            # Fallback if parsing keeps failing
            self.story.append(Paragraph(clean_news.replace('<b>', '').replace('</b>', ''), self.styles['BodyTextCustom']))
