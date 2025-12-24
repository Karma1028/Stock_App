import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import plotly.graph_objects as go
from datetime import datetime

def plotly_fig_to_image(fig, width=800, height=400):
    """
    Convert Plotly figure to image bytes for PDF embedding.
    """
    try:
        img_bytes = fig.to_image(format="png", width=width, height=height, engine="kaleido")
        return io.BytesIO(img_bytes)
    except Exception as e:
        print(f"Error converting plotly figure: {e}")
        return None

def create_portfolio_pdf(
    investment_params,
    ai_generated_content,
    sector_chart_fig,
    market_cap_fig,
    projection_fig,
    portfolio_data=None,
    treemap_fig=None
):
    """
    Creates a comprehensive PDF report for portfolio analysis.
    
    Args:
        investment_params: Dict with amount, duration, risk_profile, etc.
        ai_generated_content: String containing AI-generated markdown text
        sector_chart_fig: Plotly figure for sector distribution
        market_cap_fig: Plotly figure for market cap distribution
        projection_fig: Plotly figure for future projections
        portfolio_data: Optional DataFrame with portfolio stocks
        treemap_fig: Optional Plotly treemap figure
    
    Returns:
        BytesIO object containing the PDF
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, 
                           topMargin=0.5*inch, bottomMargin=0.5*inch,
                           leftMargin=0.5*inch, rightMargin=0.5*inch)
    
    # Container for PDF elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    normal_style = styles['BodyText']
    normal_style.fontSize = 10
    normal_style.leading = 14
    
    # Title
    elements.append(Paragraph("Portfolio Analysis Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Investment Parameters Section
    elements.append(Paragraph("Investment Parameters", heading_style))
    param_data = [
        ['Investment Amount', f"₹{investment_params.get('amount', 0):,.2f}"],
        ['Investment Type', investment_params.get('type', 'N/A')],
        ['Duration', f"{investment_params.get('duration', 0)} Years"],
        ['Expected Annual Return', f"{investment_params.get('expected_return', 0)}%"],
        ['Risk Profile', investment_params.get('risk_profile', 'N/A')]
    ]
    
    param_table = Table(param_data, colWidths=[3*inch, 3*inch])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(param_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Sector Distribution Chart
    if sector_chart_fig:
        elements.append(Paragraph("Portfolio Sector Distribution", heading_style))
        img_buffer = plotly_fig_to_image(sector_chart_fig, width=700, height=400)
        if img_buffer:
            img = Image(img_buffer, width=6*inch, height=3.4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
    
    # Market Cap Distribution Chart
    if market_cap_fig:
        elements.append(Paragraph("Market Capitalization Distribution", heading_style))
        img_buffer = plotly_fig_to_image(market_cap_fig, width=700, height=400)
        if img_buffer:
            img = Image(img_buffer, width=6*inch, height=3.4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
    
    # Portfolio Projection
    if projection_fig:
        elements.append(Paragraph("Portfolio Growth Projection", heading_style))
        img_buffer = plotly_fig_to_image(projection_fig, width=700, height=400)
        if img_buffer:
            img = Image(img_buffer, width=6*inch, height=3.4*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
    
    # Treemap if available
    if treemap_fig:
        elements.append(PageBreak())
        elements.append(Paragraph("Portfolio Stock Weightage", heading_style))
        img_buffer = plotly_fig_to_image(treemap_fig, width=700, height=500)
        if img_buffer:
            img = Image(img_buffer, width=6*inch, height=4.3*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
    
    # AI Generated Content
    elements.append(PageBreak())
    elements.append(Paragraph("AI-Powered Investment Analysis", heading_style))
    
    # Parse markdown-like content to PDF paragraphs
    if ai_generated_content:
        lines = ai_generated_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                elements.append(Spacer(1, 0.1*inch))
                continue
            
            # Handle headers
            if line.startswith('###'):
                text = line.replace('###', '').strip()
                elements.append(Paragraph(text, styles['Heading3']))
            elif line.startswith('##'):
                text = line.replace('##', '').strip()
                elements.append(Paragraph(text, heading_style))
            elif line.startswith('#'):
                text = line.replace('#', '').strip()
                elements.append(Paragraph(text, title_style))
            elif line.startswith('**') and line.endswith('**'):
                text = line.replace('**', '').strip()
                bold_style = ParagraphStyle('Bold', parent=normal_style, fontName='Helvetica-Bold')
                elements.append(Paragraph(text, bold_style))
            elif line.startswith('-') or line.startswith('*'):
                text = '• ' + line[1:].strip()
                elements.append(Paragraph(text, normal_style))
            else:
                # Regular paragraph
                elements.append(Paragraph(line, normal_style))
    
    # Portfolio Data Table if available
    if portfolio_data is not None and len(portfolio_data) > 0:
        elements.append(PageBreak())
        elements.append(Paragraph("Portfolio Holdings", heading_style))
        
        # Create table data
        table_data = [['Symbol', 'Sector', 'Quantity', 'Price', 'Value']]
        for _, row in portfolio_data.iterrows():
            table_data.append([
                str(row.get('Symbol', '')),
                str(row.get('Sector', 'N/A'))[:20],  # Truncate long sectors
                str(row.get('Quantity', '')),
                f"₹{row.get('Current Price', 0):,.2f}",
                f"₹{row.get('Value', 0):,.2f}"
            ])
        
        holdings_table = Table(table_data, colWidths=[1.2*inch, 1.8*inch, 0.8*inch, 1.2*inch, 1.5*inch])
        holdings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        elements.append(holdings_table)
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    elements.append(Paragraph("This report is generated for informational purposes only. Please consult a financial advisor before making investment decisions.", footer_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer

def get_download_link(pdf_buffer, filename="portfolio_analysis.pdf"):
    """
    Generate a download link for the PDF file.
    """
    pdf_buffer.seek(0)
    b64 = base64.b64encode(pdf_buffer.read()).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'
