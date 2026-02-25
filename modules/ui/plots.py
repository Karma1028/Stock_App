import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def _get_value(df, keys, date):
    """Helper to safely get value from dataframe for a specific date."""
    for key in keys:
        if key in df.index:
            try:
                return df.loc[key, date]
            except:
                pass
    return 0

def plot_capital_structure(balance_sheet, ticker, currency="INR"):
    """
    Plots Capital Structure as a Waterfall chart (Assets - Liabilities = Equity).
    """
    if balance_sheet.empty:
        return go.Figure()

    # Get most recent date
    recent_date = balance_sheet.columns[0]
    
    # Extract values
    total_assets = _get_value(balance_sheet, ["Total Assets"], recent_date)
    total_liab = _get_value(balance_sheet, ["Total Liabilities Net Minority Interest", "Total Liabilities"], recent_date)
    total_equity = _get_value(balance_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"], recent_date)
    
    fig = go.Figure(go.Waterfall(
        name = "Capital Structure",
        orientation = "v",
        measure = ["relative", "relative", "total"],
        x = ["Total Assets", "Total Liabilities", "Total Equity"],
        textposition = "outside",
        text = [f"{total_assets/1e9:.1f}B", f"-{total_liab/1e9:.1f}B", f"{total_equity/1e9:.1f}B"],
        y = [total_assets, -total_liab, total_equity],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"#ef553b"}}, # Red
        increasing = {"marker":{"color":"#00cc96"}}, # Green
        totals = {"marker":{"color":"#636efa"}}       # Blue
    ))

    fig.update_layout(
        title=f"Capital Structure (Assets - Liab = Equity) - {recent_date.year}",
        template="plotly_dark",
        showlegend = False,
        yaxis_title=f"Amount ({currency})"
    )
    return fig

def plot_balance_sheet_trends(balance_sheet, ticker, currency="INR"):
    """
    Plots Balance Sheet Trends: Assets vs (Liabilities + Equity).
    """
    if balance_sheet.empty:
        return go.Figure()

    dates = balance_sheet.columns
    # Sort dates ascending for chart
    dates = sorted(dates)
    
    assets = [_get_value(balance_sheet, ["Total Assets"], d) for d in dates]
    liabilities = [_get_value(balance_sheet, ["Total Liabilities Net Minority Interest", "Total Liabilities"], d) for d in dates]
    equity = [_get_value(balance_sheet, ["Stockholders Equity", "Total Equity Gross Minority Interest"], d) for d in dates]

    fig = go.Figure()

    # Assets Bar (Left side of group)
    fig.add_trace(go.Bar(
        x=dates, 
        y=assets, 
        name='Total Assets', 
        marker_color='#00cc96', # Green
        offsetgroup=0
    ))

    # Liabilities (Right side bottom)
    fig.add_trace(go.Bar(
        x=dates, 
        y=liabilities, 
        name='Total Liabilities', 
        marker_color='#ef553b', # Red
        offsetgroup=1
    ))

    # Equity (Right side top - stacked on Liab)
    fig.add_trace(go.Bar(
        x=dates, 
        y=equity, 
        name='Total Equity', 
        marker_color='#636efa', # Blue
        offsetgroup=1,
        base=liabilities # Stack on top of liabilities
    ))

    fig.update_layout(
        title=f'Balance Sheet Structure - {ticker}',
        xaxis_title='Year',
        yaxis_title=f'Amount ({currency})',
        barmode='group',
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig

def plot_income_statement_trends(income_stmt, ticker, currency="INR"):
    """
    Plots Income Statement Waterfall (Revenue - Expenses = Net Income) for latest year.
    """
    if income_stmt.empty:
        return go.Figure()

    # Get most recent date
    recent_date = income_stmt.columns[0]
    
    revenue = _get_value(income_stmt, ["Total Revenue", "Operating Revenue"], recent_date)
    net_income = _get_value(income_stmt, ["Net Income", "Net Income Common Stockholders"], recent_date)
    expenses = revenue - net_income # Simplified expenses

    fig = go.Figure(go.Waterfall(
        name = "Income Statement",
        orientation = "v",
        measure = ["relative", "relative", "total"],
        x = ["Total Revenue", "Expenses", "Net Income"],
        textposition = "outside",
        text = [f"{revenue/1e9:.1f}B", f"-{expenses/1e9:.1f}B", f"{net_income/1e9:.1f}B"],
        y = [revenue, -expenses, net_income],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
        decreasing = {"marker":{"color":"#ef553b"}}, # Red
        increasing = {"marker":{"color":"#00cc96"}}, # Green
        totals = {"marker":{"color":"#ffa15a"}}       # Yellow/Orange
    ))

    fig.update_layout(
        title=f'Income Statement Waterfall - {ticker} ({recent_date.year})',
        template="plotly_dark",
        showlegend = False,
        yaxis_title=f'Amount ({currency})'
    )
    return fig

def plot_cash_flow_trends(cash_flow, ticker, currency="INR"):
    """
    Plots Cash Flow Trends.
    """
    if cash_flow.empty:
        return go.Figure()

    dates = cash_flow.columns
    dates = sorted(dates)
    
    operating = [_get_value(cash_flow, ["Operating Cash Flow", "Total Cash From Operating Activities"], d) for d in dates]
    investing = [_get_value(cash_flow, ["Investing Cash Flow", "Total Cashflows From Investing Activities"], d) for d in dates]
    financing = [_get_value(cash_flow, ["Financing Cash Flow", "Total Cash From Financing Activities"], d) for d in dates]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=operating, name='Operating CF', marker_color='#00cc96'))
    fig.add_trace(go.Bar(x=dates, y=investing, name='Investing CF', marker_color='#ef553b'))
    fig.add_trace(go.Bar(x=dates, y=financing, name='Financing CF', marker_color='#636efa'))

    fig.update_layout(
        title=f'Cash Flow Trends - {ticker}',
        barmode='group',
        xaxis_title='Year',
        yaxis_title=f'Amount ({currency})',
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig

def plot_margins(income_stmt, ticker):
    """
    Plots Net Profit Margin trends.
    """
    if income_stmt.empty:
        return go.Figure()

    dates = income_stmt.columns
    dates = sorted(dates)
    
    revenue = [_get_value(income_stmt, ["Total Revenue", "Operating Revenue"], d) for d in dates]
    net_income = [_get_value(income_stmt, ["Net Income", "Net Income Common Stockholders"], d) for d in dates]
    
    margins = []
    for r, n in zip(revenue, net_income):
        if r != 0:
            margins.append((n / r) * 100)
        else:
            margins.append(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=margins, name='Net Margin %', mode='lines+markers', line=dict(color='#ab63fa', width=3)))

    fig.update_layout(
        title=f'Net Profit Margin Trend - {ticker}',
        xaxis_title='Year',
        yaxis_title='Margin (%)',
        template="plotly_dark",
        hovermode="x unified"
    )
    return fig

def plot_stock_chart(df, ticker, chart_type='Candlestick', show_technicals=True, sentiment_df=None):
    """
    Plots the main stock chart with interactive features.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if df.empty:
        return go.Figure()

    # Create Subplots: Main Price + Volume + (Optional) Sentiment/MACD
    # Rows: 1. Price, 2. Volume, 3. MACD/Sentiment
    rows = 3 if show_technicals else 2
    row_heights = [0.6, 0.2, 0.2] if show_technicals else [0.7, 0.3]
    
    # Determine titles
    titles = [f"{ticker} Price", "Volume"]
    if show_technicals:
        titles.append("Technical/Sentiment")

    fig = make_subplots(
        rows=rows, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=row_heights, 
        subplot_titles=titles
    )

    # 1. Price Chart
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name='OHLC'
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], name='Close Price',
            line=dict(color='#00cc96', width=2)
        ), row=1, col=1)
        
    # Moving Averages
    if 'SMA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)
    if 'SMA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200', line=dict(color='blue', width=1)), row=1, col=1)
        
    # Bollinger Bands
    if 'BB_High' in df.columns and show_technicals:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name='BB Upper', line=dict(color='gray', dash='dot', width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name='BB Lower', line=dict(color='gray', dash='dot', width=1), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=1, col=1)

    # 2. Volume Chart
    # Determine colors: Green if Close >= Open, Red otherwise
    colors = ['#00cc96' if c >= o else '#ef553b' for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)

    # 3. Technicals (MACD) & Sentiment
    if show_technicals and 'MACD' in df.columns:
         fig.add_trace(go.Bar(x=df.index, y=df['MACD_Diff'], name='MACD Hist', marker_color='gray', opacity=0.3), row=3, col=1)
         fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#636efa', width=1)), row=3, col=1)
         fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange', width=1)), row=3, col=1)

    # Plot Sentiment (Overlay or Standalone in Row 3)
    if sentiment_df is not None and not sentiment_df.empty:
        # Align sentiment dates to df index if possible, or just plot scatter
        # Sentiment is daily, df is daily.
        fig.add_trace(go.Scatter(
            x=sentiment_df['date_only'], 
            y=sentiment_df['bert_sentiment_score'], 
            name='AI Sentiment (BERT)',
            line=dict(color='#00cc96', width=2, dash='dot'),
            mode='lines+markers'
        ), row=3, col=1)
            
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig
