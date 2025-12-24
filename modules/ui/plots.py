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
