import yfinance as yf
import pandas as pd
import ta
from config import Config
from modules.utils.helpers import setup_logging
from pathlib import Path

# Setup logging
setup_logging(Config.LOGS_DIR / "data_manager.log")

class StockDataManager:
    def __init__(self):
        self.use_local = Config.USE_LOCAL_DATA
        self.default_tickers = [
            'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
            'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS',
            'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS',
            'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
            'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS',
            'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
            'MARUTI.NS', 'M&M.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS',
            'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SUNPHARMA.NS',
            'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS',
            'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'
        ]

    def get_stock_list(self):
        """
        Returns a list of stock tickers from EQUITY.csv if available, else default list.
        """
        if True: # Always try to load full list if available
            equity_file = Config.DATA_DIR / "EQUITY.csv"
            if equity_file.exists():
                try:
                    df = pd.read_csv(equity_file)
                    # CSV has 'SYMBOL' column (uppercase)
                    if 'SYMBOL' in df.columns:
                        # Ensure tickers have .NS suffix for yfinance
                        tickers = [f"{sym}.NS" if not str(sym).endswith('.NS') else sym for sym in df['SYMBOL'].tolist()]
                        return tickers
                    elif 'Symbol' in df.columns:
                        tickers = [f"{sym}.NS" if not str(sym).endswith('.NS') else sym for sym in df['Symbol'].tolist()]
                        return tickers
                except Exception as e:
                    print(f"Error reading local equity file: {e}")
        
        return self.default_tickers

    def _get_valid_symbol(self, symbol):
        """
        Ensures symbol is valid for yfinance.
        """
        if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
             # Simple heuristic, user can provide full ticker
             return f"{symbol}.NS"
        return symbol

    def get_historical_data(self, symbol, period="1y", interval="1d"):
        """
        Fetches historical data from yfinance.
        """
        symbol = self._get_valid_symbol(symbol)
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df.empty:
                return pd.DataFrame()
            

            # Flatten MultiIndex columns if present (yfinance v0.2+)
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.get_level_values(0):
                    df.columns = df.columns.get_level_values(0)
                elif df.columns.nlevels > 1 and 'Close' in df.columns.get_level_values(1):
                    df.columns = df.columns.get_level_values(1)
            
            # Ensure index is DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Add technical indicators
            try:
                df = ta.add_all_ta_features(
                    df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
                )
            except Exception as e:
                print(f"Warning: TA generation failed for {symbol} (possibly insufficient data): {e}")
            
            return df
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def get_bulk_historical_data(self, tickers, period="1y"):
        """
        Fetches historical data for multiple stocks efficiently.
        """
        try:
            # yf.download returns a MultiIndex dataframe if multiple tickers
            data = yf.download(tickers, period=period, threads=True, progress=False)
            return data
        except Exception as e:
            print(f"Error fetching bulk data: {e}")
            return pd.DataFrame()

    def get_cached_data(self, tickers, period="5y"):
        """
        Fetches data with local caching to avoid redundant API calls.
        Cache is stored in data/raw/bulk_data_{period}.pkl
        """
        cache_dir = Config.DATA_DIR / "raw"
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"bulk_data_{period}.pkl"
        
        # Check if cache exists and is fresh (e.g., < 24 hours)
        if cache_file.exists():
            try:
                import time
                file_age = time.time() - cache_file.stat().st_mtime
                if file_age < 86400: # 24 hours
                    print(f"Loading cached data from {cache_file}")
                    return pd.read_pickle(cache_file)
            except Exception as e:
                print(f"Error reading cache: {e}")
        
        # Fetch fresh data
        print("Fetching fresh data from API...")
        data = self.get_bulk_historical_data(tickers, period=period)
        
        # Save to cache
        if not data.empty:
            try:
                data.to_pickle(cache_file)
                print(f"Saved data to cache: {cache_file}")
            except Exception as e:
                print(f"Error saving cache: {e}")
                
        return data

    def get_live_data(self, symbol):
        """
        Fetches live data for a symbol.
        """
        symbol = self._get_valid_symbol(symbol)
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            # fast_info = ticker.fast_info # Alternative if info is slow
            
            # Get latest price from history if info is missing currentPrice
            current_price = info.get("currentPrice")
            if not current_price:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_price = hist["Close"].iloc[-1]
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "previous_close": info.get("previousClose"),
                "open": info.get("open"),
                "day_high": info.get("dayHigh"),
                "day_low": info.get("dayLow"),
                "volume": info.get("volume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "long_business_summary": info.get("longBusinessSummary"),
                "website": info.get("website"),
                # Extended Fields
                "currency": info.get("currency"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "float_shares": info.get("floatShares"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "book_value": info.get("bookValue"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "forward_pe": info.get("forwardPE"),
                "trailing_eps": info.get("trailingEps"),
                "forward_eps": info.get("forwardEps"),
                "enterprise_value": info.get("enterpriseValue"),
                "total_revenue": info.get("totalRevenue"),
            "P/B Ratio": info.get("priceToBook"),
            "Debt/Equity": info.get("debtToEquity"),
            "ROE": info.get("returnOnEquity"),
            "Profit Margin": info.get("profitMargins"),
            "Dividend Yield": info.get("dividendYield"),
            }
        except Exception as e:
            print(f"Error fetching live data for {symbol}: {e}")
            return {}

    def calculate_kpis(self, symbol):
        """
        Calculates KPIs based on historical data.
        """
        df = self.get_historical_data(symbol, period="1y")
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        # Example KPIs
        return {
            "RSI": latest.get("momentum_rsi"),
            "MACD": latest.get("trend_macd"),
            "SMA_50": latest.get("trend_sma_fast"),
            "SMA_200": latest.get("trend_sma_slow")
        }

    def get_balance_sheet(self, symbol):
        """
        Fetches the annual balance sheet.
        """
        symbol = self._get_valid_symbol(symbol)
        try:
            ticker = yf.Ticker(symbol)
            bs = ticker.balance_sheet
            if bs.empty:
                return pd.DataFrame()
            return bs
        except Exception as e:
            print(f"Error fetching balance sheet for {symbol}: {e}")
            return pd.DataFrame()

    def get_income_statement(self, symbol):
        """
        Fetches the annual income statement.
        """
        symbol = self._get_valid_symbol(symbol)
        try:
            ticker = yf.Ticker(symbol)
            ist = ticker.income_stmt
            if ist.empty:
                return pd.DataFrame()
            return ist
        except Exception as e:
            print(f"Error fetching income statement for {symbol}: {e}")
            return pd.DataFrame()

    def get_cash_flow(self, symbol):
        """
        Fetches the annual cash flow statement.
        """
        symbol = self._get_valid_symbol(symbol)
        try:
            ticker = yf.Ticker(symbol)
            cf = ticker.cashflow
            if cf.empty:
                return pd.DataFrame()
            return cf
        except Exception as e:
            print(f"Error fetching cash flow for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_liquidity_ratios(self, balance_sheet):
        """
        Calculates liquidity ratios from the balance sheet.
        """
        if balance_sheet.empty:
            return {}
        
        try:
            # Helper to safely get value for the most recent year (first column)
            def get_val(key_candidates):
                if isinstance(key_candidates, str):
                    key_candidates = [key_candidates]
                
                for key in key_candidates:
                    if key in balance_sheet.index:
                        return balance_sheet.loc[key].iloc[0]
                
                # Fuzzy search if exact match fails
                for key in key_candidates:
                    for idx in balance_sheet.index:
                        if key.lower() in str(idx).lower():
                            return balance_sheet.loc[idx].iloc[0]
                return 0

            current_assets = get_val(["Total Current Assets", "Current Assets"])
            current_liabilities = get_val(["Total Current Liabilities", "Current Liabilities"])
            inventory = get_val(["Inventory", "Inventories"])
            total_assets = get_val(["Total Assets"])
            total_equity = get_val(["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"])
            total_debt = get_val(["Total Debt", "Total Liabilities Net Minority Interest"]) # Fallback for debt proxy if needed

            ratios = {}
            
            # Current Ratio
            if current_liabilities and current_liabilities != 0:
                ratios["Current Ratio"] = current_assets / current_liabilities
            
            # Quick Ratio
            if current_liabilities and current_liabilities != 0:
                ratios["Quick Ratio"] = (current_assets - inventory) / current_liabilities
            
            # Debt to Equity
            if total_equity and total_equity != 0:
                ratios["Debt/Equity"] = total_debt / total_equity

            return ratios
        except Exception as e:
            print(f"Error calculating ratios: {e}")
            return {}
    def get_nifty50_sector_map(self):
        return {
            'ADANIENT.NS':     'Conglomerate',
            'ADANIPORTS.NS':   'Ports & Logistics',
            'APOLLOHOSP.NS':   'Healthcare - Hospitals',
            'ASIANPAINT.NS':   'Materials - Paints',
            'AXISBANK.NS':     'Financials - Banking',
            'BAJAJ-AUTO.NS':   'Automobiles',
            'BAJAJFINSV.NS':   'Financials - Financial Services',
            'BAJFINANCE.NS':   'Financials - NBFC',
            'BHARTIARTL.NS':   'Telecommunications',
            'BPCL.NS':         'Energy - Oil & Gas',
            'BRITANNIA.NS':    'Consumer Staples - Food (FMCG)',
            'CIPLA.NS':        'Pharmaceuticals',
            'COALINDIA.NS':    'Energy - Coal',
            'DIVISLAB.NS':     'Pharmaceuticals - Specialty Chemicals',
            'DRREDDY.NS':      'Pharmaceuticals',
            'EICHERMOT.NS':    'Automobiles',
            'GRASIM.NS':       'Materials - Cement & Diversified',
            'HCLTECH.NS':      'Information Technology',
            'HDFCBANK.NS':     'Financials - Banking',
            'HDFCLIFE.NS':     'Financials - Insurance',
            'HEROMOTOCO.NS':   'Automobiles',
            'HINDALCO.NS':     'Metals & Mining',
            'HINDUNILVR.NS':   'Consumer Staples - FMCG',
            'ICICIBANK.NS':    'Financials - Banking',
            'INDUSINDBK.NS':   'Financials - Banking',
            'INFY.NS':         'Information Technology',
            'ITC.NS':          'Consumer Staples - Diversified',
            'JSWSTEEL.NS':     'Metals & Mining - Steel',
            'KOTAKBANK.NS':    'Financials - Banking',
            'LT.NS':           'Industrials - Engineering & Infra',
            'MARUTI.NS':       'Automobiles',
            'M&M.NS':          'Automobiles & Farm Equipment',
            'NESTLEIND.NS':    'Consumer Staples - Food (FMCG)',
            'NTPC.NS':         'Utilities - Power',
            'ONGC.NS':         'Energy - Oil & Gas',
            'POWERGRID.NS':    'Utilities - Power Transmission',
            'RELIANCE.NS':     'Conglomerate - Energy & Retail',
            'SBILIFE.NS':      'Financials - Insurance',
            'SBIN.NS':         'Financials - Banking',
            'SUNPHARMA.NS':    'Pharmaceuticals',
            'TATACONSUM.NS':   'Consumer Staples - FMCG',
            'TATAMOTORS.NS':   'Automobiles',
            'TATASTEEL.NS':    'Metals & Mining - Steel',
            'TCS.NS':          'Information Technology',
            'TECHM.NS':        'Information Technology',
            'TITAN.NS':        'Consumer Discretionary - Retail & Jewellery',
            'ULTRACEMCO.NS':   'Materials - Cement',
            'WIPRO.NS':        'Information Technology'
        }

    def get_stock_news(self, symbol):
        """
        Fetches latest news for a stock.
        """
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if news:
                return [n.get('title', '') for n in news[:3] if n.get('title')] # Top 3 headlines
            return []
        except:
            return []
    
    def get_stock_name_mapping(self):
        """
        Returns a dictionary mapping company names to ticker symbols.
        This enables search by company name.
        """
        name_to_ticker = {}
        ticker_to_name = {}
        
        # Get all tickers
        tickers = self.get_stock_list()
        
        for ticker in tickers:
            try:
                t = yf.Ticker(ticker)
                info = t.info
                long_name = info.get('longName', '')
                short_name = info.get('shortName', '')
                
                if long_name:
                    name_to_ticker[long_name] = ticker
                    ticker_to_name[ticker] = long_name
                elif short_name:
                    name_to_ticker[short_name] = ticker
                    ticker_to_name[ticker] = short_name
                else:
                    # Use the base symbol as name
                    base = ticker.replace('.NS', '').replace('.BO', '')
                    name_to_ticker[base] = ticker
                    ticker_to_name[ticker] = base
            except:
                # Fallback to ticker symbol
                base = ticker.replace('.NS', '').replace('.BO', '')
                name_to_ticker[base] = ticker
                ticker_to_name[ticker] = base
        
        return {'name_to_ticker': name_to_ticker, 'ticker_to_name': ticker_to_name}
    
    def get_market_cap_category(self, market_cap):
        """
        Categorizes a stock based on market capitalization (Indian market standards).
        Large Cap: > ₹20,000 Crores
        Mid Cap: ₹5,000 - ₹20,000 Crores
        Small Cap: < ₹5,000 Crores
        """
        if market_cap is None:
            return "Unknown"
        
        # Convert to Crores (1 Crore = 10 Million)
        market_cap_crores = market_cap / 10_000_000
        
        if market_cap_crores >= 20000:
            return "Large Cap"
        elif market_cap_crores >= 5000:
            return "Mid Cap"
        else:
            return "Small Cap"
    def get_top_gainers(self, limit=5):
        """
        Identifies top gaining stocks from the tracked list for the day.
        """
        tickers = self.get_stock_list()
        # For performance, maybe limit to first 10-20 or use bulk fetch if possible
        # Using bulk fetch for efficiency
        try:
            # Fetch last 2 days to calculate % change. 
            # Note: Fetching thousands of stocks takes time.
            # We strictly filter for today's data if possible, or last close.
            # Using 5d to be safe on weekends/holidays.
            
            # Optimization: If list > 100, maybe chunk? For now, let yf handle threads.
            data = self.get_bulk_historical_data(tickers, period="5d")
            if data.empty:
                return []
            
            # yfinance bulk data is MultiIndex (Price, Ticker)
            # We need Close price
            closes = data['Close']
            
            # Drop columns with all NaNs (delisted or no data)
            closes = closes.dropna(axis=1, how='all')
            
            # Calculate % change of last available day vs previous
            pct_change = closes.pct_change().iloc[-1]
            top_gainers = pct_change.nlargest(limit)
            
            results = []
            for ticker, change in top_gainers.items():
                results.append({
                    "symbol": ticker,
                    "change_pct": change * 100,
                    "price": closes[ticker].iloc[-1]
                })
            return results
        except Exception as e:
            print(f"Error getting top gainers: {e}")
            return []

    def get_market_sentiment(self):
        """
        Derives overall market sentiment from Nifty 50 stocks performance.
        """
        try:
            tickers = self.get_stock_list() # This list is basically Nifty 50
            data = self.get_bulk_historical_data(tickers, period="5d")
            
            if data.empty:
                return {"status": "Neutral", "score": 50, "summary": "Market data unavailable."}
            
            # Calculate breadth
            closes = data['Close']
            changes = closes.pct_change().iloc[-1]
            
            advances = (changes > 0).sum()
            declines = (changes < 0).sum()
            total = len(changes)
            
            # Sentiment Score (0 to 100)
            score = (advances / total) * 100
            
            if score > 60:
                status = "Bullish"
                color = "green"
            elif score < 40:
                status = "Bearish"
                color = "red"
            else:
                status = "Neutral"
                color = "yellow"
                
            summary = f"Market Breadth: {advances} Advances vs {declines} Declines."
            
            return {
                "status": status, 
                "score": score, 
                "summary": summary,
                "color": color
            }
        except Exception as e:
             print(f"Error getting market sentiment: {e}")
             return {"status": "Unknown", "score": 50, "summary": "Error calculating sentiment."}
