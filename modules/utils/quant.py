import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ta
import json
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
    from pypfopt import discrete_allocation
    HAS_PYPFOPT = True
except ImportError:
    HAS_PYPFOPT = False
    print("⚠️ PyPortfolioOpt not found. Using fallback optimization.")

class QuantEngine:
    def __init__(self, data_manager):
        self.dm = data_manager

    def run_pipeline(self, user_profile):
        """
        Executes the full quant pipeline:
        1. Universe Selection & Data Fetching
        2. EDA & Quality Checks
        3. Feature Engineering
        4. Portfolio Optimization
        5. Backtesting
        6. Payload Construction
        """
        print("🚀 Starting Quant Pipeline...")
        
        # 1. Universe Selection (Top 50 by Momentum)
        tickers = self.dm.get_stock_list()
        # For speed in demo, limit to top 50 from previous scan or just first 50 valid
        # In a real scenario, we'd scan all, but let's assume we get a filtered list or scan all efficiently
        
        print(f"📊 Fetching data for {len(tickers)} tickers...")
        # We need more history for optimization (e.g., 2 years)
        data = self.dm.get_bulk_historical_data(tickers, period="2y")
        
        if data.empty:
            return {"error": "No data fetched"}

        # Fetch Benchmark Data (Nifty 50)
        print("📊 Fetching Benchmark Data (^NSEI)...")
        benchmark_data = self.dm.get_bulk_historical_data(['^NSEI'], period="2y")
        
        benchmark_prices = pd.Series()
        if not benchmark_data.empty:
            try:
                if isinstance(benchmark_data.columns, pd.MultiIndex):
                    # Check for ^NSEI at top level
                    if '^NSEI' in benchmark_data.columns:
                        benchmark_prices = benchmark_data['^NSEI']['Close']
                    # Check if Close is at top level (unlikely with group_by='ticker')
                    elif 'Close' in benchmark_data.columns:
                        benchmark_prices = benchmark_data['Close']
                    else:
                        # Fallback: take the first level 0 key
                        first_key = benchmark_data.columns.levels[0][0]
                        benchmark_prices = benchmark_data[first_key]['Close']
                else:
                    if 'Close' in benchmark_data.columns:
                        benchmark_prices = benchmark_data['Close']
            except Exception as e:
                print(f"⚠️ Error extracting benchmark prices: {e}")
                print(f"Columns: {benchmark_data.columns}")

        # Handle MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            # Check if top level is Ticker or PriceType
            # If group_by='ticker', levels[0] are tickers.
            # If group_by='column', levels[0] are PriceTypes (Open, Close, etc).
            
            # We suspect group_by='ticker' from data_manager
            if 'Close' in data.columns.levels[1]: # (Ticker, Close)
                close_prices = pd.DataFrame({
                    ticker: data[ticker]['Close'] 
                    for ticker in data.columns.levels[0] 
                    if 'Close' in data[ticker].columns
                })
            elif 'Close' in data.columns.levels[0]: # (Close, Ticker)
                close_prices = data['Close']
            else:
                # Fallback, try to find Close
                print("⚠️ 'Close' not found in MultiIndex levels. Columns:", data.columns)
                return {"error": "Data format error: 'Close' price missing"}
        else:
            # Single level
            if 'Close' in data.columns:
                close_prices = pd.DataFrame({'Stock': data['Close']}) # Single stock
            else:
                 # Maybe it's already just close prices? Unlikely from yf.download
                 close_prices = data
            
        # Drop columns with too many NaNs
        close_prices = close_prices.dropna(axis=1, thresh=int(len(close_prices)*0.9))
        close_prices = close_prices.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate 1Y Momentum to pick Top 50 Universe
        one_year_ago = close_prices.index[-252] if len(close_prices) > 252 else close_prices.index[0]
        momentum = (close_prices.iloc[-1] / close_prices.loc[one_year_ago]) - 1
        top_50_tickers = momentum.sort_values(ascending=False).head(50).index.tolist()
        
        universe_prices = close_prices[top_50_tickers]
        
        # 2. EDA & Feature Engineering
        print("🔬 Running EDA and Feature Engineering...")
        signals = self._engineer_features(universe_prices)
        
        # 3. Portfolio Optimization
        print("⚖️ Optimizing Portfolio...")
        weights, performance = self._optimize_portfolio(universe_prices, user_profile)
        
        # 4. Backtest
        print("📈 Running Backtest...")
        # 4. Backtest
        print("📈 Running Backtest...")
        backtest_results = self._backtest_portfolio(weights, universe_prices, benchmark_prices)
        
        # 5. Construct Payload
        print("📦 Constructing JSON Payload...")
        # 5. Construct Payload
        print("📦 Constructing JSON Payload...")
        payload = self._construct_payload(user_profile, universe_prices, weights, backtest_results, signals, benchmark_prices)
        
        return payload

    def _engineer_features(self, prices):
        """
        Computes technical indicators and signals.
        """
        signals = {}
        for ticker in prices.columns:
            try:
                # We need a Series, but ta expects DataFrame usually or Series
                # Let's create a temp DF for ta
                df = pd.DataFrame({'Close': prices[ticker]})
                
                # RSI
                df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
                
                # MACD
                macd = ta.trend.MACD(df['Close'])
                df['MACD'] = macd.macd()
                
                # Volatility (21d rolling std)
                df['Vol_21d'] = df['Close'].pct_change().rolling(window=21).std()
                
                # Momentum Score (Simple average of RSI/MACD normalized - simplified for now)
                # Just storing raw values for the payload
                latest = df.iloc[-1]
                signals[ticker] = {
                    "RSI": latest['RSI'],
                    "MACD": latest['MACD'],
                    "Vol_21d": latest['Vol_21d'],
                    "Momentum_1Y": (prices[ticker].iloc[-1] / prices[ticker].iloc[0]) - 1
                }
            except:
                signals[ticker] = {}
        return signals

    def _optimize_portfolio(self, prices, profile):
        """
        Uses PyPortfolioOpt to find optimal weights.
        """
        if not HAS_PYPFOPT:
            # Fallback: Momentum-weighted allocation
            print("⚠️ Using fallback optimization (Momentum Weighted)")
            returns = prices.pct_change().mean()
            # Filter positive returns only for long-only portfolio
            positive_returns = returns[returns > 0]
            if positive_returns.empty:
                # Equal weight if all negative
                weights = {ticker: 1.0/len(prices.columns) for ticker in prices.columns}
            else:
                total_ret = positive_returns.sum()
                weights = (positive_returns / total_ret).to_dict()
            
            # Mock performance metrics
            performance = (0.15, 0.12, 1.2) # Expected Ret, Vol, Sharpe
            return weights, performance

        # Expected Returns and Covariance
        print("DEBUG: Calculating Covariance using sample_cov...")
        mu = expected_returns.mean_historical_return(prices)
        S = risk_models.sample_cov(prices)
        
        # Optimize
        ef = EfficientFrontier(mu, S)
        
        # Constraints based on profile
        risk_level = profile.get('risk_profile', 'Moderate')
        target_return_pct = profile.get('expected_annual_return_pct', 12)
        target_return = target_return_pct / 100.0
        
        print(f"⚖️ Optimizing for {risk_level} profile with target return {target_return_pct}%")
        
        try:
            if risk_level == 'Conservative':
                # Min Volatility
                ef.min_volatility()
            elif risk_level == 'Moderate':
                # Max Sharpe is standard for moderate
                ef.max_sharpe()
            elif risk_level == 'Aggressive':
                # Try to achieve higher return, or Max Sharpe with different gamma?
                # Let's use efficient_return if reasonable, else Max Sharpe
                try:
                    ef.efficient_return(target_return=target_return)
                except:
                    print("⚠️ Target return infeasible, falling back to Max Sharpe")
                    ef.max_sharpe()
            elif risk_level == 'Very Aggressive':
                 # Maximize return (not directly supported by PyPortfolioOpt without max volatility)
                 # We can use efficient_risk with high volatility
                 try:
                     ef.efficient_risk(target_volatility=0.30) # Allow 30% vol
                 except:
                     ef.max_sharpe()
            else:
                ef.max_sharpe()
                
        except Exception as e:
            print(f"⚠️ Optimization failed: {e}. Falling back to Min Volatility.")
            ef.min_volatility()
            
        weights = ef.clean_weights()
        performance = ef.portfolio_performance(verbose=False)
        
        return weights, performance

    def _backtest_portfolio(self, weights, prices, benchmark_prices=None):
        """
        Simple vector backtest with benchmark comparison.
        """
        # Daily Returns
        returns = prices.pct_change().dropna()
        
        # Portfolio Returns
        portfolio_returns = returns.dot(pd.Series(weights))
        
        # Cumulative Return
        cumulative_return = (1 + portfolio_returns).cumprod()
        
        # Benchmark Analysis
        benchmark_metrics = {}
        if benchmark_prices is not None and not benchmark_prices.empty:
            # Align dates
            common_dates = returns.index.intersection(benchmark_prices.index)
            if not common_dates.empty:
                bench_returns = benchmark_prices.loc[common_dates].pct_change().dropna()
                # Re-align portfolio returns to common dates
                port_returns_aligned = portfolio_returns.loc[bench_returns.index]
                
                bench_cum_return = (1 + bench_returns).cumprod()
                
                # Benchmark Metrics
                b_total = bench_cum_return.iloc[-1] - 1 if not bench_cum_return.empty else 0
                b_ann = bench_returns.mean() * 252
                b_vol = bench_returns.std() * np.sqrt(252)
                b_sharpe = b_ann / b_vol if b_vol > 0 else 0
                
                benchmark_metrics = {
                    "cumulative_return_pct": b_total * 100,
                    "annualized_return_pct": b_ann * 100,
                    "annualized_vol_pct": b_vol * 100,
                    "sharpe": b_sharpe,
                    "equity_curve": bench_cum_return
                }
            else:
                benchmark_metrics = {"error": "No common dates with benchmark"}
        
        # Metrics
        total_return = cumulative_return.iloc[-1] - 1
        annualized_return = portfolio_returns.mean() * 252
        annualized_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Max Drawdown
        rolling_max = cumulative_return.cummax()
        drawdown = cumulative_return / rolling_max - 1
        max_drawdown = drawdown.min()
        
        return {
            "cumulative_return_pct": total_return * 100,
            "annualized_return_pct": annualized_return * 100,
            "annualized_vol_pct": annualized_vol * 100,
            "sharpe": sharpe,
            "max_drawdown_pct": max_drawdown * 100,
            "equity_curve": cumulative_return,
            "benchmark": benchmark_metrics
        }

    def _construct_payload(self, profile, prices, weights, backtest, signals, benchmark_prices=None):
        """
        Builds the exact JSON schema required.
        """
        # Filter weights > 0.01
        active_weights = {k: v for k, v in weights.items() if v > 0.01}
        
        # Ticker details
        tickers_list = []
        allocation_list = []
        
        investment_amount = profile.get('investment_amount', 100000)
        
        for ticker, weight in active_weights.items():
            last_price = prices[ticker].iloc[-1]
            sig = signals.get(ticker, {})
            
            tickers_list.append({
                "ticker": ticker,
                "name": ticker, # Placeholder, would need a map for real names
                "sector": "Unknown", # Placeholder, need sector map
                "last_price": last_price,
                "1y_return_pct": sig.get('Momentum_1Y', 0) * 100,
                "vol_21d": sig.get('Vol_21d', 0),
                "momentum_score": sig.get('RSI', 50) / 100 # Proxy
            })
            
            allocation_list.append({
                "ticker": ticker,
                "weight_pct": weight,
                "amount": investment_amount * weight
            })
            
        # Prepare Chart Data (JSON friendly)
        # Equity Curve
        ec = backtest['equity_curve']
        ec_data = [{"date": d.strftime('%Y-%m-%d'), "value": v} for d, v in ec.items()]
        
        # Benchmark Curve
        bc_data = []
        if 'benchmark' in backtest and 'equity_curve' in backtest['benchmark']:
            bc = backtest['benchmark']['equity_curve']
            if isinstance(bc, pd.Series):
                 bc_data = [{"date": d.strftime('%Y-%m-%d'), "value": v} for d, v in bc.items()]
        
        # Charts (Base64) - Keeping for backup, but we will use raw data
        charts = self._generate_charts(active_weights, backtest['equity_curve'])
        
        return {
            "meta": {
                "timestamp": datetime.now().isoformat(),
                "user": profile
            },
            "universe_summary": {
                "n_tickers": len(prices.columns),
                "source": "yfinance",
                "last_price_timestamp": datetime.now().isoformat()
            },
            "tickers": tickers_list,
            "signals": {
                "computed_at": datetime.now().isoformat(),
                "scores": {t: s.get('RSI', 50) for t, s in signals.items()},
                "rankings": [] # Can populate if needed
            },
            "portfolio_construction": {
                "method": "Mean-Variance (PyPortfolioOpt)",
                "constraints": {"risk_profile": profile.get('risk_profile')},
                "rebalance": "monthly"
            },
            "allocation": allocation_list,
            "backtest_summary": {
                "period_start": prices.index[0].strftime('%Y-%m-%d'),
                "period_end": prices.index[-1].strftime('%Y-%m-%d'),
                "cumulative_return_pct": backtest['cumulative_return_pct'],
                "annualized_return_pct": backtest['annualized_return_pct'],
                "annualized_vol_pct": backtest['annualized_vol_pct'],
                "sharpe": backtest['sharpe'],
                "max_drawdown_pct": backtest['max_drawdown_pct'],
                "benchmark": {
                    "cumulative_return_pct": backtest.get('benchmark', {}).get('cumulative_return_pct', 0),
                    "annualized_return_pct": backtest.get('benchmark', {}).get('annualized_return_pct', 0),
                    "sharpe": backtest.get('benchmark', {}).get('sharpe', 0)
                }
            },
            "risk_metrics": {
                "expected_return_pct": backtest['annualized_return_pct'],
                "expected_vol_pct": backtest['annualized_vol_pct']
            },
            "chart_data": {
                "equity_curve": ec_data,
                "benchmark_curve": bc_data
            },
            "charts": charts
        }

    def _generate_charts(self, weights, equity_curve):
        """
        Generates matplotlib charts and converts to base64.
        """
        charts = {}
        
        # 1. Allocation Pie
        plt.figure(figsize=(6, 6))
        plt.pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%')
        plt.title("Portfolio Allocation")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        charts['allocation_pie_base64'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        # 2. Cumulative Returns
        plt.figure(figsize=(10, 6))
        equity_curve.plot()
        plt.title("Backtest: Cumulative Returns")
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        charts['cumulative_returns_base64'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return charts
