
import pandas as pd
import numpy as np
import ta
from modules.data.manager import StockDataManager

class QuantEngine:
    """
    Quant Strategy Engine for Backtesting.
    Implements vectorized backtesting for Momentum and Mean Reversion strategies.
    """
    
    def __init__(self):
        self.dm = StockDataManager()
        
    def run_backtest(self, params):
        """
        Run a backtest based on the provided parameters.
        params: {
            "type": "Momentum" | "Mean Reversion",
            "lookback": int,
            "p1": float (RSI Threshold or BB Std Dev),
            "p2": int (SMA Window),
            "capital": float
        }
        """
        try:
            strategy_type = params.get('type')
            capital = params.get('capital', 100000)
            
            # Fetch Data for Nifty 50 (or a subset for speed)
            # For this MVP, let's pick top liquid stocks to show aggregate performance or just one?
            # The UI implies a "Universe" backtest, but that's heavy.
            # Let's run it on a single representative stock for now (e.g., NIFTY 50 index or a top stock)
            # OR iterate over a basket. The UI showed "Recent Trades" with multiple symbols.
            # To make it fast and impressive, let's run on a Basket of Top 5 stocks.
            tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
            
            all_trades = []
            portfolio_equity = pd.DataFrame()
            
            for ticker in tickers:
                df = self.dm.get_historical_data(ticker, period="2y")
                if df.empty: continue
                
                # Apply Strategy
                signals = self._apply_strategy(df, strategy_type, params)
                
                # Calculate Returns
                daily_rets, trade_log, equity_curve = self._simulate_trading(df, signals, capital / len(tickers))
                
                # Append formatted trades
                for t in trade_log:
                    all_trades.append({
                        "Date": t['date'],
                        "Symbol": ticker,
                        "Action": t['action'],
                        "Price": t['price'],
                        "PnL": t['pnl']
                    })
                
                portfolio_equity[ticker] = equity_curve
                
            # Aggregate Portfolio
            if portfolio_equity.empty:
                return {}, pd.Series(), pd.DataFrame()
                
            total_equity = portfolio_equity.sum(axis=1)
            
            # Calculate Stats
            stats = self._calculate_stats(total_equity)
            
            # Calculate Trade Win Rate
            closed_trades = [t for t in all_trades if t['Action'] == 'SELL']
            if closed_trades:
                winning_trades = [t for t in closed_trades if t['PnL'] > 0]
                win_rate = (len(winning_trades) / len(closed_trades)) * 100
                stats['Win Rate'] = f"{win_rate:.0f}%"
            else:
                stats['Win Rate'] = "0%"
            
            # Format Trades
            trades_df = pd.DataFrame(all_trades).sort_values("Date", ascending=False).head(20)
            
            return stats, total_equity, trades_df
            
        except Exception as e:
            print(f"Backtest Error: {e}")
            return {}, pd.Series(), pd.DataFrame()

    def _apply_strategy(self, df, strategy_type, params):
        """
        Generates Buy (1) / Sell (-1) / Hold (0) signals.
        """
        signals = pd.Series(0, index=df.index)
        close = df['Close']
        
        if strategy_type == "Momentum":
            # RSI + SMA Trend
            rsi_thresh = params.get('p1', 50) # Buy above this
            ma_window = int(params.get('p2', 20))
            
            rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
            sma = close.rolling(window=ma_window).mean()
            
            # Buy Condition: RSI > Threshold AND Price > SMA
            buy_cond = (rsi > rsi_thresh) & (close > sma)
            
            # Sell Condition: RSI < Threshold OR Price < SMA
            sell_cond = (rsi < rsi_thresh) | (close < sma)
            
        elif strategy_type == "Mean Reversion":
            # Bollinger Bands
            std_dev = params.get('p1', 2.0)
            ma_window = int(params.get('p2', 20))
            
            indicator_bb = ta.volatility.BollingerBands(close=close, window=ma_window, window_dev=std_dev)
            bb_high = indicator_bb.bollinger_hband()
            bb_low = indicator_bb.bollinger_lband()
            
            # Buy Condition: Price < Lower Band (Oversold)
            buy_cond = close < bb_low
            
            # Sell Condition: Price > Upper Band (Overbought) or Price > MA (Reverted to mean)
            sell_cond = (close > bb_high) | (close > indicator_bb.bollinger_mavg())
            
        else:
            return signals

        signals[buy_cond] = 1
        signals[sell_cond] = -1
        return signals

    def _simulate_trading(self, df, signals, initial_capital):
        """
        Simulates execution with transaction costs.
        """
        position = 0
        cash = initial_capital
        equity = []
        trades = []
        entry_price = 0
        
        cost = 0.001 # 0.1% per trade
        
        for i, date in enumerate(df.index):
            price = df['Close'].iloc[i]
            signal = signals.iloc[i]
            
            # Execute
            if signal == 1 and position == 0: # Buy
                shares = (cash * (1 - cost)) / price
                position = shares
                cash = 0
                entry_price = price
                trades.append({'date': date, 'action': 'BUY', 'price': round(price, 2), 'pnl': 0})
                
            elif signal == -1 and position > 0: # Sell
                revenue = position * price * (1 - cost)
                pnl = revenue - (position * entry_price)
                cash = revenue
                position = 0
                trades.append({'date': date, 'action': 'SELL', 'price': round(price, 2), 'pnl': round(pnl, 2)})
            
            # Mark to Market
            current_value = cash + (position * price)
            equity.append(current_value)
            
        return pd.Series(equity, index=df.index).pct_change(), trades, pd.Series(equity, index=df.index)

    def _calculate_stats(self, equity_curve):
        """
        Computes CAG, Sharpe, Drawdown.
        """
        if equity_curve.empty: return {}
        
        start_val = equity_curve.iloc[0]
        end_val = equity_curve.iloc[-1]
        
        # Total Return
        total_ret = ((end_val - start_val) / start_val) * 100
        
        # Daily Returns
        daily_rets = equity_curve.pct_change().dropna()
        
        # Sharpe (assuming 5% risk free rate)
        rf_daily = 0.05 / 252
        excess_ret = daily_rets - rf_daily
        sharpe = np.sqrt(252) * (excess_ret.mean() / excess_ret.std()) if excess_ret.std() != 0 else 0
        
        # Max Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100
        
        # Win Rate (Trade-based)
        # We need to pass trades to this function or calculate it before
        return {
            "Total Return": f"{total_ret:.1f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Max Drawdown": f"{max_dd:.1f}%",
            "Win Rate": "N/A" # Calculated in run_backtest now
        }
