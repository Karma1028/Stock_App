import pandas as pd
import numpy as np
import ta
from datetime import datetime

class FeatureEngineer:
    def __init__(self):
        pass

    def compute_all_features(self, df, sentiment_df=None):
        """
        Computes all technical, price, and sentiment features.
        Expects df to have MultiIndex (Ticker, Attributes) or Single Index if one ticker.
        """
        if df.empty:
            return df

        # Handle MultiIndex (Bulk Data)
        if isinstance(df.columns, pd.MultiIndex):
            # Process each ticker separately
            processed_dfs = []
            tickers = df.columns.levels[0]
            
            for ticker in tickers:
                try:
                    # Extract single ticker df
                    # Handle yfinance structure: (Ticker, PriceType)
                    single_df = df[ticker].copy()
                    
                    # Ensure we have OHLCV
                    if 'Close' not in single_df.columns:
                        continue
                        
                    # Compute features
                    single_df = self._compute_single_ticker_features(single_df)
                    
                    # Merge Sentiment if available
                    if sentiment_df is not None and not sentiment_df.empty:
                        # Filter for this ticker if sentiment_df has ticker column, 
                        # but for now assuming sentiment_df is passed per ticker or we merge on date
                        # If sentiment_df is global (all tickers), we need a ticker column
                        pass 
                    
                    # Add Ticker column for later use
                    single_df['Ticker'] = ticker
                    processed_dfs.append(single_df)
                except Exception as e:
                    print(f"Error processing features for {ticker}: {e}")
            
            if not processed_dfs:
                return pd.DataFrame()
                
            return pd.concat(processed_dfs)
        else:
            # Single Ticker
            return self._compute_single_ticker_features(df)

    def _compute_single_ticker_features(self, df):
        """
        Internal method to compute features for a single dataframe.
        """
        df = df.copy()
        
        # 1. Price Features
        df['Returns_1d'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_21d'] = df['Close'].pct_change(21)
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Rolling Stats
        df['Vol_21d'] = df['Returns_1d'].rolling(window=21).std()
        df['MA_21d'] = df['Close'].rolling(window=21).mean()
        
        # 2. Technical Indicators (using ta library)
        # Trend
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(df['Close'], window=200).sma_indicator()
        
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Diff'] = macd.macd_diff()
        
        # Momentum
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        
        # Volatility
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        
        # Volume
        if 'Volume' in df.columns:
            df['Vol_Change'] = df['Volume'].pct_change()
            df['Vol_MA_20'] = df['Volume'].rolling(20).mean()
            df['Vol_Z'] = (df['Volume'] - df['Vol_MA_20']) / (df['Volume'].rolling(20).std() + 1e-9)
            
        # 3. Calendar Features
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        
        # 4. Targets (Next 5d Return)
        # We want to predict return 5 days into the future
        df['Target_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        
        return df

    def merge_sentiment(self, feature_df, sentiment_df):
        """
        Merges daily sentiment scores into the feature dataframe.
        """
        if sentiment_df.empty:
            feature_df['sentiment_score'] = 0
            feature_df['sentiment_volatility'] = 0
            return feature_df
            
        # Ensure dates match
        # feature_df index is Datetime, sentiment_df has 'date_only'
        
        feature_df['date_only'] = feature_df.index.date
        
        merged = pd.merge(feature_df, sentiment_df[['date_only', 'sentiment_score', 'sentiment_volatility']], 
                          on='date_only', how='left')
        
        merged.index = feature_df.index # Restore index
        merged['sentiment_score'] = merged['sentiment_score'].fillna(0)
        merged['sentiment_volatility'] = merged['sentiment_volatility'].fillna(0)
        
        # Drop temp column
        merged = merged.drop(columns=['date_only'])
        
        return merged
