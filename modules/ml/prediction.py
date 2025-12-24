import pandas as pd
from prophet import Prophet
import logging

class StockPredictor:
    def __init__(self):
        pass

    def train_and_predict(self, df, periods=30):
        """
        Trains a Prophet model and predicts future prices.
        """
        if df.empty:
            return None, "Input DataFrame is empty."

        # Prepare data for Prophet
        data = df.reset_index()
        
        # Flatten MultiIndex/Tuple columns
        new_cols = []
        for col in data.columns:
            if isinstance(col, tuple):
                # Join tuple elements, filtering empty strings
                new_cols.append("_".join(str(c) for c in col if c).strip())
            else:
                new_cols.append(str(col))
        data.columns = new_cols

        # Identify 'ds' logic
        # Check explicit names first
        if 'Date' in data.columns:
            data = data.rename(columns={'Date': 'ds'})
        elif 'ds' not in data.columns:
             # Search for datetime column
             found_ds = False
             for col in data.columns:
                 if pd.api.types.is_datetime64_any_dtype(data[col]):
                     data = data.rename(columns={col: 'ds'})
                     found_ds = True
                     break
             
             if not found_ds:
                # Use index if datetime (though we reset index, so it should be in columns)
                if isinstance(df.index, pd.DatetimeIndex):
                     data['ds'] = df.index
                else:
                    cols = list(data.columns)
                    logging.error(f"No date column found. Columns: {cols}")
                    return None, f"No date column found. Available: {cols}"
        
        # Identify 'y' logic (Close)
        # Search for columns containing 'Close'
        close_col = None
        if 'Close' in data.columns:
            close_col = 'Close'
        elif 'Adj_Close' in data.columns:
            close_col = 'Adj_Close'
        else:
            # Search for anything ending in _Close
            for col in data.columns:
                if col.endswith('_Close') or col.endswith(' Close'):
                    close_col = col
                    break
        
        if close_col:
            data = data.rename(columns={close_col: 'y'})
        else:
            cols = list(data.columns)
            logging.error(f"No Close column found. Columns: {cols}")
            return None, f"No Close column found. Available: {cols}"

        data = data[['ds', 'y']]
        
        # Ensure 'ds' is timezone-naive for Prophet
        if data['ds'].dt.tz is not None:
            data['ds'] = data['ds'].dt.tz_localize(None)

        try:
            model = Prophet(daily_seasonality=True)
            model.fit(data)
            
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            return forecast, model
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            return None, str(e)
