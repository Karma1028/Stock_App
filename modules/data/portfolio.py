import json
import pandas as pd
from pathlib import Path
from config import Config
from modules.data.manager import StockDataManager

class PortfolioManager:
    def __init__(self):
        self.portfolios_dir = Config.PORTFOLIOS_DIR
        self.data_manager = StockDataManager()

    def load_or_create_portfolio(self, name):
        file_path = self.portfolios_dir / f"{name}.json"
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    return json.load(f)
            except:
                return {"holdings": {}}
        return {"holdings": {}}

    def save_portfolio(self, portfolio, name):
        file_path = self.portfolios_dir / f"{name}.json"
        with open(file_path, "w") as f:
            json.dump(portfolio, f, indent=4)

    def get_current_values(self, portfolio):
        """
        Calculates current value of portfolio holdings.
        """
        holdings = portfolio.get("holdings", {})
        results = []
        total_value = 0.0

        for symbol, qty in holdings.items():
            live_data = self.data_manager.get_live_data(symbol)
            current_price = live_data.get("current_price", 0.0)
            if current_price is None:
                current_price = 0.0
            
            value = qty * current_price
            total_value += value
            
            results.append({
                "Symbol": symbol,
                "Quantity": qty,
                "Current Price": current_price,
                "Value": value
            })
        
        return pd.DataFrame(results), total_value

    def add_stock(self, portfolio, symbol, qty):
        holdings = portfolio.get("holdings", {})
        if symbol in holdings:
            holdings[symbol] += qty
        else:
            holdings[symbol] = qty
        portfolio["holdings"] = holdings
        return portfolio

    def remove_stock(self, portfolio, symbol, qty):
        holdings = portfolio.get("holdings", {})
        if symbol in holdings:
            holdings[symbol] -= qty
            if holdings[symbol] <= 0:
                del holdings[symbol]
        portfolio["holdings"] = holdings
        return portfolio
