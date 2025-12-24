import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from config import Config

CACHE_FILE = Config.DATA_DIR / "wiki_cache.json"

def load_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)

def get_company_summary(company_name):
    """
    Scrapes Wikipedia for company summary.
    """
    cache = load_cache()
    if company_name in cache:
        timestamp = cache[company_name].get("timestamp")
        if timestamp:
            saved_time = datetime.fromisoformat(timestamp)
            if datetime.now() - saved_time < timedelta(hours=24):
                return cache[company_name]["summary"]

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        # Search Wikipedia
        search_url = f"https://en.wikipedia.org/wiki/{company_name.replace(' ', '_')}"
        driver.get(search_url)
        
        # Simple extraction logic (first paragraph)
        paragraphs = driver.find_elements("css selector", "p")
        summary = "No summary found."
        for p in paragraphs:
            text = p.text
            if text and len(text) > 50:
                summary = text
                break
        
        driver.quit()
        
        # Update cache
        cache[company_name] = {
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        save_cache(cache)
        
        return summary

    except Exception as e:
        print(f"Error scraping Wikipedia: {e}")
        
        # Fallback to yfinance
        try:
            import yfinance as yf
            ticker = yf.Ticker(company_name) # This might fail if company_name is not a symbol
            # But we often pass symbol or name. If name, yfinance might not work directly.
            # Let's try to use the symbol if possible, but here we only have company_name.
            # Actually, in stock_analysis.py we pass 'selected_company' which IS the symbol.
            # So this fallback should work!
            info = ticker.info
            summary = info.get("longBusinessSummary", "No summary available.")
            return summary
        except:
            return "Summary unavailable."
