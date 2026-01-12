import pandas as pd
import requests
from bs4 import BeautifulSoup
import io

def inspect():
    url = 'https://en.wikipedia.org/wiki/NIFTY_500'
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(resp.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        if table:
            df = pd.read_html(io.StringIO(str(table)))[0]
            print(f"Columns: {df.columns.tolist()}")
            print(f"First row: {df.iloc[0].to_dict()}")
        else:
            print("No table found")
    except Exception as e:
        print(e)
inspect()
