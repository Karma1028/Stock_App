import requests
import feedparser
import urllib.parse
from datetime import datetime

base_url = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
target_year = 2020
start_date = "2020-01-01"
end_date = "2020-02-01"

query = f"Reliance stock news after:{start_date} before:{end_date}"
encoded_query = urllib.parse.quote(query)
url = base_url.format(query=encoded_query)

print(f"Querying: {url}")
feed = feedparser.parse(url)
print(f"Found {len(feed.entries)} entries")

for entry in feed.entries[:5]:
    print(f"Title: {entry.title}")
    print(f"Published: {entry.published}")
    print("-" * 20)
