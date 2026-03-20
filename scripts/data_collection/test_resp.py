import requests

url = "https://news.google.com/rss/articles/CBMiigFBVV95cUxQMEhEMXhkckZhOHJNN3o2SVZxbUFrejVPN2hQSFRjZy1vUEpqTFlBZ09HUnNTclB1ZTdhblhTc3I0Vm85bWUyS3VrLTdzdGlsa1pNNkhlRTQzZmtaMnNaMmhCWVNZQ3l5eXhQWDFMZndYdHFmS2p4Y3VpXzJuTndLdzlKemJGNkZtMmc?oc=5"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

res = requests.get(url, headers=headers, allow_redirects=True)
with open("test_resp.html", "w", encoding="utf-8") as f:
    f.write(res.text)
print("Saved to test_resp.html")
