import asyncio
import urllib.request
import json
from bs4 import BeautifulSoup

def try_extractors(url):
    print(f"Target URL: {url}")
    
    # 1. Try newspaper3k
    try:
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()
        print("\n--- NEWSPAPER3K ---")
        print(article.text[:1000])
    except ImportError:
        print("\nnewspaper3k not installed")
    except Exception as e:
        print(f"\nnewspaper3k failed: {e}")

    # 2. Try trafilatura
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        print("\n--- TRAFILATURA ---")
        print(text[:1000] if text else "Trafilatura returned None")
    except ImportError:
        print("\ntrafilatura not installed")
    except Exception as e:
        print(f"\ntrafilatura failed: {e}")

if __name__ == "__main__":
    final_url = "https://www.business-standard.com/content/press-releases-ani/360-one-asset-achieves-iso-27001-2022-certification-strengthening-its-leadership-in-information-security-125110800015_1.html"
    try_extractors(final_url)
