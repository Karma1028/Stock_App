import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import re

async def resolve_redirect(browser, url):
    context = await browser.new_context()
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        for _ in range(12):
            if "google.com" not in page.url:
                await asyncio.sleep(1.5)
                final = page.url
                await context.close()
                return final
            await asyncio.sleep(1.0)
        final = page.url
        await context.close()
        return final
    except Exception as e:
        await context.close()
        return url

def clean_extracted_text(html_content):
    """Fallback parser to extract pure text from HTML if markdown is noisy."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unwanted tags
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        element.decompose()
        
    paragraphs = soup.find_all(['p', 'article', 'div'])
    cleaned_text = []
    
    # Very strict filtering: only grab blocks of text that look like actual sentences > 60 chars
    for p in paragraphs:
        text = p.get_text(strip=True)
        # Filter out short UI elements, copyright lines, and menus
        if len(text) > 80 and text.count(' ') > 10 and "Copyright" not in text and "All rights reserved" not in text:
            cleaned_text.append(text)
            
    # Remove duplicates preserving order
    seen = set()
    unique_text = [x for x in cleaned_text if not (x in seen or seen.add(x))]
    
    return "\n\n".join(unique_text)

async def test_first_20():
    file_path = "data/news_history/360ONE.NS_news.csv"
    df = pd.read_csv(file_path)
    
    # Get first 20 links
    links = df['Link'].dropna().head(20).tolist()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        async with AsyncWebCrawler(verbose=False) as crawler:
            for i, link in enumerate(links):
                print(f"\n[{i+1}/20] Processing Link: {link[:60]}...")
                final_url = await resolve_redirect(browser, link)
                print(f"Final URL: {final_url}")
                
                if "google.com" in final_url:
                    print("-> SKIPPED: Stuck on Google Redirect.")
                    continue
                    
                result = await crawler.arun(
                    url=final_url,
                    magic=True,
                    simulate_user=True,
                    bypass_cache=True,
                    delay_before_return_html=3.0
                )
                
                if result.success:
                    print("-> CRAWL4AI RAW MARKDOWN PREVIEW (Top 300 chars):")
                    raw_md = result.markdown[:300].replace('\n', ' ')
                    print(f"   {raw_md}")
                    
                    print("-> CUSTOM BEAUTIFULSOUP CLEAN EXTRACTION PREVIEW:")
                    # Crawl4AI returns the HTML in result.html
                    clean_text = clean_extracted_text(result.html)
                    print(f"   {clean_text[:500]}..." if clean_text else "   [No valid text blocks found]")
                else:
                    print(f"-> CRAWL4AI FAILED: {result.error_message}")
                    
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_first_20())
