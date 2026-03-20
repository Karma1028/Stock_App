import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re

async def main():
    url = "https://news.google.com/rss/articles/CBMikwJBVV95cUxQMDJOalpWUC1MZEQ5ZENfY1BCTi1mMnRQRHJvWnBUZWtnYTdvbm5PTmRLUWRJOHdyMXJsTFVsNDRHNFI1TGs5ZFptU3NqQUVaLTZEejVPQzdpbHYtTlpJd0g3M214WlY1dDhmMGw1d3o4TGxtN3d6RzBiR3JVdkxWT0dvaEU2emRvOThJZWtIMkZCMm1jSFZ0aHdMVW1VUHZoNjF2a3JkeUx3OUFPR2tmbmRTOVpTNmhubFRkZ0pfcGpqT1I5cDRDdkRKX000Z19Zd0ZTT0NDWHlWTTNVNFNKVjFrLTFNSEZVTFdHTGg5a3pycm04TmdLNF9wQzNYM0dVcEJ4U3NweWlqVnpudUdLbTlYTdIBmAJBVV95cUxNdWZKMldQS0JzZllPNzZIQm9mVnJqTEhDbXhhRjJ2SWNSdkxzMGFTVVJiN0Nmc0Nuc3E1Q0pqNWdqbDh6cU56aHlqbTZreUhjdGsycllmeFRVOFRvZm9HYm1ibVRxdHoyQkFrYmVkVVdacWJpSTRYWkdMdkdsbmt1akFWUlEyYnE5dkZ6dVA5U2gtcWVoVVAzYkU0VlBjWmdqblhvd3pNeEN5QTdDY1F4LW1fQUoxcWo1ZDJBcmF5NGJocTNpOVJfT20tckJGNUpvYkZVUDdiZnhHd0J6OGtqaUJhLUFVdUw1b3VkeHhZRzNibXVERGpjSDM4MUFQQjl3a29TdXF3bmwxSkRtTExxaTdKQmNvMTFB?oc=5"
    print("Launching Playwright...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        print("Navigating to URL...")
        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        
        # Wait for redirect
        for _ in range(12):
            if "google.com" not in page.url:
                await asyncio.sleep(2.0)
                break
            await asyncio.sleep(1.0)
            
        print(f"Final URL: {page.url}")
        
        # Extract HTML and clean text
        html = await page.content()
        soup = BeautifulSoup(html, 'html.parser')
        
        # Simple extraction: all <p> tags with enough text
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 50]
        
        print("\n--- EXTRACTED <p> TAGS ---")
        for i, p_text in enumerate(paragraphs[:10]):
            print(f"[{i}]: {p_text}")
            
        # Try generic body parsing to see what it is returning
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
