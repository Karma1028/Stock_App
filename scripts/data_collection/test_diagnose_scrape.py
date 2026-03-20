import asyncio
from playwright.async_api import async_playwright
from crawl4ai import AsyncWebCrawler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TestScraper')

async def resolve_redirect(browser, url):
    context = await browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    )
    page = await context.new_page()
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        for _ in range(12):
            curr = page.url
            if "google.com" not in curr:
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

async def main():
    url = "https://news.google.com/rss/articles/CBMikwJBVV95cUxQMDJOalpWUC1MZEQ5ZENfY1BCTi1mMnRQRHJvWnBUZWtnYTdvbm5PTmRLUWRJOHdyMXJsTFVsNDRHNFI1TGs5ZFptU3NqQUVaLTZEejVPQzdpbHYtTlpJd0g3M214WlY1dDhmMGw1d3o4TGxtN3d6RzBiR3JVdkxWT0dvaEU2emRvOThJZWtIMkZCMm1jSFZ0aHdMVW1VUHZoNjF2a3JkeUx3OUFPR2tmbmRTOVpTNmhubFRkZ0pfcGpqT1I5cDRDdkRKX000Z19Zd0ZTT0NDWHlWTTNVNFNKVjFrLTFNSEZVTFdHTGg5a3pycm04TmdLNF9wQzNYM0dVcEJ4U3NweWlqVnpudUdLbTlYTdIBmAJBVV95cUxNdWZKMldQS0JzZllPNzZIQm9mVnJqTEhDbXhhRjJ2SWNSdkxzMGFTVVJiN0Nmc0Nuc3E1Q0pqNWdqbDh6cU56aHlqbTZreUhjdGsycllmeFRVOFRvZm9HYm1ibVRxdHoyQkFrYmVkVVdacWJpSTRYWkdMdkdsbmt1akFWUlEyYnE5dkZ6dVA5U2gtcWVoVVAzYkU0VlBjWmdqblhvd3pNeEN5QTdDY1F4LW1fQUoxcWo1ZDJBcmF5NGJocTNpOVJfT20tckJGNUpvYkZVUDdiZnhHd0J6OGtqaUJhLUFVdUw1b3VkeHhZRzNibXVERGpjSDM4MUFQQjl3a29TdXF3bmwxSkRtTExxaTdKQmNvMTFB?oc=5"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        final_url = await resolve_redirect(browser, url)
        print(f"Final URL: {final_url}")
        
        async with AsyncWebCrawler(verbose=True) as crawler:
            result = await crawler.arun(
                url=final_url,
                magic=True,
                simulate_user=True,
                bypass_cache=True,
                delay_before_return_html=5.0
            )
            
            if result.success:
                print("\n--- EXTRACTED FIT MARKDOWN ---")
                if hasattr(result, 'markdown_v2') and result.markdown_v2:
                    print(result.markdown_v2.fit_markdown)
                else:
                    # Older versions of crawl4ai or alternative prop
                    print(getattr(result, 'fit_markdown', getattr(result, 'extracted_content', result.markdown)))
            else:
                print("Crawl4AI failed.")
                
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
