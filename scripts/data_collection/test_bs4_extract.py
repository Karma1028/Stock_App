import asyncio
from playwright.async_api import async_playwright
import re

async def playwright_bs_extractor():
    url = "https://www.business-standard.com/content/press-releases-ani/360-one-asset-achieves-iso-27001-2022-certification-strengthening-its-leadership-in-information-security-125110800015_1.html"
    print("Testing Playwright + Custom BeautifulSoup Extraction...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        
        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        html = await page.content()
        await browser.close()
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # 1. Nuke everything we don't want
        for elem in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            elem.decompose()
            
        # 2. Try to find the main article body. Most news sites use <article> or a div with 'content'
        main_content = soup.find('article')
        if not main_content:
            # Fallback to looking for content divs
            main_content = soup.find('div', class_=re.compile(r'content|article|story|post', re.I))
            
        if not main_content:
             main_content = soup # last resort, scan whole page
             
        # 3. Extract purely <p> tags from the main content area
        paragraphs = main_content.find_all('p')
        valid_text = []
        
        for p in paragraphs:
            text = p.get_text(separator=' ', strip=True)
            # A valid news paragraph is usually > 50 chars, has multiple words
            if len(text) > 50 and text.count(' ') > 5:
                # Filter out standard boilerplate
                if not any(b in text.lower() for b in ["copyright", "all rights reserved", "click here", "read more", "subscribe"]):
                    valid_text.append(text)
                    
        print("\n--- EXTRACTED ARTICLE TEXT ---")
        print("\n\n".join(valid_text))

if __name__ == "__main__":
    asyncio.run(playwright_bs_extractor())
