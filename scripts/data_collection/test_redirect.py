import asyncio
from playwright.async_api import async_playwright

url = "https://news.google.com/rss/articles/CBMiigFBVV95cUxQMEhEMXhkckZhOHJNN3o2SVZxbUFrejVPN2hQSFRjZy1vUEpqTFlBZ09HUnNTclB1ZTdhblhTc3I0Vm85bWUyS3VrLTdzdGlsa1pNNkhlRTQzZmtaMnNaMmhCWVNZQ3l5eXhQWDFMZndYdHFmS2p4Y3VpXzJuTndLdzlKemJGNkZtMmc?oc=5"

async def test_pw():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            print("Going to url with playwright (allowing scripts)...")
            await page.route("**/*", lambda route: route.continue_() if route.request.resource_type in ["document", "script"] else route.abort())
            await page.goto(url, timeout=10000, wait_until="commit")
            await page.wait_for_timeout(3000)
            print("Playwright final url:", page.url)
        except Exception as e:
            print("Playwright error:", e)
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(test_pw())
