import asyncio
import pandas as pd
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import CSSExtractionStrategy

async def advanced_css_extractor():
    url = "https://www.business-standard.com/content/press-releases-ani/360-one-asset-achieves-iso-27001-2022-certification-strengthening-its-leadership-in-information-security-125110800015_1.html"
    print("Testing Crawl4AI CSS Extraction...")
    
    # We want to extract just the paragraphs that are part of the main article body
    # Using a common generalized selector for news sites
    strategy = CSSExtractionStrategy(
        schema={
            "name": "News Article",
            "baseSelector": "article, main, .story-content, .article-body, .post-content, #main-content, .content-area",
            "fields": [
                {
                    "name": "text",
                    "selector": "p",
                    "type": "text",
                    "is_list": True
                }
            ]
        }
    )
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(
            url=url,
            extraction_strategy=strategy,
            bypass_cache=True,
            simulate_user=True,
            delay_before_return_html=3.0
        )
        
        if result.success and result.extracted_content:
            import json
            data = json.loads(result.extracted_content)
            print("\n--- EXTRACTED JSON ---")
            for item in data:
                if 'text' in item and isinstance(item['text'], list):
                    for i, p in enumerate(item['text']):
                        if len(p) > 50:
                            print(f"[{i}]: {p}")
        else:
            print(f"Extraction failed or empty: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(advanced_css_extractor())
