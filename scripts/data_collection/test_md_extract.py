import asyncio
import pandas as pd
from crawl4ai import AsyncWebCrawler
import re

async def clean_markdown_extractor():
    url = "https://www.business-standard.com/content/press-releases-ani/360-one-asset-achieves-iso-27001-2022-certification-strengthening-its-leadership-in-information-security-125110800015_1.html"
    print("Testing Ultra-Strict Markdown Cleaning...")
    
    async with AsyncWebCrawler(verbose=False) as crawler:
        result = await crawler.arun(
            url=url,
            magic=True,
            simulate_user=True,
            bypass_cache=True,
            delay_before_return_html=3.0,
            word_count_threshold=20,
            excluded_tags=['nav', 'footer', 'aside', 'header', 'script', 'style', 'noscript', 'meta', 'figure', 'ul', 'ol', 'li', 'button', 'form']
        )
        
        if result.success:
            print("\n--- EXTRACTED FIT MARKDOWN ---")
            
            text = getattr(result, 'fit_markdown', None)
            if not text and hasattr(result, 'markdown_v2') and result.markdown_v2:
                text = result.markdown_v2.fit_markdown
            if not text:
                text = result.markdown
                
            # Post-processing: remove Markdown links
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
            
            # Post-processing: remove Markdown Images
            text = re.sub(r'!\[.*?\]\([^\)]+\)', '', text)
            text = re.sub(r'!\[.*?\]', '', text)
            
            # Post-processing: remove remaining bold/italic stars
            text = re.sub(r'\*+', '', text)
            
            paragraphs = []
            for line in text.split('\n'):
                line = line.strip()
                # A real article paragraph:
                # 1. Does not start with common markdown symbols for lists, headers, quotes
                # 2. Is reasonably long (at least ~80 chars)
                # 3. Contains multiple words (at least 15 spaces)
                if len(line) > 80 and line.count(' ') > 15 and not re.match(r'^[>\-+#\d\.]', line):
                    # Exclude huge navigational menus lacking punctuation which are usually mashed links
                    if "News" in line and "India" in line and "World" in line and "Disclaimer" in line:
                         continue
                           
                    blocks = [
                        "subscribe to read", "access denied", "please log in", "robot", 
                        "enable javascript", "cookies", "copyright", "all rights reserved", 
                        "advertisement", "click here", "newsindia", "politics news", 
                        "budget 2024", "calculator", "rates in india", "mutual funds",
                        "newsbudget", "opinions", "sports news"
                    ]
                    if not any(b in line.lower() for b in blocks):
                        paragraphs.append(line)
                    
            print("\n\n".join(paragraphs))
            
        else:
            print(f"Extraction failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(clean_markdown_extractor())
