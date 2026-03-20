import asyncio
import os
import pandas as pd
import logging
import json
import random
import re
from playwright.async_api import async_playwright
from crawl4ai import AsyncWebCrawler
from datetime import datetime, timedelta
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console

console = Console()
os.makedirs('Universal_Engine_Workspace/mas_logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="Universal_Engine_Workspace/mas_logs/bulk_scraper.log"
)
logger = logging.getLogger('BulkContentUpdater')

# Configuration
BATCH_SIZE = 15
CONCURRENCY = 4 # Process 4 stocks at the same time
STATE_FILE = 'Universal_Engine_Workspace/scripts_scraping/scraper_state.json'

class RobustScraper:
    def __init__(self):
        self.state = self.load_state()
        self.five_years_ago = datetime.now() - timedelta(days=5*365)
        self.reputable = ['moneycontrol.com', 'economictimes.indiatimes.com', 'business-standard.com', 
                         'livemint.com', 'reuters.com', 'businesstoday.in', 'bloomberg.com', 
                         'ndtv.com', 'financialexpress.com', 'thehindubusinessline.com']
        self.state_lock = asyncio.Lock()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            except: pass
        return {"processed_urls": {}, "last_file_idx": 0}

    async def save_state(self):
        async with self.state_lock:
            temp_file = STATE_FILE + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(self.state, f)
            os.replace(temp_file, STATE_FILE)

    async def resolve_redirect(self, browser, url):
        """Resolves Google News redirects to the actual target page using Playwright."""
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        try:
            # Random delay simulates human browsing
            await asyncio.sleep(random.uniform(0.5, 2.0))
            await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            
            # Wait for JS redirects
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

    async def fetch_high_quality_content(self, crawler, url):
        """Uses Crawl4AI to extract text, and filters HTML aggressively using built-in markdown properties and Regex."""
        try:
            # ET specific interstitial trap
            if "defaultinterstitial.cms" in url:
                return None, None

            result = await crawler.arun(
                url=url,
                magic=True,
                simulate_user=True,
                bypass_cache=True,
                delay_before_return_html=random.uniform(3.0, 6.0),
                word_count_threshold=20,
                excluded_tags=['nav', 'footer', 'aside', 'header', 'script', 'style', 'noscript', 'meta', 'form', 'button']
            )
            
            if result.success:
                title = result.metadata.get('og:title') or result.metadata.get('title')
                
                # Fetch built-in cleaned markdown
                text = getattr(result, 'fit_markdown', None)
                if not text and hasattr(result, 'markdown_v2') and result.markdown_v2:
                    text = result.markdown_v2.fit_markdown
                if not text:
                    text = result.markdown
                
                if not text:
                    return None, None

                # Post-processing: remove Markdown links, leaving only the anchor text
                text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
                
                # Post-processing: remove remaining bold/italic stars and image tags
                text = re.sub(r'\*+', '', text)
                text = re.sub(r'!\[.*?\]', '', text)
                
                # Post-processing: keep only paragraphs that look like sentences
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
                
                content = "\n\n".join(paragraphs)
                
                # We enforce minimum total length to avoid just capturing stray UI elements
                if content and len(content) > 300:
                    return title, content
        except Exception as e:
            pass
        return None, None

    async def process_file(self, file_path, browser, crawler, progress_ui, task_id):
        try:
            df = pd.read_csv(file_path)
        except: return 0

        if 'Full_Title' not in df.columns: df['Full_Title'] = None
        if 'Content' not in df.columns: df['Content'] = None

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df[df['Date'] >= self.five_years_ago]
        
        # Only target rows missing content and not marked as hard fail
        mask = (df['Content'].isna() | (df['Content'].astype(str).str.len() < 200)) & df['Link'].notna()
        source_mask = df['Source'].apply(lambda s: any(r in str(s).lower() for r in self.reputable))
        
        target_indices = df[mask & source_mask].head(BATCH_SIZE).index.tolist()
        if not target_indices:
            target_indices = df[mask].head(5).index.tolist() # Fallback
            
        if not target_indices:
            progress_ui.update(task_id, description=f"[dim]Skipping {os.path.basename(file_path)}[/dim]", completed=100)
            return 0

        stock_name = os.path.basename(file_path).replace('_news.csv', '')
        progress_ui.update(task_id, description=f"[cyan]Processing {stock_name}[/cyan]", total=len(target_indices))
        
        success_count = 0
        for idx in target_indices:
            link = df.at[idx, 'Link']
            
            async with self.state_lock:
                status = self.state['processed_urls'].get(link)
                if status in ['hard_fail', 'success']:
                    progress_ui.advance(task_id, 1)
                    continue

            # 1. Resolve URL through Playwright
            final_url = await self.resolve_redirect(browser, link)
            if "google.com" in final_url:
                async with self.state_lock:
                    self.state['processed_urls'][link] = 'hard_fail'
                await self.save_state()
                progress_ui.advance(task_id, 1)
                continue

            # 2. Extract cleanly with Crawl4AI
            title, content = await self.fetch_high_quality_content(crawler, final_url)
            
            if content:
                df.at[idx, 'Full_Title'] = title
                df.at[idx, 'Content'] = " ".join(content.split())[:6000]
                success_count += 1
                async with self.state_lock:
                    self.state['processed_urls'][link] = 'success'
            else:
                async with self.state_lock:
                    self.state['processed_urls'][link] = 'hard_fail' # To avoid getting stuck retrying bad links
            
            await self.save_state()
            progress_ui.advance(task_id, 1)
            
        if success_count > 0:
            df.to_csv(file_path, index=False)
            logger.info(f"[{stock_name}] COMMITTED {success_count} valid articles.")
        
        progress_ui.update(task_id, description=f"[green]Done {stock_name} (+{success_count})[/green]")
        return success_count

    async def run(self):
        history_dir = 'data/news_history'
        files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.endswith('.csv')]
        files.sort()
        
        start_idx = self.state.get('last_file_idx', 0)
        
        console.print(f"[bold magenta]🚀 Starting Robust Scraping Engine (1900+ Stocks)[/bold magenta]")
        console.print(f"[dim]Resuming from batch index: {start_idx}[/dim]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            expand=True
        ) as progress_ui:
            
            overall_task = progress_ui.add_task("[bold yellow]Overall Progress[/bold yellow]", total=len(files), completed=start_idx)

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                async with AsyncWebCrawler(verbose=False) as crawler:
                    
                    # Batch processing logic for horizontal scaling across stocks
                    for i in range(start_idx, len(files), CONCURRENCY):
                        chunk = files[i:i + CONCURRENCY]
                        
                        # Create dynamic tasks for this batch
                        batch_ui_tasks = []
                        tasks = []
                        for f in chunk:
                            task_id = progress_ui.add_task(f"[dim]Pending {os.path.basename(f)}[/dim]", total=100)
                            batch_ui_tasks.append(task_id)
                            tasks.append(self.process_file(f, browser, crawler, progress_ui, task_id))
                        
                        await asyncio.gather(*tasks)
                        
                        # Clean up UI tasks after batch completes to prevent terminal spam
                        for t_id in batch_ui_tasks:
                            progress_ui.remove_task(t_id)
                        
                        async with self.state_lock:
                            self.state['last_file_idx'] = i + len(chunk)
                        await self.save_state()
                        
                        progress_ui.advance(overall_task, len(chunk))
                        await asyncio.sleep(5)
                        
                await browser.close()

if __name__ == "__main__":
    scraper = RobustScraper()
    asyncio.run(scraper.run())

