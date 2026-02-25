
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
import os
import concurrent.futures
import threading

# URLs to scrape
URLS = [
    "https://youtu.be/aiFpAl3mgGk",
    "https://youtu.be/q24MVGCUr4w",
    "https://youtu.be/V2l7cZxUpQs",
    "https://youtu.be/kRa3PUxNBTM",
    "https://youtu.be/1F0gYkk7YYw"
]

# Lock for driver creation to avoid patching race conditions
driver_lock = threading.Lock()

def scrape_video(url):
    print(f"[{url}] Starting...", flush=True)
    driver = None
    try:
        # Initialize driver (Thread-safe creation)
        with driver_lock:
             options = uc.ChromeOptions()
             options.add_argument("--headless")
             options.add_argument("--disable-gpu")
             options.add_argument("--no-sandbox")
             options.add_argument("--mute-audio")
             # Use a unique user-data-dir if possible or let uc handle it (uc handles temp dirs usually)
             driver = uc.Chrome(options=options)
        
        print(f"[{url}] Navigating...", flush=True)
        driver.get(url)
        time.sleep(5)
        
        # Scroll to trigger comments
        driver.execute_script("window.scrollTo(0, 500);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, 2000);")
        time.sleep(3)
        
        # Aggressive scrolling to load ALL threads
        print(f"[{url}] Scrolling...", flush=True)
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        for i in range(40):
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2.5)
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                time.sleep(2)
                new_height = driver.execute_script("return document.documentElement.scrollHeight")
                if new_height == last_height:
                    break
            last_height = new_height
            
        # Extract comments and replies
        comments_data = []
        threads = driver.find_elements(By.TAG_NAME, "ytd-comment-thread-renderer")
        print(f"[{url}] Found {len(threads)} threads. Extracting...", flush=True)
        
        for thread in threads:
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", thread)
                time.sleep(0.1) 

                # Main Comment
                main = thread.find_element(By.ID, "comment")
                author = main.find_element(By.ID, "author-text").text
                content = main.find_element(By.ID, "content-text").text
                try:
                    likes = main.find_element(By.ID, "vote-count-middle").text
                except:
                    likes = "0"
                try:
                    date = main.find_element(By.CLASS_NAME, "published-time-text").text
                except:
                    date = ""
                
                comments_data.append({
                    "Video URL": url,
                    "Type": "Comment",
                    "Author": author,
                    "Content": content,
                    "Likes": likes,
                    "Date": date,
                    "Reply To": ""
                })
                
                # Replies
                try:
                    replies_section = thread.find_element(By.ID, "replies")
                    # Try to find button with "replies" in text/aria-label
                    buttons = replies_section.find_elements(By.TAG_NAME, "button")
                    view_btn = None
                    for btn in buttons:
                        if "replies" in btn.get_attribute("aria-label") or "replies" in btn.text.lower():
                            view_btn = btn
                            break
                    if not view_btn:
                         btns = replies_section.find_elements(By.TAG_NAME, "ytd-button-renderer")
                         for b in btns:
                              if "View" in b.text or "replies" in b.text:
                                  view_btn = b
                                  break

                    if view_btn:
                        driver.execute_script("arguments[0].click();", view_btn)
                        time.sleep(1.5) # Wait for expansion
                        
                        replies = replies_section.find_elements(By.TAG_NAME, "ytd-comment-renderer")
                        for reply in replies:
                            try:
                                r_author = reply.find_element(By.ID, "author-text").text
                                r_content = reply.find_element(By.ID, "content-text").text
                                try:
                                    r_likes = reply.find_element(By.ID, "vote-count-middle").text
                                except:
                                    r_likes = "0"
                                try:
                                    r_date = reply.find_element(By.CLASS_NAME, "published-time-text").text
                                except:
                                    r_date = ""
                                comments_data.append({
                                    "Video URL": url,
                                    "Type": "Reply",
                                    "Author": r_author,
                                    "Content": r_content,
                                    "Likes": r_likes,
                                    "Date": r_date,
                                    "Reply To": author
                                })
                            except:
                                continue
                except:
                    pass
            except:
                continue
                
        print(f"[{url}] Finished. Collected {len(comments_data)} items.", flush=True)
        return comments_data
        
    except Exception as e:
        print(f"[{url}] Error: {e}", flush=True)
        return []
    finally:
        if driver:
            driver.quit()

def main():
    print("Starting Parallel Selenium Scraper (5 Workers)...", flush=True)
    all_data = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(scrape_video, url): url for url in URLS}
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                data = future.result()
                all_data.extend(data)
            except Exception as exc:
                print(f"[{url}] generated an exception: {exc}")
                
    if all_data:
        df = pd.DataFrame(all_data)
        outfile = "YouTube_Comments_Consolidated.xlsx"
        df.to_excel(outfile, index=False)
        print(f"Saved {len(df)} total comments/replies to {outfile}", flush=True)
    else:
        print("No data collected.", flush=True)

if __name__ == "__main__":
    main()
