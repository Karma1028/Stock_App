
import pandas as pd
import os
import glob

def consolidate_news():
    news_dir = os.path.join("data", "news_history")
    output_file = os.path.join("data", "all_stocks_news_consolidated.csv")
    
    print(f"Scanning {news_dir}...")
    
    all_files = glob.glob(os.path.join(news_dir, "*.csv"))
    
    if not all_files:
        print("No news files found.")
        return

    print(f"Found {len(all_files)} files. Reading and combining...")
    
    dfs = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            # If Ticker column exists, good. If not, extract from filename.
            # Filename format: TICKER.NS_news.csv
            if 'Ticker' not in df.columns:
                basename = os.path.basename(filename)
                ticker = basename.replace("_news.csv", "")
                df['Ticker'] = ticker
            
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by Date if possible
        if 'Date' in combined_df.columns:
             combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
             combined_df = combined_df.sort_values(by='Date', ascending=False)
        
        combined_df.to_csv(output_file, index=False)
        print(f"Successfully created {output_file} with {len(combined_df)} rows.")
    else:
        print("No data collected.")

if __name__ == "__main__":
    consolidate_news()
