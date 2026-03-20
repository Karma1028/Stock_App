import pandas as pd
import os
import glob

def audit_news():
    history_dir = 'data/news_history'
    files = glob.glob(os.path.join(history_dir, "*.csv"))
    
    summary = []
    
    for f in files[:20]: # Check first 20 to get a sense
        try:
            df = pd.read_csv(f)
            stats = {
                "file": os.path.basename(f),
                "total_rows": len(df),
                "with_content": df["Content"].count() if "Content" in df.columns else 0,
                "with_full_title": df["Full_Title"].count() if "Full_Title" in df.columns else 0
            }
            summary.append(stats)
        except Exception as e:
            summary.append({"file": os.path.basename(f), "error": str(e)})

    audit_df = pd.DataFrame(summary)
    print(audit_df.to_string())
    
    # Specific check for 360ONE.NS
    target = os.path.join(history_dir, "360ONE.NS_news.csv")
    if os.path.exists(target):
        df_target = pd.read_csv(target)
        if "Content" in df_target.columns:
            print("\n--- 360ONE.NS Audit ---")
            print(df_target[df_target["Content"].notna()][["Date", "Source", "Full_Title"]].head())
        else:
            print("\n'Content' column missing in 360ONE.NS_news.csv")

if __name__ == "__main__":
    audit_news()
