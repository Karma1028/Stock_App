import pandas as pd
import os
import glob
from datetime import datetime, timedelta

def audit_dates():
    files = glob.glob('data/news_history/*_news.csv')
    five_years_ago = datetime.now() - timedelta(days=5*365)
    
    missing_files = []
    gap_files = []
    
    print(f"Auditing {len(files)} CSV files for 5-year data completeness...")
    
    total_valid_rows = 0
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty or 'Date' not in df.columns:
                missing_files.append(os.path.basename(f))
                continue
                
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            valid = df.dropna(subset=['Date'])
            
            if valid.empty:
                missing_files.append(os.path.basename(f))
                continue
                
            total_valid_rows += len(valid)
            min_d = valid['Date'].min()
            max_d = valid['Date'].max()
            
            # If the oldest date is more recent than 5 years ago (plus a 30 day grace period)
            if min_d > pd.Timestamp(five_years_ago) + pd.Timedelta(days=30):
                gap_files.append({
                    'File': os.path.basename(f),
                    'MinDate': min_d.strftime('%Y-%m-%d'),
                    'MaxDate': max_d.strftime('%Y-%m-%d'),
                    'Rows': len(valid)
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")
            missing_files.append(os.path.basename(f))
            
    with open("Universal_Engine_Workspace/mas_logs/date_audit_results.txt", "w", encoding="utf-8") as out_f:
        out_f.write(f"\n--- AUDIT RESULTS ---\n")
        out_f.write(f"Total Files Checked: {len(files)}\n")
        out_f.write(f"Total Valid Rows Across All Files: {total_valid_rows}\n")
        out_f.write(f"Files completely empty/missing dates: {len(missing_files)}\n")
        out_f.write(f"Files with a gap (starting after {five_years_ago.strftime('%Y-%m-%d')}): {len(gap_files)}\n")
        
        if gap_files:
            out_f.write("\nSample of files missing 5 years of history:\n")
            for gap in gap_files[:100]:
                out_f.write(f"  {gap['File']}: Oldest Record = {gap['MinDate']} ({gap['Rows']} total rows)\n")


if __name__ == '__main__':
    audit_dates()
