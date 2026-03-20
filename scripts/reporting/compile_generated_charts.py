
import os
import re
import math

figures_dir = r"d:/stock/stock project dnyanesh/stock_app/reports/figures_gen"
report_path = r"d:/stock/stock project dnyanesh/stock_app/reports/Master_Chart_Book.md"

# Threshold for blank images (bytes)
# User reported ~6KB as blank. Setting safer limit at 10KB.
MIN_FILE_SIZE = 10 * 1024 

def format_title(filename):
    """Convert filename to readable title."""
    # Remove extension and notebook prefix if present
    name = os.path.splitext(filename)[0]
    # Remove prefix pattern like "01_Data_Extraction_"
    name = re.sub(r'^\d+_[a-zA-Z0-9]+_[a-zA-Z0-9]+_', '', name)
    # Replace underscores/hyphens with spaces
    title = name.replace('_', ' ').replace('-', ' ').title()
    return title

def generate_report():
    print("Compiling Master Chart Book from generated figures...")
    
    if not os.path.exists(figures_dir):
        print(f"Directory not found: {figures_dir}")
        return

    # Get all png files
    all_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
    valid_files = []
    
    print(f"Found {len(all_files)} total files.")
    
    skipped_count = 0
    for f in all_files:
        path = os.path.join(figures_dir, f)
        size = os.path.getsize(path)
        if size < MIN_FILE_SIZE:
            skipped_count += 1
            # Optional: delete blank files to clean up?
            # os.remove(path) 
        else:
            valid_files.append(f)
            
    print(f"Skipped {skipped_count} blank/small files (<10KB).")
    print(f"Processing {len(valid_files)} valid charts.")

    # Sort files by modification time - ensures chronological order of execution
    # This is better than name sort because execution order matters for flow
    valid_files.sort(key=lambda x: os.path.getmtime(os.path.join(figures_dir, x)))

    # Group by notebook
    notebooks = {}
    for f in valid_files:
        # Extract notebook name (first 2-3 parts usually: 01_Data_Extraction_...)
        # Regex: Start with digits, then some words
        match = re.match(r"((\d+_[a-zA-Z0-9]+(?:_[a-zA-Z0-9]+)?)).*", f)
        if match:
            nb_key = match.group(1)
        else:
            nb_key = "Uncategorized"
            
        if nb_key not in notebooks:
            notebooks[nb_key] = []
        notebooks[nb_key].append(f)
            
    # Write Markdown
    with open(report_path, 'w', encoding='utf-8') as md:
        md.write("# Master Chart Book: Authentic Notebook Outputs\n\n")
        md.write(f"**Generated:** {os.path.basename(figures_dir)}\n")
        md.write(f"**Total Valid Charts:** {len(valid_files)}\n\n")
        md.write("This document contains authentic visualizations generated directly from the notebooks, filtered for validity.\n\n")
        
        # Sort notebooks by number
        sorted_keys = sorted(notebooks.keys())
        
        for nb in sorted_keys:
            clean_nb_name = nb.replace('_', ' ')
            md.write(f"## {clean_nb_name}\n\n")
            
            for img in notebooks[nb]:
                title = format_title(img)
                md.write(f"### {title}\n")
                md.write(f"![{title}](figures_gen/{img})\n\n")
                
    print(f"✅ Report generated at: {report_path}")

if __name__ == "__main__":
    generate_report()
