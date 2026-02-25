
import os

target_file = r"d:\stock\stock project dnyanesh\stock_app\generate_final_report.py"

with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find start of NOTEBOOK_CHARTS
start_idx = -1
for i, line in enumerate(lines):
    if line.strip().startswith("NOTEBOOK_CHARTS = {"):
        start_idx = i
        break

# Find start of fetch_data (which is after add_notebook_charts)
end_idx = -1
for i, line in enumerate(lines):
    if i > start_idx and line.strip().startswith("def fetch_data():"):
        end_idx = i
        break

# We want to keep lines up to start_idx (exclusive, but actually inclusive of previous lines)
# And lines from 'def fetch_data' upwards, but fetch_data starts a new section.
# The previous section (add_notebook_charts) ended before the block separator.
# Let's find the block separator before fetch_data
separator_idx = -1
for i in range(end_idx - 1, start_idx, -1):
    if "════" in lines[i]:
        separator_idx = i
        # Go back to the start of the comment block
        if i > 0 and lines[i-1].strip().startswith("#"):
             separator_idx = i-1
        break

if start_idx != -1 and separator_idx != -1:
    new_content = lines[:start_idx]
    
    # Insert new code
    new_code = """from chart_definitions import NOTEBOOK_CHARTS, PREFIX_TO_CHAPTER

def add_notebook_charts(story, s, chapter_num):
    \"\"\"Add all notebook-generated charts matching the given chapter number.\"\"\"
    prefix = f'{chapter_num:02d}_'
    matched = sorted([f for f in NOTEBOOK_CHARTS if f.startswith(prefix)])
    if not matched:
        return
    story.append(Spacer(1, 12))
    story.append(Paragraph(f'<b>Notebook Visualizations — Additional Analysis Charts</b>', s['Sec']))
    fig_counter = 1
    for fname in matched:
        fpath = os.path.join(FIGURES_GEN, fname)
        if not os.path.exists(fpath) or os.path.getsize(fpath) < 10240:
            continue
        title, explanation = NOTEBOOK_CHARTS[fname]
        cap = f'Figure {chapter_num}.N{fig_counter}: {title}'
        story.append(RLImage(fpath, width=6.0*inch, height=3.3*inch))
        story.append(Paragraph(f'<i>{cap}</i>', s['Cap']))
        story.append(Paragraph(explanation, s['Insight']))
        story.append(Spacer(1, 8))
        fig_counter += 1

"""
    new_content.append(new_code)
    new_content.extend(lines[separator_idx:])
    
    with open(target_file, 'w', encoding='utf-8') as f:
        f.writelines(new_content)
    print("Successfully updated generate_final_report.py")

else:
    print(f"Could not find start or end markers. Start: {start_idx}, Separator: {separator_idx}")
