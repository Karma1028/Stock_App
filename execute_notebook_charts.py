
import nbformat
import os
import glob
import time
import re
from nbconvert.preprocessors import ExecutePreprocessor

# Configuration
notebooks_dir = r"d:/stock/stock project dnyanesh/stock_app/notebooks"
figures_dir = r"d:/stock/stock project dnyanesh/stock_app/reports/figures_gen"
os.makedirs(figures_dir, exist_ok=True)

def extract_title(source):
    """Attempt to extract a chart title from the code source."""
    # Pattern 1: plt.title('...') or ax.set_title('...')
    # Using raw string for regex
    # Capture content inside quotes
    match = re.search(r"(?:plt\.title|set_title)\s*\(\s*['\"](.*?)['\"]", source)
    if match:
        return match.group(1)
    
    # Pattern 2: variable assignment like title = "..." (less common for plots but possible)
    return None

def sanitize_filename(text):
    """Make text safe for filenames."""
    if not text: return "chart"
    # Remove non-alphanumeric, replace spaces with underscores
    s = re.sub(r'[^\w\s-]', '', text)
    s = re.sub(r'[-\s]+', '_', s)
    return s.strip('_')

def get_indentation(line):
    """Return the leading whitespace of a line."""
    match = re.match(r"^(\s*)", line)
    return match.group(1) if match else ""

def process_notebook(notebook_name):
    print(f"Processing {notebook_name}...")
    nb_path = os.path.join(notebooks_dir, notebook_name)
    
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            
        # Inject imports at the top
        setup_code = f"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
FIGURES_DIR = r"{figures_dir}"
if not os.path.exists(FIGURES_DIR):
    try: os.makedirs(FIGURES_DIR)
    except: pass
# Ensure consistent style
try:
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (14, 7)
except: pass
"""
        nb.cells.insert(0, nbformat.v4.new_code_cell(setup_code))
        
        # Iterate and inject savefig
        nb_name_clean = notebook_name.replace('.ipynb', '')
        # Sort prefix: 01_Data -> 01_Data
        
        for cell in nb.cells:
            if cell.cell_type == 'code':
                source = cell.source
                if not source.strip(): continue
                
                # Check if it's likely a plot
                if 'plt.show()' in source or '.plot(' in source or 'sns.' in source:
                    # Extract title
                    title = extract_title(source)
                    safe_title = sanitize_filename(title) if title else f"fig_{int(time.time()*1000)%10000}" # fallback unique
                    
                    filename = f"{nb_name_clean}_{safe_title}.png"
                    
                    # Prepare save command
                    # bbox_inches='tight' is good but sometimes cuts off. 
                    save_cmd_template = "plt.savefig(os.path.join(FIGURES_DIR, '{}'), bbox_inches='tight', dpi=100)"
                    print_cmd_template = "print(f'Saved {}')"
                    
                    full_save_cmd = f"{save_cmd_template.format(filename)}\n{print_cmd_template.format(filename)}"

                    # Injection Strategy
                    if 'plt.show()' in source:
                        # Find the indentation of plt.show()
                        lines = source.split('\n')
                        new_lines = []
                        injected = False
                        
                        for line in lines:
                            if 'plt.show()' in line:
                                indent = get_indentation(line)
                                # Inject BEFORE show, with same indentation
                                new_lines.append(f"{indent}{save_cmd_template.format(filename)}")
                                new_lines.append(f"{indent}{print_cmd_template.format(filename)}")
                                new_lines.append(line) # output existing show
                                injected = True
                            else:
                                new_lines.append(line)
                        
                        if injected:
                            cell.source = '\n'.join(new_lines)
                        else:
                            # Fallback if plt.show() logic failed (e.g. inline)
                            cell.source += f"\n{full_save_cmd}"
                            
                    else:
                        # No show(), append at end
                        # Check last line indentation? Usually append at end is safe if top level.
                        # If inside a loop/function, we might miss it.
                        # Heuristic: if last line is indented, match it?
                        lines = source.rstrip().split('\n')
                        last_line = lines[-1]
                        indent = get_indentation(last_line)
                        # Append with same indent
                        cell.source += f"\n{indent}{full_save_cmd}"

        # Save temp notebook
        temp_nb_path = os.path.join(notebooks_dir, f"_exec_{notebook_name}")
        with open(temp_nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        # Execute
        print(f"Executing {notebook_name}...")
        # Timeout 600s (10 mins) per notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        
        ep.preprocess(nb, {'metadata': {'path': notebooks_dir}})
        print(f"✅ Successfully executed {notebook_name}")
        
    except Exception as e:
        print(f"❌ Execution failed for {notebook_name}: {e}")
        
    finally:
        if os.path.exists(temp_nb_path):
            try: os.remove(temp_nb_path)
            except: pass

if __name__ == "__main__":
    # Get all notebooks
    notebooks = sorted([f for f in os.listdir(notebooks_dir) if f.endswith('.ipynb') and not f.startswith('_exec')])
    
    print(f"Found {len(notebooks)} notebooks. Starting sequential execution...")
    
    for nb in notebooks:
        process_notebook(nb)
        time.sleep(2)
        
    print("\n🎉 Total Execution Complete.")
