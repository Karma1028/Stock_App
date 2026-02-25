import nbformat, os

nb_dir = 'notebooks'
files = sorted([f for f in os.listdir(nb_dir) if f.endswith('.ipynb')])
print(f'Found {len(files)} notebooks\n')

for f in files:
    fp = os.path.join(nb_dir, f)
    size = os.path.getsize(fp) / 1024
    with open(fp, encoding='utf-8') as fh:
        nb = nbformat.read(fh, as_version=4)
    cells = len(nb.cells)
    md_cells = sum(1 for c in nb.cells if c.cell_type == 'markdown')
    code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
    print(f'  {f:45s} | {size:6.1f} KB | {cells:3d} cells (md:{md_cells}, code:{code_cells})')

print(f'\nTotal: {len(files)} notebooks')
