import nbformat as nbf
import os

class NotebookBuilder:
    def __init__(self):
        self.nb = nbf.v4.new_notebook()
        self.cells = []

    def add_markdown(self, text):
        self.cells.append(nbf.v4.new_markdown_cell(text))

    def add_code(self, code):
        self.cells.append(nbf.v4.new_code_cell(code))

    def save(self, filename):
        self.nb['cells'] = self.cells
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            nbf.write(self.nb, f)
        print(f"Notebook saved to {filename}")
