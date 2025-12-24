
import sys
import os

print(f"Running with: {sys.executable}")
try:
    import typing_extensions
    print(f"typing_extensions version: {typing_extensions.__version__}")
    import openai
    print("OpenAI imported successfully!")
    print(f"Version: {openai.__version__}")
except ImportError as e:
    print(f"FAILED to import: {e}")
except Exception as e:
    print(f"Error: {e}")
