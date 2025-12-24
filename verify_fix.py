
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    print("Attempting to import modules.utils.ai_insights...")
    from modules.utils import ai_insights
    print("Import successful.")
    
    if ai_insights.OpenAI is None:
        print("OpenAI module is correctly identified as missing/None (Graceful Fallback).")
    else:
        print("OpenAI module found.")
        
    print("Test passed: No ModuleNotFoundError raised.")
    
except ImportError as e:
    print(f"FAILED: ImportError raised: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FAILED: Unexpected error: {e}")
    sys.exit(1)
