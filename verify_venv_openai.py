
import sys
try:
    import openai
    print("SUCCESS")
except ImportError as e:
    print(f"FAIL: {e}")
except Exception as e:
    print(f"ERROR: {e}")
