try:
    import ta
    print("✅ 'ta' module imported successfully!")
    print(f"Version: {ta.__version__}")
except ImportError as e:
    print(f"❌ Failed to import 'ta': {e}")
except Exception as e:
    print(f"❌ An error occurred: {e}")
