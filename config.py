import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    PAGES_DIR = BASE_DIR / "pages"
    PORTFOLIOS_DIR = BASE_DIR / "portfolios"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"
    SCRAPERS_DIR = BASE_DIR / "scrapers"

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    SITE_TITLE = os.getenv("SITE_TITLE", "Stock Analysis Tool")
    USE_LOCAL_DATA = os.getenv("USE_LOCAL_DATA", "False").lower() == "true"
    
    # API Caching Configuration
    API_CACHE_ENABLED = os.getenv("API_CACHE_ENABLED", "True").lower() == "true"
    API_CACHE_TTL = int(os.getenv("API_CACHE_TTL", "3600"))  # 1 hour default
    API_CACHE_DIR = BASE_DIR / "data" / "api_cache"
    
    # Fallback AI Models (in order of preference)
    # Fallback AI Models (in order of preference)
    AI_MODEL_FALLBACKS = [
        "xiaomi/mimo-v2-flash:free",
        "mistralai/devstral-2512:free",
        "tngtech/deepseek-r1t2-chimera:free",
        "z-ai/glm-4.5-air:free",
        "deepseek/deepseek-r1-0528:free",
        "google/gemma-3-27b-it:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "openai/gpt-oss-120b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free"
    ]
    DEFAULT_AI_MODEL = AI_MODEL_FALLBACKS[0]
    
    # Performance Configuration
    MAX_DASHBOARD_STOCKS = int(os.getenv("MAX_DASHBOARD_STOCKS", "100"))
    CACHE_REFRESH_INTERVAL = int(os.getenv("CACHE_REFRESH_INTERVAL", "900"))  # 15 minutes

    @classmethod
    def ensure_directories(cls):
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.PAGES_DIR.mkdir(exist_ok=True)
        cls.PORTFOLIOS_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.SCRAPERS_DIR.mkdir(exist_ok=True)
        cls.API_CACHE_DIR.mkdir(exist_ok=True)

    @classmethod
    def validate_config(cls):
        if not cls.OPENROUTER_API_KEY:
            print("WARNING: OPENROUTER_API_KEY is not set. AI features will be disabled.")

# Ensure directories exist on import
Config.ensure_directories()
