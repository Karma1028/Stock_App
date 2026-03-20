import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Config:
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    PAGES_DIR = BASE_DIR / "pages"
    PORTFOLIOS_DIR = BASE_DIR / "portfolios"
    LOGS_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"
    SCRAPERS_DIR = BASE_DIR / "scrapers"

    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    # Pool of API keys for rotation (avoids per-key rate limits)
    OPENROUTER_API_KEYS = [
        k for k in [
            os.getenv("OPENROUTER_API_KEY"),
            os.getenv("OPENROUTER_API_KEY_2"),
            os.getenv("OPENROUTER_API_KEY_3"),
            os.getenv("OPENROUTER_API_KEY_4"),
        ] if k  # filter out None/empty
    ]
    
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    SITE_TITLE = os.getenv("SITE_TITLE", "Stock Analysis Tool")
    USE_LOCAL_DATA = os.getenv("USE_LOCAL_DATA", "False").lower() == "true"
    
    # API Caching Configuration
    API_CACHE_ENABLED = os.getenv("API_CACHE_ENABLED", "True").lower() == "true"
    API_CACHE_TTL = int(os.getenv("API_CACHE_TTL", "3600"))  # 1 hour default
    API_CACHE_DIR = BASE_DIR / "data" / "api_cache"
    
    # Fallback AI Models (in order of preference) — OpenRouter free models
    AI_MODEL_FALLBACKS = [
        "deepseek/deepseek-r1-0528:free",
        "qwen/qwen3-235b-a22b:free",
        "google/gemma-3-27b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "qwen/qwen3-30b-a3b:free",
        "meta-llama/llama-4-maverick:free",
        "meta-llama/llama-4-scout:free",
        "microsoft/phi-4-reasoning:free",
        "nvidia/llama-3.1-nemotron-70b-instruct:free",
        "qwen/qwen3-coder:free",
        "google/gemini-2.0-flash-exp:free",
        "deepseek/deepseek-chat-v3-0324:free",
        "meta-llama/llama-3.3-70b-instruct:free",
    ]
    DEFAULT_AI_MODEL = AI_MODEL_FALLBACKS[0]

    # ── Tier 2: Groq (free, ultra-fast) ──
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    GROQ_MODELS = [
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
    ]

    # ── Tier 3: LM Studio (local, offline-capable) ──
    LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
    
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
