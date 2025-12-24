import logging
from pathlib import Path
from datetime import datetime
import pytz

# Setup logging
def setup_logging(log_file: Path):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def get_current_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def format_currency(value):
    return f"₹{value:,.2f}"

def format_large_number(num):
    """
    Formats a large number into a readable string with suffixes (K, M, B, T).
    """
    if num is None:
        return "N/A"
    try:
        num = float(num)
    except (ValueError, TypeError):
        return str(num)

    if num >= 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.2f}T"
    elif num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"

