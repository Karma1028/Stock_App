"""
Global API Usage Stats Store — persists across sessions/users via JSON file.
Thread-safe with file locking for concurrent access on Streamlit Cloud.
Auto-resets daily.
"""
import json
import os
import threading
from datetime import date, datetime

_STATS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'api_stats.json')
_lock = threading.Lock()


def _default_stats():
    return {
        "date": date.today().isoformat(),
        "total_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "keys_exhausted": 0,
        "last_model": "None",
        "last_error": "",
        "call_history": [],  # list of { "time": "HH:MM", "success": bool, "model": "...", "tokens": N }
        "model_usage": {},   # { "model_name": count }
    }


def _ensure_dir():
    d = os.path.dirname(os.path.abspath(_STATS_FILE))
    os.makedirs(d, exist_ok=True)


def load_stats() -> dict:
    """Load stats from file. Auto-reset if date changed."""
    with _lock:
        try:
            _ensure_dir()
            if os.path.exists(_STATS_FILE):
                with open(_STATS_FILE, 'r') as f:
                    stats = json.load(f)
                # Auto-reset on new day
                if stats.get('date') != date.today().isoformat():
                    stats = _default_stats()
                    _save_stats_unsafe(stats)
                return stats
        except Exception:
            pass
        stats = _default_stats()
        _save_stats_unsafe(stats)
        return stats


def _save_stats_unsafe(stats: dict):
    """Save stats without locking (caller must hold lock)."""
    try:
        _ensure_dir()
        with open(_STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass


def record_call(success: bool, model: str = "", input_tokens: int = 0,
                output_tokens: int = 0, key_exhausted: bool = False, error: str = ""):
    """Record an API call into the global store."""
    with _lock:
        try:
            _ensure_dir()
            if os.path.exists(_STATS_FILE):
                with open(_STATS_FILE, 'r') as f:
                    stats = json.load(f)
                if stats.get('date') != date.today().isoformat():
                    stats = _default_stats()
            else:
                stats = _default_stats()

            stats['total_calls'] += 1
            if success:
                stats['successful_calls'] += 1
                short_model = model.split("/")[-1].replace(":free", "")
                stats['last_model'] = short_model
            else:
                stats['failed_calls'] += 1
                stats['last_error'] = error[:80]
            
            stats['input_tokens'] += input_tokens
            stats['output_tokens'] += output_tokens
            
            if key_exhausted:
                stats['keys_exhausted'] += 1
            
            # Append to history (keep last 50)
            stats['call_history'].append({
                "time": datetime.now().strftime("%H:%M"),
                "success": success,
                "model": model.split("/")[-1].replace(":free", "")[:20],
                "tokens": input_tokens + output_tokens,
            })
            stats['call_history'] = stats['call_history'][-50:]
            
            # Model usage counter
            short = model.split("/")[-1].replace(":free", "")
            stats['model_usage'][short] = stats['model_usage'].get(short, 0) + 1
            
            _save_stats_unsafe(stats)
        except Exception:
            pass
