"""
API Call Logger — per-call granular log, stored as a JSON-lines file.
- Rolls over at midnight (new date = new file, old file deleted).
- Light-weight: each call appends one JSON line to the log file.
- Thread-safe file append.
"""
import json
import os
import threading
import time
from datetime import datetime, date

_LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'api_logs')
_lock = threading.Lock()


def _today_log_path() -> str:
    os.makedirs(_LOG_DIR, exist_ok=True)
    return os.path.join(_LOG_DIR, f"api_calls_{date.today().isoformat()}.jsonl")


def _cleanup_old_logs():
    """Delete log files from previous days (keep only today's)."""
    today_file = os.path.basename(_today_log_path())
    try:
        for fname in os.listdir(_LOG_DIR):
            if fname.startswith("api_calls_") and fname.endswith(".jsonl") and fname != today_file:
                try:
                    os.remove(os.path.join(_LOG_DIR, fname))
                except Exception:
                    pass
    except Exception:
        pass


def log_call(
    key_label: str,
    model: str,
    status: str,          # "success" | "rate_limited" | "auth_error" | "error" | "empty"
    prompt_type: str = "",# e.g. "Quick Analysis", "Deep Report", "News Sentiment"
    ticker: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: int = 0,
    error: str = "",
    attempt: int = 1,
):
    """Append one API call record to today's log file."""
    _cleanup_old_logs()
    record = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "key": key_label,
        "model": model.split("/")[-1].replace(":free", "") if model else "—",
        "model_full": model,
        "prompt_type": prompt_type or "AI Query",
        "ticker": ticker.upper() if ticker else "",
        "status": status,
        "in_tokens": input_tokens,
        "out_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "latency_ms": latency_ms,
        "error": error[:120] if error else "",
        "attempt": attempt,
    }
    with _lock:
        try:
            with open(_today_log_path(), 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass


def read_today_logs(limit: int = 200) -> list[dict]:
    """Read today's log entries, most recent first. Returns up to `limit` entries."""
    path = _today_log_path()
    entries = []
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except Exception:
                            pass
    except Exception:
        pass
    return entries[-limit:][::-1]   # most recent first


def get_log_summary() -> dict:
    """Return a quick summary of today's calls for the sidebar badge."""
    logs = read_today_logs(limit=5000)
    total = len(logs)
    ok = sum(1 for e in logs if e.get("status") == "success")
    rate_limited = sum(1 for e in logs if "rate" in e.get("status", ""))
    auth_errors = sum(1 for e in logs if "auth" in e.get("status", ""))
    errors = total - ok - rate_limited - auth_errors
    in_tok = sum(e.get("in_tokens", 0) for e in logs)
    out_tok = sum(e.get("out_tokens", 0) for e in logs)
    last_ts = logs[0].get("ts", "") if logs else ""
    model_counts: dict[str, int] = {}
    for e in logs:
        m = e.get("model", "—")
        model_counts[m] = model_counts.get(m, 0) + 1
    top_model = max(model_counts, key=model_counts.get) if model_counts else "—"
    return {
        "total": total,
        "ok": ok,
        "rate_limited": rate_limited,
        "auth_errors": auth_errors,
        "other_errors": errors,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "last_ts": last_ts,
        "top_model": top_model,
    }
