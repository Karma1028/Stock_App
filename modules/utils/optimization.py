"""
Backend optimization utilities
"""
from functools import lru_cache, wraps
import time
from typing import Callable, Any
import threading

# Request rate limiting
class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        with self.lock:
            now = time.time()
            # Remove old requests
            self.requests = [
                (cid, timestamp) 
                for cid, timestamp in self.requests 
                if now - timestamp < self.window_seconds
            ]
            
            # Count requests from this client
            client_requests = sum(1 for cid, _ in self.requests if cid == client_id)
            
            if client_requests >= self.max_requests:
                return False
            
            self.requests.append((client_id, now))
            return True

# Memory-efficient caching decorator
def cache_with_ttl(ttl_seconds: int = 300):
    """Decorator to cache function results with TTL."""
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            with lock:
                # Check if cached and not expired
                if key in cache:
                    if time.time() - cache_times[key] < ttl_seconds:
                        return cache[key]
                    else:
                        # Expired, remove from cache
                        del cache[key]
                        del cache_times[key]
            
            # Call function and cache result
            result = func(*args, **kwargs)
            
            with lock:
                cache[key] = result
                cache_times[key] = time.time()
                
                # Limit cache size to prevent memory issues
                if len(cache) > 1000:
                    # Remove oldest entry
                    oldest_key = min(cache_times, key=cache_times.get)
                    del cache[oldest_key]
                    del cache_times[oldest_key]
            
            return result
        
        return wrapper
    return decorator

# Batch request processor
class BatchProcessor:
    """Process multiple requests in batches to reduce API calls."""
    
    def __init__(self, batch_size: int = 10, delay_seconds: float = 0.1):
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds
    
    def process_batch(self, items: list, processor_func: Callable) -> list:
        """Process items in batches."""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = processor_func(batch)
            results.extend(batch_results)
            
            # Delay between batches to avoid rate limits
            if i + self.batch_size < len(items):
                time.sleep(self.delay_seconds)
        
        return results

# Query optimization
@lru_cache(maxsize=128)
def get_cached_stock_info(symbol: str) -> dict:
    """Cached stock info retrieval."""
    # This would normally call the data manager
    # Cached to reduce redundant API calls
    pass

# Lazy loading helper
def lazy_load_data(data_func: Callable, chunk_size: int = 50):
    """Generator for lazy loading large datasets."""
    offset = 0
    while True:
        chunk = data_func(offset, chunk_size)
        if not chunk:
            break
        yield chunk
        offset += chunk_size
