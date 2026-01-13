import os
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Any, Callable
from config import Config

class CacheManager:
    """Centralized cache manager with LRU eviction and TTL support."""
    
    def __init__(self, cache_dir: Optional[Path] = None, default_ttl: int = 3600):
        self.cache_dir = cache_dir or Config.API_CACHE_DIR
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self._memory_cache = {}  # In-memory cache for faster access
        
    def _get_cache_key(self, key: str) -> str:
        """Generate a safe filename from cache key."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache if not expired."""
        # Check memory cache first
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if time.time() < entry['expires_at']:
                return entry['value']
            else:
                del self._memory_cache[key]
        
        # Check file cache
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                entry = json.load(f)
                
            if time.time() < entry['expires_at']:
                # Load into memory cache
                self._memory_cache[key] = entry
                return entry['value']
            else:
                # Expired, delete file
                cache_path.unlink()
                return None
        except Exception as e:
            print(f"Cache read error for {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Store value in cache with TTL."""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        
        entry = {
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time()
        }
        
        # Store in memory
        self._memory_cache[key] = entry
        
        # Store in file
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(entry, f)
            return True
        except Exception as e:
            print(f"Cache write error for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        # Remove from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
        
        # Remove file
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                cache_path.unlink()
                return True
            except Exception as e:
                print(f"Cache delete error for {key}: {e}")
                return False
        return False
    
    def clear(self) -> int:
        """Clear all cache entries. Returns number of files deleted."""
        count = 0
        
        # Clear memory cache
        self._memory_cache.clear()
        
        # Clear file cache
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception as e:
                print(f"Error deleting cache file {cache_file}: {e}")
        
        return count
    
    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: Optional[int] = None) -> Any:
        """Get from cache or compute and store if missing."""
        value = self.get(key)
        if value is not None:
            return value
        
        # Compute value
        value = factory()
        self.set(key, value, ttl)
        return value
    
    def warm_cache(self, keys_and_factories: dict[str, Callable[[], Any]], ttl: Optional[int] = None):
        """Pre-populate cache with multiple values."""
        for key, factory in keys_and_factories.items():
            if self.get(key) is None:
                try:
                    value = factory()
                    self.set(key, value, ttl)
                except Exception as e:
                    print(f"Error warming cache for {key}: {e}")
    
    def cleanup_expired(self) -> int:
        """Remove expired cache files. Returns number of files deleted."""
        count = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    entry = json.load(f)
                
                if current_time >= entry['expires_at']:
                    cache_file.unlink()
                    count += 1
            except Exception as e:
                # If we can't read it, delete it
                cache_file.unlink()
                count += 1
        
        return count

# Global cache instance
_global_cache = None

def get_cache() -> CacheManager:
    """Get the global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheManager()
    return _global_cache
