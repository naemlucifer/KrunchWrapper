#!/usr/bin/env python3
"""
Persistent Token Cache for KrunchWrap

Provides disk-backed token caching with lazy loading to persist across server restarts.
Cache files are stored in the temp folder and loaded only when needed.
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from functools import wraps
import threading

logger = logging.getLogger('token_cache')

class PersistentTokenCache:
    """
    Disk-backed cache with lazy loading for token counts and symbol efficiency data.
    """
    
    def __init__(self, temp_dir: str = "temp", max_ram_entries: int = 1000, 
                 cache_ttl_hours: int = 24):
        """
        Initialize persistent token cache.
        
        Args:
            temp_dir: Directory to store cache files
            max_ram_entries: Maximum entries to keep in RAM
            cache_ttl_hours: Hours after which cache entries expire
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_ram_entries = max_ram_entries
        self.cache_ttl_seconds = cache_ttl_hours * 3600
        
        # In-memory cache for frequently accessed items
        self._ram_cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._creation_times: Dict[str, float] = {}
        self._cache_lock = threading.RLock()
        
        # Cache statistics
        self.stats = {
            'ram_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'saves': 0,
            'evictions': 0
        }
        
        # Cache initialized (details shown in formatted startup section)
    
    def _get_cache_key(self, key: str, prefix: str = "token") -> str:
        """Generate a safe cache key."""
        # Create hash of the key to avoid filesystem issues
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return f"{prefix}_{key_hash}"
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get the full path to a cache file."""
        return self.temp_dir / f"{cache_key}.json"
    
    def _is_cache_valid(self, file_path: Path) -> bool:
        """Check if a cache file is still valid (not expired)."""
        if not file_path.exists():
            return False
        
        try:
            file_age = time.time() - file_path.stat().st_mtime
            return file_age < self.cache_ttl_seconds
        except Exception:
            return False
    
    def _evict_lru_from_ram(self):
        """Remove least recently used items from RAM cache."""
        with self._cache_lock:
            if len(self._ram_cache) <= self.max_ram_entries:
                return
            
            # Sort by access time and remove oldest entries
            sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
            items_to_remove = len(self._ram_cache) - self.max_ram_entries
            
            for cache_key, _ in sorted_items[:items_to_remove]:
                if cache_key in self._ram_cache:
                    del self._ram_cache[cache_key]
                    if cache_key in self._access_times:
                        del self._access_times[cache_key]
                    if cache_key in self._creation_times:
                        del self._creation_times[cache_key]
                    self.stats['evictions'] += 1
    
    def get(self, key: str, prefix: str = "token") -> Optional[Any]:
        """
        Get value from cache, checking RAM first, then disk.
        
        Args:
            key: The cache key
            prefix: Cache prefix for organizing different types of cached data
            
        Returns:
            Cached value or None if not found/expired
        """
        cache_key = self._get_cache_key(key, prefix)
        current_time = time.time()
        
        # Check RAM cache first
        with self._cache_lock:
            if cache_key in self._ram_cache:
                # Check if the RAM cache entry is still valid based on when it was created
                if cache_key in self._creation_times:
                    entry_age = current_time - self._creation_times[cache_key]
                    if entry_age < self.cache_ttl_seconds:
                        self._access_times[cache_key] = current_time
                        self.stats['ram_hits'] += 1
                        logger.debug(f"RAM cache hit for {prefix}:{key[:50]}...")
                        return self._ram_cache[cache_key]
                    else:
                        # Entry is expired, remove from RAM cache
                        del self._ram_cache[cache_key]
                        if cache_key in self._access_times:
                            del self._access_times[cache_key]
                        del self._creation_times[cache_key]
                        logger.debug(f"RAM cache entry expired for {prefix}:{key[:50]}...")
                else:
                    # No creation time recorded, assume valid and update
                    self._access_times[cache_key] = current_time
                    self._creation_times[cache_key] = current_time
                    self.stats['ram_hits'] += 1
                    logger.debug(f"RAM cache hit for {prefix}:{key[:50]}...")
                    return self._ram_cache[cache_key]
        
        # Check disk cache
        cache_file = self._get_cache_file_path(cache_key)
        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Load into RAM cache
                with self._cache_lock:
                    self._ram_cache[cache_key] = data['value']
                    self._access_times[cache_key] = current_time
                    self._creation_times[cache_key] = data.get('timestamp', current_time)
                    self._evict_lru_from_ram()
                
                self.stats['disk_hits'] += 1
                logger.debug(f"Disk cache hit for {prefix}:{key[:50]}...")
                return data['value']
                
            except Exception as e:
                logger.warning(f"Failed to load cache from {cache_file}: {e}")
                # Clean up corrupted cache file
                try:
                    cache_file.unlink()
                except Exception:
                    pass
        
        self.stats['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, prefix: str = "token"):
        """
        Set value in cache, storing in both RAM and disk.
        
        Args:
            key: The cache key
            value: The value to cache
            prefix: Cache prefix for organizing different types of cached data
        """
        cache_key = self._get_cache_key(key, prefix)
        current_time = time.time()
        
        # Store in RAM cache
        with self._cache_lock:
            self._ram_cache[cache_key] = value
            self._access_times[cache_key] = current_time
            self._creation_times[cache_key] = current_time
            self._evict_lru_from_ram()
        
        # Store on disk asynchronously (don't block)
        try:
            cache_file = self._get_cache_file_path(cache_key)
            cache_data = {
                'key': key,
                'value': value,
                'timestamp': current_time,
                'prefix': prefix
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, separators=(',', ':'))
            
            self.stats['saves'] += 1
            logger.debug(f"Saved cache entry to disk: {prefix}:{key[:50]}...")
            
        except Exception as e:
            logger.warning(f"Failed to save cache to disk for {prefix}:{key[:50]}...: {e}")
    
    def clear_expired(self):
        """Remove expired cache files from disk."""
        expired_count = 0
        try:
            for cache_file in self.temp_dir.glob("*.json"):
                if not self._is_cache_valid(cache_file):
                    try:
                        cache_file.unlink()
                        expired_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove expired cache file {cache_file}: {e}")
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired cache files")
                
        except Exception as e:
            logger.error(f"Failed to clean expired cache files: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            ram_size = len(self._ram_cache)
        
        disk_files = len(list(self.temp_dir.glob("*.json")))
        
        total_requests = self.stats['ram_hits'] + self.stats['disk_hits'] + self.stats['misses']
        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = (self.stats['ram_hits'] + self.stats['disk_hits']) / total_requests * 100
        
        return {
            'ram_entries': ram_size,
            'disk_files': disk_files,
            'ram_hits': self.stats['ram_hits'],
            'disk_hits': self.stats['disk_hits'],
            'misses': self.stats['misses'],
            'saves': self.stats['saves'],
            'evictions': self.stats['evictions'],
            'hit_rate_percent': round(hit_rate, 2),
            'temp_dir': str(self.temp_dir)
        }
    
    def clear_all(self):
        """Clear all cached data from RAM and disk."""
        # Clear RAM
        with self._cache_lock:
            self._ram_cache.clear()
            self._access_times.clear()
            self._creation_times.clear()
        
        # Clear disk
        try:
            for cache_file in self.temp_dir.glob("*.json"):
                cache_file.unlink()
            logger.info("Cleared all cache data")
        except Exception as e:
            logger.error(f"Failed to clear disk cache: {e}")


def persistent_cache(cache_instance: PersistentTokenCache, prefix: str = "token"):
    """
    Decorator to add persistent caching to any function.
    
    Args:
        cache_instance: The PersistentTokenCache instance to use
        prefix: Cache prefix for this function's cached data
    
    Returns:
        Decorated function with persistent caching
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            
            # Add args to key (convert to strings)
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
                else:
                    # For complex objects, use their string representation
                    key_parts.append(str(type(arg).__name__))
            
            # Add kwargs to key
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}:{v}")
                else:
                    key_parts.append(f"{k}:{type(v).__name__}")
            
            cache_key = "|".join(key_parts)
            
            # Try to get from cache first
            cached_result = cache_instance.get(cache_key, prefix)
            if cached_result is not None:
                return cached_result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, prefix)
            
            return result
        
        return wrapper
    return decorator


# Global cache instance
_global_cache = None

def get_persistent_cache(temp_dir: str = "temp") -> PersistentTokenCache:
    """Get the global persistent cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = PersistentTokenCache(temp_dir)
    return _global_cache 