# mylath/performance/cache.py
import time
from typing import Any, Optional, Dict
from functools import wraps
import redis


class CacheManager:
    """Intelligent caching layer for MyLath"""
    
    def __init__(self, redis_client: redis.Redis, default_ttl: int = 3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.cache_prefix = "cache:"
        
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        cache_key = f"{self.cache_prefix}{key}"
        try:
            data = self.redis.get(cache_key)
            if data:
                import json
                return json.loads(data.decode())
        except:
            pass
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set cached value"""
        cache_key = f"{self.cache_prefix}{key}"
        ttl = ttl or self.default_ttl
        try:
            import json
            self.redis.setex(cache_key, ttl, json.dumps(value))
            return True
        except:
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cached value"""
        cache_key = f"{self.cache_prefix}{key}"
        return bool(self.redis.delete(cache_key))
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern"""
        cache_pattern = f"{self.cache_prefix}{pattern}"
        keys = self.redis.keys(cache_pattern)
        if keys:
            return self.redis.delete(*keys)
        return 0


def cached(cache_manager: CacheManager, ttl: int = None, key_func=None):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator
