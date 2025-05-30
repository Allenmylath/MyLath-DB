# mylath/performance/connection_pool.py
import redis.connection
from typing import Optional


class OptimizedRedisStorage:
    """Redis storage with connection pooling and optimizations"""
    
    def __init__(self, host='localhost', port=6379, db=0, 
                 max_connections=50, socket_keepalive=True, 
                 socket_keepalive_options=None, **kwargs):
        
        # Connection pool configuration
        pool_kwargs = {
            'host': host,
            'port': port,
            'db': db,
            'max_connections': max_connections,
            'socket_keepalive': socket_keepalive,
            'socket_keepalive_options': socket_keepalive_options or {},
            'retry_on_timeout': True,
            'health_check_interval': 30,
        }
        pool_kwargs.update(kwargs)
        
        self.pool = redis.ConnectionPool(**pool_kwargs)
        self.redis = redis.Redis(connection_pool=self.pool)
        
        # Initialize cache and metrics
        from .cache import CacheManager
        from .metrics import PerformanceMetrics
        
        self.cache = CacheManager(self.redis)
        self.metrics = PerformanceMetrics(self.redis)
