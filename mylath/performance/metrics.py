# mylath/performance/metrics.py
import time
from typing import Dict, Any
from functools import wraps
import redis


class PerformanceMetrics:
    """Performance monitoring and metrics collection"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.metrics_prefix = "metrics:"
    
    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record operation metrics"""
        timestamp = int(time.time())
        
        # Record operation count
        self.redis.incr(f"{self.metrics_prefix}ops:{operation}:count")
        
        # Record duration
        self.redis.lpush(f"{self.metrics_prefix}ops:{operation}:durations", duration)
        self.redis.ltrim(f"{self.metrics_prefix}ops:{operation}:durations", 0, 999)  # Keep last 1000
        
        # Record success/failure
        if success:
            self.redis.incr(f"{self.metrics_prefix}ops:{operation}:success")
        else:
            self.redis.incr(f"{self.metrics_prefix}ops:{operation}:errors")
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for an operation"""
        count = int(self.redis.get(f"{self.metrics_prefix}ops:{operation}:count") or 0)
        success = int(self.redis.get(f"{self.metrics_prefix}ops:{operation}:success") or 0)
        errors = int(self.redis.get(f"{self.metrics_prefix}ops:{operation}:errors") or 0)
        
        durations = self.redis.lrange(f"{self.metrics_prefix}ops:{operation}:durations", 0, -1)
        durations = [float(d) for d in durations]
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "operation": operation,
            "total_count": count,
            "success_count": success,
            "error_count": errors,
            "success_rate": success / count if count > 0 else 0,
            "average_duration": avg_duration,
            "recent_durations": durations[:10]  # Last 10
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all operation statistics"""
        pattern = f"{self.metrics_prefix}ops:*:count"
        keys = self.redis.keys(pattern)
        
        operations = set()
        for key in keys:
            operation = key.decode().split(':')[2]  # Extract operation name
            operations.add(operation)
        
        stats = {}
        for operation in operations:
            stats[operation] = self.get_operation_stats(operation)
        
        return stats


def monitor_performance(metrics: PerformanceMetrics, operation_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                metrics.record_operation(operation_name, duration, success)
        return wrapper
    return decorator
