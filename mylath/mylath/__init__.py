# mylath/mylath/__init__.py
"""
MyLath - A Graph Database with Vector Search using Redis
Now with Redis native vector search for 10-100x performance boost
"""

__version__ = "0.1.0"

from .graph import Graph
from .storage import RedisStorage
from .vector import VectorCore

# Check if Redis Stack is available for high-performance vector search
def _check_redis_stack():
    """Check if Redis Stack/RediSearch is available"""
    try:
        import redis
        r = redis.Redis(decode_responses=False)
        r.execute_command("FT._LIST")
        return True
    except:
        return False

# Auto-detect Redis Stack on import
_redis_stack_available = _check_redis_stack()

if _redis_stack_available:
    print("🚀 MyLath: Redis Stack detected - high-performance vector search enabled!")
else:
    print("⚙️  MyLath: Standard mode (install Redis Stack for 10-100x vector search speedup)")
    print("   Quick install: docker run -d -p 6379:6379 redis/redis-stack-server")

__all__ = ["Graph", "RedisStorage", "VectorCore"]
