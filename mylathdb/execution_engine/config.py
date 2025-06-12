# execution_engine/config.py

"""
MyLathDB Execution Engine Configuration
"""

import os
from typing import Dict, Any

class MyLathDBExecutionConfig:
    """Configuration for MyLathDB execution engine"""
    
    # Redis settings
    REDIS_HOST = os.getenv('MYLATHDB_REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('MYLATHDB_REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('MYLATHDB_REDIS_DB', 0))
    
    # Execution settings
    MAX_EXECUTION_TIME = float(os.getenv('MYLATHDB_MAX_EXECUTION_TIME', 300.0))
    MAX_PARALLEL_OPERATIONS = int(os.getenv('MYLATHDB_MAX_PARALLEL_OPS', 4))
    ENABLE_CACHING = os.getenv('MYLATHDB_ENABLE_CACHING', 'true').lower() == 'true'
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration for MyLathDB"""
        return {
            'host': cls.REDIS_HOST,
            'port': cls.REDIS_PORT,
            'db': cls.REDIS_DB,
            'decode_responses': True
        }
