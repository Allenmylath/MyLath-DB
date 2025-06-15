# mylathdb/execution_engine/config.py

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
    AUTO_START_REDIS = os.getenv('MYLATHDB_AUTO_START_REDIS', 'true').lower() == 'true'
    
    # Execution settings
    MAX_EXECUTION_TIME = float(os.getenv('MYLATHDB_MAX_EXECUTION_TIME', 300.0))
    MAX_PARALLEL_OPERATIONS = int(os.getenv('MYLATHDB_MAX_PARALLEL_OPS', 4))
    ENABLE_CACHING = os.getenv('MYLATHDB_ENABLE_CACHING', 'true').lower() == 'true'
    
    # GraphBLAS settings
    GRAPHBLAS_THREADS = int(os.getenv('MYLATHDB_GRAPHBLAS_THREADS', 4))
    ENABLE_MATRIX_PERSISTENCE = os.getenv('MYLATHDB_MATRIX_PERSISTENCE', 'false').lower() == 'true'
    MATRIX_PERSISTENCE_PATH = os.getenv('MYLATHDB_MATRIX_PERSISTENCE_PATH', './mylathdb_matrices')
    
    # Performance settings
    NODE_CREATION_BUFFER = int(os.getenv('MYLATHDB_NODE_BUFFER', 10000))
    EDGE_CREATION_BUFFER = int(os.getenv('MYLATHDB_EDGE_BUFFER', 50000))
    BATCH_SIZE = int(os.getenv('MYLATHDB_BATCH_SIZE', 1000))
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration for MyLathDB"""
        return {
            'host': cls.REDIS_HOST,
            'port': cls.REDIS_PORT,
            'db': cls.REDIS_DB,
            'decode_responses': True,
            'retry_on_timeout': True
        }
