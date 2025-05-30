# mylath/config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class MyLathConfig:
    """Configuration for MyLath database"""
    redis_host: str = 'localhost'
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Vector index configuration
    vector_m: int = 16
    vector_ef_construction: int = 128
    vector_ef_search: int = 768
    
    # API configuration
    api_host: str = 'localhost'
    api_port: int = 5000
    api_debug: bool = False
    
    # Performance settings
    connection_pool_size: int = 10
    query_timeout: int = 30
