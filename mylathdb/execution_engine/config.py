# mylathdb/execution_engine/config.py - FIXED VERSION

"""
Fixed Configuration for MyLathDB Execution Engine
Provides proper initialization support for testing
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class MyLathDBExecutionConfig:
    """
    FIXED: Configuration class with proper initialization support
    """
    
    # Redis Configuration
    REDIS_HOST: str = field(default_factory=lambda: os.getenv('MYLATH_REDIS_HOST', 'localhost'))
    REDIS_PORT: int = field(default_factory=lambda: int(os.getenv('MYLATH_REDIS_PORT', '6379')))
    REDIS_DB: int = field(default_factory=lambda: int(os.getenv('MYLATH_REDIS_DB', '0')))
    REDIS_PASSWORD: Optional[str] = field(default_factory=lambda: os.getenv('MYLATH_REDIS_PASSWORD'))
    REDIS_SOCKET_TIMEOUT: float = field(default_factory=lambda: float(os.getenv('MYLATH_REDIS_TIMEOUT', '5.0')))
    
    # Redis Behavior
    AUTO_START_REDIS: bool = field(default_factory=lambda: os.getenv('MYLATH_AUTO_START_REDIS', 'false').lower() == 'true')
    REDIS_PERSISTENCE: bool = field(default_factory=lambda: os.getenv('MYLATH_REDIS_PERSISTENCE', 'true').lower() == 'true')
    REDIS_MAX_MEMORY: Optional[str] = field(default_factory=lambda: os.getenv('MYLATH_REDIS_MAX_MEMORY'))
    
    # GraphBLAS Configuration  
    GRAPHBLAS_THREADS: int = field(default_factory=lambda: int(os.getenv('MYLATH_GRAPHBLAS_THREADS', '4')))
    GRAPHBLAS_CHUNK_SIZE: int = field(default_factory=lambda: int(os.getenv('MYLATH_GRAPHBLAS_CHUNK_SIZE', '1000')))
    GRAPHBLAS_BACKEND: str = field(default_factory=lambda: os.getenv('MYLATH_GRAPHBLAS_BACKEND', 'suitesparse'))
    
    # Execution Engine Settings
    BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv('MYLATH_BATCH_SIZE', '1000')))
    MAX_MEMORY_USAGE: str = field(default_factory=lambda: os.getenv('MYLATH_MAX_MEMORY', '2GB'))
    QUERY_TIMEOUT: float = field(default_factory=lambda: float(os.getenv('MYLATH_QUERY_TIMEOUT', '300.0')))
    
    # Logging and Debug
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv('MYLATH_LOG_LEVEL', 'INFO'))
    ENABLE_QUERY_LOGGING: bool = field(default_factory=lambda: os.getenv('MYLATH_QUERY_LOG', 'false').lower() == 'true')
    ENABLE_PERFORMANCE_METRICS: bool = field(default_factory=lambda: os.getenv('MYLATH_PERF_METRICS', 'false').lower() == 'true')
    
    # Development and Testing
    DEVELOPMENT_MODE: bool = field(default_factory=lambda: os.getenv('MYLATH_DEV_MODE', 'false').lower() == 'true')
    ENABLE_DEBUG_OPERATIONS: bool = field(default_factory=lambda: os.getenv('MYLATH_DEBUG_OPS', 'false').lower() == 'true')
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Validate Redis configuration
        if not isinstance(self.REDIS_PORT, int) or self.REDIS_PORT <= 0:
            raise ValueError(f"Invalid Redis port: {self.REDIS_PORT}")
        
        if not isinstance(self.REDIS_DB, int) or self.REDIS_DB < 0:
            raise ValueError(f"Invalid Redis database: {self.REDIS_DB}")
        
        # Validate timeout values
        if self.REDIS_SOCKET_TIMEOUT <= 0:
            raise ValueError(f"Invalid Redis timeout: {self.REDIS_SOCKET_TIMEOUT}")
        
        if self.QUERY_TIMEOUT <= 0:
            raise ValueError(f"Invalid query timeout: {self.QUERY_TIMEOUT}")
        
        # Validate thread and batch settings
        if self.GRAPHBLAS_THREADS <= 0:
            raise ValueError(f"Invalid GraphBLAS threads: {self.GRAPHBLAS_THREADS}")
        
        if self.BATCH_SIZE <= 0:
            raise ValueError(f"Invalid batch size: {self.BATCH_SIZE}")
    
    @classmethod
    def create_test_config(cls, redis_db: int = 15, **overrides) -> 'MyLathDBExecutionConfig':
        """
        Create a configuration specifically for testing
        
        Args:
            redis_db: Redis database number for testing (default 15)
            **overrides: Additional configuration overrides
            
        Returns:
            MyLathDBExecutionConfig instance for testing
        """
        config = cls(
            REDIS_HOST='localhost',
            REDIS_PORT=6379,
            REDIS_DB=redis_db,
            AUTO_START_REDIS=False,
            DEVELOPMENT_MODE=True,
            ENABLE_DEBUG_OPERATIONS=True,
            LOG_LEVEL='DEBUG',
            BATCH_SIZE=100,  # Smaller batches for testing
            QUERY_TIMEOUT=30.0,  # Shorter timeout for tests
            **overrides
        )
        return config
    
    @classmethod
    def create_production_config(cls, **overrides) -> 'MyLathDBExecutionConfig':
        """
        Create a configuration optimized for production
        
        Args:
            **overrides: Configuration overrides
            
        Returns:
            MyLathDBExecutionConfig instance for production
        """
        config = cls(
            REDIS_HOST=os.getenv('MYLATH_REDIS_HOST', 'localhost'),
            REDIS_PORT=int(os.getenv('MYLATH_REDIS_PORT', '6379')),
            REDIS_DB=int(os.getenv('MYLATH_REDIS_DB', '0')),
            AUTO_START_REDIS=False,
            DEVELOPMENT_MODE=False,
            ENABLE_DEBUG_OPERATIONS=False,
            LOG_LEVEL='INFO',
            BATCH_SIZE=10000,  # Larger batches for production
            QUERY_TIMEOUT=300.0,  # 5 minute timeout
            ENABLE_PERFORMANCE_METRICS=True,
            **overrides
        )
        return config
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        else:
            return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    def get_redis_connection_kwargs(self) -> Dict[str, Any]:
        """Get Redis connection keyword arguments"""
        kwargs = {
            'host': self.REDIS_HOST,
            'port': self.REDIS_PORT,
            'db': self.REDIS_DB,
            'socket_timeout': self.REDIS_SOCKET_TIMEOUT,
            'socket_connect_timeout': self.REDIS_SOCKET_TIMEOUT,
            'decode_responses': True,
            'retry_on_timeout': True,
            'health_check_interval': 30
        }
        
        if self.REDIS_PASSWORD:
            kwargs['password'] = self.REDIS_PASSWORD
        
        return kwargs
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary
        
        Args:
            config_dict: Dictionary of configuration values
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def __repr__(self) -> str:
        """String representation of configuration"""
        return f"MyLathDBExecutionConfig(redis={self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB})"


# Alternative simple configuration class for backward compatibility
class SimpleMyLathDBConfig:
    """
    Simple configuration class for backward compatibility
    Can be initialized with or without arguments
    """
    
    def __init__(self, **kwargs):
        """Initialize with optional keyword arguments"""
        # Default values
        self.REDIS_HOST = kwargs.get('REDIS_HOST', 'localhost')
        self.REDIS_PORT = kwargs.get('REDIS_PORT', 6379)
        self.REDIS_DB = kwargs.get('REDIS_DB', 0)
        self.AUTO_START_REDIS = kwargs.get('AUTO_START_REDIS', False)
        self.REDIS_PASSWORD = kwargs.get('REDIS_PASSWORD', None)
        self.REDIS_SOCKET_TIMEOUT = kwargs.get('REDIS_SOCKET_TIMEOUT', 5.0)
        
        # GraphBLAS settings
        self.GRAPHBLAS_THREADS = kwargs.get('GRAPHBLAS_THREADS', 4)
        self.GRAPHBLAS_CHUNK_SIZE = kwargs.get('GRAPHBLAS_CHUNK_SIZE', 1000)
        self.GRAPHBLAS_BACKEND = kwargs.get('GRAPHBLAS_BACKEND', 'suitesparse')
        
        # Execution settings
        self.BATCH_SIZE = kwargs.get('BATCH_SIZE', 1000)
        self.MAX_MEMORY_USAGE = kwargs.get('MAX_MEMORY_USAGE', '2GB')
        self.QUERY_TIMEOUT = kwargs.get('QUERY_TIMEOUT', 300.0)
        
        # Logging and debug
        self.LOG_LEVEL = kwargs.get('LOG_LEVEL', 'INFO')
        self.ENABLE_QUERY_LOGGING = kwargs.get('ENABLE_QUERY_LOGGING', False)
        self.ENABLE_PERFORMANCE_METRICS = kwargs.get('ENABLE_PERFORMANCE_METRICS', False)
        self.DEVELOPMENT_MODE = kwargs.get('DEVELOPMENT_MODE', False)
        self.ENABLE_DEBUG_OPERATIONS = kwargs.get('ENABLE_DEBUG_OPERATIONS', False)
    
    def get_redis_connection_kwargs(self):
        """Get Redis connection keyword arguments"""
        kwargs = {
            'host': self.REDIS_HOST,
            'port': self.REDIS_PORT,
            'db': self.REDIS_DB,
            'socket_timeout': self.REDIS_SOCKET_TIMEOUT,
            'socket_connect_timeout': self.REDIS_SOCKET_TIMEOUT,
            'decode_responses': True,
            'retry_on_timeout': True
        }
        
        if self.REDIS_PASSWORD:
            kwargs['password'] = self.REDIS_PASSWORD
        
        return kwargs


# Factory functions for easy configuration creation
def create_test_config(redis_db: int = 15, **kwargs) -> MyLathDBExecutionConfig:
    """Create test configuration"""
    return MyLathDBExecutionConfig.create_test_config(redis_db=redis_db, **kwargs)


def create_production_config(**kwargs) -> MyLathDBExecutionConfig:
    """Create production configuration"""
    return MyLathDBExecutionConfig.create_production_config(**kwargs)


def create_simple_config(**kwargs) -> SimpleMyLathDBConfig:
    """Create simple configuration for backward compatibility"""
    return SimpleMyLathDBConfig(**kwargs)


