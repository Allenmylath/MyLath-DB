# mylathdb/execution_engine/__init__.py

"""
MyLathDB Execution Engine
Real execution engine using Python GraphBLAS + Redis, based on FalkorDB architecture
"""

from .engine import ExecutionEngine, ExecutionResult, ExecutionContext
from .redis_executor import RedisExecutor
from .graphblas_executor import GraphBLASExecutor, GraphBLASGraph
from .coordinator import ExecutionCoordinator
from .data_bridge import DataBridge, EntityManager
from .result_formatter import ResultFormatter, ResultSet
from .config import MyLathDBExecutionConfig
from .exceptions import MyLathDBExecutionError
from .utils import setup_mylathdb_logging

# Main API exports
__all__ = [
    # Core Engine
    "ExecutionEngine",
    "ExecutionResult", 
    "ExecutionContext",
    
    # Executors
    "RedisExecutor",
    "GraphBLASExecutor",
    "GraphBLASGraph",
    "ExecutionCoordinator",
    
    # Data Management
    "DataBridge",
    "EntityManager",
    "ResultFormatter",
    "ResultSet",
    
    # Configuration and Exceptions
    "MyLathDBExecutionConfig",
    "MyLathDBExecutionError",
    
    # Utilities
    "setup_mylathdb_logging",
    
    # Convenience Functions
    "execute_physical_plan",
    "create_mylathdb_engine",
]

def execute_physical_plan(physical_plan, parameters=None, graph_data=None, **kwargs):
    """
    Convenience function to execute a physical plan
    
    Args:
        physical_plan: Physical plan from PhysicalPlanner
        parameters: Query parameters
        graph_data: Optional graph data for GraphBLAS
        **kwargs: Additional execution options
        
    Returns:
        ExecutionResult
    """
    engine = create_mylathdb_engine(**kwargs)
    return engine.execute(physical_plan, parameters=parameters, graph_data=graph_data)

def create_mylathdb_engine(redis_host="localhost", redis_port=6379, redis_db=0,
                          enable_caching=True, auto_start_redis=True, **kwargs):
    """
    Create and configure MyLathDB execution engine
    
    Args:
        redis_host: Redis server host
        redis_port: Redis server port
        redis_db: Redis database number
        enable_caching: Enable result caching
        auto_start_redis: Auto-start local Redis if needed
        **kwargs: Additional configuration
        
    Returns:
        ExecutionEngine instance
    """
    config = MyLathDBExecutionConfig()
    config.REDIS_HOST = redis_host
    config.REDIS_PORT = redis_port  
    config.REDIS_DB = redis_db
    config.ENABLE_CACHING = enable_caching
    config.AUTO_START_REDIS = auto_start_redis
    
    # Update with any additional config
    for key, value in kwargs.items():
        if hasattr(config, key.upper()):
            setattr(config, key.upper(), value)
    
    return ExecutionEngine(config)
