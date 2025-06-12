# execution_engine/__init__.py

"""
MyLathDB Execution Engine
Executes physical plans from the Cypher planner using Redis + GraphBLAS
"""

# Core execution components
from .engine import ExecutionEngine, ExecutionResult, ExecutionContext
from .redis_executor import RedisExecutor, RedisResult
from .graphblas_executor import GraphBLASExecutor, GraphBLASResult
from .coordinator import ExecutionCoordinator, CoordinationResult
from .data_bridge import DataBridge, ResultSet
from .result_formatter import ResultFormatter

# Configuration and utilities
from .config import MyLathDBExecutionConfig
from .exceptions import (
    MyLathDBExecutionError, MyLathDBRedisError, 
    MyLathDBGraphBLASError, MyLathDBTimeoutError
)
from .utils import mylathdb_measure_time, setup_mylathdb_logging

# Version
__version__ = "1.0.0"

# Main exports
__all__ = [
    # === CORE ENGINE ===
    "ExecutionEngine",
    "ExecutionResult", 
    "ExecutionContext",
    
    # === EXECUTORS ===
    "RedisExecutor",
    "RedisResult",
    "GraphBLASExecutor", 
    "GraphBLASResult",
    "ExecutionCoordinator",
    "CoordinationResult",
    
    # === DATA MANAGEMENT ===
    "DataBridge",
    "ResultSet",
    "ResultFormatter",
    
    # === CONVENIENCE FUNCTIONS ===
    "execute_physical_plan",
    "create_mylathdb_engine",
    
    # === CONFIGURATION ===
    "MyLathDBExecutionConfig",
    
    # === EXCEPTIONS ===
    "MyLathDBExecutionError",
    "MyLathDBRedisError",
    "MyLathDBGraphBLASError", 
    "MyLathDBTimeoutError",
    
    # === UTILITIES ===
    "mylathdb_measure_time",
    "setup_mylathdb_logging",
    
    # === VERSION ===
    "__version__"
]


# =============================================================================
# CONVENIENCE FUNCTIONS FOR MYLATHDB
# =============================================================================

def execute_physical_plan(physical_plan, redis_client=None, graph_data=None, **kwargs):
    """
    Execute a physical plan (convenience function)
    
    Args:
        physical_plan: Physical plan from PhysicalPlanner
        redis_client: Redis client instance (optional)
        graph_data: Graph data for GraphBLAS operations (optional)
        **kwargs: Additional execution options
        
    Returns:
        ExecutionResult
        
    Example:
        >>> from cypher_planner import parse_cypher_query, LogicalPlanner, PhysicalPlanner
        >>> from execution_engine import execute_physical_plan
        >>> 
        >>> # Parse and plan
        >>> ast = parse_cypher_query("MATCH (n:Person) RETURN n")
        >>> logical_plan = LogicalPlanner().create_logical_plan(ast)
        >>> physical_plan = PhysicalPlanner().create_physical_plan(logical_plan)
        >>> 
        >>> # Execute
        >>> result = execute_physical_plan(physical_plan)
        >>> print(f"Success: {result.success}, Records: {len(result.data)}")
    """
    engine = ExecutionEngine(redis_client=redis_client)
    return engine.execute(physical_plan, graph_data=graph_data, **kwargs)

def create_mylathdb_engine(redis_host="localhost", redis_port=6379, redis_db=0, 
                          enable_caching=True, max_parallel_ops=4, debug=False, **redis_kwargs):
    """
    Create execution engine optimized for MyLathDB
    
    Args:
        redis_host: Redis server host
        redis_port: Redis server port
        redis_db: Redis database number
        enable_caching: Enable operation caching
        max_parallel_ops: Maximum parallel operations
        debug: Enable debug logging
        **redis_kwargs: Additional Redis connection options
        
    Returns:
        ExecutionEngine configured for MyLathDB
        
    Example:
        >>> engine = create_mylathdb_engine(enable_caching=True, debug=True)
        >>> result = engine.execute(physical_plan)
    """
    try:
        import redis
        
        # Create Redis connection
        redis_config = {
            'host': redis_host,
            'port': redis_port,
            'db': redis_db,
            'decode_responses': True,
            'socket_timeout': 30,
            'socket_connect_timeout': 10,
            **redis_kwargs
        }
        
        redis_client = redis.Redis(**redis_config)
        
        # Test connection
        redis_client.ping()
        
        if debug:
            print(f"✅ Connected to Redis at {redis_host}:{redis_port}/{redis_db}")
            
    except ImportError:
        if debug:
            print("⚠️  Redis package not available, creating engine without Redis")
        redis_client = None
    except Exception as e:
        if debug:
            print(f"⚠️  Redis connection failed: {e}, creating engine without Redis")
        redis_client = None
    
    # Create engine
    engine = ExecutionEngine(
        redis_client=redis_client,
        enable_caching=enable_caching,
        max_parallel_ops=max_parallel_ops,
        debug=debug
    )
    
    if debug:
        print(f"✅ MyLathDB execution engine created")
        print(f"   - Caching: {enable_caching}")
        print(f"   - Max parallel ops: {max_parallel_ops}")
        print(f"   - Redis available: {redis_client is not None}")
    
    return engine


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def integrate_with_cypher_planner():
    """
    Integration helper to ensure compatibility with cypher_planner
    
    Returns:
        Dict with integration status and available components
    """
    integration_status = {
        'cypher_planner_available': False,
        'redis_available': False,
        'graphblas_available': False,
        'components': []
    }
    
    # Check cypher_planner
    try:
        from cypher_planner import LogicalPlanner, PhysicalPlanner
        integration_status['cypher_planner_available'] = True
        integration_status['components'].append('cypher_planner')
    except ImportError:
        pass
    
    # Check Redis
    try:
        import redis
        integration_status['redis_available'] = True
        integration_status['components'].append('redis')
    except ImportError:
        pass
    
    # Check GraphBLAS
    try:
        import graphblas
        integration_status['graphblas_available'] = True
        integration_status['components'].append('graphblas')
    except ImportError:
        pass
    
    return integration_status

def validate_mylathdb_setup():
    """
    Validate MyLathDB setup and dependencies
    
    Returns:
        Tuple of (is_valid: bool, issues: List[str], recommendations: List[str])
    """
    issues = []
    recommendations = []
    
    # Check integration
    integration = integrate_with_cypher_planner()
    
    if not integration['cypher_planner_available']:
        issues.append("cypher_planner module not found")
        recommendations.append("Ensure cypher_planner is in the same project")
    
    if not integration['redis_available']:
        issues.append("Redis package not available")
        recommendations.append("Install Redis: pip install redis>=4.0.0")
    
    if not integration['graphblas_available']:
        recommendations.append("Install GraphBLAS for better performance: pip install graphblas")
    
    # Check Redis connectivity
    if integration['redis_available']:
        try:
            engine = create_mylathdb_engine(debug=False)
            if engine.redis_executor.redis is None:
                issues.append("Redis server not accessible")
                recommendations.append("Start Redis server: redis-server")
        except Exception as e:
            issues.append(f"Redis connection error: {e}")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues, recommendations


# =============================================================================
# AUTO-VALIDATION ON IMPORT
# =============================================================================

def _auto_validate_on_import():
    """Auto-validate setup when module is imported"""
    try:
        import os
        if os.getenv('MYLATHDB_VALIDATE_ON_IMPORT', 'false').lower() == 'true':
            is_valid, issues, recommendations = validate_mylathdb_setup()
            
            if not is_valid:
                print("⚠️  MyLathDB Execution Engine - Setup Issues Detected:")
                for issue in issues:
                    print(f"   ❌ {issue}")
                
                if recommendations:
                    print("\n💡 Recommendations:")
                    for rec in recommendations:
                        print(f"   ✅ {rec}")
                print()
    except Exception:
        pass  # Silent fail

# Run auto-validation
_auto_validate_on_import()


# =============================================================================
# USAGE EXAMPLES (in module docstring)
# =============================================================================

"""
USAGE EXAMPLES:

1. Basic Query Execution:
   ```python
   from mylathdb import create_database
   
   db = create_database()
   result = db.execute_query("MATCH (n:Person) RETURN n LIMIT 10")
   print(f"Found {len(result.data)} people")
   ```

2. Direct Engine Usage:
   ```python
   from execution_engine import create_mylathdb_engine, execute_physical_plan
   from cypher_planner import parse_cypher_query, LogicalPlanner, PhysicalPlanner
   
   # Create engine
   engine = create_mylathdb_engine(debug=True)
   
   # Parse and plan query
   ast = parse_cypher_query("MATCH (n:Person) WHERE n.age > 25 RETURN n")
   logical_plan = LogicalPlanner().create_logical_plan(ast)
   physical_plan = PhysicalPlanner().create_physical_plan(logical_plan)
   
   # Execute
   result = engine.execute(physical_plan)
   ```

3. With Graph Data:
   ```python
   from mylathdb import MyLathDB
   
   db = MyLathDB()
   
   # Load data
   nodes = [
       {"id": "1", "name": "Alice", "age": 30, "_labels": ["Person"]},
       {"id": "2", "name": "Bob", "age": 25, "_labels": ["Person"]}
   ]
   edges = [("1", "KNOWS", "2")]
   
   db.load_graph_data(nodes=nodes, edges=edges)
   
   # Query
   result = db.execute_query("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
   ```

4. Configuration:
   ```python
   from execution_engine.config import MyLathDBExecutionConfig
   
   config = MyLathDBExecutionConfig()
   print(f"Redis host: {config.REDIS_HOST}")
   print(f"Max execution time: {config.MAX_EXECUTION_TIME}")
   ```

5. Error Handling:
   ```python
   from mylathdb import execute_query
   from execution_engine.exceptions import MyLathDBExecutionError
   
   try:
       result = execute_query("INVALID CYPHER QUERY")
   except MyLathDBExecutionError as e:
       print(f"Execution failed: {e}")
   ```
"""