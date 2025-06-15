# mylathdb/execution_engine/utils.py

"""
MyLathDB Execution Engine Utilities
"""

import time
import logging
from functools import wraps
from typing import Any, Dict

def mylathdb_measure_time(func):
    """Decorator to measure execution time for MyLathDB operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        if hasattr(result, 'execution_time'):
            result.execution_time = execution_time
        
        # Log slow operations
        if execution_time > 1.0:  # Log operations taking more than 1 second
            logging.getLogger(__name__).warning(
                f"Slow operation: {func.__name__} took {execution_time:.3f}s"
            )
            
        return result
    return wrapper

def setup_mylathdb_logging(level: str = "INFO"):
    """Setup logging for MyLathDB execution engine"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - MyLathDB - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set specific loggers
    logging.getLogger('mylathdb.execution_engine').setLevel(level.upper())
    
    # Suppress noisy third-party loggers
    logging.getLogger('redis').setLevel(logging.WARNING)
    logging.getLogger('graphblas').setLevel(logging.WARNING)

def format_execution_time(seconds: float) -> str:
    """Format execution time in human-readable format"""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.2f}s"

def extract_node_id(node_data: Any) -> str:
    """Extract node ID from various node data formats"""
    if isinstance(node_data, dict):
        return str(node_data.get('_id') or node_data.get('id', ''))
    elif isinstance(node_data, str):
        return node_data
    else:
        return str(node_data)

def safe_get_nested(data: Dict[str, Any], path: str, default=None):
    """Safely get nested dictionary value using dot notation"""
    try:
        value = data
        for key in path.split('.'):
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
                    joined_results
