# execution_engine/utils.py

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
            
        return result
    return wrapper

def setup_mylathdb_logging(level: str = "INFO"):
    """Setup logging for MyLathDB execution engine"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - MyLathDB - %(name)s - %(levelname)s - %(message)s'
    )
