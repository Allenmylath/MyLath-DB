# mylathdb/execution_engine/exceptions.py

"""
MyLathDB Execution Engine Exceptions
"""

class MyLathDBExecutionError(Exception):
    """Base exception for MyLathDB execution errors"""
    pass

class MyLathDBRedisError(MyLathDBExecutionError):
    """Redis-related execution errors"""
    pass

class MyLathDBGraphBLASError(MyLathDBExecutionError):
    """GraphBLAS-related execution errors"""
    pass

class MyLathDBTimeoutError(MyLathDBExecutionError):
    """Execution timeout errors"""
    pass

class MyLathDBDataError(MyLathDBExecutionError):
    """Data-related execution errors"""
    pass
