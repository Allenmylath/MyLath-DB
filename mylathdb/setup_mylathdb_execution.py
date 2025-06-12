# setup_mylathdb_execution.py

"""
Setup script to integrate execution engine into MyLathDB
"""

import os
import sys
from pathlib import Path
import shutil

def setup_mylathdb_execution_engine():
    """Setup execution engine for MyLathDB"""
    
    print("🚀 Setting up MyLathDB Execution Engine...")
    
    # Check if we're in the right directory
    if not Path("cypher_planner").exists():
        print("❌ Error: cypher_planner directory not found!")
        print("Please run this script from your MyLathDB root directory")
        return False
    
    # Create execution_engine directory
    execution_dir = Path("execution_engine")
    execution_dir.mkdir(exist_ok=True)
    print(f"✅ Created {execution_dir}")
    
    # Create subdirectories
    subdirs = ["tests", "examples", "docs"]
    for subdir in subdirs:
        (execution_dir / subdir).mkdir(exist_ok=True)
        print(f"✅ Created {execution_dir / subdir}")
    
    # Create __init__.py files
    init_files = [
        "execution_engine/__init__.py",
        "execution_engine/tests/__init__.py",
        "execution_engine/examples/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = Path(init_file)
        if not init_path.exists():
            init_path.write_text('"""MyLathDB Execution Engine"""\n')
            print(f"✅ Created {init_file}")
    
    # Create core module files (placeholders for now)
    core_files = {
        "execution_engine/engine.py": get_engine_template(),
        "execution_engine/redis_executor.py": get_redis_executor_template(),
        "execution_engine/graphblas_executor.py": get_graphblas_executor_template(),
        "execution_engine/coordinator.py": get_coordinator_template(),
        "execution_engine/data_bridge.py": get_data_bridge_template(),
        "execution_engine/result_formatter.py": get_result_formatter_template(),
        "execution_engine/config.py": get_config_template(),
        "execution_engine/exceptions.py": get_exceptions_template(),
        "execution_engine/utils.py": get_utils_template(),
    }
    
    for file_path, content in core_files.items():
        path = Path(file_path)
        if not path.exists():
            path.write_text(content)
            print(f"✅ Created {file_path}")
    
    # Update main package __init__.py
    update_main_init()
    
    # Update requirements.txt
    update_requirements()
    
    # Update setup.py
    update_setup_py()
    
    # Create integration examples
    create_integration_examples()
    
    # Create tests
    create_integration_tests()
    
    print("\n🎉 MyLathDB Execution Engine setup complete!")
    print("\n📝 Next steps:")
    print("1. Copy the full implementation from the artifacts")
    print("2. Install new dependencies: pip install -r requirements.txt")
    print("3. Run tests: python -m pytest execution_engine/tests/")
    print("4. Try examples: python execution_engine/examples/mylathdb_demo.py")
    
    return True

def get_engine_template():
    return '''# execution_engine/engine.py

"""
MyLathDB Execution Engine
Main execution engine for MyLathDB that executes physical plans
"""

# TODO: Copy implementation from artifacts
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ExecutionResult:
    """Result of executing a physical plan"""
    success: bool
    data: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    operations_executed: int = 0
    error: Optional[str] = None

class ExecutionEngine:
    """Main execution engine for MyLathDB"""
    
    def __init__(self, redis_client=None, enable_caching=True):
        self.redis_client = redis_client
        self.enable_caching = enable_caching
        # TODO: Initialize other components
    
    def execute(self, physical_plan, **kwargs) -> ExecutionResult:
        """Execute a physical plan"""
        # TODO: Implement execution logic
        return ExecutionResult(success=True, data=[])
'''

def get_redis_executor_template():
    return '''# execution_engine/redis_executor.py

"""
MyLathDB Redis Executor
Executes Redis operations for node and property access
"""

# TODO: Copy implementation from artifacts
'''

def get_graphblas_executor_template():
    return '''# execution_engine/graphblas_executor.py

"""
MyLathDB GraphBLAS Executor
Executes GraphBLAS operations for graph traversals
"""

# TODO: Copy implementation from artifacts
'''

def get_coordinator_template():
    return '''# execution_engine/coordinator.py

"""
MyLathDB Execution Coordinator
Coordinates execution between Redis and GraphBLAS
"""

# TODO: Copy implementation from artifacts
'''

def get_data_bridge_template():
    return '''# execution_engine/data_bridge.py

"""
MyLathDB Data Bridge
Handles data conversion between different execution contexts
"""

# TODO: Copy implementation from artifacts
'''

def get_result_formatter_template():
    return '''# execution_engine/result_formatter.py

"""
MyLathDB Result Formatter
Formats execution results for different output requirements
"""

# TODO: Copy implementation from artifacts
'''

def get_config_template():
    return '''# execution_engine/config.py

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
'''

def get_exceptions_template():
    return '''# execution_engine/exceptions.py

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
'''

def get_utils_template():
    return '''# execution_engine/utils.py

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
'''

def update_main_init():
    """Update main __init__.py to include execution engine"""
    
    init_path = Path("__init__.py")
    
    # Read existing content if file exists
    existing_content = ""
    if init_path.exists():
        existing_content = init_path.read_text()
    
    # Add execution engine imports if not already present
    execution_imports = '''
# MyLathDB Execution Engine
from .execution_engine import (
    ExecutionEngine, 
    ExecutionResult,
    execute_physical_plan,
    create_mylathdb_engine
)
'''
    
    if "execution_engine" not in existing_content:
        # Append to existing content
        new_content = existing_content + execution_imports
        init_path.write_text(new_content)
        print("✅ Updated main __init__.py")

def update_requirements():
    """Update requirements.txt with execution engine dependencies"""
    
    req_path = Path("requirements.txt")
    
    new_deps = [
        "# MyLathDB Execution Engine dependencies",
        "redis>=4.0.0",
        "numpy>=1.20.0", 
        "psutil>=5.8.0",
        "# Optional: GraphBLAS for high-performance operations",
        "# graphblas>=2022.12.0",
        ""
    ]
    
    if req_path.exists():
        existing = req_path.read_text()
        if "redis>=" not in existing:
            new_content = existing + "\n" + "\n".join(new_deps)
            req_path.write_text(new_content)
            print("✅ Updated requirements.txt")
    else:
        req_path.write_text("\n".join(new_deps))
        print("✅ Created requirements.txt")

def update_setup_py():
    """Update setup.py to include execution engine"""
    
    setup_path = Path("setup.py")
    
    if not setup_path.exists():
        # Create new setup.py
        setup_content = '''
from setuptools import setup, find_packages

setup(
    name="mylathdb",
    version="1.0.0",
    description="MyLathDB - Graph Database with Cypher Support",
    packages=find_packages(),
    install_requires=[
        "redis>=4.0.0",
        "numpy>=1.20.0",
        "psutil>=5.8.0",
    ],
    extras_require={
        "graphblas": ["graphblas>=2022.12.0"],
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.0.0"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
'''
        setup_path.write_text(setup_content)
        print("✅ Created setup.py")

def create_integration_examples():
    """Create integration examples"""
    
    examples_dir = Path("execution_engine/examples")
    
    # MyLathDB demo
    demo_content = '''# execution_engine/examples/mylathdb_demo.py

"""
MyLathDB Execution Engine Demo
Complete example showing query execution
"""

import sys
from pathlib import Path

# Add parent directory to path to import MyLathDB modules
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from cypher_planner import parse_cypher_query, LogicalPlanner, PhysicalPlanner
    from execution_engine import ExecutionEngine
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the MyLathDB root directory")
    sys.exit(1)

def mylathdb_demo():
    """Demonstrate MyLathDB query execution"""
    
    print("🗄️  MyLathDB Execution Demo")
    print("=" * 50)
    
    # Sample query
    query = "MATCH (n:Person) WHERE n.age > 25 RETURN n.name, n.age"
    print(f"Query: {query}")
    
    try:
        # Parse query
        ast = parse_cypher_query(query)
        print("✅ Query parsed successfully")
        
        # Create logical plan
        logical_planner = LogicalPlanner()
        logical_plan = logical_planner.create_logical_plan(ast)
        print("✅ Logical plan created")
        
        # Create physical plan
        physical_planner = PhysicalPlanner()
        physical_plan = physical_planner.create_physical_plan(logical_plan)
        print("✅ Physical plan created")
        
        # Execute query
        engine = ExecutionEngine()
        result = engine.execute(physical_plan)
        
        print(f"✅ Query executed: {result.success}")
        print(f"📊 Execution time: {result.execution_time:.3f}s")
        print(f"📋 Results: {len(result.data)} records")
        
        # Show results
        for i, record in enumerate(result.data[:3]):
            print(f"   {i+1}: {record}")
        
        if len(result.data) > 3:
            print(f"   ... and {len(result.data) - 3} more")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    mylathdb_demo()
'''
    
    (examples_dir / "mylathdb_demo.py").write_text(demo_content)
    print("✅ Created MyLathDB demo example")
    
    # Performance benchmark
    benchmark_content = '''# execution_engine/examples/performance_benchmark.py

"""
MyLathDB Performance Benchmark
Benchmark different query types and patterns
"""

import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

def benchmark_mylathdb():
    """Benchmark MyLathDB execution performance"""
    
    print("⚡ MyLathDB Performance Benchmark")
    print("=" * 50)
    
    test_queries = [
        ("Node Scan", "MATCH (n:Person) RETURN n"),
        ("Property Filter", "MATCH (n:Person) WHERE n.age > 30 RETURN n"),
        ("Graph Traversal", "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b"),
        ("Optional Match", "MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) RETURN p, c"),
    ]
    
    # TODO: Implement benchmarking logic
    for name, query in test_queries:
        print(f"🔄 Testing: {name}")
        print(f"   Query: {query}")
        # Benchmark execution time
        print(f"   ⏱️  Time: N/A (TODO: implement)")
        print()

if __name__ == "__main__":
    benchmark_mylathdb()
'''
    
    (examples_dir / "performance_benchmark.py").write_text(benchmark_content)
    print("✅ Created performance benchmark example")

def create_integration_tests():
    """Create integration tests"""
    
    tests_dir = Path("execution_engine/tests")
    
    test_content = '''# execution_engine/tests/test_mylathdb_integration.py

"""
MyLathDB Integration Tests
Tests for the complete execution pipeline
"""

import unittest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from cypher_planner import parse_cypher_query, LogicalPlanner, PhysicalPlanner
    from execution_engine import ExecutionEngine, ExecutionResult
except ImportError as e:
    import pytest
    pytest.skip(f"MyLathDB modules not available: {e}", allow_module_level=True)

class TestMyLathDBIntegration(unittest.TestCase):
    """Test MyLathDB execution integration"""
    
    def setUp(self):
        """Setup test environment"""
        self.engine = ExecutionEngine(redis_client=None)
    
    def test_simple_query_execution(self):
        """Test executing a simple query"""
        query = "MATCH (n:Person) RETURN n"
        
        # Parse and plan
        ast = parse_cypher_query(query)
        logical_planner = LogicalPlanner()
        logical_plan = logical_planner.create_logical_plan(ast)
        physical_planner = PhysicalPlanner()
        physical_plan = physical_planner.create_physical_plan(logical_plan)
        
        # Execute
        result = self.engine.execute(physical_plan)
        
        # Verify
        self.assertIsInstance(result, ExecutionResult)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.data, list)
    
    def test_filter_query_execution(self):
        """Test executing a query with filters"""
        query = "MATCH (n:Person) WHERE n.age > 25 RETURN n"
        
        # Parse and plan
        ast = parse_cypher_query(query)
        logical_planner = LogicalPlanner()
        logical_plan = logical_planner.create_logical_plan(ast)
        physical_planner = PhysicalPlanner()
        physical_plan = physical_planner.create_physical_plan(logical_plan)
        
        # Execute
        result = self.engine.execute(physical_plan)
        
        # Verify
        self.assertIsInstance(result, ExecutionResult)
    
    def test_engine_initialization(self):
        """Test engine initializes correctly"""
        engine = ExecutionEngine()
        self.assertIsNotNone(engine)

class TestMyLathDBConfiguration(unittest.TestCase):
    """Test MyLathDB configuration"""
    
    def test_config_loading(self):
        """Test configuration loads correctly"""
        from execution_engine.config import MyLathDBExecutionConfig
        
        config = MyLathDBExecutionConfig()
        redis_config = config.get_redis_config()
        
        self.assertIn('host', redis_config)
        self.assertIn('port', redis_config)
        self.assertIn('db', redis_config)

if __name__ == '__main__':
    unittest.main()
'''
    
    (tests_dir / "test_mylathdb_integration.py").write_text(test_content)
    print("✅ Created integration tests")

if __name__ == "__main__":
    setup_mylathdb_execution_engine()