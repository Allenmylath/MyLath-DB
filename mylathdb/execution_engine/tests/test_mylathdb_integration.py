# execution_engine/tests/test_mylathdb_integration.py

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
