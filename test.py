#!/usr/bin/env python3
"""
Simple One-at-a-Time Cypher Test
Test individual Cypher queries step by step
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import redis
    from mylathdb.execution_engine.redis_executor import RedisExecutor
    from mylathdb.execution_engine.config import MyLathDBExecutionConfig
    from mylathdb.execution_engine.engine import ExecutionContext
    from mylathdb.cypher_planner import parse_cypher_query, LogicalPlanner, PhysicalPlanner
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class SimpleCypherTest:
    def __init__(self):
        # Setup
        self.config = MyLathDBExecutionConfig()
        self.config.REDIS_HOST = 'localhost'
        self.config.REDIS_PORT = 6379
        self.config.REDIS_DB = 15
        self.executor = RedisExecutor(self.config)
        self.executor.initialize()
        self.executor.redis.flushdb()
        
        self.logical_planner = LogicalPlanner()
        self.physical_planner = PhysicalPlanner()
    
    def load_simple_data(self):
        """Load simple test data - NO BOOLEANS"""
        nodes = [
            {'id': '1', 'name': 'Alice', 'age': 30, 'department': 'Engineering', '_labels': ['Person']},
            {'id': '2', 'name': 'Bob', 'age': 25, 'department': 'Marketing', '_labels': ['Person']},
            {'id': '3', 'name': 'Charlie', 'age': 35, 'department': 'Engineering', '_labels': ['Person']}
        ]
        self.executor.load_nodes(nodes)
        print(f"✅ Loaded {len(nodes)} simple nodes")
    
    def execute_query(self, query):
        """Execute a Cypher query"""
        print(f"\n🔍 Query: {query}")
        try:
            ast = parse_cypher_query(query)
            logical_plan = self.logical_planner.create_logical_plan(ast)
            physical_plan = self.physical_planner.create_physical_plan(logical_plan)
            context = ExecutionContext()
            results = self.executor.execute_operation(physical_plan, context)
            
            print(f"✅ Results: {len(results)} rows")
            for i, result in enumerate(results[:3]):  # Show max 3 results
                print(f"   {i+1}: {result}")
            if len(results) > 3:
                print(f"   ... and {len(results)-3} more")
            return results
        except Exception as e:
            print(f"❌ Error: {e}")
            return None


def test_basic_queries():
    """Test 1: Basic queries"""
    print("🧪 TEST 1: Basic Queries")
    test = SimpleCypherTest()
    test.load_simple_data()
    
    # Query 1: Simple match
    test.execute_query("MATCH (p:Person) RETURN p.name")
    
    # Query 2: With property filter
    test.execute_query("MATCH (p:Person) WHERE p.department = 'Engineering' RETURN p.name")
    
    # Query 3: Age filter
    test.execute_query("MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age")


def test_range_queries():
    """Test 2: Range queries"""
    print("\n🧪 TEST 2: Range Queries")
    test = SimpleCypherTest()
    
    # Load data with salaries
    nodes = [
        {'id': '1', 'name': 'Alice', 'salary': 80000, '_labels': ['Person']},
        {'id': '2', 'name': 'Bob', 'salary': 120000, '_labels': ['Person']},
        {'id': '3', 'name': 'Charlie', 'salary': 95000, '_labels': ['Person']}
    ]
    test.executor.load_nodes(nodes)
    print(f"✅ Loaded {len(nodes)} salary nodes")
    
    # High earners
    test.execute_query("MATCH (p:Person) WHERE p.salary > 100000 RETURN p.name, p.salary")
    
    # Mid range
    test.execute_query("MATCH (p:Person) WHERE p.salary >= 80000 AND p.salary <= 100000 RETURN p.name")


def test_string_queries():
    """Test 3: String operations"""
    print("\n🧪 TEST 3: String Operations")
    test = SimpleCypherTest()
    
    # Load data with roles
    nodes = [
        {'id': '1', 'name': 'Alice', 'role': 'Senior Developer', '_labels': ['Person']},
        {'id': '2', 'name': 'Bob', 'role': 'Product Manager', '_labels': ['Person']},
        {'id': '3', 'name': 'Charlie', 'role': 'Engineering Manager', '_labels': ['Person']}
    ]
    test.executor.load_nodes(nodes)
    print(f"✅ Loaded {len(nodes)} role nodes")
    
    # Contains
    test.execute_query("MATCH (p:Person) WHERE p.role CONTAINS 'Manager' RETURN p.name, p.role")
    
    # Starts with
    test.execute_query("MATCH (p:Person) WHERE p.role STARTS WITH 'Senior' RETURN p.name")


def test_projection():
    """Test 4: Projections and aliases"""
    print("\n🧪 TEST 4: Projections")
    test = SimpleCypherTest()
    test.load_simple_data()
    
    # With aliases
    test.execute_query("MATCH (p:Person) RETURN p.name AS person_name, p.age AS person_age")
    
    # Multiple properties
    test.execute_query("MATCH (p:Person) WHERE p.department = 'Engineering' RETURN p.name, p.age, p.department")


def test_ordering():
    """Test 5: Ordering"""
    print("\n🧪 TEST 5: Ordering")
    test = SimpleCypherTest()
    test.load_simple_data()
    
    # Order by age
    test.execute_query("MATCH (p:Person) RETURN p.name, p.age ORDER BY p.age")
    
    # Limit
    test.execute_query("MATCH (p:Person) RETURN p.name ORDER BY p.name LIMIT 2")


if __name__ == "__main__":
    print("🧪 Simple Cypher Tests - One at a Time")
    print("=" * 50)
    
    # Check Redis
    try:
        r = redis.Redis(host='localhost', port=6379, db=15)
        r.ping()
        print("✅ Redis connected")
    except:
        print("❌ Redis not available")
        sys.exit(1)
    
    # Run tests one by one
    try:
        test_basic_queries()
        
        input("\nPress Enter to continue to Range Queries...")
        test_range_queries()
        
        input("\nPress Enter to continue to String Operations...")
        test_string_queries()
        
        input("\nPress Enter to continue to Projections...")
        test_projection()
        
        input("\nPress Enter to continue to Ordering...")
        test_ordering()
        
        print("\n🎉 All simple tests completed!")
        
    except KeyboardInterrupt:
        print("\n👋 Tests stopped by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")