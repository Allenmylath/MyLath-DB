#!/usr/bin/env python3
"""
MyLathDB Debug and Fix Script
Diagnoses data loading and query execution issues
"""

import sys
from pathlib import Path

# Add the mylathdb directory to Python path
current_dir = Path(__file__).parent
mylathdb_dir = current_dir / "mylathdb"
sys.path.insert(0, str(mylathdb_dir))

def debug_data_loading():
    """Debug the data loading process"""
    print("🔍 Debugging Data Loading Process...")
    
    try:
        from mylathdb import MyLathDB
        
        # Create database
        db = MyLathDB(auto_start_redis=False)
        print("   ✅ MyLathDB instance created")
        
        # Test data
        test_nodes = [
            {"id": "1", "name": "Alice", "age": 30, "country": "USA", "_labels": ["Person"]},
            {"id": "2", "name": "Bob", "age": 25, "country": "USA", "_labels": ["Person"]},
        ]
        
        test_edges = [
            ("1", "KNOWS", "2"),
        ]
        
        print(f"   📝 Loading {len(test_nodes)} nodes and {len(test_edges)} edges...")
        
        # Load data
        db.load_graph_data(nodes=test_nodes, edges=test_edges)
        print("   ✅ Data loading completed")
        
        # Check Redis data directly
        print("\n   🔍 Checking Redis data storage...")
        
        if db.engine.redis_executor.redis:
            redis_client = db.engine.redis_executor.redis
            
            # Check if nodes were stored
            print("   🔍 Checking stored nodes:")
            node_keys = list(redis_client.scan_iter(match="node:*"))
            print(f"      Found {len(node_keys)} node keys: {node_keys}")
            
            for node_key in node_keys[:3]:  # Check first 3
                node_data = redis_client.hgetall(node_key)
                print(f"      {node_key}: {node_data}")
            
            # Check label indexes
            print("   🔍 Checking label indexes:")
            label_keys = list(redis_client.scan_iter(match="label:*"))
            print(f"      Found {len(label_keys)} label keys: {label_keys}")
            
            for label_key in label_keys:
                members = redis_client.smembers(label_key)
                print(f"      {label_key}: {list(members)}")
            
            # Check property indexes
            print("   🔍 Checking property indexes:")
            prop_keys = list(redis_client.scan_iter(match="prop:*"))
            print(f"      Found {len(prop_keys)} property keys: {prop_keys[:5]}...")
            
            for prop_key in prop_keys[:3]:
                members = redis_client.smembers(prop_key)
                print(f"      {prop_key}: {list(members)}")
        else:
            print("   ⚠️ Redis not connected - cannot check data storage")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data loading debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_simple_query():
    """Debug a simple query step by step"""
    print("\n🔍 Debugging Simple Query Execution...")
    
    try:
        from mylathdb import MyLathDB
        from mylathdb.cypher_planner import parse_cypher_query, LogicalPlanner, PhysicalPlanner
        
        # Setup
        db = MyLathDB(auto_start_redis=False)
        
        # Load test data
        test_nodes = [
            {"id": "1", "name": "Alice", "age": 30, "_labels": ["Person"]},
            {"id": "2", "name": "Bob", "age": 25, "_labels": ["Person"]},
        ]
        db.load_graph_data(nodes=test_nodes)
        
        # Simple query
        query = "MATCH (n:Person) RETURN n.name"
        print(f"   🧪 Testing query: {query}")
        
        # Step 1: Parse query
        print("   🔧 Step 1: Parsing query...")
        try:
            ast = parse_cypher_query(query)
            print(f"      ✅ AST created: {type(ast).__name__}")
            print(f"      📋 Match clauses: {len(ast.match_clauses)}")
            print(f"      📋 Return clause: {ast.return_clause is not None}")
        except Exception as e:
            print(f"      ❌ Parsing failed: {e}")
            return False
        
        # Step 2: Create logical plan
        print("   🔧 Step 2: Creating logical plan...")
        try:
            logical_planner = LogicalPlanner()
            logical_plan = logical_planner.create_logical_plan(ast)
            print(f"      ✅ Logical plan created: {type(logical_plan).__name__}")
            
            # Print plan structure
            def print_plan_debug(op, indent=0):
                prefix = "  " * indent
                print(f"      {prefix}- {type(op).__name__}")
                if hasattr(op, 'variable'):
                    print(f"      {prefix}  Variable: {op.variable}")
                if hasattr(op, 'labels'):
                    print(f"      {prefix}  Labels: {op.labels}")
                for child in getattr(op, 'children', []):
                    print_plan_debug(child, indent + 1)
            
            print_plan_debug(logical_plan)
            
        except Exception as e:
            print(f"      ❌ Logical planning failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 3: Create physical plan
        print("   🔧 Step 3: Creating physical plan...")
        try:
            physical_planner = PhysicalPlanner()
            physical_plan = physical_planner.create_physical_plan(logical_plan)
            print(f"      ✅ Physical plan created: {type(physical_plan).__name__}")
            print(f"      📋 Target: {getattr(physical_plan, 'target', 'unknown')}")
            print(f"      📋 Operation: {getattr(physical_plan, 'operation_type', 'unknown')}")
        except Exception as e:
            print(f"      ❌ Physical planning failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 4: Execute query
        print("   🔧 Step 4: Executing query...")
        try:
            result = db.execute_query(query)
            print(f"      ✅ Execution completed")
            print(f"      📊 Success: {result.success}")
            print(f"      📊 Records: {len(result.data)}")
            print(f"      📊 Time: {result.execution_time:.3f}s")
            
            if result.error:
                print(f"      ⚠️ Error: {result.error}")
            
            if result.data:
                print(f"      📋 Sample results:")
                for i, record in enumerate(result.data[:3]):
                    print(f"         {i+1}: {record}")
            else:
                print(f"      ⚠️ No results returned")
                
        except Exception as e:
            print(f"      ❌ Execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Query debug failed: {e}")
        return False

def debug_redis_executor():
    """Debug Redis executor specifically"""
    print("\n🔍 Debugging Redis Executor...")
    
    try:
        from mylathdb.execution_engine.config import MyLathDBExecutionConfig
        from mylathdb.execution_engine.redis_executor import RedisExecutor
        
        # Create Redis executor
        config = MyLathDBExecutionConfig()
        config.AUTO_START_REDIS = False
        redis_executor = RedisExecutor(config)
        
        print("   🔧 Initializing Redis executor...")
        try:
            redis_executor.initialize()
            print("   ✅ Redis executor initialized")
        except Exception as e:
            print(f"   ⚠️ Redis initialization warning: {e}")
            print("   ℹ️ Continuing with limited testing...")
        
        # Test Redis status
        status = redis_executor.get_status()
        print(f"   📊 Redis Status:")
        print(f"      Connected: {status.get('connected', False)}")
        print(f"      Host: {status.get('host', 'unknown')}")
        print(f"      Port: {status.get('port', 'unknown')}")
        
        if status.get('connected'):
            print("   🧪 Testing Redis operations...")
            
            # Test basic Redis operations
            test_data = {"name": "TestNode", "age": "25"}
            key = "test:node:1"
            
            try:
                redis_executor.redis.hset(key, mapping=test_data)
                retrieved = redis_executor.redis.hgetall(key)
                print(f"      ✅ Redis read/write test: {retrieved}")
                
                # Cleanup
                redis_executor.redis.delete(key)
                
            except Exception as e:
                print(f"      ❌ Redis operation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Redis executor debug failed: {e}")
        return False

def debug_data_retrieval():
    """Debug data retrieval specifically"""
    print("\n🔍 Debugging Data Retrieval...")
    
    try:
        from mylathdb import MyLathDB
        
        # Create database
        db = MyLathDB(auto_start_redis=False)
        
        # Load minimal test data
        test_nodes = [
            {"id": "1", "name": "Alice", "_labels": ["Person"]},
        ]
        
        print("   📝 Loading test node...")
        db.load_graph_data(nodes=test_nodes)
        
        # Test direct Redis access
        if db.engine.redis_executor.redis:
            redis_client = db.engine.redis_executor.redis
            
            print("   🔍 Direct Redis check:")
            
            # Check if node exists
            node_key = "node:1"
            node_data = redis_client.hgetall(node_key)
            print(f"      Node data: {node_data}")
            
            # Check label index
            label_key = "label:Person"
            label_members = redis_client.smembers(label_key)
            print(f"      Label Person members: {list(label_members)}")
            
            # Test Redis executor operation
            print("   🧪 Testing Redis executor operations...")
            
            # Create a simple logical operation for testing
            class MockLogicalOp:
                def __init__(self):
                    self.variable = "n"
                    self.labels = ["Person"]
                    self.properties = {}
            
            class MockContext:
                def __init__(self):
                    self.parameters = {}
            
            mock_op = MockLogicalOp()
            mock_context = MockContext()
            
            try:
                # Test node scan directly
                results = db.engine.redis_executor._execute_node_scan_fixed(mock_op, mock_context)
                print(f"      ✅ Direct node scan returned: {len(results)} results")
                
                if results:
                    print(f"      📋 Sample result: {results[0]}")
                else:
                    print("      ⚠️ No results from direct scan")
                
            except Exception as e:
                print(f"      ❌ Direct scan failed: {e}")
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data retrieval debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_property_expression_error():
    """Debug the PropertyExpression error specifically"""
    print("\n🔍 Debugging PropertyExpression Error...")
    
    try:
        from mylathdb.cypher_planner import parse_cypher_query
        
        # Test the problematic query
        problematic_query = "MATCH (p1:Person)-[:WORKS_AT]->(c:Company)<-[:WORKS_AT]-(p2:Person) WHERE p1.name < p2.name RETURN p1.name, p2.name, c.name"
        
        print(f"   🧪 Parsing problematic query...")
        print(f"      Query: {problematic_query}")
        
        try:
            ast = parse_cypher_query(problematic_query)
            print("   ✅ Query parsed successfully")
            
            # Check WHERE clause
            if ast.where_clause:
                print(f"   🔍 WHERE clause condition: {ast.where_clause.condition}")
                print(f"   🔍 Condition type: {type(ast.where_clause.condition).__name__}")
                
                # Check if it's a binary expression
                if hasattr(ast.where_clause.condition, 'left'):
                    left = ast.where_clause.condition.left
                    print(f"   🔍 Left side: {left} (type: {type(left).__name__})")
                    
                    # Check if left side has required attributes
                    if hasattr(left, 'variable'):
                        print(f"      Variable: {left.variable}")
                    if hasattr(left, 'property_name'):
                        print(f"      Property: {left.property_name}")
                    if hasattr(left, 'value'):
                        print(f"      Value: {left.value}")
                    else:
                        print(f"      ⚠️ Missing 'value' attribute!")
                
                if hasattr(ast.where_clause.condition, 'right'):
                    right = ast.where_clause.condition.right
                    print(f"   🔍 Right side: {right} (type: {type(right).__name__})")
                    
                    if hasattr(right, 'variable'):
                        print(f"      Variable: {right.variable}")
                    if hasattr(right, 'property_name'):
                        print(f"      Property: {right.property_name}")
                    if hasattr(right, 'value'):
                        print(f"      Value: {right.value}")
                    else:
                        print(f"      ⚠️ Missing 'value' attribute!")
            
        except Exception as e:
            print(f"   ❌ Query parsing failed: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"   ❌ PropertyExpression debug failed: {e}")
        return False

def suggest_fixes():
    """Suggest fixes based on debug results"""
    print("\n💡 Suggested Fixes:")
    print("=" * 50)
    
    print("\n1. 🔧 **Data Loading Issue:**")
    print("   - Data may not be stored correctly in Redis")
    print("   - Check Redis connection and storage methods")
    print("   - Verify data format and indexing")
    
    print("\n2. 🔧 **Query Execution Routing:**")
    print("   - Queries may not be routed to correct executor")
    print("   - Check physical plan generation")
    print("   - Verify Redis executor operations")
    
    print("\n3. 🔧 **PropertyExpression Error:**")
    print("   - Fix missing 'value' attribute in PropertyExpression")
    print("   - Update property filter handling")
    print("   - Check AST node structure")
    
    print("\n4. 🔧 **Coordination Issues:**")
    print("   - Improve Redis/GraphBLAS coordination")
    print("   - Fix data bridge synchronization")
    print("   - Check result formatting")

def main():
    """Run comprehensive debug analysis"""
    print("🔍 MyLathDB Debug and Analysis")
    print("=" * 50)
    
    debug_results = {}
    
    # Run debug tests
    debug_results['data_loading'] = debug_data_loading()
    debug_results['simple_query'] = debug_simple_query()
    debug_results['redis_executor'] = debug_redis_executor()
    debug_results['data_retrieval'] = debug_data_retrieval()
    debug_results['property_error'] = debug_property_expression_error()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Debug Results Summary:")
    
    for test_name, result in debug_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")
    
    passed = sum(1 for r in debug_results.values() if r)
    total = len(debug_results)
    
    print(f"\n🎯 Debug Success Rate: {passed}/{total} ({(passed/total)*100:.1f}%)")
    
    # Suggest fixes
    suggest_fixes()
    
    return passed >= total * 0.6  # 60% pass rate for debugging

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)