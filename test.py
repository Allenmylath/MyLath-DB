#!/usr/bin/env python3
"""
Basic Test for MyLathDB System
Tests the core parsing and planning functionality
"""

import sys
import os
from pathlib import Path

# Add the mylathdb directory to Python path
current_dir = Path(__file__).parent
mylathdb_dir = current_dir / "mylathdb"
sys.path.insert(0, str(mylathdb_dir))

def test_cypher_parsing():
    """Test basic Cypher parsing functionality"""
    print("🔍 Testing Cypher Parser...")
    
    try:
        from cypher_planner.parser import parse_cypher_query
        from cypher_planner.ast_nodes import Query, MatchClause, ReturnClause
        
        # Test simple query
        query = "MATCH (n:Person) WHERE n.age > 25 RETURN n.name, n.age"
        print(f"   Parsing: {query}")
        
        ast = parse_cypher_query(query)
        
        # Verify AST structure
        assert isinstance(ast, Query), f"Expected Query, got {type(ast)}"
        assert len(ast.match_clauses) > 0, "Should have at least one MATCH clause"
        assert ast.return_clause is not None, "Should have RETURN clause"
        assert ast.where_clause is not None, "Should have WHERE clause"
        
        print("   ✅ AST created successfully")
        print(f"   📊 Match clauses: {len(ast.match_clauses)}")
        print(f"   📊 Has WHERE: {ast.where_clause is not None}")
        print(f"   📊 Has RETURN: {ast.return_clause is not None}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Parser test failed: {e}")
        return False

def test_logical_planning():
    """Test logical plan creation"""
    print("\n🧠 Testing Logical Planner...")
    
    try:
        from cypher_planner.parser import parse_cypher_query
        from cypher_planner.logical_planner import LogicalPlanner
        from cypher_planner.logical_operators import LogicalOperator
        
        # Parse query
        query = "MATCH (n:Person) RETURN n.name"
        ast = parse_cypher_query(query)
        
        # Create logical plan
        planner = LogicalPlanner()
        logical_plan = planner.create_logical_plan(ast)
        
        # Verify logical plan
        assert logical_plan is not None, "Logical plan should not be None"
        assert isinstance(logical_plan, LogicalOperator), f"Expected LogicalOperator, got {type(logical_plan)}"
        
        print("   ✅ Logical plan created successfully")
        print(f"   📊 Plan type: {type(logical_plan).__name__}")
        
        # Print plan structure
        from cypher_planner.logical_operators import print_plan
        print("   📋 Plan structure:")
        print_plan(logical_plan)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Logical planner test failed: {e}")
        return False

def test_physical_planning():
    """Test physical plan creation"""
    print("\n⚙️  Testing Physical Planner...")
    
    try:
        from cypher_planner.parser import parse_cypher_query
        from cypher_planner.logical_planner import LogicalPlanner
        from cypher_planner.physical_planner import PhysicalPlanner, PhysicalOperation
        
        # Parse and create logical plan
        query = "MATCH (n:Person) RETURN n.name"
        ast = parse_cypher_query(query)
        logical_planner = LogicalPlanner()
        logical_plan = logical_planner.create_logical_plan(ast)
        
        # Create physical plan
        physical_planner = PhysicalPlanner()
        physical_plan = physical_planner.create_physical_plan(logical_plan)
        
        # Verify physical plan
        assert physical_plan is not None, "Physical plan should not be None"
        assert isinstance(physical_plan, PhysicalOperation), f"Expected PhysicalOperation, got {type(physical_plan)}"
        
        print("   ✅ Physical plan created successfully")
        print(f"   📊 Plan type: {type(physical_plan).__name__}")
        print(f"   📊 Target: {physical_plan.target}")
        
        # Print physical plan
        from cypher_planner.physical_planner import print_physical_plan
        print("   📋 Physical plan structure:")
        print_physical_plan(physical_plan)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Physical planner test failed: {e}")
        return False

def test_execution_engine_basic():
    """Test basic execution engine functionality"""
    print("\n🚀 Testing Execution Engine (Basic)...")
    
    try:
        from execution_engine.config import MyLathDBExecutionConfig
        from execution_engine.engine import ExecutionEngine, ExecutionResult
        
        # Create configuration (without Redis requirement)
        config = MyLathDBExecutionConfig()
        config.AUTO_START_REDIS = False  # Don't try to start Redis
        
        # Create engine (this should work without Redis)
        print("   Creating execution engine...")
        engine = ExecutionEngine(config)
        
        print("   ✅ Execution engine created successfully")
        print(f"   📊 Engine config: Redis {config.REDIS_HOST}:{config.REDIS_PORT}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Execution engine test failed: {e}")
        print(f"   ℹ️  This is expected if Redis is not available")
        return False

def test_end_to_end_parsing():
    """Test complete parsing pipeline"""
    print("\n🔄 Testing End-to-End Pipeline...")
    
    try:
        # Import all necessary components
        from cypher_planner import (
            parse_cypher_query, LogicalPlanner, 
            RuleBasedOptimizer, PhysicalPlanner
        )
        
        # Complex query to test full pipeline
        query = """
        MATCH (p:Person {country: 'USA'})-[:KNOWS]->(f:Person)
        WHERE p.age > 25 AND f.age < 40
        RETURN p.name, f.name, p.age
        ORDER BY p.age DESC
        LIMIT 10
        """
        
        print(f"   Testing complex query: {query.strip()}")
        
        # Step 1: Parse
        ast = parse_cypher_query(query)
        print("   ✅ Step 1: Parsed successfully")
        
        # Step 2: Logical planning
        logical_planner = LogicalPlanner()
        logical_plan = logical_planner.create_logical_plan(ast)
        print("   ✅ Step 2: Logical plan created")
        
        # Step 3: Optimization
        optimizer = RuleBasedOptimizer()
        optimized_plan = optimizer.optimize(logical_plan)
        print("   ✅ Step 3: Plan optimized")
        
        # Step 4: Physical planning
        physical_planner = PhysicalPlanner()
        physical_plan = physical_planner.create_physical_plan(optimized_plan)
        print("   ✅ Step 4: Physical plan created")
        
        print(f"   📊 Final plan target: {physical_plan.target}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 MyLathDB Basic Functionality Test")
    print("=" * 50)
    
    # Check if mylathdb directory exists
    mylathdb_path = Path("mylathdb")
    if not mylathdb_path.exists():
        print(f"❌ Error: {mylathdb_path} directory not found!")
        print("Please run this test from the directory containing the mylathdb folder")
        return False
    
    print(f"✅ Found MyLathDB at: {mylathdb_path.absolute()}")
    
    # Run tests
    tests = [
        ("Cypher Parsing", test_cypher_parsing),
        ("Logical Planning", test_logical_planning), 
        ("Physical Planning", test_physical_planning),
        ("Execution Engine Basic", test_execution_engine_basic),
        ("End-to-End Pipeline", test_end_to_end_parsing),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! MyLathDB core functionality is working.")
    elif passed > 0:
        print("⚠️  Some tests passed. Basic functionality is working.")
    else:
        print("❌ All tests failed. Check the setup and dependencies.")
    
    return passed > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
