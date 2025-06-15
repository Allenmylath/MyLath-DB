# execution_engine/examples/mylathdb_demo.py

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
