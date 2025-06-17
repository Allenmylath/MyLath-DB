# filter_execution_tracer.py - Trace exactly where filters are lost

"""
Filter Execution Tracer
Detailed tracing to understand exactly where in the execution chain filters are being bypassed
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def trace_filter_execution():
    """Trace filter execution step by step"""
    print("🔬 === FILTER EXECUTION TRACER ===\n")
    
    from mylathdb import MyLathDB
    
    # Setup minimal test case
    db = MyLathDB()
    if db.engine.redis_executor.redis:
        db.engine.redis_executor.redis.flushdb()
    
    # Load exactly 2 nodes for crystal clear testing
    nodes = [
        {'id': '1', 'name': 'Alice', 'age': 30, '_labels': ['Person']},
        {'id': '2', 'name': 'Bob', 'age': 25, '_labels': ['Person']},
    ]
    db.load_graph_data(nodes=nodes)
    
    print("✅ Test data loaded:")
    print("   Node 1: Alice, age=30")
    print("   Node 2: Bob, age=25")
    print()
    
    # Test the problematic query
    query = "MATCH (n:Person) WHERE n.name = 'Alice' RETURN n.name"
    print(f"🧪 Testing: {query}")
    print("Expected: 1 result (Alice only)")
    print()
    
    # STEP 1: Verify Redis has the right data
    print("1. 📊 Redis Data Verification:")
    redis = db.engine.redis_executor.redis
    print(f"   All Person nodes: {list(redis.smembers('label:Person'))}")
    print(f"   Nodes with name=Alice: {list(redis.smembers('prop:name:Alice'))}")
    print(f"   Nodes with name=Bob: {list(redis.smembers('prop:name:Bob'))}")
    print()
    
    # STEP 2: Parse and analyze the plan structure
    print("2. 🧠 Plan Structure Analysis:")
    from mylathdb.cypher_planner import parse_cypher_query, LogicalPlanner, PhysicalPlanner
    
    ast = parse_cypher_query(query)
    logical_planner = LogicalPlanner()
    logical_plan = logical_planner.create_logical_plan(ast)
    physical_planner = PhysicalPlanner()
    physical_plan = physical_planner.create_physical_plan(logical_plan)
    
    print("   Logical Plan Structure:")
    print_plan_structure(logical_plan, "     ")
    
    print("   Physical Plan Structure:")
    print_plan_structure(physical_plan, "     ", is_physical=True)
    print()
    
    # STEP 3: Manual execution tracing
    print("3. 🔧 Manual Execution Tracing:")
    
    # Get Redis executor
    redis_executor = db.engine.redis_executor
    
    # Step 3a: Test NodeByLabelScan alone
    print("   3a. Testing NodeByLabelScan alone:")
    from mylathdb.execution_engine.engine import ExecutionContext
    context = ExecutionContext()
    
    # Find the NodeByLabelScan operation
    scan_op = find_operation_by_type(physical_plan, "NodeByLabelScan")
    if scan_op and scan_op.logical_op:
        scan_results = redis_executor._execute_node_by_label_scan_fixed(scan_op.logical_op, context)
        print(f"       Scan results: {len(scan_results)} records")
        for i, result in enumerate(scan_results):
            entity = result.get('n', {})
            print(f"         {i+1}: {entity.get('name', 'Unknown')} (age={entity.get('age', 'Unknown')})")
    print()
    
    # Step 3b: Test PropertyFilter if it exists
    print("   3b. Testing PropertyFilter:")
    filter_op = find_operation_by_type(physical_plan, "PropertyFilter")
    if filter_op and filter_op.logical_op:
        print(f"       Found PropertyFilter: {filter_op.logical_op.property_key} {filter_op.logical_op.operator} {filter_op.logical_op.value}")
        
        # Try to execute the filter using the scan results as input
        print("       Attempting filter execution with scan results as input...")
        try:
            # This is where we'll see if the filter logic works
            filter_results = redis_executor._execute_property_filter_fixed(filter_op.logical_op, context)
            print(f"       Filter results: {len(filter_results)} records")
            for i, result in enumerate(filter_results):
                entity = result.get('n', {})
                print(f"         {i+1}: {entity.get('name', 'Unknown')} (age={entity.get('age', 'Unknown')})")
        except Exception as e:
            print(f"       Filter execution failed: {e}")
    else:
        print("       No PropertyFilter found in physical plan!")
    print()
    
    # Step 3c: Test Project operation
    print("   3c. Testing Project operation:")
    project_op = find_operation_by_type(physical_plan, "Project")
    if project_op:
        print("       Found Project operation")
        # The project operation should execute its children and then project
        print("       Project should execute children first, then apply projections")
    print()
    
    # STEP 4: Execute the full query and compare
    print("4. 🚀 Full Query Execution:")
    result = db.execute_query(query)
    print(f"   Success: {result.success}")
    print(f"   Result count: {len(result.data)}")
    print(f"   Results: {result.data}")
    
    # STEP 5: Analysis
    print("\n5. 🎯 Analysis:")
    if len(result.data) == 1:
        entity_name = None
        for record in result.data:
            if 'n.name' in record:
                entity_name = record['n.name']
        
        if entity_name == 'Alice':
            print("   ✅ Filter working correctly - got Alice only")
        else:
            print(f"   ⚠️  Got 1 result but wrong entity: {entity_name}")
    elif len(result.data) == 2:
        print("   ❌ Filter NOT working - got both Alice and Bob")
        print("   🔍 This confirms filters are being bypassed")
    elif len(result.data) == 0:
        print("   ⚠️  Over-filtering - got no results")
    else:
        print(f"   ⚠️  Unexpected result count: {len(result.data)}")


def print_plan_structure(op, prefix="", is_physical=False):
    """Print plan structure with details"""
    op_name = type(op).__name__
    details = []
    
    if is_physical:
        if hasattr(op, 'operation_type'):
            details.append(f"type={op.operation_type}")
        if hasattr(op, 'logical_op') and op.logical_op:
            details.append(f"logical={type(op.logical_op).__name__}")
    else:
        if hasattr(op, 'variable'):
            details.append(f"var={op.variable}")
        if hasattr(op, 'property_key'):
            details.append(f"prop={op.property_key}")
        if hasattr(op, 'operator'):
            details.append(f"op={op.operator}")
        if hasattr(op, 'value'):
            details.append(f"val={op.value}")
    
    detail_str = f" ({', '.join(details)})" if details else ""
    print(f"{prefix}{op_name}{detail_str}")
    
    for child in getattr(op, 'children', []):
        print_plan_structure(child, prefix + "  ", is_physical)


def find_operation_by_type(plan, operation_type):
    """Find operation by type in plan tree"""
    if hasattr(plan, 'operation_type') and plan.operation_type == operation_type:
        return plan
    
    for child in getattr(plan, 'children', []):
        result = find_operation_by_type(child, operation_type)
        if result:
            return result
    
    return None


def test_multiple_filter_scenarios():
    """Test multiple filter scenarios to establish patterns"""
    print("🧪 === MULTIPLE FILTER SCENARIO TESTING ===\n")
    
    scenarios = [
        ("String equality", "MATCH (n:Person) WHERE n.name = 'Alice' RETURN n.name", 1),
        ("Number equality", "MATCH (n:Person) WHERE n.age = 30 RETURN n.name", 1),
        ("Number range", "MATCH (n:Person) WHERE n.age > 25 RETURN n.name", 1),
        ("No filter baseline", "MATCH (n:Person) RETURN n.name", 2),
    ]
    
    from mylathdb import MyLathDB
    
    for scenario_name, query, expected_count in scenarios:
        print(f"🔬 Testing: {scenario_name}")
        print(f"   Query: {query}")
        print(f"   Expected count: {expected_count}")
        
        # Fresh setup for each test
        db = MyLathDB()
        if db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        nodes = [
            {'id': '1', 'name': 'Alice', 'age': 30, '_labels': ['Person']},
            {'id': '2', 'name': 'Bob', 'age': 25, '_labels': ['Person']},
        ]
        db.load_graph_data(nodes=nodes)
        
        # Execute
        result = db.execute_query(query)
        actual_count = len(result.data)
        
        print(f"   Actual count: {actual_count}")
        print(f"   Match: {'✅' if actual_count == expected_count else '❌'}")
        
        if actual_count != expected_count:
            print(f"   Results: {result.data}")
        
        print()


def main():
    """Run comprehensive filter tracing"""
    print("🎯 === COMPREHENSIVE FILTER TRACING ===\n")
    
    # Detailed execution tracing
    trace_filter_execution()
    
    print("\n" + "="*80 + "\n")
    
    # Multiple scenario testing
    test_multiple_filter_scenarios()
    
    print("🎯 === TRACING COMPLETE ===")


if __name__ == "__main__":
    main()