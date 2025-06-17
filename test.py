# execution_order_test.py - Test to confirm the execution order premise

"""
This test will trace exactly what's happening during query execution
to confirm whether filters are being bypassed due to execution order issues
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_execution_order_premise():
    """Test to confirm the premise about execution order"""
    print("🔬 === EXECUTION ORDER PREMISE TEST ===\n")
    
    from mylathdb import MyLathDB
    from mylathdb.cypher_planner import parse_cypher_query, LogicalPlanner, PhysicalPlanner
    
    # Setup minimal test case
    db = MyLathDB()
    if db.engine.redis_executor.redis:
        db.engine.redis_executor.redis.flushdb()
    
    # Load exactly 2 nodes for clear testing
    nodes = [
        {'id': '1', 'name': 'Alice', 'age': 30, '_labels': ['Person']},
        {'id': '2', 'name': 'Bob', 'age': 25, '_labels': ['Person']},
    ]
    db.load_graph_data(nodes=nodes)
    
    print("✅ Test data loaded:")
    print("   Node 1: Alice, age=30")
    print("   Node 2: Bob, age=25")
    print()
    
    # Parse the problematic query
    query = "MATCH (n:Person) WHERE n.name = 'Alice' RETURN n.name"
    print(f"🧪 Query: {query}")
    print("Expected: 1 result (Alice only)")
    print()
    
    # STEP 1: Analyze the plan structure
    print("1. 📋 Plan Structure Analysis:")
    ast = parse_cypher_query(query)
    logical_planner = LogicalPlanner()
    logical_plan = logical_planner.create_logical_plan(ast)
    physical_planner = PhysicalPlanner()
    physical_plan = physical_planner.create_physical_plan(logical_plan)
    
    print("   Logical Plan:")
    print_plan_tree(logical_plan, "     ")
    print()
    
    print("   Physical Plan:")
    print_plan_tree(physical_plan, "     ", is_physical=True)
    print()
    
    # STEP 2: Trace the actual execution path
    print("2. 🔍 Execution Path Tracing:")
    print("   Let's manually trace what should happen vs what actually happens...")
    
    # Create an instrumented Redis executor to trace calls
    original_execute_child = db.engine.redis_executor._execute_child_operation
    call_trace = []
    
    def traced_execute_child(self, child_operation, context):
        """Instrumented version to trace execution order"""
        op_name = type(child_operation).__name__
        op_type = getattr(child_operation, 'operation_type', 'Unknown')
        logical_type = type(getattr(child_operation, 'logical_op', None)).__name__ if hasattr(child_operation, 'logical_op') and child_operation.logical_op else 'None'
        
        trace_entry = f"{op_name}({op_type}, logical={logical_type})"
        call_trace.append(trace_entry)
        print(f"     🔧 Executing: {trace_entry}")
        
        # Call the original method
        result = original_execute_child(child_operation, context)
        print(f"       ↳ Returned {len(result)} results")
        
        return result
    
    # Monkey patch for tracing
    db.engine.redis_executor._execute_child_operation = traced_execute_child.__get__(db.engine.redis_executor, type(db.engine.redis_executor))
    
    # Execute the query
    print("   Actual execution trace:")
    result = db.execute_query(query)
    
    print()
    print("3. 📊 Execution Results:")
    print(f"   Success: {result.success}")
    print(f"   Result count: {len(result.data)}")
    print(f"   Results: {result.data}")
    print()
    
    print("4. 🔍 Call Trace Analysis:")
    print("   Execution order was:")
    for i, call in enumerate(call_trace, 1):
        print(f"     {i}. {call}")
    print()
    
    # STEP 3: Analyze what went wrong
    print("5. 🎯 Premise Analysis:")
    
    # Check if filter was executed
    filter_executed = any('PropertyFilter' in call for call in call_trace)
    scan_executed = any('NodeByLabelScan' in call or 'NodeScan' in call for call in call_trace)
    project_executed = any('Project' in call for call in call_trace)
    
    print(f"   Filter executed: {'✅' if filter_executed else '❌'}")
    print(f"   Scan executed: {'✅' if scan_executed else '❌'}")
    print(f"   Project executed: {'✅' if project_executed else '❌'}")
    print()
    
    # Check execution order
    if len(result.data) == 2:
        print("   🔍 PREMISE ANALYSIS:")
        print("   ❌ Filter bypassed - got both Alice and Bob")
        
        if filter_executed:
            print("   📋 Filter WAS executed but didn't work correctly")
            print("   💡 Issue: Filter may be executing on wrong data or in wrong order")
        else:
            print("   📋 Filter was NOT executed at all")
            print("   💡 Issue: Filter is being skipped in execution path")
        
        print()
        print("   🎯 CONFIRMED PREMISE:")
        print("   The execution order is incorrect - filters are not being applied")
        print("   before projection gets the full scan results.")
        
        return False
        
    elif len(result.data) == 1 and result.data[0].get('n.name') == 'Alice':
        print("   ✅ PREMISE INCORRECT:")
        print("   Filter is working correctly - got Alice only")
        print("   The issue may be somewhere else in the codebase")
        
        return True
        
    else:
        print("   ⚠️  UNEXPECTED RESULT:")
        print(f"   Got {len(result.data)} results, expected 1 or 2")
        print("   Need to investigate further")
        
        return None

def print_plan_tree(op, prefix="", is_physical=False):
    """Print plan tree structure"""
    if not op:
        return
        
    op_name = type(op).__name__
    details = []
    
    if is_physical:
        if hasattr(op, 'operation_type'):
            details.append(f"type={op.operation_type}")
        if hasattr(op, 'logical_op') and op.logical_op:
            logical_name = type(op.logical_op).__name__
            details.append(f"logical={logical_name}")
            
            # Add logical operation details
            if hasattr(op.logical_op, 'variable'):
                details.append(f"var={op.logical_op.variable}")
            if hasattr(op.logical_op, 'property_key'):
                details.append(f"prop={op.logical_op.property_key}")
            if hasattr(op.logical_op, 'operator'):
                details.append(f"op={op.logical_op.operator}")
            if hasattr(op.logical_op, 'value'):
                details.append(f"val={op.logical_op.value}")
    else:
        # Logical plan details
        if hasattr(op, 'variable'):
            details.append(f"var={op.variable}")
        if hasattr(op, 'property_key'):
            details.append(f"prop={op.property_key}")
        if hasattr(op, 'operator'):
            details.append(f"op={op.operator}")
        if hasattr(op, 'value'):
            details.append(f"val={op.value}")
        if hasattr(op, 'projections'):
            details.append(f"proj={len(op.projections)}")
    
    detail_str = f" ({', '.join(details)})" if details else ""
    print(f"{prefix}{op_name}{detail_str}")
    
    # Print children
    children = getattr(op, 'children', [])
    for child in children:
        print_plan_tree(child, prefix + "  ", is_physical)

def test_individual_operations():
    """Test individual operations to see which ones work correctly"""
    print("🔧 === INDIVIDUAL OPERATION TEST ===\n")
    
    from mylathdb import MyLathDB
    from mylathdb.execution_engine.engine import ExecutionContext
    
    # Setup
    db = MyLathDB()
    if db.engine.redis_executor.redis:
        db.engine.redis_executor.redis.flushdb()
    
    nodes = [
        {'id': '1', 'name': 'Alice', 'age': 30, '_labels': ['Person']},
        {'id': '2', 'name': 'Bob', 'age': 25, '_labels': ['Person']},
    ]
    db.load_graph_data(nodes=nodes)
    
    context = ExecutionContext()
    redis_executor = db.engine.redis_executor
    
    print("Testing individual operations in isolation:")
    print()
    
    # Test 1: NodeByLabelScan alone
    print("1. 🔍 NodeByLabelScan Test:")
    from mylathdb.cypher_planner.logical_operators import NodeByLabelScan
    scan_op = NodeByLabelScan('n', 'Person')
    scan_results = redis_executor._execute_node_by_label_scan_fixed(scan_op, context)
    print(f"   Results: {len(scan_results)} nodes")
    for i, result in enumerate(scan_results):
        entity = result.get('n', {})
        print(f"     {i+1}: {entity.get('name', 'Unknown')} (age={entity.get('age', 'Unknown')})")
    print()
    
    # Test 2: PropertyFilter alone
    print("2. 🔍 PropertyFilter Test:")
    from mylathdb.cypher_planner.logical_operators import PropertyFilter
    filter_op = PropertyFilter('n', 'name', '=', 'Alice')
    filter_results = redis_executor._execute_property_filter_fixed(filter_op, context)
    print(f"   Results: {len(filter_results)} nodes")
    for i, result in enumerate(filter_results):
        entity = result.get('n', {})
        print(f"     {i+1}: {entity.get('name', 'Unknown')} (age={entity.get('age', 'Unknown')})")
    print()
    
    # Test 3: Manual filter application
    print("3. 🔍 Manual Filter Application Test:")
    print("   Applying filter manually to scan results:")
    
    filtered_manual = []
    for result in scan_results:
        entity = result.get('n', {})
        if entity.get('name') == 'Alice':
            filtered_manual.append(result)
            print(f"     ✅ {entity.get('name')} passed filter")
        else:
            print(f"     ❌ {entity.get('name')} failed filter")
    
    print(f"   Manual filter result: {len(filtered_manual)} nodes")
    print()
    
    # Analysis
    print("4. 🎯 Analysis:")
    if len(scan_results) == 2:
        print("   ✅ NodeByLabelScan works correctly")
    else:
        print("   ❌ NodeByLabelScan not working")
    
    if len(filter_results) == 1:
        print("   ✅ PropertyFilter works correctly when called directly")
    else:
        print("   ❌ PropertyFilter not working when called directly")
    
    if len(filtered_manual) == 1:
        print("   ✅ Manual filter application works")
        print("   💡 Issue is likely in the execution flow, not the filter logic")
    else:
        print("   ❌ Even manual filtering fails")
        print("   💡 Issue is in the filter logic itself")

def main():
    """Run premise confirmation tests"""
    print("🚀 === PREMISE CONFIRMATION TEST SUITE ===\n")
    
    # Test 1: Execution order premise
    premise_confirmed = test_execution_order_premise()
    
    print("\n" + "="*60 + "\n")
    
    # Test 2: Individual operations
    test_individual_operations()
    
    print("\n" + "="*60 + "\n")
    
    # Summary
    print("🎯 === PREMISE CONFIRMATION SUMMARY ===")
    
    if premise_confirmed is False:
        print("✅ PREMISE CONFIRMED!")
        print("🔍 The issue IS execution order - filters are being bypassed")
        print("💡 The fix should focus on correcting the execution flow")
        print("📋 Next: Implement the execution order fix")
    elif premise_confirmed is True:
        print("❌ PREMISE INCORRECT!")
        print("🔍 Filters are actually working correctly")
        print("💡 The issue is elsewhere - need to investigate further")
        print("📋 Next: Look for other causes of the filter bypass")
    else:
        print("⚠️  PREMISE UNCLEAR!")
        print("🔍 Results are unexpected - need more investigation")
        print("📋 Next: Debug individual components more thoroughly")

if __name__ == "__main__":
    main()