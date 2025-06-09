# STEP 7: Replace main.py with this enhanced version

#!/usr/bin/env python3
"""
Enhanced Cypher Planner with FalkorDB-inspired improvements
Step-by-step implementation completed
"""

import sys
import time
from typing import List, Dict, Any

# Import enhanced components
from cypher_planner.parser import CypherParser
from cypher_planner.logical_planner import LogicalPlanner
from cypher_planner.optimizer import RuleBasedOptimizer
from cypher_planner.physical_planner import PhysicalPlanner, print_physical_plan
from cypher_planner.logical_operators import print_plan, analyze_plan_execution_targets
from cypher_planner.execution_statistics import ExecutionStatistics
from cypher_planner.filter_placement import FilterOptimizer, FilterNode, FilterType

class EnhancedCypherPlanner:
    """Enhanced Cypher Planner with FalkorDB-inspired architecture"""
    
    def __init__(self):
        # Core components
        self.parser = CypherParser()
        self.logical_planner = LogicalPlanner()
        self.optimizer = RuleBasedOptimizer()  # Now includes FilterOptimizer
        
        # Enhanced statistics
        self.statistics = ExecutionStatistics()
        self.physical_planner = PhysicalPlanner(self.statistics)
    
    def plan_query(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """Plan a query with enhanced capabilities"""
        if verbose:
            print(f"🔍 Planning Query: {query}")
            print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Parse the query
            if verbose:
                print("📝 Step 1: Parsing...")
            parse_start = time.time()
            ast = self.parser.parse(query)
            parse_time = time.time() - parse_start
            
            # Step 2: Create initial logical plan
            if verbose:
                print("🏗️  Step 2: Creating Logical Plan...")
            logical_start = time.time()
            initial_plan = self.logical_planner.create_logical_plan(ast)
            logical_time = time.time() - logical_start
            
            if verbose:
                print("📋 Initial Logical Plan:")
                print_plan(initial_plan, 1)
            
            # Step 3: Apply enhanced optimizations
            if verbose:
                print("⚡ Step 3: Enhanced Optimizations...")
            opt_start = time.time()
            optimized_plan = self.optimizer.optimize(initial_plan)
            opt_time = time.time() - opt_start
            
            if verbose:
                print("📋 Optimized Logical Plan:")
                print_plan(optimized_plan, 1)
            
            # Step 4: Generate enhanced physical plan
            if verbose:
                print("🚀 Step 4: Enhanced Physical Planning...")
            physical_start = time.time()
            physical_plan = self.physical_planner.create_physical_plan(optimized_plan)
            physical_time = time.time() - physical_start
            
            if verbose:
                print("🔧 Enhanced Physical Plan:")
                print_physical_plan(physical_plan, 1)
            
            total_time = time.time() - start_time
            
            # Analyze results
            targets = analyze_plan_execution_targets(optimized_plan)
            complexity = self._calculate_complexity(optimized_plan)
            
            if verbose:
                print(f"\n📊 Performance Analysis:")
                print(f"  Total Planning Time: {total_time * 1000:.2f}ms")
                print(f"  - Parse: {parse_time * 1000:.2f}ms")
                print(f"  - Logical: {logical_time * 1000:.2f}ms")
                print(f"  - Optimization: {opt_time * 1000:.2f}ms")
                print(f"  - Physical: {physical_time * 1000:.2f}ms")
                print(f"  Complexity Score: {complexity}/10")
                print(f"  Execution Targets: {targets}")
            
            return {
                'success': True,
                'ast': ast,
                'initial_logical_plan': initial_plan,
                'optimized_logical_plan': optimized_plan,
                'physical_plan': physical_plan,
                'timing': {
                    'parse': parse_time * 1000,
                    'logical': logical_time * 1000,
                    'optimization': opt_time * 1000,
                    'physical': physical_time * 1000,
                    'total': total_time * 1000
                },
                'analysis': {
                    'complexity': complexity,
                    'execution_targets': targets,
                    'estimated_cardinality': getattr(physical_plan, 'estimated_cardinality', 'Unknown')
                }
            }
            
        except Exception as e:
            if verbose:
                print(f"❌ Error: {e}")
            return {
                'success': False,
                'error': str(e),
                'timing': {'total': (time.time() - start_time) * 1000}
            }
    
    def _calculate_complexity(self, plan) -> int:
        """Calculate query complexity score"""
        complexity = 0
        
        def count_ops(op):
            nonlocal complexity
            complexity += 1
            
            # Add extra complexity for certain operations
            op_name = type(op).__name__
            if 'VarLen' in op_name:
                complexity += 3
            elif 'Apply' in op_name:
                complexity += 2
            elif 'Filter' in op_name:
                complexity += 1
            
            for child in op.children:
                count_ops(child)
        
        count_ops(plan)
        return min(complexity, 10)

def run_enhanced_demonstrations():
    """Run demonstrations of enhanced capabilities"""
    
    print("🚀 Enhanced Cypher Planner - Step-by-Step Implementation Complete!")
    print("Showcasing FalkorDB-inspired improvements")
    print("=" * 80)
    
    planner = EnhancedCypherPlanner()
    
    # Test queries showcasing enhanced features
    test_cases = [
        {
            'title': 'Enhanced Property Filtering',
            'query': "MATCH (p:Person) WHERE p.age > 30 AND p.country = 'USA' RETURN p.name",
            'features': ['NodeByLabelScan', 'PropertyFilter', 'Index optimization']
        },
        {
            'title': 'Variable-Length Path with GraphBLAS',
            'query': "MATCH (a:Actor {name: 'Tom Hanks'})-[:ACTED_IN*1..3]->(m:Movie) RETURN m.title",
            'features': ['ConditionalVarLenTraverse', 'Matrix operations', 'Path algorithms']
        },
        {
            'title': 'Optional Match with Semi-Apply',
            'query': "MATCH (p:Person) OPTIONAL MATCH (p)-[:OWNS]->(c:Car) RETURN p.name, c.make",
            'features': ['Optional operation', 'Left outer join', 'NULL handling']
        },
        {
            'title': 'Complex Multi-Pattern Query',
            'query': "MATCH (u:User)-[:FOLLOWS]->(f:User), (f)-[:POSTED]->(t:Tweet) WHERE t.hashtags CONTAINS 'AI' RETURN u.name, t.content",
            'features': ['Multiple patterns', 'Join optimization', 'Filter placement']
        },
        {
            'title': 'Structural Path Filtering',
            'query': "MATCH (p1:Person)-[r:FRIENDS_WITH]->(p2:Person) WHERE r.since > 2020 AND EXISTS((p1)-[:KNOWS]->(:Person)) RETURN p1.name",
            'features': ['PathFilter', 'StructuralFilter', 'EXISTS pattern']
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['title']}")
        print(f"Enhanced Features: {', '.join(test_case['features'])}")
        print(f"Query: {test_case['query']}")
        print("-" * 80)
        
        result = planner.plan_query(test_case['query'], verbose=True)
        
        if result['success']:
            print(f"\n✅ Planning completed successfully!")
            print(f"   Enhanced features demonstrated in {result['timing']['total']:.2f}ms")
        else:
            print(f"❌ Planning failed: {result['error']}")
        
        print("\n" + "=" * 80)

def run_performance_comparison():
    """Compare performance across different query types"""
    
    print("\n⚡ Performance Analysis")
    print("Testing enhanced planner performance")
    print("=" * 50)
    
    planner = EnhancedCypherPlanner()
    
    test_queries = [
        ("Simple", "MATCH (n:Person) RETURN n.name"),
        ("Property Filter", "MATCH (p:Person) WHERE p.age > 25 RETURN p.name"),
        ("Single Traversal", "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, m.title"),
        ("Variable Length", "MATCH (p1:Person)-[:KNOWS*1..2]-(p2:Person) RETURN p1.name, p2.name"),
        ("Optional Match", "MATCH (u:User) OPTIONAL MATCH (u)-[:POSTED]->(t:Tweet) RETURN u.name, t.content"),
        ("Complex Multi-Pattern", "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie)<-[:DIRECTED]-(d:Director) WHERE m.year > 2000 RETURN a.name, d.name")
    ]
    
    results = []
    
    for query_type, query in test_queries:
        print(f"\n🔥 Testing {query_type}:")
        print(f"   {query}")
        
        result = planner.plan_query(query, verbose=False)
        
        if result['success']:
            timing = result['timing']
            analysis = result['analysis']
            
            results.append({
                'type': query_type,
                'total_time': timing['total'],
                'complexity': analysis['complexity'],
                'targets': analysis['execution_targets']
            })
            
            print(f"   ✅ Completed in {timing['total']:.2f}ms")
            print(f"      Complexity: {analysis['complexity']}/10")
            print(f"      Targets: R={analysis['execution_targets']['redis']}, "
                  f"GB={analysis['execution_targets']['graphblas']}, "
                  f"M={analysis['execution_targets']['mixed']}")
        else:
            print(f"   ❌ Failed: {result['error']}")
    
    # Summary
    if results:
        print(f"\n📈 Performance Summary:")
        print("-" * 40)
        for result in results:
            print(f"{result['type']:20} | {result['total_time']:6.2f}ms | "
                  f"Complexity: {result['complexity']}/10")
        
        avg_time = sum(r['total_time'] for r in results) / len(results)
        print(f"\nAverage planning time: {avg_time:.2f}ms")

def interactive_enhanced_mode():
    """Enhanced interactive mode"""
    
    print("\n🎯 Enhanced Interactive Mode")
    print("Test the enhanced Cypher planner")
    print("Commands: 'quit', 'help', 'features'")
    print("-" * 50)
    
    planner = EnhancedCypherPlanner()
    
    while True:
        try:
            query = input("\nenhanced> ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\n📚 Commands:")
                print("  - Enter any Cypher query for enhanced planning")
                print("  - 'features' - Show enhanced features")
                print("  - 'quit' - Exit")
                continue
            
            if query.lower() == 'features':
                print("\n⭐ Enhanced Features:")
                print("  🏷️  NodeByLabelScan - Optimized label-based scanning")
                print("  🔍 PropertyFilter - Index-aware property filtering")
                print("  🌐 ConditionalTraverse - GraphBLAS matrix operations")
                print("  📏 VarLenTraverse - Variable-length path algorithms")
                print("  🔗 SemiApply - EXISTS pattern optimization")
                print("  📊 Advanced Statistics - Cost-based optimization")
                print("  ⚡ Filter Placement - Optimal filter positioning")
                continue
            
            if not query:
                continue
            
            # Plan the query with enhanced features
            result = planner.plan_query(query, verbose=True)
            
            if result['success']:
                print(f"\n🎉 Enhanced planning completed!")
                print(f"   Features used: {result['analysis']['execution_targets']}")
            else:
                print(f"❌ Error: {result['error']}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function with enhanced options"""
    print("🚀 Enhanced Cypher Planner")
    print("FalkorDB-inspired execution planning")
    print("Step-by-step implementation complete!")
    print("=" * 50)
    
    print("\nChoose mode:")
    print("1. Enhanced demonstrations (default)")
    print("2. Performance analysis")
    print("3. Enhanced interactive mode")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "2":
        run_performance_comparison()
    elif choice == "3":
        interactive_enhanced_mode()
    else:
        run_enhanced_demonstrations()
        
        # Offer additional modes
        next_mode = input("\nTry another mode? (p)erformance, (i)nteractive, or (n)o: ").lower()
        if next_mode in ['p', 'performance']:
            run_performance_comparison()
        elif next_mode in ['i', 'interactive']:
            interactive_enhanced_mode()

if __name__ == "__main__":
    main()