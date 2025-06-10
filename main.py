# main.py

"""
Enhanced Cypher Planner Demo
Comprehensive demonstration of parsing, planning, and error handling capabilities
"""

import time
import sys
from typing import List, Dict, Any

# Import all cypher_planner components
from cypher_planner import (
   CypherParser, QueryPlanner, 
   parse_cypher_query, validate_cypher_query, get_cypher_errors,
   CypherParserError, ErrorCode,
   format_query, optimize_plan, estimate_cost,
   get_package_info, demo_enhanced_features
)

def print_header(title: str, char: str = "="):
   """Print a formatted header"""
   print(f"\n{char * 60}")
   print(f" {title}")
   print(f"{char * 60}")

def print_section(title: str):
   """Print a section header"""
   print(f"\n🔹 {title}")
   print("-" * 40)

def run_enhanced_demonstrations():
   """Run comprehensive demonstrations of all features"""
   
   print_header("🚀 Enhanced Cypher Planner Demonstration")
   
   # Show package info
   print_section("Package Information")
   info = get_package_info()
   print(f"Version: {info['version']}")
   print(f"Description: {info['description']}")
   print(f"Supported Features: {len([f for f, v in info['features'].items() if v])}")
   
   # Demo queries - from simple to complex
   demo_queries = [
       {
           "name": "Simple Node Match",
           "query": "MATCH (n:Person) RETURN n.name",
           "description": "Basic node pattern with label and property return"
       },
       {
           "name": "Relationship Pattern", 
           "query": "MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE a.age > 25 RETURN a.name, b.name, r.since",
           "description": "Node-relationship-node pattern with filtering"
       },
       {
           "name": "Variable Length Path",
           "query": "MATCH (start:Person {name: 'Alice'})-[:FOLLOWS*1..3]->(end:Person) RETURN start.name, end.name",
           "description": "Variable length relationship path"
       },
       {
           "name": "Optional Match",
           "query": "MATCH (p:Person) OPTIONAL MATCH (p)-[:HAS_PHONE]->(phone:Phone) RETURN p.name, phone.number",
           "description": "Optional pattern matching"
       },
       {
           "name": "Aggregation with Grouping",
           "query": "MATCH (p:Person)-[:WORKS_AT]->(c:Company) RETURN c.name, count(p) as employee_count ORDER BY employee_count DESC",
           "description": "Aggregation function with grouping and ordering"
       },
       {
           "name": "Complex Multi-Clause",
           "query": """
           MATCH (p:Person {country: 'USA'})-[:KNOWS*1..3]->(friend:Person)
           WHERE p.age > 21 AND friend.age < p.age
           WITH p, collect(friend) as friends
           WHERE size(friends) > 2
           RETURN p.name, size(friends) as friend_count, 
                  [f IN friends | f.name] as friend_names
           ORDER BY friend_count DESC
           LIMIT 10
           """,
           "description": "Complex query with multiple clauses, collections, and list comprehension"
       }
   ]
   
   parser = CypherParser()
   planner = QueryPlanner()
   
   for i, demo in enumerate(demo_queries, 1):
       print_section(f"Demo {i}: {demo['name']}")
       print(f"Description: {demo['description']}")
       print(f"Query: {demo['query'].strip()}")
       
       try:
           # Parse the query
           start_time = time.time()
           ast = parser.parse(demo['query'])
           parse_time = time.time() - start_time
           
           print(f"✅ Parse successful! ({parse_time*1000:.1f}ms)")
           
           # Show AST structure
           print(f"   📊 AST Structure:")
           print(f"      - MATCH clauses: {len(ast.match_clauses)}")
           print(f"      - Optional MATCH: {len(ast.optional_match_clauses)}")
           print(f"      - WHERE clause: {'Yes' if ast.where_clause else 'No'}")
           print(f"      - WITH clauses: {len(ast.with_clauses)}")
           print(f"      - RETURN clause: {'Yes' if ast.return_clause else 'No'}")
           
           if ast.return_clause:
               print(f"      - Return items: {len(ast.return_clause.items)}")
               print(f"      - DISTINCT: {ast.return_clause.distinct}")
               print(f"      - ORDER BY: {'Yes' if ast.return_clause.order_by else 'No'}")
               print(f"      - LIMIT: {ast.return_clause.limit if ast.return_clause.limit else 'None'}")
           
           # Generate execution plan
           start_time = time.time()
           plan = planner.plan(ast)
           plan_time = time.time() - start_time
           
           print(f"   🎯 Execution Plan: ({plan_time*1000:.1f}ms)")
           for j, step in enumerate(plan.steps):
               print(f"      {j+1}. {step.operation}")
               if hasattr(step, 'details') and step.details:
                   for detail in step.details[:2]:  # Show first 2 details
                       print(f"         • {detail}")
           
           # Cost estimation
           cost = estimate_cost(plan)
           print(f"   💰 Estimated Cost: {cost}")
           
           # Try optimization
           optimized_plan = optimize_plan(plan)
           if optimized_plan != plan:
               print(f"   ⚡ Optimization: Applied {len(optimized_plan.optimizations)} optimizations")
               for opt in optimized_plan.optimizations[:2]:
                   print(f"      • {opt}")
           else:
               print(f"   ⚡ Optimization: No optimizations applied")
           
       except Exception as e:
           print(f"❌ Error: {str(e)}")
           
           # Show detailed error information
           error_details = get_cypher_errors(demo['query'])
           if error_details.get('errors'):
               print("   📋 Error Details:")
               for error in error_details['errors'][:3]:  # Show first 3 errors
                   print(f"      • {error['code']}: {error['message']}")
                   if error['suggestion']:
                       print(f"        💡 Suggestion: {error['suggestion']}")

def run_performance_comparison():
   """Compare parsing performance across different query types"""
   
   print_header("⚡ Performance Analysis", "=")
   
   test_queries = [
       ("Simple", "MATCH (n:Person) RETURN n.name"),
       ("Medium", "MATCH (a:Person)-[r:KNOWS]->(b:Person) WHERE a.age > 25 RETURN a.name, b.name"),
       ("Complex", "MATCH (p:Person)-[:KNOWS*1..3]->(f) WITH p, collect(f) as friends WHERE size(friends) > 2 RETURN p.name, size(friends)"),
       ("Very Complex", """
           MATCH (p:Person {country: 'USA'})-[:KNOWS*1..3]->(friend:Person)
           WHERE p.age > 21 AND friend.age < p.age
           WITH p, collect(friend) as friends
           WHERE size(friends) > 2
           RETURN p.name, size(friends) as friend_count
           ORDER BY friend_count DESC
           LIMIT 10
       """)
   ]
   
   parser = CypherParser()
   planner = QueryPlanner()
   iterations = 100
   
   print(f"🔍 Running {iterations} iterations per query type...\n")
   
   results = []
   
   for query_type, query in test_queries:
       print(f"Testing {query_type} Query...")
       
       # Parse timing
       parse_times = []
       for _ in range(iterations):
           start = time.time()
           try:
               ast = parser.parse(query)
               parse_time = time.time() - start
               parse_times.append(parse_time)
           except:
               parse_times.append(float('inf'))  # Mark failures
       
       # Plan timing
       plan_times = []
       try:
           ast = parser.parse(query)
           for _ in range(iterations):
               start = time.time()
               plan = planner.plan(ast)
               plan_time = time.time() - start
               plan_times.append(plan_time)
       except:
           plan_times = [float('inf')] * iterations
       
       # Calculate statistics
       valid_parse_times = [t for t in parse_times if t != float('inf')]
       valid_plan_times = [t for t in plan_times if t != float('inf')]
       
       if valid_parse_times:
           avg_parse = sum(valid_parse_times) / len(valid_parse_times)
           min_parse = min(valid_parse_times)
           max_parse = max(valid_parse_times)
       else:
           avg_parse = min_parse = max_parse = 0
           
       if valid_plan_times:
           avg_plan = sum(valid_plan_times) / len(valid_plan_times)
           min_plan = min(valid_plan_times)
           max_plan = max(valid_plan_times)
       else:
           avg_plan = min_plan = max_plan = 0
       
       results.append({
           'type': query_type,
           'parse_avg': avg_parse * 1000,  # Convert to ms
           'parse_min': min_parse * 1000,
           'parse_max': max_parse * 1000,
           'plan_avg': avg_plan * 1000,
           'plan_min': min_plan * 1000, 
           'plan_max': max_plan * 1000,
           'success_rate': len(valid_parse_times) / iterations * 100
       })
       
       print(f"  Parse: {avg_parse*1000:.2f}ms avg ({min_parse*1000:.2f}-{max_parse*1000:.2f}ms)")
       print(f"  Plan:  {avg_plan*1000:.2f}ms avg ({min_plan*1000:.2f}-{max_plan*1000:.2f}ms)")
       print(f"  Success: {len(valid_parse_times)}/{iterations} ({len(valid_parse_times)/iterations*100:.1f}%)")
       print()
   
   # Summary table
   print_section("Performance Summary")
   print(f"{'Query Type':<12} {'Parse (ms)':<12} {'Plan (ms)':<12} {'Success %':<10}")
   print("-" * 50)
   for result in results:
       print(f"{result['type']:<12} {result['parse_avg']:<12.2f} {result['plan_avg']:<12.2f} {result['success_rate']:<10.1f}")

def interactive_enhanced_mode():
   """Interactive mode with enhanced error handling"""
   
   print_header("🎮 Interactive Enhanced Mode")
   print("Enter Cypher queries to see parsing, planning, and error analysis.")
   print("Commands: 'help', 'stats', 'demo', 'quit'")
   
   parser = CypherParser()
   planner = QueryPlanner()
   session_stats = {
       'queries_parsed': 0,
       'successful_parses': 0,
       'syntax_errors': 0,
       'semantic_errors': 0
   }
   
   while True:
       try:
           query = input("\ncypher> ").strip()
           
           if not query:
               continue
               
           if query.lower() == 'quit':
               break
           elif query.lower() == 'help':
               print("\nCommands:")
               print("  help     - Show this help")
               print("  stats    - Show session statistics")
               print("  demo     - Run error handling demo")
               print("  quit     - Exit interactive mode")
               print("\nOr enter any Cypher query to analyze it.")
               continue
           elif query.lower() == 'stats':
               print(f"\n📊 Session Statistics:")
               for key, value in session_stats.items():
                   print(f"  {key.replace('_', ' ').title()}: {value}")
               if session_stats['queries_parsed'] > 0:
                   success_rate = session_stats['successful_parses'] / session_stats['queries_parsed'] * 100
                   print(f"  Success Rate: {success_rate:.1f}%")
               continue
           elif query.lower() == 'demo':
               demonstrate_enhanced_error_handling()
               continue
           
           session_stats['queries_parsed'] += 1
           
           # Quick validation first
           print(f"\n🔍 Quick Validation: {'✅ Valid' if validate_cypher_query(query) else '❌ Invalid'}")
           
           # Detailed parsing
           try:
               start_time = time.time()
               ast = parser.parse(query)
               parse_time = time.time() - start_time
               
               session_stats['successful_parses'] += 1
               
               print(f"✅ Parse Successful! ({parse_time*1000:.1f}ms)")
               
               # Show query structure
               print("📊 Query Structure:")
               if ast.match_clauses:
                   print(f"   • {len(ast.match_clauses)} MATCH clause(s)")
               if ast.optional_match_clauses:
                   print(f"   • {len(ast.optional_match_clauses)} OPTIONAL MATCH clause(s)")
               if ast.where_clause:
                   print(f"   • WHERE clause present")
               if ast.with_clauses:
                   print(f"   • {len(ast.with_clauses)} WITH clause(s)")
               if ast.return_clause:
                   print(f"   • RETURN clause with {len(ast.return_clause.items)} item(s)")
               
               # Generate execution plan
               try:
                   start_time = time.time()
                   plan = planner.plan(ast)
                   plan_time = time.time() - start_time
                   
                   print(f"🎯 Execution Plan: ({plan_time*1000:.1f}ms)")
                   for i, step in enumerate(plan.steps[:5]):  # Show first 5 steps
                       print(f"   {i+1}. {step.operation}")
                   if len(plan.steps) > 5:
                       print(f"   ... and {len(plan.steps) - 5} more steps")
                   
                   # Cost estimation
                   cost = estimate_cost(plan)
                   print(f"💰 Estimated Cost: {cost}")
                   
               except Exception as e:
                   print(f"⚠️  Planning failed: {str(e)}")
               
           except Exception as e:
               print(f"❌ Parse Failed: {str(e)}")
               
               # Detailed error analysis
               error_details = get_cypher_errors(query)
               
               if error_details.get('errors'):
                   has_syntax_error = any('syntax' in err['message'].lower() for err in error_details['errors'])
                   has_semantic_error = any('semantic' in err['message'].lower() or 'undefined' in err['message'].lower() for err in error_details['errors'])
                   
                   if has_syntax_error:
                       session_stats['syntax_errors'] += 1
                   elif has_semantic_error:
                       session_stats['semantic_errors'] += 1
                   
                   print("🔍 Error Analysis:")
                   for error in error_details['errors'][:3]:  # Show first 3 errors
                       print(f"   • {error['code']}: {error['message']}")
                       if error['suggestion']:
                           print(f"     💡 {error['suggestion']}")
                       if error['context']:
                           print(f"     📍 Context: {error['context'][:50]}...")
               
               if error_details.get('warnings'):
                   print("⚠️  Warnings:")
                   for warning in error_details['warnings'][:2]:  # Show first 2 warnings
                       print(f"   • {warning['code']}: {warning['message']}")
       
       except KeyboardInterrupt:
           print("\n\nExiting interactive mode...")
           break
       except Exception as e:
           print(f"Unexpected error: {str(e)}")
   
   # Show final statistics
   print(f"\n📊 Final Session Statistics:")
   for key, value in session_stats.items():
       print(f"   {key.replace('_', ' ').title()}: {value}")

def demonstrate_enhanced_error_handling():
   """Demonstrate the enhanced error handling capabilities"""
   
   print_header("🔍 Enhanced Error Handling Demonstration")
   
   test_cases = [
       {
           'name': 'Valid Query',
           'query': "MATCH (n:Person) WHERE n.age > 25 RETURN n.name",
           'expected': 'success'
       },
       {
           'name': 'Syntax Error - Missing Parenthesis',
           'query': "MATCH (n:Person WHERE n.age > 25 RETURN n.name",
           'expected': 'syntax_error'
       },
       {
           'name': 'Syntax Error - Unbalanced Brackets',
           'query': "MATCH (n:Person)-[r:KNOWS->(b:Person) RETURN n, b",
           'expected': 'syntax_error'
       },
       {
           'name': 'Semantic Error - Undefined Variable',
           'query': "MATCH (n:Person) RETURN m.name",
           'expected': 'semantic_error'
       },
       {
           'name': 'Semantic Error - Mixed Aggregation',
           'query': "MATCH (n:Person) RETURN count(n) + n.name",
           'expected': 'semantic_error'
       },
       {
           'name': 'Empty WHERE Clause',
           'query': "MATCH (n:Person) WHERE RETURN n.name",
           'expected': 'syntax_error'
       },
       {
           'name': 'Invalid Variable Length',
           'query': "MATCH (a)-[*-1..3]->(b) RETURN a, b",
           'expected': 'syntax_error'
       },
       {
           'name': 'Performance Warning - Potential Cartesian Product',
           'query': "MATCH (n) MATCH (m) RETURN n, m",
           'expected': 'warning'
       },
       {
           'name': 'Complex Valid Query',
           'query': """
           MATCH (p:Person {country: 'USA'})-[:KNOWS*1..3]->(friend:Person)
           WHERE p.age > 21 AND friend.age < p.age
           WITH p, collect(friend) as friends
           WHERE size(friends) > 2
           RETURN p.name, size(friends) as friend_count
           ORDER BY friend_count DESC
           LIMIT 10
           """,
           'expected': 'success'
       }
   ]
   
   parser = CypherParser()
   
   for i, test_case in enumerate(test_cases, 1):
       print_section(f"Test {i}: {test_case['name']}")
       print(f"Query: {test_case['query'].strip()}")
       print(f"Expected: {test_case['expected']}")
       
       # Quick validation
       is_valid = validate_cypher_query(test_case['query'])
       print(f"Quick validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
       
       # Detailed parsing
       try:
           result = parser.parse(test_case['query'])
           print("✅ Parse successful!")
           print(f"   Query type: {type(result).__name__}")
           
           # Show basic structure for successful parses
           if result.match_clauses:
               print(f"   MATCH clauses: {len(result.match_clauses)}")
           if result.where_clause:
               print(f"   WHERE clause: Present")
           if result.return_clause:
               print(f"   RETURN items: {len(result.return_clause.items)}")
               
       except Exception as e:
           print("❌ Parse failed!")
           print(f"   Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")
           
           # Get detailed error information
           error_details = get_cypher_errors(test_case['query'])
           
           if error_details.get('errors'):
               print("   📋 Detailed Errors:")
               for error in error_details['errors'][:2]:  # Show first 2 errors
                   print(f"      • {error['code']}: {error['message']}")
                   if error['suggestion']:
                       print(f"        💡 {error['suggestion']}")
                   if error.get('position') and error['position']['line']:
                       print(f"        📍 Line {error['position']['line']}, Column {error['position']['column']}")
                       
           if error_details.get('warnings'):
               print("   ⚠️  Warnings:")
               for warning in error_details['warnings'][:2]:  # Show first 2 warnings
                   print(f"      • {warning['code']}: {warning['message']}")
   
   # Show parser statistics
   print_section("Parser Statistics")
   stats = parser.get_parse_statistics()
   for key, value in stats.items():
       print(f"   {key.replace('_', ' ').title()}: {value}")

def main():
   """Main entry point for the enhanced Cypher planner demo"""
   
   print("🚀 Welcome to Enhanced Cypher Planner!")
   print("=" * 50)
   print("A comprehensive Cypher query parser and planner with")
   print("enhanced error handling inspired by FalkorDB.")
   
   # Show package information
   info = get_package_info()
   print(f"\nVersion: {info['version']}")
   print(f"Features: {len([f for f, v in info['features'].items() if v])} enabled")
   
   while True:
       print("\nChoose mode:")
       print("1. Enhanced demonstrations (default)")
       print("2. Performance analysis") 
       print("3. Interactive mode")
       print("4. Error handling demonstration")
       print("5. Quick feature demo")
       print("6. Exit")
       
       try:
           choice = input("\nEnter choice (1-6): ").strip()
           
           if choice == "" or choice == "1":
               run_enhanced_demonstrations()
           elif choice == "2":
               run_performance_comparison()
           elif choice == "3":
               interactive_enhanced_mode()
           elif choice == "4":
               demonstrate_enhanced_error_handling()
           elif choice == "5":
               demo_enhanced_features()
           elif choice == "6":
               print("\n👋 Thank you for using Enhanced Cypher Planner!")
               break
           else:
               print("❌ Invalid choice. Please enter 1-6.")
               continue
               
           # Ask if user wants to try another mode
           next_choice = input("\n🔄 Try another mode? (y/n): ").lower().strip()
           if next_choice not in ['y', 'yes', '']:
               print("\n👋 Thank you for using Enhanced Cypher Planner!")
               break
               
       except KeyboardInterrupt:
           print("\n\n👋 Goodbye!")
           break
       except Exception as e:
           print(f"\n❌ Unexpected error: {str(e)}")
           print("Please try again or contact support.")

if __name__ == "__main__":
   main()