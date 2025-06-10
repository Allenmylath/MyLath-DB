#!/usr/bin/env python3
"""
Quick Test Runner for Cypher Parser
Run this to quickly test the parser and find logical errors
"""

import sys
import os

# Add the current directory to path
sys.path.insert(0, '.')

def test_parser_basic():
    """Basic functionality test"""
    print("🧪 Basic Parser Test")
    print("=" * 30)
    
    try:
        from cypher_planner.parser import CypherParser
        from cypher_planner.logical_planner import LogicalPlanner
        
        parser = CypherParser()
        planner = LogicalPlanner()
        
        # Test queries
        test_queries = [
            "MATCH (n:Person) RETURN n.name",
            "MATCH (a)-[r:KNOWS]->(b) WHERE a.age > 25 RETURN a, b",
            "MATCH (a)-[*1..3]->(b) RETURN a, b",
            "OPTIONAL MATCH (a)-[r]->(b) RETURN a, b",
            "MATCH (n) WHERE n.name = 'John' AND n.age > 30 RETURN n",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            try:
                ast = parser.parse(query)
                plan = planner.create_logical_plan(ast)
                print("  ✅ Parsed and planned successfully")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're in the cypher_planner directory")
        return False

def test_logical_errors():
    """Test for logical errors"""
    print("\n🔍 Logical Error Detection")
    print("=" * 30)
    
    try:
        from cypher_planner.parser import CypherParser
        
        parser = CypherParser()
        
        # Queries that should expose logical issues
        problematic_queries = [
            ("MATCH (n)", "Missing RETURN clause"),
            ("RETURN undefined_var", "Undefined variable"),
            ("MATCH (a)-[*5..2]->(b) RETURN a", "Invalid variable length range"),
            ("MATCH (n) WHERE 1 = 2 RETURN n", "Always false condition"),
            ("MATCH (a), (a) RETURN a", "Duplicate variable"),
            ("MATCH (a)-[r]->(b)-[s] RETURN a", "Incomplete relationship pattern"),
        ]
        
        issues_found = 0
        
        for query, expected_issue in problematic_queries:
            print(f"\nTesting: {query}")
            print(f"Expected issue: {expected_issue}")
            
            try:
                ast = parser.parse(query)
                
                # Basic checks
                if not ast.return_clause and ast.match_clauses:
                    print("  🔍 Found: Missing RETURN clause")
                    issues_found += 1
                elif not ast.match_clauses and ast.return_clause:
                    # Check for undefined variables in RETURN
                    return_vars = set()
                    for item in ast.return_clause.items:
                        if hasattr(item.expression, 'name'):
                            return_vars.add(item.expression.name)
                    if return_vars:
                        print(f"  🔍 Found: Potentially undefined variables: {return_vars}")
                        issues_found += 1
                else:
                    print("  ⚠️  Parser accepted query (might be a logical issue)")
                    
            except Exception as e:
                print(f"  ✅ Parser correctly rejected: {e}")
        
        print(f"\n📊 Summary: Found {issues_found} potential logical issues")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False

def interactive_test():
    """Interactive testing mode"""
    print("\n🎯 Interactive Test Mode")
    print("Enter Cypher queries to test (type 'quit' to exit)")
    print("-" * 40)
    
    try:
        from cypher_planner.parser import CypherParser
        from cypher_planner.logical_planner import LogicalPlanner
        
        parser = CypherParser()
        planner = LogicalPlanner()
        
        while True:
            try:
                query = input("\ncypher> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                # Parse
                try:
                    ast = parser.parse(query)
                    print("  ✅ Parsing: SUCCESS")
                    
                    # Basic AST analysis
                    match_count = len(ast.match_clauses) if ast.match_clauses else 0
                    optional_count = len(ast.optional_match_clauses) if ast.optional_match_clauses else 0
                    has_where = bool(ast.where_clause)
                    has_return = bool(ast.return_clause)
                    
                    print(f"     MATCH clauses: {match_count}")
                    if optional_count > 0:
                        print(f"     OPTIONAL MATCH clauses: {optional_count}")
                    print(f"     WHERE clause: {'Yes' if has_where else 'No'}")
                    print(f"     RETURN clause: {'Yes' if has_return else 'No'}")
                    
                    # Logical planning
                    try:
                        plan = planner.create_logical_plan(ast)
                        print("  ✅ Logical Planning: SUCCESS")
                    except Exception as e:
                        print(f"  ❌ Logical Planning: FAILED - {e}")
                    
                    # Basic logical checks
                    if match_count > 0 and not has_return:
                        print("  ⚠️  Logical Issue: MATCH without RETURN")
                    
                except Exception as e:
                    print(f"  ❌ Parsing: FAILED - {e}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except ImportError as e:
        print(f"❌ Import Error: {e}")

def main():
    """Main test runner"""
    print("🚀 Cypher Parser Quick Test")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('cypher_planner'):
        print("❌ Error: cypher_planner directory not found")
        print("Please run this script from the cypher_planner project root directory")
        sys.exit(1)
    
    # Run basic tests
    if not test_parser_basic():
        return
    
    # Run logical error tests
    test_logical_errors()
    
    # Offer interactive mode
    if input("\nTry interactive mode? (y/n): ").lower().startswith('y'):
        interactive_test()
    
    print("\n🎉 Testing complete!")
    print("\nTo run more comprehensive tests:")
    print("  python cypher_parser_tests.py")
    print("  python logical_error_finder.py")

if __name__ == "__main__":
    main()