# main.py

"""Main test runner using new parser"""

from cypher_planner import (
    CypherParser, 
    parse_cypher_query,
    LogicalPlanner, 
    RuleBasedOptimizer,
    ParseError,
    LexerError
)

def main():
    print("🚀 Cypher Parser - Production Version")
    print("=" * 40)
    
    parser = CypherParser()
    planner = LogicalPlanner()
    optimizer = RuleBasedOptimizer()
    
    test_queries = [
        "MATCH (n:Person) RETURN n.name",
        "MATCH (a)-[r:KNOWS*1..3]->(b) WHERE a.age > 25 RETURN a.name, b.name",
        "OPTIONAL MATCH (a)-[r]->(b) RETURN a, b",
        # Unicode test
        "MATCH (café:Restaurant) WHERE café.type = 'français' RETURN café.name",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query[:50]}...")
        print("-" * 50)
        
        try:
            ast = parser.parse(query)
            print("✅ Parsing: SUCCESS")
            
            logical_plan = planner.create_logical_plan(ast)
            print("✅ Planning: SUCCESS")
            
            optimized_plan = optimizer.optimize(logical_plan)
            print("✅ Optimization: SUCCESS")
            
        except (ParseError, LexerError) as e:
            print(f"❌ Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()