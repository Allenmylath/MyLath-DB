# ==========================================
# examples/basic_queries.py
# ==========================================

"""
Basic Query Examples for Cypher Planner
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cypher_planner import *


def run_basic_examples():
    """Run basic query examples"""

    print("🔍 Basic Query Examples")
    print("=" * 50)

    parser = CypherParser()
    planner = LogicalPlanner()
    optimizer = RuleBasedOptimizer()

    examples = [
        ("Simple node scan", "MATCH (n:Person) RETURN n.name"),
        ("Property filter", "MATCH (u:User {country: 'USA'}) RETURN u.name"),
        ("Simple relationship", "MATCH (u:User)-[:FOLLOWS]->(f) RETURN f.name"),
        ("With WHERE clause", "MATCH (p:Product) WHERE p.price > 100 RETURN p.name"),
        (
            "Multiple conditions",
            "MATCH (u:User) WHERE u.age > 18 AND u.active = true RETURN u.name",
        ),
    ]

    for title, query in examples:
        print(f"\n📝 {title}")
        print(f"Query: {query}")
        print("-" * 40)

        try:
            ast = parser.parse(query)
            plan = planner.create_logical_plan(ast)
            optimized = optimizer.optimize(plan)

            print("Optimized Plan:")
            print_plan(optimized)

            targets = analyze_plan_execution_targets(optimized)
            print(
                f"Targets: Redis={targets['redis']}, GraphBLAS={targets['graphblas']}"
            )

        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_basic_examples()
