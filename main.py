#!/usr/bin/env python3
"""
Updated Main entry point for Cypher to Logical Execution Plan Converter
Testing 5 representative queries from the comprehensive query set
"""

from cypher_planner.parser import CypherParser
from cypher_planner.logical_planner import LogicalPlanner
from cypher_planner.optimizer import RuleBasedOptimizer
from cypher_planner.physical_planner import PhysicalPlanner
from cypher_planner.logical_operators import print_plan, analyze_plan_execution_targets
from cypher_planner.physical_planner import print_physical_plan


def main():
    """Main function to test 5 representative Cypher queries"""

    print("🚀 Cypher to Logical Execution Plan Converter")
    print("Testing 5 Representative Queries from Comprehensive Set")
    print("=" * 70)

    # Initialize components
    parser = CypherParser()
    logical_planner = LogicalPlanner()
    optimizer = RuleBasedOptimizer()
    physical_planner = PhysicalPlanner()

    # 5 Selected test queries covering different query patterns
    test_queries = [
        # Query 1: Basic node filtering (Query #8 from your list)
        {
            "title": "Basic Property Filtering",
            "query": "MATCH (p:Person) WHERE p.age > 30 RETURN p.name, p.age",
            "description": "Simple node scan with property filter",
        },
        # Query 2: Relationship traversal with multiple conditions (Query #14 adapted)
        {
            "title": "Co-Actor Discovery",
            "query": "MATCH (actor:Actor {name: 'Tom Hanks'})-[:ACTED_IN]->(movie)<-[:ACTED_IN]-(coActor) RETURN coActor.name",
            "description": "Complex pattern with outgoing and incoming relationships",
        },
        # Query 3: Variable-length paths (Query #17 from your list)
        {
            "title": "Variable-Length Path Traversal",
            "query": "MATCH (person1:Person {name: 'Alice'})-[:KNOWS*1..3]-(person2) RETURN person2.name",
            "description": "Multi-hop traversal with variable path length",
        },
        # Query 4: Optional relationships (Query #19 from your list)
        {
            "title": "Optional Relationship Matching",
            "query": "MATCH (p:Person) OPTIONAL MATCH (p)-[:OWNS]->(c:Car) RETURN p.name, c.make",
            "description": "Left outer join pattern with optional relationships",
        },
        # Query 5: Relationship property filtering (Query #20 from your list)
        {
            "title": "Relationship Property Filtering",
            "query": "MATCH (p1:Person)-[r:FRIENDS_WITH]->(p2:Person) WHERE r.since > 2020 RETURN p1.name, p2.name",
            "description": "Filtering based on relationship properties",
        },
    ]

    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {test_case['title']}")
        print(f"Description: {test_case['description']}")
        print(f"Cypher: {test_case['query']}")
        print("-" * 80)

        try:
            # Parse the query
            print("🔍 Parsing query...")
            ast = parser.parse(test_case["query"])
            print("✅ Parsing successful")

            # Create initial logical plan
            print("🏗️  Creating logical plan...")
            initial_plan = logical_planner.create_logical_plan(ast)
            print("✅ Initial logical plan created")

            print("\n📋 Initial Logical Plan:")
            print_plan(initial_plan)

            # Apply optimizations
            print("\n⚡ Applying optimizations...")
            optimized_plan = optimizer.optimize(initial_plan)
            print("✅ Optimizations applied")

            print("\n📋 Optimized Logical Plan:")
            print_plan(optimized_plan)

            # Generate physical plan
            print("\n🔧 Generating physical plan...")
            physical_plan = physical_planner.create_physical_plan(optimized_plan)
            print("✅ Physical plan generated")

            print("\n📋 Physical Execution Plan:")
            print_physical_plan(physical_plan)

            # Show execution target analysis
            initial_targets = analyze_plan_execution_targets(initial_plan)
            optimized_targets = analyze_plan_execution_targets(optimized_plan)

            print("\n📊 Execution Target Analysis:")
            print(
                f"  Initial plan   - Redis: {initial_targets['redis']}, GraphBLAS: {initial_targets['graphblas']}, Mixed: {initial_targets['mixed']}"
            )
            print(
                f"  Optimized plan - Redis: {optimized_targets['redis']}, GraphBLAS: {optimized_targets['graphblas']}, Mixed: {optimized_targets['mixed']}"
            )

            # Show query complexity assessment
            print("\n💡 Query Complexity Assessment:")
            complexity = assess_query_complexity(test_case["query"], optimized_plan)
            print(f"  Complexity Score: {complexity['score']}/10")
            print(f"  Primary Operations: {', '.join(complexity['operations'])}")
            print(f"  Execution Strategy: {complexity['strategy']}")

        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback

            traceback.print_exc()

        print("\n" + "=" * 80)


def assess_query_complexity(query: str, plan) -> dict:
    """Assess the complexity of a query based on its patterns and execution plan"""

    complexity_score = 0
    operations = []

    # Analyze query patterns
    if "OPTIONAL" in query.upper():
        complexity_score += 2
        operations.append("Optional Matching")

    if "*" in query:
        complexity_score += 3
        operations.append("Variable-Length Paths")

    if query.count("-") > 2:  # Multiple relationships
        complexity_score += 2
        operations.append("Multi-Hop Traversal")

    if "WHERE" in query.upper():
        complexity_score += 1
        operations.append("Property Filtering")

    if "<-" in query:
        complexity_score += 1
        operations.append("Incoming Relationships")

    # Analyze execution plan
    targets = analyze_plan_execution_targets(plan)
    if targets["graphblas"] > 0 and targets["redis"] > 0:
        complexity_score += 1
        operations.append("Hybrid Execution")

    # Determine execution strategy
    if targets["graphblas"] > targets["redis"]:
        strategy = "GraphBLAS-Heavy (Matrix Operations Dominant)"
    elif targets["redis"] > targets["graphblas"]:
        strategy = "Redis-Heavy (Property Operations Dominant)"
    else:
        strategy = "Balanced Hybrid (Equal Redis/GraphBLAS Operations)"

    return {
        "score": min(complexity_score, 10),
        "operations": operations or ["Basic Pattern Matching"],
        "strategy": strategy,
    }


def interactive_mode():
    """Enhanced interactive mode with query suggestions"""

    print("\n🎯 Interactive Mode")
    print("Enter Cypher queries (type 'exit' to quit, 'examples' for query examples)")
    print("-" * 70)

    parser = CypherParser()
    logical_planner = LogicalPlanner()
    optimizer = RuleBasedOptimizer()
    physical_planner = PhysicalPlanner()

    # Query examples for users
    example_queries = [
        "MATCH (n:Person) RETURN n.name",
        "MATCH (p:Person) WHERE p.age > 25 RETURN p",
        "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, m.title",
        "MATCH (p1:Person)-[:KNOWS*1..2]-(p2:Person) RETURN p1.name, p2.name",
        "MATCH (u:User) OPTIONAL MATCH (u)-[:POSTED]->(t:Tweet) RETURN u.name, t.content",
    ]

    while True:
        try:
            query = input("\ncypher> ").strip()

            if query.lower() in ["exit", "quit", "q"]:
                print("👋 Goodbye!")
                break

            if query.lower() in ["examples", "help"]:
                print("\n📚 Example queries you can try:")
                for i, example in enumerate(example_queries, 1):
                    print(f"  {i}. {example}")
                continue

            if not query:
                continue

            print(f"\n🔍 Processing: {query}")

            # Parse and plan
            ast = parser.parse(query)
            initial_plan = logical_planner.create_logical_plan(ast)
            optimized_plan = optimizer.optimize(initial_plan)
            physical_plan = physical_planner.create_physical_plan(optimized_plan)

            # Show results
            print("\n📋 Optimized Logical Plan:")
            print_plan(optimized_plan)

            print("\n🔧 Physical Execution Plan:")
            print_physical_plan(physical_plan)

            # Analysis
            targets = analyze_plan_execution_targets(optimized_plan)
            complexity = assess_query_complexity(query, optimized_plan)

            print(f"\n📊 Analysis:")
            print(
                f"  Execution Targets - Redis: {targets['redis']}, GraphBLAS: {targets['graphblas']}, Mixed: {targets['mixed']}"
            )
            print(f"  Complexity Score: {complexity['score']}/10")
            print(f"  Strategy: {complexity['strategy']}")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("💡 Try 'examples' to see valid query patterns")


def benchmark_mode():
    """Benchmark mode to test query performance characteristics"""

    print("\n⚡ Benchmark Mode")
    print("Testing query performance characteristics")
    print("-" * 50)

    parser = CypherParser()
    logical_planner = LogicalPlanner()
    optimizer = RuleBasedOptimizer()

    # Benchmark queries with different complexity levels
    benchmark_queries = [
        ("Simple", "MATCH (n:Person) RETURN n.name"),
        (
            "Medium",
            "MATCH (p:Person)-[:KNOWS]->(f:Person) WHERE p.age > 30 RETURN f.name",
        ),
        (
            "Complex",
            "MATCH (a:Actor)-[:ACTED_IN*1..3]->(m:Movie)<-[:DIRECTED]-(d:Director) RETURN a.name, d.name",
        ),
        (
            "Very Complex",
            "MATCH (u:User)-[:FOLLOWS*2..4]->(f:User) OPTIONAL MATCH (f)-[:POSTED]->(t:Tweet) WHERE t.hashtags CONTAINS 'AI' RETURN u.name, collect(t.content)",
        ),
    ]

    results = []

    for level, query in benchmark_queries:
        try:
            print(f"\n🔥 Testing {level} Query:")
            print(f"   {query}")

            import time

            start_time = time.time()

            # Process query
            ast = parser.parse(query)
            logical_plan = logical_planner.create_logical_plan(ast)
            optimized_plan = optimizer.optimize(logical_plan)

            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # Convert to milliseconds

            # Analyze complexity
            targets = analyze_plan_execution_targets(optimized_plan)
            complexity = assess_query_complexity(query, optimized_plan)

            result = {
                "level": level,
                "processing_time_ms": processing_time,
                "complexity_score": complexity["score"],
                "redis_ops": targets["redis"],
                "graphblas_ops": targets["graphblas"],
                "mixed_ops": targets["mixed"],
            }

            results.append(result)

            print(f"   ✅ Processed in {processing_time:.2f}ms")
            print(f"   📊 Complexity: {complexity['score']}/10")
            print(
                f"   🎯 Operations: R={targets['redis']}, GB={targets['graphblas']}, M={targets['mixed']}"
            )

        except Exception as e:
            print(f"   ❌ Failed: {e}")

    # Summary
    print(f"\n📈 Benchmark Summary:")
    print("-" * 30)
    for result in results:
        print(
            f"{result['level']:12} | {result['processing_time_ms']:6.2f}ms | Score: {result['complexity_score']}/10 | Ops: {result['redis_ops']+result['graphblas_ops']+result['mixed_ops']}"
        )


if __name__ == "__main__":
    print("Choose mode:")
    print("1. Run 5 test queries (default)")
    print("2. Interactive mode")
    print("3. Benchmark mode")

    choice = input("\nEnter choice (1, 2, or 3): ").strip()

    if choice == "2":
        interactive_mode()
    elif choice == "3":
        benchmark_mode()
    else:
        main()

        # Offer other modes after tests
        next_mode = input(
            "\nTry another mode? (i)nteractive, (b)enchmark, or (n)o: "
        ).lower()
        if next_mode in ["i", "interactive"]:
            interactive_mode()
        elif next_mode in ["b", "benchmark"]:
            benchmark_mode()
