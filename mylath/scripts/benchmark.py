

# scripts/benchmark.py
"""
Benchmarking script for MyLath performance testing
"""

import time
import random
import statistics
from mylath import Graph, RedisStorage
import numpy as np


def benchmark_node_operations(graph, num_nodes=10000):
    """Benchmark node CRUD operations"""
    print(f"Benchmarking node operations with {num_nodes} nodes...")
    
    # Create nodes
    start_time = time.time()
    node_ids = []
    for i in range(num_nodes):
        node = graph.create_node("person", {
            "name": f"User_{i}",
            "age": random.randint(18, 80),
            "score": random.random()
        })
        node_ids.append(node.id)
    
    create_time = time.time() - start_time
    print(f"  Create {num_nodes} nodes: {create_time:.2f}s ({num_nodes/create_time:.0f} ops/s)")
    
    # Read nodes
    start_time = time.time()
    sample_ids = random.sample(node_ids, min(1000, len(node_ids)))
    for node_id in sample_ids:
        graph.get_node(node_id)
    
    read_time = time.time() - start_time
    print(f"  Read {len(sample_ids)} nodes: {read_time:.2f}s ({len(sample_ids)/read_time:.0f} ops/s)")
    
    return node_ids


def benchmark_edge_operations(graph, node_ids, num_edges=50000):
    """Benchmark edge operations"""
    print(f"Benchmarking edge operations with {num_edges} edges...")
    
    start_time = time.time()
    edge_ids = []
    
    for i in range(num_edges):
        from_node = random.choice(node_ids)
        to_node = random.choice(node_ids)
        
        if from_node != to_node:
            edge = graph.create_edge("knows", from_node, to_node, {
                "weight": random.random(),
                "created": time.time()
            })
            edge_ids.append(edge.id)
    
    create_time = time.time() - start_time
    print(f"  Create {len(edge_ids)} edges: {create_time:.2f}s ({len(edge_ids)/create_time:.0f} ops/s)")
    
    return edge_ids


def benchmark_traversals(graph, node_ids, num_queries=1000):
    """Benchmark graph traversals"""
    print(f"Benchmarking traversals with {num_queries} queries...")
    
    times = []
    
    for i in range(num_queries):
        start_node = random.choice(node_ids)
        
        start_time = time.time()
        results = graph.V(start_node).out("knows").limit(10).to_list()
        query_time = time.time() - start_time
        times.append(query_time)
    
    avg_time = statistics.mean(times)
    p95_time = statistics.quantiles(times, n=20)[18]  # 95th percentile
    
    print(f"  Average query time: {avg_time*1000:.2f}ms")
    print(f"  95th percentile: {p95_time*1000:.2f}ms")
    print(f"  Queries per second: {1/avg_time:.0f}")


def benchmark_vector_operations(graph, num_vectors=10000, dimension=128):
    """Benchmark vector operations"""
    print(f"Benchmarking vector operations with {num_vectors} vectors of dimension {dimension}...")
    
    # Add vectors
    start_time = time.time()
    vector_ids = []
    
    for i in range(num_vectors):
        vector_data = np.random.random(dimension).tolist()
        vector = graph.vectors.add_vector(vector_data, 
                                         metadata={"type": "test"},
                                         properties={"index": i})
        vector_ids.append(vector.id)
    
    add_time = time.time() - start_time
    print(f"  Add {num_vectors} vectors: {add_time:.2f}s ({num_vectors/add_time:.0f} ops/s)")
    
    # Search vectors
    start_time = time.time()
    num_searches = 100
    
    for i in range(num_searches):
        query_vector = np.random.random(dimension).tolist()
        results = graph.vectors.search_vectors(query_vector, k=10)
    
    search_time = time.time() - start_time
    avg_search_time = search_time / num_searches
    print(f"  Average search time: {avg_search_time*1000:.2f}ms")
    print(f"  Searches per second: {1/avg_search_time:.0f}")


def run_full_benchmark():
    """Run complete benchmark suite"""
    print("MyLath Performance Benchmark")
    print("=" * 40)
    
    # Initialize
    storage = RedisStorage(db=14)  # Use separate DB for benchmarks
    graph = Graph(storage)
    
    try:
        # Clear any existing data
        storage.redis.flushdb()
        
        # Run benchmarks
        node_ids = benchmark_node_operations(graph, num_nodes=10000)
        edge_ids = benchmark_edge_operations(graph, node_ids, num_edges=20000)
        benchmark_traversals(graph, node_ids, num_queries=1000)
        benchmark_vector_operations(graph, num_vectors=5000, dimension=128)
        
        # Final stats
        stats = graph.get_stats()
        print(f"\nFinal graph stats: {stats}")
        
    finally:
        # Cleanup
        storage.redis.flushdb()
    
    print("\nBenchmark completed! 🎉")


if __name__ == "__main__":
    run_full_benchmark()