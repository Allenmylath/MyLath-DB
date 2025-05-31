#!/usr/bin/env python3
"""
Final Quick Vector Test for MyLath with Redis Stack Support
Fixed to use DB=0 for Redis Stack compatibility
"""

import time
import random
import numpy as np

# Assuming mylath package structure is accessible
# For local testing, ensure 'mylath' is in your PYTHONPATH or run from parent directory
from mylath.mylath.storage.redis_storage import RedisStorage
from mylath.mylath.graph.graph import Graph # Graph class is expected to use VectorCore internally

def generate_random_vector(dimension=128):
    """Generate a random normalized vector."""
    # Normalize the vector to ensure consistent similarity calculations (e.g., cosine similarity)
    vec = np.random.random(dimension)
    return (vec / np.linalg.norm(vec)).tolist()

def quick_test():
    """Ultimate MyLath vector performance test."""
    try:
        print("🚀 MyLath Ultimate Vector Test")
        print("=" * 60)
        
        # CRITICAL: Use db=0 for Redis Stack compatibility.
        # Ensure your Redis Stack instance is running on localhost:6379.
        storage = RedisStorage(host='localhost', port=6379, db=0)
        
        # Flush the database to ensure a clean test environment.
        storage.redis.flushdb()
        
        # The Graph class is expected to instantiate and use VectorCore,
        # which now handles Redis Search integration.
        graph = Graph(storage)
        
        # Test parameters
        num_vectors = 1000
        vector_dim = 128
        
        print(f"📊 Parameters: {num_vectors:,} vectors × {vector_dim} dimensions")
        
        # Check Redis Stack status by trying to list RediSearch indices.
        # This confirms if the RediSearch module is loaded and available.
        try:
            # FT._LIST is a RediSearch command to list all indices.
            indices = storage.redis.execute_command("FT._LIST")
            print(f"✅ Redis Stack available (DB=0)")
            redis_stack_available = True
        except Exception as e:
            print(f"❌ Redis Stack not available or RediSearch module not loaded: {e}")
            redis_stack_available = False
        
        # PHASE 1: INGESTION TEST - Adding vectors to MyLath.
        print(f"\n📥 INGESTION TEST")
        print("-" * 40)
        
        categories = ["document", "image", "audio", "video", "text"]
        vectors = [] # To store references to added vectors
        
        start_time = time.time()
        
        for i in range(num_vectors):
            # Generate a random vector for ingestion.
            vector_data = generate_random_vector(vector_dim)
            
            # Add metadata and properties to each vector.
            category = random.choice(categories)
            metadata = {"type": category, "source": f"file_{i}"}
            properties = {"title": f"{category}_{i}", "score": round(random.uniform(0.1, 1.0), 3)}
            
            # Add the vector to MyLath (which uses VectorCore internally).
            vector = graph.vectors.add_vector(vector_data, metadata, properties)
            vectors.append(vector)
            
            # Print progress updates periodically.
            if (i + 1) % 200 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"    {i + 1:4d} vectors | {rate:6.1f} vec/sec | {elapsed:5.1f}s")
        
        ingestion_time = time.time() - start_time
        ingestion_rate = num_vectors / ingestion_time
        
        print(f"\n📊 INGESTION RESULTS:")
        print(f"    Time: {ingestion_time:.2f} seconds")
        print(f"    Rate: {ingestion_rate:.1f} vectors/second")
        print(f"    Per vector: {(ingestion_time/num_vectors)*1000:.2f} ms")
        
        # PHASE 2: BACKEND DETECTION - Verify which backend MyLath is using.
        print(f"\n🔍 BACKEND ANALYSIS")
        print("-" * 40)
        
        # Check Redis keys to infer backend usage.
        # 'vec:*' keys indicate Redis Search is being used.
        # 'vectors:*' keys indicate Python fallback storage.
        redis_search_keys = len(storage.redis.keys("vec:*"))
        python_keys = len(storage.redis.keys("vectors:*"))
        
        # Check the internal 'vector_index' set used by the Python fallback.
        index_keys = storage.redis.scard("vector_index")
        
        backend_used = "Unknown"
        expected_search_time = "N/A"

        # Determine the backend based on the presence of specific key patterns.
        if redis_search_keys > 0:
            backend_used = "Redis Search"
            print(f"✅ Backend: Redis Search")
            print(f"    vec:* keys: {redis_search_keys}")
            expected_search_time = "5-20ms"
        elif python_keys > 0:
            backend_used = "Python"
            print(f"⚠️  Backend: Python fallback")
            print(f"    vectors:* keys: {python_keys}")
            expected_search_time = "100-500ms"
        else:
            print(f"❓ No vector keys found in Redis. Check ingestion process.")
            
        print(f"    vector_index entries (Python fallback set): {index_keys}")
        print(f"    Expected search time: {expected_search_time}")
        
        # Verify if the RediSearch index 'mylath_vectors' exists.
        if redis_stack_available:
            try:
                info = storage.redis.ft("mylath_vectors").info()
                print(f"✅ MyLath vector index 'mylath_vectors' created and accessible.")
            except redis.exceptions.ResponseError as e:
                print(f"❌ MyLath vector index 'mylath_vectors' missing or inaccessible: {e}")
        
        # PHASE 3: SEARCH PERFORMANCE TEST - Measure similarity search speed.
        print(f"\n🔍 SEARCH PERFORMANCE TEST")
        print("-" * 40)
        
        search_times = []
        num_searches = 10
        k_results = 10 # Number of top similar results to retrieve
        
        for i in range(num_searches):
            # Generate a random query vector.
            query_vector = generate_random_vector(vector_dim)
            
            # Time the search operation.
            start_time = time.time()
            results = graph.vectors.search_vectors(query_vector, k=k_results)
            search_time = time.time() - start_time
            
            search_times.append(search_time)
            print(f"    Search {i+1:2d}: {search_time*1000:7.2f}ms | {len(results)} results")
        
        # Calculate search statistics.
        avg_search = sum(search_times) / len(search_times)
        min_search = min(search_times)
        max_search = max(search_times)
        
        print(f"\n📊 SEARCH RESULTS:")
        print(f"    Average: {avg_search*1000:7.2f} ms")
        print(f"    Min:     {min_search*1000:7.2f} ms")
        print(f"    Max:     {max_search*1000:7.2f} ms")
        print(f"    Rate:    {1/avg_search:7.1f} searches/sec")
        
        # PHASE 4: FILTERED SEARCH TEST - Measure search speed with metadata filters.
        print(f"\n🎯 FILTERED SEARCH TEST")
        print("-" * 40)
        
        filter_tests = [
            {"type": "document"},
            {"type": "image"},
            {"score": "0.5"} # Example filter for a property
        ]
        
        for filter_dict in filter_tests:
            query_vector = generate_random_vector(vector_dim)
            
            start_time = time.time()
            filtered_results = graph.vectors.search_vectors(query_vector, k=5, filters=filter_dict)
            filter_time = time.time() - start_time
            
            filter_key_str = ", ".join([f"{k}={v}" for k,v in filter_dict.items()])
            print(f"    Filter ({filter_key_str}): {filter_time*1000:6.2f}ms | {len(filtered_results)} results")
        
        # PHASE 5: SAMPLE RESULTS - Display top search results for inspection.
        print(f"\n📋 TOP 10 SEARCH RESULTS")
        print("-" * 40)
        
        query_vector = generate_random_vector(vector_dim)
        top_results = graph.vectors.search_vectors(query_vector, k=10)
        
        print(f"Query: [{', '.join([f'{x:.3f}' for x in query_vector[:6]])}...]")
        print(f"Top {len(top_results)} matches:")
        
        for i, (vector, score) in enumerate(top_results, 1):
            title = vector.properties.get("title", "Unknown")[:20]
            vector_type = vector.metadata.get("type", "unknown")
            vector_preview = [f'{x:.3f}' for x in vector.data[:4]]
            
            print(f"    {i:2d}. Score: {score:.4f} | Type: {vector_type:8s} | {title}")
            print(f"          Data: [{', '.join(vector_preview)}...] | ID: {vector.id[:8]}...")
        
        # PHASE 6: PERFORMANCE ANALYSIS - Summarize and grade performance.
        print(f"\n🏆 PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Determine search performance grade.
        search_grade = "N/A"
        performance_note = "No search performed or backend unknown."
        if backend_used == "Redis Search":
            if avg_search < 0.01:  # < 10ms
                search_grade = "🚀 Excellent"
                performance_note = "Redis Stack working optimally for search!"
            elif avg_search < 0.05:  # < 50ms
                search_grade = "⚡ Good"
                performance_note = "Redis Stack working well for search."
            else:
                search_grade = "⚠️  Slow"
                performance_note = "Redis Stack search may have issues or needs tuning."
        elif backend_used == "Python":
            if avg_search < 0.1:  # < 100ms
                search_grade = "⚡ Fast"
                performance_note = "Python fallback performing surprisingly well."
            elif avg_search < 0.5:  # < 500ms
                search_grade = "✅ Normal"
                performance_note = "Typical Python fallback performance."
            else:
                search_grade = "⚠️  Slow"
                performance_note = "Python fallback is slow. Consider Redis Stack."
        
        # Determine ingestion performance grade.
        ingestion_grade = "N/A"
        if ingestion_rate > 1000:
            ingestion_grade = "🚀 Excellent"
        elif ingestion_rate > 500:
            ingestion_grade = "⚡ Good"
        elif ingestion_rate > 200:
            ingestion_grade = "✅ Fair"
        else:
            ingestion_grade = "⚠️  Slow"
        
        print(f"Backend: {backend_used}")
        print(f"Database: Redis DB 0 ({'✅ Redis Stack compatible' if redis_stack_available else '❌ No Redis Stack'})")
        print(f"")
        print(f"📥 Ingestion Performance:")
        print(f"    Grade: {ingestion_grade}")
        print(f"    Rate:  {ingestion_rate:7.1f} vectors/second")
        print(f"    Time:  {ingestion_time:7.2f} seconds")
        print(f"")
        print(f"🔍 Search Performance:")
        print(f"    Grade: {search_grade}")
        print(f"    Time:  {avg_search*1000:7.2f} ms average")
        print(f"    Rate:  {1/avg_search:7.1f} searches/second")
        print(f"    Note:  {performance_note}")
        print(f"")
        print(f"💾 Storage Efficiency:")
        print(f"    Vectors: {len(vectors):,}")
        # Rough estimate for memory usage (FLOAT32 is 4 bytes per dimension)
        print(f"    Memory:  ~{len(vectors) * vector_dim * 4 / (1024*1024):.1f} MB (estimated vector data)")
        print(f"    Density: {len(vectors) / (ingestion_time + sum(search_times)):.1f} ops/sec overall")
        
        # PHASE 7: RECOMMENDATIONS - Provide actionable advice.
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 40)
        
        if backend_used == "Python" and redis_stack_available:
            print("🔥 PRIORITY: MyLath is not fully leveraging Redis Stack!")
            print("    • Ensure 'mylath_vectors' index is created and populated.")
            print("    • Verify VectorCore initialization and Redis Search integration.")
            print("    • A potential 10-100x speedup is available with Redis Search.")
        elif backend_used == "Python" and not redis_stack_available:
            print("🚀 Install Redis Stack for massive speedup:")
            print("    docker run -d -p 6379:6379 redis/redis-stack-server")
        elif backend_used == "Redis Search" and avg_search > 0.05:
            print("⚙️  Redis Stack optimization opportunities:")
            print("    • Tune HNSW parameters (M, EF_CONSTRUCTION, EF_SEARCH) in VectorCore.")
            print("    • Consider vector dimensionality reduction if appropriate.")
        else:
            print("✅ System optimally configured!")
            print(f"    • Ingestion: {ingestion_rate:.0f} vectors/sec")
            print(f"    • Search: {avg_search*1000:.1f}ms average")
        
        # Capacity projections based on current performance.
        print(f"")
        print(f"📈 Capacity Projections:")
        print(f"    • Vectors per minute: {ingestion_rate * 60:,.0f}")
        print(f"    • Searches per minute: {(1/avg_search) * 60:,.0f}")
        print(f"    • Daily vector capacity: {ingestion_rate * 3600 * 8:,.0f} (8hr workday)")
        
        # Cleanup: Flush the database after the test.
        storage.redis.flushdb()
        print(f"\n🧹 Test data cleaned up")
        
        # Final summary of the test.
        print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"")
        print(f"📊 FINAL SUMMARY:")
        print(f"    Backend: {backend_used}")
        print(f"    Ingestion: {ingestion_rate:.0f} vec/sec ({ingestion_grade})")
        print(f"    Search: {avg_search*1000:.1f}ms ({search_grade})")
        print(f"    Overall: {performance_note}")
        
        return {
            'backend': backend_used,
            'ingestion_rate': ingestion_rate,
            'search_time': avg_search,
            'redis_stack_available': redis_stack_available
        }
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

if __name__ == "__main__":
    print("MyLath Ultimate Vector Performance Test")
    print("Testing Redis Stack integration and performance")
    print()
    
    # Quick Redis connection check before running the full test.
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection OK")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Please ensure Redis Stack is running on localhost:6379 and accessible.")
        exit(1)
    
    # Run the comprehensive performance test.
    results = quick_test()
    
    if results:
        print(f"\n🎉 MyLath is ready for production use (with Redis Stack if configured correctly)!")
    else:
        print(f"\n❌ Issues detected - check output above for details.")

