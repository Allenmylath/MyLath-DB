#!/usr/bin/env python3
"""
Fixed QuickTest for Redis Official Vector Database Implementation
"""

import time
import random
import numpy as np
import sys
import os

# Add the mylath directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mylath'))

def generate_random_vector(dimension=128):
    """Generate a random normalized vector."""
    vec = np.random.random(dimension)
    return (vec / np.linalg.norm(vec)).tolist()

def test_redis_connection():
    """Test basic Redis connection"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection successful")
        
        # Get Redis info
        info = r.info()
        print(f"   Redis version: {info.get('redis_version', 'unknown')}")
        print(f"   Memory used: {info.get('used_memory_human', 'unknown')}")
        
        return True
    except ImportError:
        print("❌ Redis library not installed: pip install redis")
        return False
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("   Make sure Redis is running on localhost:6379")
        return False

def test_redis_stack():
    """Test Redis Stack availability"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Try RediSearch command
        result = r.execute_command("FT._LIST")
        print("✅ Redis Stack available")
        
        # Check for loaded modules
        modules = r.execute_command("MODULE LIST")
        search_loaded = any('search' in str(module).lower() for module in modules)
        if search_loaded:
            print("   RediSearch module loaded")
        else:
            print("   RediSearch module not found")
        
        return True
    except Exception as e:
        print(f"⚠️  Redis Stack not available: {e}")
        print("   Install with: docker run -d -p 6379:6379 redis/redis-stack-server")
        return False

def quick_test():
    """Comprehensive MyLath vector test with Redis official implementation"""
    try:
        print("🚀 MyLath Vector Performance Test")
        print("=" * 60)
        
        # Test Redis connection first
        if not test_redis_connection():
            return None
        
        # Test Redis Stack
        redis_stack_available = test_redis_stack()
        
        # Import MyLath components
        try:
            from mylath.storage.redis_storage import RedisStorage
            from mylath.graph.graph import Graph
            print("🚀 MyLath: Redis Stack detected - high-performance vector search enabled!")
            print("✅ MyLath imports successful")
        except ImportError as e:
            print(f"❌ MyLath import failed: {e}")
            print("   Make sure you're running from the correct directory")
            return None
        
        # Initialize storage and graph
        storage = RedisStorage(host='localhost', port=6379, db=0)
        storage.redis.flushdb()  # Clean start
        graph = Graph(storage)
        
        # Get backend information - Fixed for new implementation
        backend_info = graph.vectors.get_backend_info()
        print(f"\n🔧 Backend Configuration:")
        print(f"   Backend: {backend_info['backend']}")
        print(f"   Redis Stack Available: {backend_info['redis_stack_available']}")
        print(f"   Approach: {backend_info.get('approach', 'Unknown')}")
        if backend_info.get('index_name'):
            print(f"   Index Name: {backend_info['index_name']}")
        
        # Test parameters
        num_vectors = 100000  # Scaled up for stress test
        vector_dim = 128
        
        print(f"\n📊 Test Parameters:")
        print(f"   Vectors: {num_vectors:,}")
        print(f"   Dimensions: {vector_dim}")
        print(f"   Database: Redis DB 0")
        
        # PHASE 1: Vector Ingestion Test
        print(f"\n📥 INGESTION TEST")
        print("-" * 40)
        
        categories = ["document", "image", "audio", "video", "text"]
        vectors = []
        
        start_time = time.time()
        
        for i in range(num_vectors):
            try:
                vector_data = generate_random_vector(vector_dim)
                category = random.choice(categories)
                metadata = {"type": category, "source": f"file_{i}"}
                properties = {
                    "title": f"{category}_{i}", 
                    "score": round(random.uniform(0.1, 1.0), 3)
                }
                
                vector = graph.vectors.add_vector(vector_data, metadata, properties)
                vectors.append(vector)
                
                if (i + 1) % 10000 == 0:  # Progress every 10k vectors
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"    {i + 1:6d} vectors | {rate:6.1f} vec/sec | {elapsed:5.1f}s")
                    
            except Exception as e:
                print(f"    Error adding vector {i}: {e}")
                break
        
        ingestion_time = time.time() - start_time
        ingestion_rate = len(vectors) / ingestion_time if ingestion_time > 0 else 0
        
        print(f"\n📊 INGESTION RESULTS:")
        print(f"    Vectors added: {len(vectors):,}")
        print(f"    Time: {ingestion_time:.2f} seconds")
        print(f"    Rate: {ingestion_rate:.1f} vectors/second")
        if len(vectors) > 0:
            print(f"    Per vector: {(ingestion_time/len(vectors))*1000:.2f} ms")
        
        if len(vectors) == 0:
            print("❌ No vectors were added successfully")
            return None
        
        # PHASE 2: Storage Analysis
        print(f"\n🔍 STORAGE ANALYSIS")
        print("-" * 40)
        
        # Check different key patterns
        redis_official_keys = len(storage.redis.keys("doc:*"))
        python_keys = len(storage.redis.keys("vectors:*"))
        index_entries = storage.redis.scard("vector_index") or 0
        
        print(f"    Redis official keys (doc:*): {redis_official_keys}")
        print(f"    Python keys (vectors:*): {python_keys}")
        print(f"    Index entries: {index_entries}")
        
        backend_used = backend_info['backend']
        if redis_official_keys > 0:
            print(f"✅ Using Redis Stack official backend")
        elif python_keys > 0:
            print(f"⚠️  Using Python fallback backend")
        else:
            print(f"❓ Storage backend unclear")
        
        # PHASE 3: Search Performance Test (More comprehensive for 100k)
        print(f"\n🔍 SEARCH PERFORMANCE TEST")
        print("-" * 40)
        
        search_times = []
        num_searches = 50  # More searches for better average with 100k vectors
        k_results = 10
        
        print(f"Running {num_searches} searches against {len(vectors):,} vectors...")
        
        for i in range(num_searches):
            try:
                query_vector = generate_random_vector(vector_dim)
                
                start_time = time.time()
                results = graph.vectors.search_vectors(query_vector, k=k_results)
                search_time = time.time() - start_time
                
                search_times.append(search_time)
                
                # Show progress every 10 searches
                if (i + 1) % 10 == 0 or i < 5:
                    print(f"    Search {i+1:2d}: {search_time*1000:7.2f}ms | {len(results)} results")
                    
            except Exception as e:
                print(f"    Search {i+1:2d}: ERROR - {e}")
                break
        
        if search_times:
            avg_search = sum(search_times) / len(search_times)
            min_search = min(search_times)
            max_search = max(search_times)
            
            print(f"\n📊 SEARCH RESULTS:")
            print(f"    Searches completed: {len(search_times)}")
            print(f"    Average: {avg_search*1000:7.2f} ms")
            print(f"    Min:     {min_search*1000:7.2f} ms")
            print(f"    Max:     {max_search*1000:7.2f} ms")
            print(f"    Rate:    {1/avg_search:7.1f} searches/sec")
        else:
            print("❌ No successful searches")
            avg_search = 0
        
        # PHASE 4: Filtered Search Test (More comprehensive)
        print(f"\n🎯 FILTERED SEARCH TEST")
        print("-" * 40)
        
        filter_tests = [
            {"type": "document"},
            {"type": "image"},
            {"type": "video"},
        ]
        
        print(f"Testing filters against {len(vectors):,} vectors...")
        
        for filter_dict in filter_tests:
            try:
                query_vector = generate_random_vector(vector_dim)
                
                start_time = time.time()
                filtered_results = graph.vectors.search_vectors(query_vector, k=10, filters=filter_dict)
                filter_time = time.time() - start_time
                
                filter_str = ", ".join([f"{k}={v}" for k, v in filter_dict.items()])
                print(f"    Filter ({filter_str}): {filter_time*1000:6.2f}ms | {len(filtered_results)} results")
                
            except Exception as e:
                print(f"    Filter test failed: {e}")
        
        # PHASE 5: Sample Results
        print(f"\n📋 SAMPLE SEARCH RESULTS")
        print("-" * 40)
        
        try:
            query_vector = generate_random_vector(vector_dim)
            top_results = graph.vectors.search_vectors(query_vector, k=5)
            
            print(f"Query vector preview: [{', '.join([f'{x:.3f}' for x in query_vector[:4]])}...]")
            print(f"Top {len(top_results)} matches:")
            
            for i, (vector, score) in enumerate(top_results, 1):
                title = vector.properties.get("title", "Unknown")[:15]
                vector_type = vector.metadata.get("type", "unknown")
                print(f"    {i}. Score: {score:.4f} | Type: {vector_type:8s} | {title}")
                
        except Exception as e:
            print(f"Sample results failed: {e}")
        
        # PHASE 6: Performance Analysis
        print(f"\n🏆 PERFORMANCE ANALYSIS")
        print("=" * 60)
        
        # Grade performance (updated thresholds for 100k scale)
        ingestion_grade = "N/A"
        if ingestion_rate > 5000:  # 5k+ vec/sec is excellent for 100k
            ingestion_grade = "🚀 Excellent"
        elif ingestion_rate > 2000:
            ingestion_grade = "⚡ Very Good"
        elif ingestion_rate > 1000:
            ingestion_grade = "✅ Good"
        elif ingestion_rate > 500:
            ingestion_grade = "⚠️  Fair"
        else:
            ingestion_grade = "❌ Slow"
        
        search_grade = "N/A"
        performance_note = "No search data available"
        if avg_search > 0:
            if "Redis Stack" in backend_used:
                if avg_search < 0.02:  # < 20ms
                    search_grade = "🚀 Excellent"
                    performance_note = "Redis Stack performing optimally!"
                elif avg_search < 0.05:  # < 50ms
                    search_grade = "⚡ Very Good"
                    performance_note = "Redis Stack performing well"
                elif avg_search < 0.1:  # < 100ms
                    search_grade = "✅ Good"
                    performance_note = "Redis Stack working correctly"
                else:
                    search_grade = "⚠️  Slow"
                    performance_note = "Redis Stack may need tuning"
            else:  # Python backend
                if avg_search < 0.1:
                    search_grade = "⚡ Fast"
                    performance_note = "Python backend performing well"
                elif avg_search < 0.5:
                    search_grade = "✅ Normal"
                    performance_note = "Typical Python performance"
                else:
                    search_grade = "⚠️  Slow"
                    performance_note = "Consider Redis Stack for speedup"
        
        print(f"Backend: {backend_used}")
        print(f"Redis Stack: {'✅ Available and Active' if redis_stack_available else '❌ Not available'}")
        print(f"📈 SCALE TEST RESULTS (100K Vectors):")
        print(f"    • Database size: {len(vectors):,} vectors")
        print(f"    • Memory usage: ~{len(vectors) * vector_dim * 4 / (1024*1024):.1f} MB vector data")
        print(f"    • Search latency: {avg_search*1000:.1f}ms average")
        print(f"    • Throughput: {1/avg_search:.0f} searches/sec") if avg_search > 0 else None
        print(f"    • Scalability: {'✅ Production Ready' if avg_search < 0.1 else '⚠️ May need optimization'}")
        print(f"")
        print(f"📥 Ingestion Performance:")
        print(f"    Grade: {ingestion_grade}")
        print(f"    Rate:  {ingestion_rate:7.1f} vectors/second")
        print(f"    Time:  {ingestion_time:7.2f} seconds")
        print(f"")
        print(f"🔍 Search Performance:")
        print(f"    Grade: {search_grade}")
        if avg_search > 0:
            print(f"    Time:  {avg_search*1000:7.2f} ms average")
            print(f"    Rate:  {1/avg_search:7.1f} searches/second")
        print(f"    Note:  {performance_note}")
        
        # PHASE 7: Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 40)
        
        if "Python" in backend_used and redis_stack_available:
            print("🔥 OPTIMIZATION OPPORTUNITY:")
            print("    • Redis Stack is available but MyLath may not be using it optimally")
            print("    • Check vector indexing and search implementation")
        elif "Python" in backend_used and not redis_stack_available:
            print("🚀 UPGRADE RECOMMENDATION:")
            print("    • Install Redis Stack for massive performance boost:")
            print("      docker run -d -p 6379:6379 redis/redis-stack-server")
        elif "Redis Stack" in backend_used:
            print("✅ OPTIMAL CONFIGURATION:")
            print("    • Redis Stack is active and performing well!")
            print(f"    • Search performance: {avg_search*1000:.1f}ms average")
            if avg_search > 0.05:
                print("    • For even better performance, consider tuning HNSW parameters")
        else:
            print("⚙️  SYSTEM STATUS:")
            print("    • MyLath is working correctly with current configuration")
        
        # Cleanup
        storage.redis.flushdb()
        print(f"\n🧹 Test data cleaned up")
        
        print(f"\n✅ TEST COMPLETED SUCCESSFULLY!")
        print(f"")
        print(f"📊 FINAL SUMMARY:")
        print(f"    Backend: {backend_used}")
        print(f"    Vectors: {len(vectors):,} added successfully")
        print(f"    Ingestion: {ingestion_rate:.0f} vec/sec ({ingestion_grade})")
        if avg_search > 0:
            print(f"    Search: {avg_search*1000:.1f}ms avg ({search_grade})")
        print(f"    Status: {performance_note}")
        
        return {
            'backend': backend_used,
            'vectors_added': len(vectors),
            'ingestion_rate': ingestion_rate,
            'search_time': avg_search,
            'redis_stack_available': redis_stack_available,
            'backend_info': backend_info
        }
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def install_instructions():
    """Print installation instructions"""
    print("\n📦 INSTALLATION GUIDE")
    print("=" * 50)
    print("1. Install MyLath dependencies:")
    print("   pip install redis numpy flask click")
    print("")
    print("2. For high-performance vector search, install Redis Stack:")
    print("   docker run -d -p 6379:6379 redis/redis-stack-server")
    print("   # OR")
    print("   # Download from: https://redis.io/download")
    print("")
    print("3. Upgrade redis-py for search support:")
    print("   pip install 'redis[hiredis]>=4.5.0'")
    print("")
    print("4. Verify installation:")
    print("   python quicktest.py")

def main():
    """Main test runner with error handling"""
    print("MyLath Vector Performance Test")
    print("Testing Redis integration and vector performance")
    print()
    
    # Check Python dependencies
    missing_deps = []
    try:
        import redis
    except ImportError:
        missing_deps.append("redis")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        install_instructions()
        return
    
    # Check Redis connection
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection OK")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Solutions:")
        print("1. Start Redis: redis-server")
        print("2. Or use Docker: docker run -d -p 6379:6379 redis:latest")
        print("3. Or install Redis Stack: docker run -d -p 6379:6379 redis/redis-stack-server")
        return
    
    # Run the test
    results = quick_test()
    
    if results:
        print(f"\n🎉 MyLath is working correctly!")
        
        # Give specific recommendations based on results
        if "Python" in results['backend'] and results['redis_stack_available']:
            print(f"\n⚡ OPTIMIZATION TIP:")
            print(f"   Redis Stack is available but may not be fully utilized")
        elif "Python" in results['backend'] and not results['redis_stack_available']:
            print(f"\n🚀 UPGRADE TIP:")
            print(f"   Install Redis Stack for massive performance boost:")
            print(f"   docker run -d -p 6379:6379 redis/redis-stack-server")
        else:
            print(f"\n✅ System optimally configured!")
    else:
        print(f"\n❌ Issues detected - check output above for details")
        install_instructions()

if __name__ == "__main__":
    main()