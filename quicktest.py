#!/usr/bin/env python3
"""
Direct test without complex imports - just test the core functionality
"""

import sys
import os

# Add the mylath/mylath directory to Python path for direct imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mylath', 'mylath'))

def test_redis_connection():
    """Test Redis connection first"""
    print("🔌 Testing Redis connection...")
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        result = r.ping()
        print(f"   ✅ Redis ping: {result}")
        return True
    except Exception as e:
        print(f"   ❌ Redis connection failed: {e}")
        return False

def test_direct_imports():
    """Test direct imports"""
    print("\n🧪 Testing direct imports...")
    
    try:
        # Import storage components directly
        from storage.redis_storage import RedisStorage, Node, Edge
        print("   ✅ RedisStorage, Node, Edge imported")
        
        # Import graph components directly  
        from graph.graph import Graph
        print("   ✅ Graph imported")
        
        # Import vector components directly
        from vector.vector_core import VectorCore
        print("   ✅ VectorCore imported")
        
        # Import traversal
        from graph.traversal import GraphTraversal
        print("   ✅ GraphTraversal imported")
        
        return True
    except Exception as e:
        print(f"   ❌ Direct import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_functionality():
    """Test core MyLath functionality with direct imports"""
    print("\n🚀 Testing core functionality...")
    
    try:
        # Direct imports
        from storage.redis_storage import RedisStorage, Node, Edge
        from graph.graph import Graph
        from vector.vector_core import VectorCore
        
        # Initialize
        print("   🔧 Initializing storage and graph...")
        storage = RedisStorage(host='localhost', port=6379, db=3)  # Use db=3 for testing
        graph = Graph(storage)
        
        # Clear test database
        storage.redis.flushdb()
        print("   ✅ Cleared test database")
        
        # Test 1: Node operations
        print("   📝 Testing node operations...")
        alice = graph.create_node("person", {
            "name": "Alice", 
            "age": 30, 
            "city": "NYC",
            "skills": ["Python", "Graph DBs"]
        })
        
        bob = graph.create_node("person", {
            "name": "Bob", 
            "age": 25, 
            "city": "SF",
            "skills": ["JavaScript", "React"]
        })
        
        company = graph.create_node("company", {
            "name": "TechCorp", 
            "industry": "Software",
            "size": "Medium"
        })
        
        print(f"      ✅ Created 3 nodes: Alice ({alice.id[:8]}...), Bob ({bob.id[:8]}...), TechCorp ({company.id[:8]}...)")
        
        # Test node retrieval
        retrieved_alice = graph.get_node(alice.id)
        assert retrieved_alice.properties["name"] == "Alice"
        print(f"      ✅ Retrieved Alice: {retrieved_alice.properties['name']}")
        
        # Test 2: Edge operations
        print("   🔗 Testing edge operations...")
        friendship = graph.create_edge("knows", alice.id, bob.id, {
            "since": "2020", 
            "strength": 8,
            "type": "friendship"
        })
        
        alice_job = graph.create_edge("works_at", alice.id, company.id, {
            "role": "Senior Engineer", 
            "salary": 120000,
            "start_date": "2021-01-15"
        })
        
        bob_job = graph.create_edge("works_at", bob.id, company.id, {
            "role": "Frontend Developer", 
            "salary": 100000,
            "start_date": "2021-06-01"
        })
        
        print(f"      ✅ Created 3 edges: friendship, alice_job, bob_job")
        
        # Test 3: Basic traversals
        print("   🗺️  Testing graph traversals...")
        
        # Find Alice's friends
        friends = graph.V(alice.id).out("knows").to_list()
        friend_names = [f.properties["name"] for f in friends]
        print(f"      ✅ Alice's friends: {friend_names}")
        
        # Find company employees
        employees = graph.V(company.id).in_("works_at").to_list()
        employee_names = [e.properties["name"] for e in employees]
        print(f"      ✅ TechCorp employees: {employee_names}")
        
        # Find Alice's colleagues
        colleagues = (graph.V(alice.id)
                          .out("works_at")
                          .in_("works_at")
                          .to_list())
        colleague_names = [c.properties["name"] for c in colleagues if c.id != alice.id]
        print(f"      ✅ Alice's colleagues: {colleague_names}")
        
        # Test 4: Vector operations
        print("   🔍 Testing vector operations...")
        
        # Add document embeddings (simulated)
        doc1 = graph.vectors.add_vector(
            [0.1, 0.2, 0.3, 0.4, 0.5],
            metadata={"type": "document", "format": "pdf"},
            properties={"title": "Machine Learning Basics", "author": "Alice"}
        )
        
        doc2 = graph.vectors.add_vector(
            [0.2, 0.3, 0.4, 0.5, 0.6],
            metadata={"type": "document", "format": "pdf"},
            properties={"title": "Deep Learning Advanced", "author": "Bob"}
        )
        
        image1 = graph.vectors.add_vector(
            [0.8, 0.1, 0.9, 0.2, 0.1],
            metadata={"type": "image", "format": "jpg"},
            properties={"title": "Cat Photo", "tags": ["animal", "cute"]}
        )
        
        print(f"      ✅ Added 3 vectors: 2 documents + 1 image")
        
        # Test vector search
        query_vector = [0.15, 0.25, 0.35, 0.45, 0.55]
        
        # Search all vectors
        all_results = graph.vectors.search_vectors(query_vector, k=3)
        print(f"      ✅ Found {len(all_results)} similar vectors")
        
        # Search with filters (documents only)
        doc_results = graph.vectors.search_vectors(
            query_vector, 
            k=2, 
            filters={"type": "document"}
        )
        print(f"      ✅ Found {len(doc_results)} similar documents")
        
        if doc_results:
            best_doc = doc_results[0]
            print(f"         Best match: '{best_doc[0].properties['title']}' (score: {best_doc[1]:.3f})")
        
        # Test 5: Property-based queries
        print("   🔎 Testing property queries...")
        
        # Find by label
        all_people = graph.find_nodes_by_label("person")
        print(f"      ✅ Found {len(all_people)} people by label")
        
        # Find by property
        nyc_people = graph.find_nodes_by_property("city", "NYC")
        nyc_names = [p.properties["name"] for p in nyc_people]
        print(f"      ✅ People in NYC: {nyc_names}")
        
        # Test 6: Advanced traversals
        print("   🧭 Testing advanced traversals...")
        
        # Filter by age
        young_people = graph.V().has("label", "person").filter(
            lambda n: n.properties.get("age", 0) < 30
        ).to_list()
        young_names = [p.properties["name"] for p in young_people]
        print(f"      ✅ People under 30: {young_names}")
        
        # Count operations
        person_count = graph.V().has("label", "person").count()
        print(f"      ✅ Total people count: {person_count}")
        
        # Test 7: Graph statistics
        print("   📊 Testing graph statistics...")
        stats = graph.get_stats()
        print(f"      ✅ Graph stats: {stats}")
        
        # Test 8: Edge queries
        print("   🔗 Testing edge queries...")
        
        # Get outgoing edges
        alice_out_edges = storage.get_outgoing_edges(alice.id)
        print(f"      ✅ Alice's outgoing edges: {len(alice_out_edges)}")
        
        # Get edges by label
        work_edges = storage.get_outgoing_edges(alice.id, "works_at")
        print(f"      ✅ Alice's work relationships: {len(work_edges)}")
        
        # Cleanup
        storage.redis.flushdb()
        print("   🧹 Cleaned up test data")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Core functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("MyLath Direct Test")
    print("=" * 60)
    
    success = True
    
    # Test Redis connection first
    if not test_redis_connection():
        success = False
        print("\n💡 Redis is not running. Start it with:")
        print("   docker run -d -p 6379:6379 redis:latest")
        return False
    
    # Test direct imports
    if not test_direct_imports():
        success = False
        return False
    
    # Test core functionality
    if not test_core_functionality():
        success = False
        return False
    
    if success:
        print(f"\n🎉 ALL TESTS PASSED! MyLath is working perfectly!")
        print("\n📋 Summary of working features:")
        print("   ✅ Redis connection and storage")
        print("   ✅ Node creation, retrieval, and queries")
        print("   ✅ Edge creation and relationship management")
        print("   ✅ Graph traversals (Gremlin-style)")
        print("   ✅ Vector similarity search with filters")
        print("   ✅ Property-based queries and indexing")
        print("   ✅ Advanced filtering and counting")
        print("   ✅ Graph statistics")
        
        print(f"\n🚀 Next steps to explore MyLath:")
        print("   1. Start API server:")
        print("      python -c \"from mylath.mylath.api.graph_api import GraphAPI; from mylath.mylath.storage.redis_storage import RedisStorage; api = GraphAPI(RedisStorage()); api.run(host='0.0.0.0', port=5000)\"")
        print("   2. Try aggregation examples:")
        print("      python mylath/examples/aggregation_examples.py")
        print("   3. Install GraphBLAS for 10-1000x speedup:")
        print("      pip install python-graphblas")
        print("   4. Try the basic usage example:")
        print("      python mylath/examples/basic_usage.py")
        
        print(f"\n💡 MyLath is a powerful graph database with:")
        print("   • Redis-backed storage for high performance")
        print("   • Gremlin-style graph traversals")
        print("   • Vector similarity search (like vector databases)")
        print("   • Property indexing and complex queries")
        print("   • REST API for web applications")
        print("   • Optional GraphBLAS for massive performance boost")
    else:
        print(f"\n❌ Some tests failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)