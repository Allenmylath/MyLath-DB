#!/usr/bin/env python3
"""
Test script for the fixed Redis executor
"""

import sys
from pathlib import Path

# Add the mylathdb directory to Python path
current_dir = Path(__file__).parent
mylathdb_dir = current_dir / "mylathdb"
sys.path.insert(0, str(mylathdb_dir))

def test_fixed_redis_executor():
    """Test the fixed Redis executor"""
    print("🔧 Testing Fixed Redis Executor...")
    
    try:
        # Import MyLathDB
        from mylathdb import MyLathDB
        
        # Create MyLathDB instance
        db = MyLathDB()
        print("✅ MyLathDB created")
        
        # Clear database
        if hasattr(db.engine, 'redis_executor') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
            print("🧹 Database cleared")
        
        # Test data
        test_nodes = [
            {"id": "1", "name": "Alice", "age": 30, "country": "USA", "_labels": ["Person"]},
            {"id": "2", "name": "Bob", "age": 25, "country": "USA", "_labels": ["Person"]},
            {"id": "3", "name": "Charlie", "age": 35, "country": "UK", "_labels": ["Person"]},
        ]
        
        test_edges = [
            ("1", "KNOWS", "2"),
        ]
        
        print(f"📝 Loading {len(test_nodes)} nodes and {len(test_edges)} edges...")
        
        # Load data using MyLathDB
        db.load_graph_data(nodes=test_nodes, edges=test_edges)
        print("✅ Test data loaded")
        
        # Test queries
        test_queries = [
            "MATCH (n:Person) RETURN n.name",
            "MATCH (n:Person) WHERE n.country = 'USA' RETURN n.name, n.age",
            "MATCH (n:Person) WHERE n.age > 25 RETURN n.name, n.age",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🧪 Test {i}: {query}")
            
            try:
                result = db.execute_query(query)
                print(f"   ✅ Query executed: {result.success}")
                print(f"   ⏱️  Execution time: {result.execution_time:.3f}s")
                print(f"   📊 Results: {len(result.data)}")
                
                # Show first few results
                for j, record in enumerate(result.data[:3]):
                    print(f"   📋 Result {j+1}: {record}")
                
                if len(result.data) > 3:
                    print(f"   ... and {len(result.data) - 3} more")
                
                if result.error:
                    print(f"   ❌ Error: {result.error}")
                
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✅ All tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fixed_redis_executor()