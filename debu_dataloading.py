# Debug script to check data loading
# Run this to see what's happening with data loading

import sys
from pathlib import Path

# Add the mylathdb directory to Python path
current_dir = Path(__file__).parent
mylathdb_dir = current_dir / "mylathdb"
sys.path.insert(0, str(mylathdb_dir))

def debug_data_loading():
    """Debug data loading process"""
    print("🔍 Debugging MyLathDB Data Loading...")
    
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
        ]
        
        test_edges = [
            ("1", "KNOWS", "2"),
        ]
        
        print(f"📝 Loading {len(test_nodes)} nodes and {len(test_edges)} edges...")
        
        # Try to load data directly into Redis
        redis_client = db.engine.redis_executor.redis
        
        # Load nodes manually
        for node in test_nodes:
            node_id = node["id"]
            print(f"  Loading node {node_id}: {node}")
            
            # Store node properties
            node_key = f"node:{node_id}"
            properties = {k: v for k, v in node.items() if k not in ['id', '_labels']}
            if properties:
                redis_client.hset(node_key, mapping=properties)
                print(f"    Stored properties: {properties}")
            
            # Store labels
            labels = node.get('_labels', [])
            if labels:
                labels_key = f"node_labels:{node_id}"
                redis_client.sadd(labels_key, *labels)
                print(f"    Stored labels: {labels}")
                
                # Create label indexes
                for label in labels:
                    label_key = f"label:{label}"
                    redis_client.sadd(label_key, node_id)
                    print(f"    Added to label index {label_key}: {node_id}")
            
            # Create property indexes
            for prop_key, prop_value in properties.items():
                prop_index_key = f"prop:{prop_key}:{prop_value}"
                redis_client.sadd(prop_index_key, node_id)
                print(f"    Added to property index {prop_index_key}: {node_id}")
        
        # Load edges manually
        for edge in test_edges:
            src_id, rel_type, dest_id = edge
            edge_id = f"{src_id}_{rel_type}_{dest_id}"
            print(f"  Loading edge {edge_id}: {edge}")
            
            # Store edge endpoints
            endpoints_key = f"edge_endpoints:{edge_id}"
            endpoints_value = f"{src_id}|{dest_id}|{rel_type}"
            redis_client.set(endpoints_key, endpoints_value)
            print(f"    Stored endpoints: {endpoints_value}")
        
        print("\n🔍 Checking what's in Redis...")
        
        # Check what keys exist
        all_keys = list(redis_client.scan_iter())
        print(f"📋 All Redis keys: {all_keys}")
        
        # Check specific data
        person_nodes = redis_client.smembers("label:Person")
        print(f"👥 Person nodes: {person_nodes}")
        
        if person_nodes:
            for node_id in person_nodes:
                node_key = f"node:{node_id}"
                node_data = redis_client.hgetall(node_key)
                print(f"📄 Node {node_id} data: {node_data}")
        
        print("\n🧪 Testing Query Execution...")
        
        # Try a simple query
        query = "MATCH (n:Person) RETURN n.name"
        print(f"Query: {query}")
        
        result = db.execute_query(query)
        print(f"✅ Query executed: {result.success}")
        print(f"📊 Results: {len(result.data)}")
        print(f"📋 Data: {result.data}")
        
        if result.error:
            print(f"❌ Error: {result.error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_data_loading()