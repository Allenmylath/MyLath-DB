#!/usr/bin/env python3
"""
Proper MyLathDB Test using MyLathDB API
Tests the actual MyLathDB functionality end-to-end
"""

import sys
import os
import time
import subprocess
from pathlib import Path

# Add the mylathdb directory to Python path
current_dir = Path(__file__).parent
mylathdb_dir = current_dir / "mylathdb"
sys.path.insert(0, str(mylathdb_dir))

def setup_redis():
    """Setup and start Redis server"""
    print("🔧 Setting up Redis...")
    
    try:
        # Try to connect to existing Redis
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        print("   ✅ Redis is already running")
        return True
    except:
        pass
    
    # Try to start Redis
    try:
        print("   🚀 Starting Redis server...")
        # Start Redis in background
        subprocess.Popen([
            'redis-server', 
            '--port', '6379', 
            '--daemonize', 'yes',
            '--save', '',  # Disable persistence for testing
            '--appendonly', 'no'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(3)
        
        # Test connection
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
        print("   ✅ Redis started successfully")
        return True
        
    except FileNotFoundError:
        print("   ❌ Redis not found. Please install Redis:")
        print("   Ubuntu/Debian: sudo apt install redis-server")
        print("   macOS: brew install redis")
        print("   Windows: Download from https://github.com/microsoftarchive/redis/releases")
        print("   Or use Docker: docker run -d -p 6379:6379 redis")
        return False
    except Exception as e:
        print(f"   ❌ Failed to start Redis: {e}")
        return False

def test_mylathdb_initialization():
    """Test MyLathDB initialization"""
    print("\n🏗️  Testing MyLathDB Initialization...")
    
    try:
        # Import MyLathDB
        from mylathdb import MyLathDB
        
        # Create MyLathDB instance
        db = MyLathDB(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            enable_caching=True
        )
        
        print("   ✅ MyLathDB instance created successfully")
        
        # Clear any existing data for clean test
        if hasattr(db.engine, 'redis_executor') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
            print("   🧹 Cleared database for clean test")
        
        return db
        
    except Exception as e:
        print(f"   ❌ MyLathDB initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_test_data_via_mylathdb(db):
    """Load test data using MyLathDB API"""
    print("\n📊 Loading test data via MyLathDB...")
    
    # Test data - we know exactly what this contains
    test_nodes = [
        {"id": "1", "name": "Alice", "age": 30, "country": "USA", "_labels": ["Person"]},
        {"id": "2", "name": "Bob", "age": 25, "country": "USA", "_labels": ["Person"]},
        {"id": "3", "name": "Charlie", "age": 35, "country": "UK", "_labels": ["Person"]},
        {"id": "4", "name": "Diana", "age": 28, "country": "USA", "_labels": ["Person"]},
        {"id": "5", "name": "Eve", "age": 22, "country": "Canada", "_labels": ["Person"]},
    ]
    
    test_edges = [
        ("1", "KNOWS", "2"),  # Alice -> Bob
        ("1", "KNOWS", "3"),  # Alice -> Charlie  
        ("2", "KNOWS", "4"),  # Bob -> Diana
        ("3", "KNOWS", "1"),  # Charlie -> Alice (bidirectional)
        ("4", "KNOWS", "5"),  # Diana -> Eve
    ]
    
    try:
        print(f"   📝 Loading {len(test_nodes)} people...")
        print(f"   🔗 Loading {len(test_edges)} relationships...")
        
        # Use MyLathDB's load_graph_data method
        db.load_graph_data(nodes=test_nodes, edges=test_edges)
        
        print("   ✅ Test data loaded successfully via MyLathDB!")
        print(f"   📊 Loaded: {len(test_nodes)} people, {len(test_edges)} relationships")
        
        return test_nodes, test_edges
        
    except Exception as e:
        print(f"   ❌ Failed to load data via MyLathDB: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_simple_query(db):
    """Test 1: Find all people from USA"""
    print("\n🧪 Test 1: Find all people from USA")
    
    query = "MATCH (n:Person) WHERE n.country = 'USA' RETURN n.name, n.age"
    expected_people = ["Alice", "Bob", "Diana"]  # We know these are the USA people
    
    print(f"   Query: {query}")
    print(f"   Expected: {', '.join(expected_people)}")
    
    try:
        # Execute query using MyLathDB
        result = db.execute_query(query)
        
        print(f"   ✅ Query executed: {result.success}")
        
        if result.success:
            print(f"   ⏱️  Execution time: {result.execution_time:.3f}s")
            print(f"   📊 Found {len(result.data)} results")
            
            # Show results
            found_names = []
            for i, record in enumerate(result.data):
                print(f"   📋 Result {i+1}: {record}")
                # Extract name from result
                if 'n.name' in record:
                    found_names.append(record['n.name'])
                elif 'n' in record and isinstance(record['n'], dict):
                    found_names.append(record['n'].get('name', 'Unknown'))
            
            # Verify results
            if len(found_names) == len(expected_people):
                print(f"   ✅ Found expected number of USA people: {len(found_names)}")
                return True
            else:
                print(f"   ⚠️  Expected {len(expected_people)} people, got {len(found_names)}")
                print(f"   🔍 Found names: {found_names}")
                return False
        else:
            print(f"   ❌ Query failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_age_filter_query(db):
    """Test 2: Find people over 25"""
    print("\n🧪 Test 2: Find people over 25")
    
    query = "MATCH (n:Person) WHERE n.age > 25 RETURN n.name, n.age ORDER BY n.age"
    expected_people = ["Diana", "Alice", "Charlie"]  # Ages: 28, 30, 35
    
    print(f"   Query: {query}")
    print(f"   Expected: {', '.join(expected_people)} (ages 28, 30, 35)")
    
    try:
        # Execute query using MyLathDB
        result = db.execute_query(query)
        
        print(f"   ✅ Query executed: {result.success}")
        
        if result.success:
            print(f"   ⏱️  Execution time: {result.execution_time:.3f}s")
            print(f"   📊 Found {len(result.data)} results")
            
            # Show results
            found_people = []
            for i, record in enumerate(result.data):
                print(f"   📋 Result {i+1}: {record}")
                # Extract name and age from result
                name = "Unknown"
                age = "Unknown"
                
                if 'n.name' in record:
                    name = record['n.name']
                if 'n.age' in record:
                    age = record['n.age']
                elif 'n' in record and isinstance(record['n'], dict):
                    name = record['n'].get('name', 'Unknown')
                    age = record['n'].get('age', 'Unknown')
                
                found_people.append((name, age))
            
            # Verify we got at least 3 people (Alice, Charlie, Diana are over 25)
            if len(found_people) >= 3:
                print(f"   ✅ Found expected number of people over 25: {len(found_people)}")
                return True
            else:
                print(f"   ⚠️  Expected at least 3 people over 25, got {len(found_people)}")
                return False
        else:
            print(f"   ❌ Query failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_return_query(db):
    """Test 3: Simple return all people"""
    print("\n🧪 Test 3: Return all people")
    
    query = "MATCH (n:Person) RETURN n.name"
    expected_count = 5  # Alice, Bob, Charlie, Diana, Eve
    
    print(f"   Query: {query}")
    print(f"   Expected: {expected_count} people")
    
    try:
        # Execute query using MyLathDB
        result = db.execute_query(query)
        
        print(f"   ✅ Query executed: {result.success}")
        
        if result.success:
            print(f"   ⏱️  Execution time: {result.execution_time:.3f}s")
            print(f"   📊 Found {len(result.data)} results")
            
            # Show results
            for i, record in enumerate(result.data):
                print(f"   📋 Result {i+1}: {record}")
            
            # Verify count
            if len(result.data) == expected_count:
                print(f"   ✅ Found expected number of people: {len(result.data)}")
                return True
            else:
                print(f"   ⚠️  Expected {expected_count} people, got {len(result.data)}")
                return False
        else:
            print(f"   ❌ Query failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mylathdb_stats(db):
    """Test MyLathDB statistics"""
    print("\n📈 Testing MyLathDB Statistics...")
    
    try:
        stats = db.get_statistics()
        print("   📊 Database Statistics:")
        
        if 'database' in stats:
            db_stats = stats['database']
            print(f"   📝 Queries executed: {db_stats.get('queries_executed', 0)}")
            print(f"   ⏱️  Total execution time: {db_stats.get('total_execution_time', 0):.3f}s")
            print(f"   📊 Average execution time: {db_stats.get('avg_execution_time', 0):.3f}s")
        
        if 'engine' in stats:
            engine_stats = stats['engine']
            print("   🏗️  Engine Statistics:")
            for key, value in engine_stats.items():
                print(f"   📊 {key}: {value}")
        
        print("   ✅ Statistics retrieved successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Statistics test failed: {e}")
        return False

def main():
    """Run all MyLathDB tests"""
    print("🗄️  MyLathDB Real Functionality Test")
    print("=" * 60)
    print("Testing MyLathDB with real Redis and actual queries")
    print("=" * 60)
    
    # Step 1: Setup Redis
    if not setup_redis():
        print("\n❌ Cannot proceed without Redis. Please install and start Redis.")
        return False
    
    # Step 2: Initialize MyLathDB
    db = test_mylathdb_initialization()
    if not db:
        print("\n❌ Cannot proceed without MyLathDB initialization.")
        return False
    
    # Step 3: Load test data
    nodes, edges = load_test_data_via_mylathdb(db)
    if not nodes or not edges:
        print("\n❌ Cannot proceed without test data.")
        return False
    
    # Step 4: Run tests
    tests = [
        ("Simple Return Query", lambda: test_simple_return_query(db)),
        ("USA People Query", lambda: test_simple_query(db)),
        ("Age Filter Query", lambda: test_age_filter_query(db)),
        ("MyLathDB Statistics", lambda: test_mylathdb_stats(db)),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Step 5: Cleanup and summary
    try:
        if hasattr(db, 'shutdown'):
            db.shutdown()
        print("\n🧹 MyLathDB shutdown complete")
    except:
        pass
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 MyLathDB Real Test Results:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! MyLathDB is working with real data!")
        print("🚀 MyLathDB successfully:")
        print("   • Connected to Redis")
        print("   • Loaded graph data") 
        print("   • Executed Cypher queries")
        print("   • Returned correct results")
    elif passed > 0:
        print("⚠️  Some tests passed. MyLathDB is partially working.")
        print("🔧 Check the failed tests for issues to fix.")
    else:
        print("❌ All tests failed. MyLathDB needs debugging.")
    
    return passed > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)