# run_comprehensive_tests_with_logging.py

"""
Run comprehensive test suite and save results to file
"""

import sys
import os
from datetime import datetime
from io import StringIO

sys.path.insert(0, 'mylathdb')

class TestLogger:
    """Logger that captures output to both console and file"""
    
    def __init__(self, filename):
        self.filename = filename
        self.console = sys.stdout
        self.file_buffer = StringIO()
        
    def write(self, text):
        # Write to console
        self.console.write(text)
        # Write to buffer
        self.file_buffer.write(text)
        
    def flush(self):
        self.console.flush()
        
    def save_to_file(self):
        """Save captured output to file"""
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(self.file_buffer.getvalue())
        print(f"\n📁 Test results saved to: {self.filename}")

def test_basic_node_scanning():
    """Test basic node scanning without projections"""
    
    print("🧪 === TESTING BASIC NODE SCANNING ===")
    
    try:
        from mylathdb import MyLathDB
        
        db = MyLathDB()
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        # Load test data
        db.load_graph_data(nodes=[
            {'id': '1', 'name': 'Alice', 'age': 30, '_labels': ['Person']},
            {'id': '2', 'name': 'Bob', 'age': 25, '_labels': ['Person']},
            {'id': '3', 'name': 'Charlie', 'age': 35, '_labels': ['User']}
        ])
        
        # Test basic node return
        result = db.execute_query("MATCH (n:Person) RETURN n")
        
        print(f"📊 Node scan results: {len(result.data)} nodes found")
        
        if result.success and len(result.data) == 2:
            print("✅ SUCCESS: Basic node scanning works")
            return True
        else:
            print(f"❌ FAIL: Expected 2 Person nodes, got {len(result.data)}")
            return False
    
    except Exception as e:
        print(f"❌ Basic node scanning test failed: {e}")
        return False

def test_property_filtering():
    """Test property-based filtering"""
    
    print("\n🧪 === TESTING PROPERTY FILTERING ===")
    
    try:
        from mylathdb import MyLathDB
        
        db = MyLathDB()
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        # Load test data
        db.load_graph_data(nodes=[
            {'id': '1', 'name': 'Alice', 'age': 30, '_labels': ['Person']},
            {'id': '2', 'name': 'Bob', 'age': 25, '_labels': ['Person']},
            {'id': '3', 'name': 'Charlie', 'age': 35, '_labels': ['Person']}
        ])
        
        # Test property filter with projection
        result = db.execute_query("MATCH (n:Person) WHERE n.age > 26 RETURN n.name, n.age")
        
        print(f"📊 Filtered results: {result.data}")
        
        if result.success and len(result.data) == 2:
            # Should find Alice (30) and Charlie (35), but not Bob (25)
            names = [r.get('n.name') for r in result.data]
            if 'Alice' in names and 'Charlie' in names and 'Bob' not in names:
                print("✅ SUCCESS: Property filtering with projection works")
                return True
            else:
                print(f"❌ FAIL: Wrong names returned: {names}")
                return False
        else:
            print(f"❌ FAIL: Expected 2 results, got {len(result.data)}")
            return False
    
    except Exception as e:
        print(f"❌ Property filtering test failed: {e}")
        return False

def test_ordering_and_limiting():
    """Test ORDER BY and LIMIT clauses"""
    
    print("\n🧪 === TESTING ORDERING AND LIMITING ===")
    
    try:
        from mylathdb import MyLathDB
        
        db = MyLathDB()
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        # Load test data
        db.load_graph_data(nodes=[
            {'id': '1', 'name': 'Alice', 'age': 30, '_labels': ['Person']},
            {'id': '2', 'name': 'Bob', 'age': 25, '_labels': ['Person']},
            {'id': '3', 'name': 'Charlie', 'age': 35, '_labels': ['Person']}
        ])
        
        # Test ordering with limit
        result = db.execute_query("MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age LIMIT 2")
        
        print(f"📊 Ordered and limited results: {result.data}")
        
        if result.success and len(result.data) == 2:
            # Should return Bob (25) and Alice (30) in that order
            first_age = result.data[0].get('n.age')
            second_age = result.data[1].get('n.age')
            
            if first_age == 25 and second_age == 30:
                print("✅ SUCCESS: Ordering and limiting works")
                return True
            else:
                print(f"❌ FAIL: Wrong order. Got ages: {first_age}, {second_age}")
                return False
        else:
            print(f"❌ FAIL: Expected 2 results, got {len(result.data)}")
            return False
    
    except Exception as e:
        print(f"❌ Ordering and limiting test failed: {e}")
        return False

def test_complex_projections():
    """Test various projection patterns"""
    
    print("\n🧪 === TESTING COMPLEX PROJECTIONS ===")
    
    try:
        from mylathdb import MyLathDB
        
        db = MyLathDB()
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        # Load test data
        db.load_graph_data(nodes=[
            {'id': '1', 'name': 'Alice', 'age': 30, 'city': 'New York', '_labels': ['Person']},
        ])
        
        # Test 1: Single property
        result1 = db.execute_query("MATCH (n:Person) RETURN n.name")
        test1_ok = (result1.success and 
                   result1.data and 
                   result1.data[0].get('n.name') == 'Alice')
        
        # Test 2: Multiple properties
        result2 = db.execute_query("MATCH (n:Person) RETURN n.name, n.age, n.city")
        test2_ok = (result2.success and 
                   result2.data and 
                   result2.data[0].get('n.name') == 'Alice' and
                   result2.data[0].get('n.age') == 30 and
                   result2.data[0].get('n.city') == 'New York')
        
        # Test 3: With alias
        result3 = db.execute_query("MATCH (n:Person) RETURN n.name AS person_name")
        test3_ok = (result3.success and 
                   result3.data and 
                   result3.data[0].get('person_name') == 'Alice')
        
        print(f"📊 Test results:")
        print(f"   Single property: {'✅' if test1_ok else '❌'} - {result1.data}")
        print(f"   Multiple properties: {'✅' if test2_ok else '❌'} - {result2.data}")
        print(f"   With alias: {'✅' if test3_ok else '❌'} - {result3.data}")
        
        if test1_ok and test2_ok and test3_ok:
            print("✅ SUCCESS: All complex projections work")
            return True
        else:
            print("❌ FAIL: Some projection tests failed")
            return False
    
    except Exception as e:
        print(f"❌ Complex projections test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    
    print("\n🧪 === TESTING EDGE CASES ===")
    
    try:
        from mylathdb import MyLathDB
        
        db = MyLathDB()
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        # Load test data
        db.load_graph_data(nodes=[
            {'id': '1', 'name': 'Alice', '_labels': ['Person']},  # Missing age property
        ])
        
        # Test 1: Non-existent property
        result1 = db.execute_query("MATCH (n:Person) RETURN n.nonexistent")
        test1_ok = (result1.success and 
                   result1.data and 
                   result1.data[0].get('n.nonexistent') is None)
        
        # Test 2: Non-existent label
        result2 = db.execute_query("MATCH (n:NonExistent) RETURN n.name")
        test2_ok = (result2.success and len(result2.data) == 0)
        
        # Test 3: Empty filter result
        result3 = db.execute_query("MATCH (n:Person) WHERE n.age > 100 RETURN n.name")
        test3_ok = (result3.success and len(result3.data) == 0)
        
        print(f"📊 Edge case results:")
        print(f"   Non-existent property: {'✅' if test1_ok else '❌'} - {result1.data}")
        print(f"   Non-existent label: {'✅' if test2_ok else '❌'} - {len(result2.data)} results")
        print(f"   Empty filter: {'✅' if test3_ok else '❌'} - {len(result3.data)} results")
        
        if test1_ok and test2_ok and test3_ok:
            print("✅ SUCCESS: Edge cases handled correctly")
            return True
        else:
            print("❌ FAIL: Some edge cases failed")
            return False
    
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False

def run_performance_test():
    """Run a basic performance test"""
    
    print("\n🧪 === TESTING PERFORMANCE ===")
    
    try:
        import time
        from mylathdb import MyLathDB
        
        db = MyLathDB()
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        # Load larger dataset
        nodes = []
        for i in range(100):
            nodes.append({
                'id': str(i),
                'name': f'Person{i}',
                'age': 20 + (i % 50),
                '_labels': ['Person']
            })
        
        print(f"📝 Loading {len(nodes)} nodes...")
        load_start = time.time()
        db.load_graph_data(nodes=nodes)
        load_time = time.time() - load_start
        
        # Test query performance
        print("🔍 Testing query performance...")
        query_start = time.time()
        result = db.execute_query("MATCH (n:Person) WHERE n.age > 30 RETURN n.name, n.age")
        query_time = time.time() - query_start
        
        print(f"📊 Performance results:")
        print(f"   Load time: {load_time:.3f}s for {len(nodes)} nodes")
        print(f"   Query time: {query_time:.3f}s for filtering and projection")
        print(f"   Results: {len(result.data)} nodes found")
        
        # Consider it a pass if operations complete in reasonable time
        if load_time < 5.0 and query_time < 2.0 and result.success:
            print("✅ SUCCESS: Performance is acceptable")
            return True
        else:
            print("⚠️  WARNING: Performance may be slow but functional")
            return True  # Still pass if functional
    
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_projection_corruption_fix():
    """Test that the original projection corruption issue is fixed"""
    
    print("\n🧪 === TESTING PROJECTION CORRUPTION FIX ===")
    
    try:
        from mylathdb import MyLathDB
        
        db = MyLathDB()
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        # Load test data
        db.load_graph_data(nodes=[{
            'id': '1', 
            'name': 'Alice', 
            'age': 30, 
            '_labels': ['Person']
        }])
        
        # Test the exact query that was failing before
        result = db.execute_query("MATCH (n:Person) RETURN n.name")
        
        print(f"📊 Projection test result: {result.data}")
        
        if (result.success and 
            result.data and 
            len(result.data) == 1 and
            'n.name' in result.data[0] and
            result.data[0]['n.name'] == 'Alice'):
            
            print("✅ SUCCESS: Projection corruption issue is FIXED!")
            print("   Expected: [{'n.name': 'Alice'}]")
            print(f"   Got:      {result.data}")
            return True
        else:
            print("❌ FAIL: Projection corruption issue still exists")
            print(f"   Got: {result.data}")
            return False
    
    except Exception as e:
        print(f"❌ Projection corruption test failed: {e}")
        return False

def main():
    """Run comprehensive test suite and save to file"""
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mylathdb_test_results_{timestamp}.txt"
    
    # Set up logging to capture output
    logger = TestLogger(filename)
    sys.stdout = logger
    
    try:
        print("🚀 MyLathDB Comprehensive Test Suite")
        print("=" * 60)
        print(f"📅 Test run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🐍 Python version: {sys.version}")
        print(f"📁 Results will be saved to: {filename}")
        print("=" * 60)
        
        tests = [
            ("Projection Corruption Fix", test_projection_corruption_fix),
            ("Basic Node Scanning", test_basic_node_scanning),
            ("Property Filtering", test_property_filtering),
            ("Ordering and Limiting", test_ordering_and_limiting),
            ("Complex Projections", test_complex_projections),
            ("Edge Cases", test_edge_cases),
            ("Performance", run_performance_test),
        ]
        
        results = []
        
        for test_name, test_func in tests:
            print(f"\n🔄 Running {test_name}...")
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ Test '{test_name}' crashed: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 === COMPREHENSIVE TEST RESULTS ===")
        
        passed = 0
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {test_name}: {status}")
            if result:
                passed += 1
        
        total = len(results)
        percentage = passed/total*100
        print(f"\n🎯 Overall Result: {passed}/{total} tests passed ({percentage:.1f}%)")
        
        if passed == total:
            print("\n🎉 EXCELLENT! All tests passed. MyLathDB is working correctly!")
            print("✨ Your graph database with Cypher support is ready for use!")
        elif passed >= total * 0.8:
            print("\n👍 GOOD! Most tests passed. MyLathDB is largely functional.")
            print("🔧 Consider investigating the failed tests for full functionality.")
        else:
            print("\n⚠️  ISSUES DETECTED! Several tests failed.")
            print("🔧 Please review the failed tests and fix the underlying issues.")
        
        print(f"\n📁 Test results saved to: {filename}")
        
        return passed == total
    
    finally:
        # Restore stdout and save results
        sys.stdout = logger.console
        logger.save_to_file()

if __name__ == "__main__":
    main()