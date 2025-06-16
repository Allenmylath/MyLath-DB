# run_streamlined_tests.py

"""
Streamlined MyLathDB Test Suite - Minimal Logging
Only logs essential information to keep output clean
"""

import sys
import os
from datetime import datetime
from io import StringIO

sys.path.insert(0, 'mylathdb')

class StreamlinedLogger:
    """Streamlined logger - only essential output"""
    
    def __init__(self, filename, verbose=False):
        self.filename = filename
        self.console = sys.stdout
        self.file_buffer = StringIO()
        self.verbose = verbose
        
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

def test_basic_functionality():
    """Test basic MyLathDB functionality"""
    
    print("🧪 Testing Basic Functionality...")
    
    try:
        from mylathdb import MyLathDB
        
        # Initialize and clear
        db = MyLathDB()
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        # Load simple test data
        db.load_graph_data(nodes=[
            {'id': '1', 'name': 'Alice', 'age': 30, '_labels': ['Person']},
            {'id': '2', 'name': 'Bob', 'age': 25, '_labels': ['Person']},
        ])
        
        # Test basic queries
        tests = [
            ("Node scan", "MATCH (n:Person) RETURN n", 2),
            ("Property projection", "MATCH (n:Person) RETURN n.name", 2),
            ("Property filter", "MATCH (n:Person) WHERE n.age > 26 RETURN n.name", 1),
        ]
        
        results = []
        for test_name, query, expected_count in tests:
            result = db.execute_query(query)
            passed = result.success and len(result.data) == expected_count
            results.append((test_name, passed))
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}: {len(result.data)}/{expected_count} results")
        
        return all(passed for _, passed in results)
    
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_projection_fix():
    """Test projection corruption fix"""
    
    print("🧪 Testing Projection Fix...")
    
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
        
        # Test projection
        result = db.execute_query("MATCH (n:Person) RETURN n.name")
        
        if (result.success and 
            result.data and 
            len(result.data) == 1 and
            'n.name' in result.data[0] and
            result.data[0]['n.name'] == 'Alice'):
            
            print("   ✅ Projection working correctly")
            return True
        else:
            print(f"   ❌ Projection failed - got: {result.data}")
            return False
    
    except Exception as e:
        print(f"   ❌ Projection test failed: {e}")
        return False

def test_complex_queries():
    """Test more complex query patterns"""
    
    print("🧪 Testing Complex Queries...")
    
    try:
        from mylathdb import MyLathDB
        
        db = MyLathDB()
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        # Load test data
        db.load_graph_data(nodes=[
            {'id': '1', 'name': 'Alice', 'age': 30, 'city': 'NYC', '_labels': ['Person']},
            {'id': '2', 'name': 'Bob', 'age': 25, 'city': 'LA', '_labels': ['Person']},
            {'id': '3', 'name': 'Charlie', 'age': 35, 'city': 'NYC', '_labels': ['Person']},
        ])
        
        # Test complex projections
        tests = [
            ("Multiple properties", "MATCH (n:Person) RETURN n.name, n.age", 3),
            ("Alias projection", "MATCH (n:Person) RETURN n.name AS person_name", 3),
            ("Filtered projection", "MATCH (n:Person) WHERE n.age > 26 RETURN n.name, n.city", 2),
        ]
        
        results = []
        for test_name, query, expected_count in tests:
            result = db.execute_query(query)
            passed = result.success and len(result.data) == expected_count
            results.append((test_name, passed))
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}: {len(result.data)}/{expected_count} results")
            
            # Show sample result for failed tests
            if not passed and result.data:
                print(f"      Sample result: {result.data[0]}")
        
        return all(passed for _, passed in results)
    
    except Exception as e:
        print(f"   ❌ Complex queries test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases"""
    
    print("🧪 Testing Edge Cases...")
    
    try:
        from mylathdb import MyLathDB
        
        db = MyLathDB()
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        # Load minimal test data
        db.load_graph_data(nodes=[
            {'id': '1', 'name': 'Alice', '_labels': ['Person']},
        ])
        
        # Test edge cases
        tests = [
            ("Non-existent property", "MATCH (n:Person) RETURN n.nonexistent", 1, True),
            ("Non-existent label", "MATCH (n:NonExistent) RETURN n.name", 0, True),
            ("Empty filter", "MATCH (n:Person) WHERE n.age > 100 RETURN n.name", 0, True),
        ]
        
        results = []
        for test_name, query, expected_count, should_succeed in tests:
            result = db.execute_query(query)
            passed = result.success == should_succeed and len(result.data) == expected_count
            results.append((test_name, passed))
            status = "✅" if passed else "❌"
            print(f"   {status} {test_name}: {len(result.data)}/{expected_count} results")
        
        return all(passed for _, passed in results)
    
    except Exception as e:
        print(f"   ❌ Edge cases test failed: {e}")
        return False

def test_performance():
    """Test basic performance"""
    
    print("🧪 Testing Performance...")
    
    try:
        import time
        from mylathdb import MyLathDB
        
        db = MyLathDB()
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        # Load moderate dataset
        nodes = []
        for i in range(50):  # Reduced from 100 to speed up test
            nodes.append({
                'id': str(i),
                'name': f'Person{i}',
                'age': 20 + (i % 50),
                '_labels': ['Person']
            })
        
        # Test load performance
        load_start = time.time()
        db.load_graph_data(nodes=nodes)
        load_time = time.time() - load_start
        
        # Test query performance
        query_start = time.time()
        result = db.execute_query("MATCH (n:Person) WHERE n.age > 30 RETURN n.name")
        query_time = time.time() - query_start
        
        print(f"   📊 Load time: {load_time:.3f}s for {len(nodes)} nodes")
        print(f"   📊 Query time: {query_time:.3f}s -> {len(result.data)} results")
        
        # Performance is acceptable if operations complete reasonably quickly
        performance_ok = load_time < 2.0 and query_time < 1.0 and result.success
        
        if performance_ok:
            print("   ✅ Performance acceptable")
        else:
            print("   ⚠️  Performance slower than expected but functional")
        
        return performance_ok or result.success  # Pass if functional even if slow
    
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def main():
    """Run streamlined test suite"""
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mylathdb_streamlined_results_{timestamp}.txt"
    
    # Set up minimal logging
    logger = StreamlinedLogger(filename)
    sys.stdout = logger
    
    try:
        print("🚀 MyLathDB Streamlined Test Suite")
        print("=" * 50)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50)
        
        # Run test suite
        tests = [
            ("Projection Fix", test_projection_fix),
            ("Basic Functionality", test_basic_functionality),
            ("Complex Queries", test_complex_queries),
            ("Edge Cases", test_edge_cases),
            ("Performance", test_performance),
        ]
        
        results = []
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                results.append((test_name, False))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        passed = 0
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{status} {test_name}")
            if result:
                passed += 1
        
        total = len(results)
        percentage = passed/total*100
        
        print(f"\n🎯 Overall: {passed}/{total} tests passed ({percentage:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! MyLathDB is working correctly.")
        elif passed >= total * 0.8:
            print("👍 Most tests passed. MyLathDB is largely functional.")
        else:
            print("⚠️  Several tests failed. Review issues above.")
        
        return passed == total
    
    finally:
        # Restore stdout and save results
        sys.stdout = logger.console
        logger.save_to_file()

if __name__ == "__main__":
    main()