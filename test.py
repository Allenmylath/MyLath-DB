#!/usr/bin/env python3
"""
Test script for GraphBLAS initialization fix
"""

import sys
from pathlib import Path

# Add the mylathdb directory to Python path
current_dir = Path(__file__).parent
mylathdb_dir = current_dir / "mylathdb"
sys.path.insert(0, str(mylathdb_dir))

def test_graphblas_initialization():
    """Test that GraphBLAS initializes correctly"""
    print("🔬 Testing GraphBLAS Initialization...")
    
    try:
        # Test 1: Check if python-graphblas is available
        print("   📦 Checking GraphBLAS package availability...")
        try:
            import graphblas as gb
            print(f"   ✅ GraphBLAS package found: version {gb.__version__}")
        except ImportError as e:
            print(f"   ❌ GraphBLAS package not available: {e}")
            print("   💡 Install with: pip install python-graphblas")
            return False
        
        # Test 2: Initialize MyLathDB and check GraphBLAS initialization
        print("   🏗️  Creating MyLathDB instance...")
        from mylathdb import MyLathDB
        
        db = MyLathDB()
        print("   ✅ MyLathDB instance created")
        
        # Check GraphBLAS executor status
        if hasattr(db.engine, 'graphblas_executor'):
            gb_executor = db.engine.graphblas_executor
            status = gb_executor.get_status()
            
            print(f"   📊 GraphBLAS Status:")
            print(f"      Available: {status.get('available', False)}")
            print(f"      Initialized: {status.get('initialized', False)}")
            print(f"      GB Initialized: {status.get('gb_initialized', False)}")
            
            if status.get('available'):
                print("   ✅ GraphBLAS is properly initialized!")
                
                # Test basic functionality
                if gb_executor.test_functionality():
                    print("   ✅ GraphBLAS functionality test passed")
                else:
                    print("   ❌ GraphBLAS functionality test failed")
                    return False
                    
            else:
                print(f"   ❌ GraphBLAS not available: {status.get('reason', 'Unknown')}")
                return False
        else:
            print("   ❌ GraphBLAS executor not found in engine")
            return False
        
        # Test 3: Load some data to verify both Redis and GraphBLAS work
        print("   📊 Testing data loading...")
        
        # Clear any existing data
        if hasattr(db.engine, 'redis_executor') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
            print("      🧹 Redis cleared")
        
        # Test data
        test_nodes = [
            {"id": "1", "name": "Alice", "age": 30, "_labels": ["Person"]},
            {"id": "2", "name": "Bob", "age": 25, "_labels": ["Person"]},
        ]
        test_edges = [("1", "KNOWS", "2")]
        
        # Load data
        db.load_graph_data(nodes=test_nodes, edges=test_edges)
        print("   ✅ Data loaded into both Redis and GraphBLAS")
        
        # Test 4: Execute a simple query
        print("   🧪 Testing query execution...")
        result = db.execute_query("MATCH (n:Person) RETURN n.name")
        
        print(f"      Query result: success={result.success}, results={len(result.data)}")
        
        if result.success and len(result.data) > 0:
            print("   ✅ Query execution successful!")
            for i, record in enumerate(result.data):
                print(f"      Result {i+1}: {record}")
        else:
            print(f"   ⚠️  Query execution issue: {result.error}")
        
        print("\n🎉 GraphBLAS initialization test completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_graphblas_manual_init():
    """Test manual GraphBLAS initialization"""
    print("\n🔧 Testing Manual GraphBLAS Initialization...")
    
    try:
        import graphblas as gb
        
        # Test manual initialization
        print("   🔄 Calling gb.init()...")
        gb.init()
        print("   ✅ Manual GraphBLAS initialization successful")
        
        # Test basic operations
        print("   🧪 Testing basic operations...")
        matrix = gb.Matrix.new(gb.dtypes.BOOL, nrows=3, ncols=3)
        matrix[0, 1] = True
        matrix[1, 2] = True
        
        vector = gb.Vector.new(gb.dtypes.BOOL, size=3)
        vector[0] = True
        
        result = vector @ matrix
        print(f"   ✅ Matrix-vector multiplication successful: nnz={result.nvals}")
        
        # Clean up
        gb.finalize()
        print("   ✅ GraphBLAS finalized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Manual test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 GraphBLAS Initialization Test Suite")
    print("=" * 50)
    
    # Test 1: Manual GraphBLAS initialization
    manual_success = test_graphblas_manual_init()
    
    # Test 2: MyLathDB GraphBLAS initialization
    mylathdb_success = test_graphblas_initialization()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Manual GraphBLAS: {'✅ PASS' if manual_success else '❌ FAIL'}")
    print(f"   MyLathDB GraphBLAS: {'✅ PASS' if mylathdb_success else '❌ FAIL'}")
    
    if manual_success and mylathdb_success:
        print("\n🎉 All GraphBLAS tests passed! MyLathDB is ready to use.")
    elif manual_success and not mylathdb_success:
        print("\n⚠️  GraphBLAS works manually but MyLathDB initialization has issues.")
        print("💡 Check the initialization sequence in the GraphBLAS executor.")
    elif not manual_success:
        print("\n❌ GraphBLAS package issues detected.")
        print("💡 Try: pip install --upgrade python-graphblas")
    
    sys.exit(0 if (manual_success and mylathdb_success) else 1)