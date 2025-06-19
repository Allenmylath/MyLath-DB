#!/usr/bin/env python3
"""
Permanent Fixes Verification Test
Verify that all permanent fixes are working
"""

import sys
import os
from pathlib import Path

# Add mylathdb to path
current_dir = Path(__file__).parent
mylathdb_dir = current_dir / "mylathdb"
sys.path.insert(0, str(mylathdb_dir))

def verify_fixes():
    """Verify all permanent fixes are in place"""
    print("🔍 Verifying Permanent Fixes...")
    
    fixes_verified = 0
    total_fixes = 4
    
    # Fix 1: Verify _get_node_data method exists
    try:
        from mylathdb.execution_engine.redis_executor import RedisExecutor
        if hasattr(RedisExecutor, '_get_node_data'):
            print("✅ Fix 1: RedisExecutor._get_node_data method exists")
            fixes_verified += 1
        else:
            print("❌ Fix 1: RedisExecutor._get_node_data method missing")
    except Exception as e:
        print(f"❌ Fix 1: Error checking RedisExecutor: {e}")
    
    # Fix 2: Verify MAX_EXECUTION_TIME in config
    try:
        from mylathdb.execution_engine.config import MyLathDBExecutionConfig
        config = MyLathDBExecutionConfig()
        if hasattr(config, 'MAX_EXECUTION_TIME'):
            print(f"✅ Fix 2: MAX_EXECUTION_TIME exists: {config.MAX_EXECUTION_TIME}")
            fixes_verified += 1
        else:
            print("❌ Fix 2: MAX_EXECUTION_TIME missing from config")
    except Exception as e:
        print(f"❌ Fix 2: Error checking config: {e}")
    
    # Fix 3: Verify DataBridge matrix method works
    try:
        from mylathdb.execution_engine.data_bridge import DataBridge
        from mylathdb.execution_engine.config import MyLathDBExecutionConfig
        from mylathdb.execution_engine.redis_executor import RedisExecutor
        from mylathdb.execution_engine.graphblas_executor import GraphBLASExecutor
        
        config = MyLathDBExecutionConfig()
        redis_exec = RedisExecutor(config)
        gb_exec = GraphBLASExecutor(config)
        bridge = DataBridge(redis_exec, gb_exec)
        
        # Test matrix entry setting with None matrix (should not crash)
        result = bridge._set_matrix_entry(None, 0, 0, True)
        print(f"✅ Fix 3: DataBridge._set_matrix_entry works: {result}")
        fixes_verified += 1
    except Exception as e:
        print(f"❌ Fix 3: Error checking DataBridge: {e}")
    
    # Fix 4: Verify ExecutionContext has max_execution_time
    try:
        from mylathdb.execution_engine.engine import ExecutionContext
        context = ExecutionContext()
        if hasattr(context, 'max_execution_time'):
            print(f"✅ Fix 4: ExecutionContext.max_execution_time exists: {context.max_execution_time}")
            fixes_verified += 1
        else:
            print("❌ Fix 4: ExecutionContext.max_execution_time missing")
    except Exception as e:
        print(f"❌ Fix 4: Error checking ExecutionContext: {e}")
    
    print(f"\n📊 Fixes Summary: {fixes_verified}/{total_fixes} verified")
    return fixes_verified == total_fixes

def test_basic_functionality():
    """Test basic functionality after fixes"""
    print("\n🧪 Testing Basic Functionality...")
    
    try:
        from mylathdb import MyLathDB
        
        # Create database
        db = MyLathDB(redis_db=15)
        print("✅ MyLathDB instance created")
        
        # Clear and load test data
        if hasattr(db.engine, 'redis_executor') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        
        test_data = [{"id": "1", "name": "Test", "age": 30, "_labels": ["Person"]}]
        db.load_graph_data(nodes=test_data)
        print("✅ Test data loaded")
        
        # Test simple query
        result = db.execute_query("MATCH (n:Person) RETURN n.name")
        print(f"✅ Query executed: success={result.success}, results={len(result.data)}")
        
        if result.success and len(result.data) > 0:
            print(f"✅ Query result: {result.data[0]}")
            return True
        else:
            print(f"❌ Query failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔧 Permanent Fixes Verification")
    print("=" * 50)
    
    # Check prerequisites
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=15)
        r.ping()
        print("✅ Redis available")
    except Exception as e:
        print(f"❌ Redis not available: {e}")
        print("Please start Redis before running this test")
        sys.exit(1)
    
    # Verify fixes
    fixes_ok = verify_fixes()
    
    if not fixes_ok:
        print("\n❌ Some fixes are missing. Please apply the permanent fixes as described.")
        sys.exit(1)
    
    # Test functionality
    functionality_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    if fixes_ok and functionality_ok:
        print("🎉 ALL FIXES VERIFIED AND WORKING!")
        print("Your MyLathDB system is now properly configured.")
    else:
        print("❌ Issues remain. Check the output above.")
    
    return fixes_ok and functionality_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)