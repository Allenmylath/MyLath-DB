#!/usr/bin/env python3
"""
Check what imports are missing from the cypher_planner package
"""

import os
import sys

def check_imports():
    """Check for missing imports and files"""
    
    print("🔍 Checking cypher_planner imports...")
    print("=" * 50)
    
    # Required files based on __init__.py imports
    required_files = [
        'cypher_planner/__init__.py',
        'cypher_planner/ast_nodes.py',
        'cypher_planner/parser.py',
        'cypher_planner/planner.py',
        'cypher_planner/error_context.py',
        'cypher_planner/query_validator.py',
        'cypher_planner/semantic_validator.py',
        'cypher_planner/integrated_parser.py',
        'cypher_planner/enhanced_parser.py',
        'cypher_planner/logical_planner.py'
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path} - MISSING")
    
    print(f"\n📊 Summary:")
    print(f"   Existing: {len(existing_files)}/{len(required_files)}")
    print(f"   Missing: {len(missing_files)}")
    
    if missing_files:
        print(f"\n🚨 Missing files that need to be created:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        
        print(f"\n💡 Quick fix suggestions:")
        print(f"   1. Create the missing files listed above")
        print(f"   2. Each missing file should at least have basic imports and empty classes")
        print(f"   3. Check the __init__.py to see what each file should export")
    else:
        print(f"\n✅ All required files exist!")
        
        # Try to import and check for other issues
        print(f"\n🧪 Testing imports...")
        try:
            sys.path.insert(0, '.')
            import cypher_planner
            print("✅ cypher_planner imports successfully!")
            
            # Test main classes
            from cypher_planner import CypherParser, QueryPlanner
            print("✅ Main classes import successfully!")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("   This suggests there are issues inside the files themselves")
        except Exception as e:
            print(f"❌ Other error: {e}")

if __name__ == "__main__":
    check_imports()