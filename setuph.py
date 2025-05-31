#!/usr/bin/env python3
"""
MyLath Setup and Dependency Checker
Ensures all dependencies are properly installed
"""

import subprocess
import sys
import importlib

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   MyLath requires Python 3.8 or higher")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True

def install_package(package_name, version_spec=None):
    """Install a package using pip"""
    try:
        if version_spec:
            install_cmd = [sys.executable, "-m", "pip", "install", f"{package_name}{version_spec}"]
        else:
            install_cmd = [sys.executable, "-m", "pip", "install", package_name]
        
        print(f"Installing {package_name}...")
        result = subprocess.run(install_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {package_name} installed successfully")
            return True
        else:
            print(f"❌ Failed to install {package_name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing {package_name}: {e}")
        return False

def check_and_install_dependencies():
    """Check and install required dependencies"""
    dependencies = [
        ("redis", ">=4.5.0"),
        ("numpy", ">=1.21.0"),
        ("flask", ">=2.0.0"),
        ("click", ">=8.0.0")
    ]
    
    print("🔍 Checking dependencies...")
    
    all_good = True
    
    for package, version_spec in dependencies:
        try:
            # Try to import the package
            module = importlib.import_module(package)
            
            # Check version if available
            if hasattr(module, '__version__'):
                print(f"✅ {package} {module.__version__} (installed)")
            else:
                print(f"✅ {package} (installed, version unknown)")
                
        except ImportError:
            print(f"❌ {package} not found")
            if install_package(package, version_spec):
                print(f"✅ {package} installed")
            else:
                all_good = False
    
    return all_good

def check_redis_search_support():
    """Check if Redis supports search operations"""
    try:
        import redis
        
        # Try importing Redis Search components
        try:
            from redis.commands.search.field import VectorField
            from redis.commands.search.query import Query
            from redis.commands.search.indexDefinition import IndexDefinition
            print("✅ Redis Search support available")
            return True
        except ImportError as e:
            print(f"⚠️  Redis Search support missing: {e}")
            print("   Consider upgrading: pip install 'redis[hiredis]>=4.5.0'")
            return False
            
    except ImportError:
        print("❌ Redis not installed")
        return False

def test_redis_connection():
    """Test Redis server connection"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        
        info = r.info()
        print(f"✅ Redis server connected")
        print(f"   Version: {info.get('redis_version', 'unknown')}")
        print(f"   Mode: {info.get('redis_mode', 'unknown')}")
        
        # Test Redis Stack
        try:
            modules = r.execute_command("MODULE LIST")
            search_module = None
            for module in modules:
                if b'search' in str(module).lower():
                    search_module = module
                    break
            
            if search_module:
                print("✅ Redis Stack detected (RediSearch available)")
                return "redis_stack"
            else:
                print("⚠️  Standard Redis (no RediSearch)")
                return "redis_standard"
                
        except Exception:
            print("⚠️  Standard Redis (no module support)")
            return "redis_standard"
            
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return None

def setup_mylath():
    """Complete MyLath setup process"""
    print("🚀 MyLath Setup and Dependency Checker")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not check_and_install_dependencies():
        print("❌ Failed to install some dependencies")
        return False
    
    # Check Redis Search support
    redis_search_ok = check_redis_search_support()
    
    # Test Redis connection
    redis_status = test_redis_connection()
    
    print("\n📊 SETUP SUMMARY")
    print("-" * 30)
    print(f"Python: ✅ Compatible")
    print(f"Dependencies: ✅ Installed")
    print(f"Redis Search: {'✅ Available' if redis_search_ok else '⚠️  Limited'}")
    print(f"Redis Server: {'✅ ' + redis_status if redis_status else '❌ Not available'}")
    
    if redis_status == "redis_stack":
        print(f"\n🎉 OPTIMAL SETUP!")
        print(f"   MyLath will use Redis Stack for high-performance vector search")
    elif redis_status == "redis_standard":
        print(f"\n⚡ GOOD SETUP")
        print(f"   MyLath will work with Python fallback")
        print(f"   For better performance, consider Redis Stack:")
        print(f"   docker run -d -p 6379:6379 redis/redis-stack-server")
    else:
        print(f"\n⚠️  REDIS NEEDED")
        print(f"   Start Redis server to use MyLath:")
        print(f"   • Standard Redis: docker run -d -p 6379:6379 redis:latest")
        print(f"   • Redis Stack: docker run -d -p 6379:6379 redis/redis-stack-server")
    
    print(f"\n🧪 NEXT STEPS")
    print(f"   Run the test: python quicktest.py")
    
    return True

if __name__ == "__main__":
    setup_mylath()