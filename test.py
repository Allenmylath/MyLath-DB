"""
MyLathDB Matrix Creation and Query Test
Tests matrix creation during data loading and verifies query execution with known results
"""

import pytest
import sys
import os

# Add your mylathdb package to path
# sys.path.append('/path/to/your/mylathdb/package')

from mylathdb import MyLathDB, ExecutionResult


class TestMyLathDBMatrixCreation:
    """Test suite for MyLathDB matrix creation and query execution"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample graph data with known structure"""
        
        # Sample nodes - 6 people with different ages and roles
        nodes = [
            {"id": "1", "name": "Alice", "age": 30, "role": "Engineer", "_labels": ["Person"]},
            {"id": "2", "name": "Bob", "age": 25, "role": "Designer", "_labels": ["Person"]},
            {"id": "3", "name": "Charlie", "age": 35, "role": "Manager", "_labels": ["Person"]},
            {"id": "4", "name": "Diana", "age": 28, "role": "Engineer", "_labels": ["Person"]},
            {"id": "5", "name": "Eve", "age": 22, "role": "Intern", "_labels": ["Person"]},
            {"id": "6", "name": "Frank", "age": 45, "role": "Director", "_labels": ["Person"]}
        ]
        
        # Sample edges - relationships between people
        edges = [
            ("1", "KNOWS", "2"),      # Alice knows Bob
            ("1", "KNOWS", "4"),      # Alice knows Diana  
            ("2", "KNOWS", "3"),      # Bob knows Charlie
            ("2", "WORKS_WITH", "4"), # Bob works with Diana
            ("3", "MANAGES", "1"),    # Charlie manages Alice
            ("3", "MANAGES", "4"),    # Charlie manages Diana
            ("4", "MENTORS", "5"),    # Diana mentors Eve
            ("6", "MANAGES", "3"),    # Frank manages Charlie
        ]
        
        return {"nodes": nodes, "edges": edges}
    
    @pytest.fixture
    def expected_results(self):
        """Define expected query results for verification"""
        return {
            # Simple traversal: Who does Alice know?
            "alice_knows": [
                {"name": "Bob", "age": 25},
                {"name": "Diana", "age": 28}
            ],
            
            # Property filter: People Alice knows who are older than 26
            "alice_knows_older_26": [
                {"name": "Diana", "age": 28}
            ],
            
            # All people managed by Charlie
            "charlie_manages": [
                {"name": "Alice", "age": 30},
                {"name": "Diana", "age": 28}
            ],
            
            # All Engineers
            "all_engineers": [
                {"name": "Alice", "age": 30, "role": "Engineer"},
                {"name": "Diana", "age": 28, "role": "Engineer"}
            ],
            
            # People older than 30
            "older_than_30": [
                {"name": "Charlie", "age": 35},
                {"name": "Frank", "age": 45}
            ],
            
            # Two-hop traversal: Who do Alice's connections know?
            "alice_two_hop": [
                {"name": "Charlie"},  # Bob knows Charlie
                {"name": "Eve"}       # Diana mentors Eve
            ]
        }
    
    @pytest.fixture
    def database(self, sample_data):
        """Create and initialize MyLathDB with sample data - ISOLATED PER TEST"""
        db = MyLathDB()
        
        # CLEANUP: Clear Redis and GraphBLAS before loading data
        self._cleanup_database(db)
        
        # Load sample data
        db.load_graph_data(
            nodes=sample_data["nodes"], 
            edges=sample_data["edges"]
        )
        
        yield db  # Use yield instead of return for cleanup
        
        # CLEANUP: Clear Redis and GraphBLAS after test
        self._cleanup_database(db)
        db.shutdown()
    
    def _cleanup_database(self, db):
        """Clean Redis and GraphBLAS data to prevent test contamination"""
        
        # Clear Redis data
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            try:
                db.engine.redis_executor.redis.flushdb()
                print("✅ Redis database flushed")
            except Exception as e:
                print(f"⚠️ Redis flush failed: {e}")
        
        # Clear GraphBLAS matrices
        if db.engine.graphblas_executor.is_available():
            try:
                db.engine.graphblas_executor.clear_matrices()
                print("✅ GraphBLAS matrices cleared")
            except Exception as e:
                print(f"⚠️ GraphBLAS clear failed: {e}")
        
        # Clear caches
        try:
            db.clear_cache()
            print("✅ Database caches cleared")
        except Exception as e:
            print(f"⚠️ Cache clear failed: {e}")
    
    def test_database_initialization(self, database):
        """Test that database initializes correctly"""
        assert database is not None
        assert database.engine is not None
        assert database.logical_planner is not None
        assert database.physical_planner is not None
    
    def test_redis_data_loading(self, database):
        """Test that data is loaded correctly into Redis"""
        
        # Debug Redis state
        print("\n=== Redis State Debug ===")
        database.debug_redis_state()
        
        # Verify Redis has data
        redis_client = database.engine.redis_executor.redis
        if redis_client:
            # Check for node data
            node_keys = [key for key in redis_client.scan_iter() if key.startswith('node:')]
            assert len(node_keys) >= 6, f"Expected at least 6 nodes, found {len(node_keys)}"
            
            # Check for label indexes
            label_keys = [key for key in redis_client.scan_iter() if key.startswith('label:')]
            assert len(label_keys) >= 1, f"Expected label indexes, found {len(label_keys)}"
            
            # Verify specific node exists
            alice_data = redis_client.hgetall('node:1')
            assert alice_data.get('name') == 'Alice'
            assert alice_data.get('age') == '30'
    
    def test_graphblas_matrix_creation(self, database):
        """Test that GraphBLAS matrices are created correctly"""
        
        gb_executor = database.engine.graphblas_executor
        
        # Check GraphBLAS status
        status = gb_executor.get_status()
        print(f"\n=== GraphBLAS Status ===")
        for key, value in status.items():
            print(f"{key}: {value}")
        
        if gb_executor.is_available():
            # Verify core matrices exist
            assert gb_executor.graph is not None
            assert gb_executor.graph.adjacency_matrix is not None
            assert gb_executor.graph.node_labels_matrix is not None
            
            # Verify relation matrices were created
            assert len(gb_executor.graph.relation_matrices) >= 3  # KNOWS, WORKS_WITH, MANAGES, MENTORS
            
            # Check specific relation matrices
            assert "KNOWS" in gb_executor.graph.relation_matrices
            assert "MANAGES" in gb_executor.graph.relation_matrices
            
            # Verify matrix has data (non-zero values)
            knows_matrix = gb_executor.graph.relation_matrices["KNOWS"]
            assert knows_matrix.nvals > 0, "KNOWS matrix should have edges"
            
            print(f"\nRelation matrices created: {list(gb_executor.graph.relation_matrices.keys())}")
            print(f"Adjacency matrix non-zeros: {gb_executor.graph.adjacency_matrix.nvals}")
        else:
            pytest.skip("GraphBLAS not available, skipping matrix tests")
    
    def test_simple_traversal_query(self, database, expected_results):
        """Test simple traversal: Who does Alice know?"""
        
        query = "MATCH (alice:Person {name: 'Alice'})-[:KNOWS]->(friend) RETURN friend.name, friend.age"
        
        print(f"\n=== Testing Query ===")
        print(f"Query: {query}")
        
        # Execute with debug
        result = database.debug_query_execution(query)
        
        assert result is not None
        assert result.success, f"Query failed: {result.error}"
        assert len(result.data) > 0, "No results returned"
        
        # Extract names and ages from results
        actual_friends = []
        for record in result.data:
            # Handle different possible result formats
            name = record.get('friend.name') or record.get('name')
            age_val = record.get('friend.age') or record.get('age')
            age = int(age_val) if age_val else None
            
            if name and age:
                actual_friends.append({"name": name, "age": age})
        
        expected_friends = expected_results["alice_knows"]
        
        print(f"Expected: {expected_friends}")
        print(f"Actual: {actual_friends}")
        
        # Verify we got the right people
        assert len(actual_friends) == len(expected_friends), f"Expected {len(expected_friends)} friends, got {len(actual_friends)}"
        
        # Check each expected friend is present
        for expected_friend in expected_friends:
            assert expected_friend in actual_friends, f"Missing expected friend: {expected_friend}"
    
    def test_property_filter_query(self, database, expected_results):
        """Test traversal with property filter"""
        
        query = "MATCH (alice:Person {name: 'Alice'})-[:KNOWS]->(friend) WHERE friend.age > 26 RETURN friend.name, friend.age"
        
        print(f"\n=== Testing Property Filter Query ===")
        print(f"Query: {query}")
        
        result = database.execute_query(query)
        
        assert result.success, f"Query failed: {result.error}"
        
        # Extract filtered results
        filtered_friends = []
        for record in result.data:
            name = record.get('friend.name') or record.get('name')
            age_val = record.get('friend.age') or record.get('age')
            age = int(age_val) if age_val else None
            
            if name and age:
                filtered_friends.append({"name": name, "age": age})
        
        expected_filtered = expected_results["alice_knows_older_26"]
        
        print(f"Expected: {expected_filtered}")
        print(f"Actual: {filtered_friends}")
        
        assert len(filtered_friends) == len(expected_filtered)
        for expected in expected_filtered:
            assert expected in filtered_friends
    
    def test_label_based_query(self, database, expected_results):
        """Test label-based node matching"""
        
        query = "MATCH (person:Person) WHERE person.role = 'Engineer' RETURN person.name, person.age, person.role"
        
        print(f"\n=== Testing Label-Based Query ===")
        print(f"Query: {query}")
        
        result = database.execute_query(query)
        
        assert result.success, f"Query failed: {result.error}"
        
        # Extract engineers
        engineers = []
        for record in result.data:
            name = record.get('person.name') or record.get('name')
            age_val = record.get('person.age') or record.get('age')
            role = record.get('person.role') or record.get('role')
            age = int(age_val) if age_val else None
            
            if name and age and role:
                engineers.append({"name": name, "age": age, "role": role})
        
        expected_engineers = expected_results["all_engineers"]
        
        print(f"Expected: {expected_engineers}")
        print(f"Actual: {engineers}")
        
        assert len(engineers) == len(expected_engineers)
        for expected in expected_engineers:
            assert expected in engineers
    
    def test_range_query(self, database, expected_results):
        """Test numeric range queries"""
        
        query = "MATCH (person:Person) WHERE person.age > 30 RETURN person.name, person.age ORDER BY person.age"
        
        print(f"\n=== Testing Range Query ===")
        print(f"Query: {query}")
        
        result = database.execute_query(query)
        
        assert result.success, f"Query failed: {result.error}"
        
        # Extract older people
        older_people = []
        for record in result.data:
            name = record.get('person.name') or record.get('name')
            age_val = record.get('person.age') or record.get('age')
            age = int(age_val) if age_val else None
            
            if name and age:
                older_people.append({"name": name, "age": age})
        
        expected_older = expected_results["older_than_30"]
        
        print(f"Expected: {expected_older}")
        print(f"Actual: {older_people}")
        
        assert len(older_people) == len(expected_older)
        for expected in expected_older:
            assert expected in older_people
    
    @pytest.fixture(scope="function")  # Ensure fresh instance per test
    def isolated_database(self):
        """Create a completely isolated database instance for tests that need clean state"""
        db = MyLathDB(redis_db=1)  # Use different Redis DB to avoid conflicts
        
        # Ensure clean state
        self._cleanup_database(db)
        
        yield db
        
        # Cleanup after test
        self._cleanup_database(db)
        db.shutdown()
        """Test that execution statistics are tracked"""
        
        # Execute a few queries
        database.execute_query("MATCH (n:Person) RETURN n.name LIMIT 5")
        database.execute_query("MATCH (a:Person)-[:KNOWS]->(b) RETURN a.name, b.name")
        
        # Get statistics
        stats = database.get_statistics()
        
        assert 'database' in stats
        assert 'engine' in stats
        assert stats['database']['queries_executed'] >= 2
        assert stats['database']['total_execution_time'] > 0
        
        print(f"\n=== Execution Statistics ===")
        print(f"Queries executed: {stats['database']['queries_executed']}")
        print(f"Total execution time: {stats['database']['total_execution_time']:.4f}s")
        print(f"Average execution time: {stats['database']['avg_execution_time']:.4f}s")
    
    def test_matrix_functionality(self, database):
        """Test GraphBLAS matrix operations work correctly"""
        
        gb_executor = database.engine.graphblas_executor
        
        if gb_executor.is_available():
            # Test basic functionality
            functionality_works = gb_executor.test_functionality()
            assert functionality_works, "GraphBLAS functionality test failed"
            
            print(f"\n=== GraphBLAS Functionality Test ===")
            print(f"✅ GraphBLAS matrices and operations working correctly")
        else:
            pytest.skip("GraphBLAS not available, skipping matrix functionality tests")


# Additional utility test with isolated database
def test_projection_fix_verification():
    """Standalone test for projection functionality with complete isolation"""
    
    db = MyLathDB(redis_db=4)  # Separate Redis DB
    
    # Ensure clean state
    if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
        db.engine.redis_executor.redis.flushdb()
    
    if db.engine.graphblas_executor.is_available():
        db.engine.graphblas_executor.clear_matrices()
    
    # Load minimal test data
    db.load_graph_data(nodes=[{
        'id': '1', 
        'name': 'TestUser', 
        'age': 25, 
        '_labels': ['Person']
    }])
    
    # Test the projection fix
    projection_works = db.test_projection_fix()
    assert projection_works, "Projection fix is not working correctly"
    
    # Cleanup
    db.shutdown()


def test_concurrent_database_instances():
    """Test that multiple database instances don't interfere with each other"""
    
    # Create two separate database instances with different Redis DBs
    db1 = MyLathDB(redis_db=5)
    db2 = MyLathDB(redis_db=6)
    
    # Clear both databases
    for db in [db1, db2]:
        if hasattr(db.engine.redis_executor, 'redis') and db.engine.redis_executor.redis:
            db.engine.redis_executor.redis.flushdb()
        if db.engine.graphblas_executor.is_available():
            db.engine.graphblas_executor.clear_matrices()
    
    # Load different data into each
    db1.load_graph_data(nodes=[{"id": "1", "name": "User1", "_labels": ["Type1"]}])
    db2.load_graph_data(nodes=[{"id": "1", "name": "User2", "_labels": ["Type2"]}])
    
    # Query each database
    result1 = db1.execute_query("MATCH (n) RETURN n.name")
    result2 = db2.execute_query("MATCH (n) RETURN n.name")
    
    # Verify isolation
    assert result1.success and result2.success
    assert len(result1.data) == 1 and len(result2.data) == 1
    
    name1 = result1.data[0].get('n.name') or result1.data[0].get('name')
    name2 = result2.data[0].get('n.name') or result2.data[0].get('name')
    
    assert name1 == "User1"
    assert name2 == "User2"
    assert name1 != name2  # Confirm they're different
    
    # Cleanup
    db1.shutdown()
    db2.shutdown()
    
    print("✅ Concurrent database instances properly isolated")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])