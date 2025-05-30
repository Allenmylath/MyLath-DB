# tests/test_storage.py
import pytest
import redis
from mylath.storage.redis_storage import RedisStorage, Node, Edge


@pytest.fixture
def storage():
    """Create a test storage instance"""
    # Use a different DB for testing
    storage = RedisStorage(db=15)
    yield storage
    # Cleanup after test
    storage.redis.flushdb()


def test_create_node(storage):
    """Test node creation"""
    node = storage.create_node("person", {"name": "Alice", "age": 30})
    
    assert node.id is not None
    assert node.label == "person"
    assert node.properties["name"] == "Alice"
    assert node.properties["age"] == 30
    assert node.created_at is not None


def test_get_node(storage):
    """Test node retrieval"""
    original = storage.create_node("person", {"name": "Bob"})
    retrieved = storage.get_node(original.id)
    
    assert retrieved is not None
    assert retrieved.id == original.id
    assert retrieved.label == original.label
    assert retrieved.properties == original.properties


def test_update_node(storage):
    """Test node update"""
    node = storage.create_node("person", {"name": "Charlie", "age": 25})
    
    success = storage.update_node(node.id, {"age": 26, "city": "NYC"})
    assert success
    
    updated = storage.get_node(node.id)
    assert updated.properties["age"] == 26
    assert updated.properties["city"] == "NYC"
    assert updated.properties["name"] == "Charlie"


def test_delete_node(storage):
    """Test node deletion"""
    node = storage.create_node("person", {"name": "David"})
    
    success = storage.delete_node(node.id)
    assert success
    
    retrieved = storage.get_node(node.id)
    assert retrieved is None


def test_create_edge(storage):
    """Test edge creation"""
    node1 = storage.create_node("person", {"name": "Alice"})
    node2 = storage.create_node("person", {"name": "Bob"})
    
    edge = storage.create_edge("knows", node1.id, node2.id, {"since": "2020"})
    
    assert edge.id is not None
    assert edge.label == "knows"
    assert edge.from_node == node1.id
    assert edge.to_node == node2.id
    assert edge.properties["since"] == "2020"


def test_get_outgoing_edges(storage):
    """Test getting outgoing edges"""
    node1 = storage.create_node("person", {"name": "Alice"})
    node2 = storage.create_node("person", {"name": "Bob"})
    node3 = storage.create_node("person", {"name": "Charlie"})
    
    edge1 = storage.create_edge("knows", node1.id, node2.id)
    edge2 = storage.create_edge("knows", node1.id, node3.id)
    edge3 = storage.create_edge("likes", node1.id, node2.id)
    
    # Get all outgoing edges
    all_edges = storage.get_outgoing_edges(node1.id)
    assert len(all_edges) == 3
    
    # Get edges by label
    knows_edges = storage.get_outgoing_edges(node1.id, "knows")
    assert len(knows_edges) == 2
    
    likes_edges = storage.get_outgoing_edges(node1.id, "likes")
    assert len(likes_edges) == 1


def test_find_nodes_by_property(storage):
    """Test finding nodes by property"""
    node1 = storage.create_node("person", {"name": "Alice", "age": 30})
    node2 = storage.create_node("person", {"name": "Bob", "age": 25})
    node3 = storage.create_node("person", {"name": "Charlie", "age": 30})
    
    nodes_age_30 = storage.find_nodes_by_property("age", 30)
    assert len(nodes_age_30) == 2
    
    names = [n.properties["name"] for n in nodes_age_30]
    assert "Alice" in names
    assert "Charlie" in names
