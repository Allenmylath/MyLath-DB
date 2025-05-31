# tests/test_traversal.py
import pytest
from mylath.mylath.storage.redis_storage import RedisStorage
from mylath.mylath.graph.traversal import GraphTraversal


@pytest.fixture
def storage():
    storage = RedisStorage(db=15)
    yield storage
    storage.redis.flushdb()


@pytest.fixture
def sample_graph(storage):
    """Create a sample graph for testing"""
    # Create nodes
    alice = storage.create_node("person", {"name": "Alice", "age": 30})
    bob = storage.create_node("person", {"name": "Bob", "age": 25})
    charlie = storage.create_node("person", {"name": "Charlie", "age": 35})
    company = storage.create_node("company", {"name": "TechCorp"})
    
    # Create edges
    storage.create_edge("knows", alice.id, bob.id, {"since": "2020"})
    storage.create_edge("knows", bob.id, charlie.id, {"since": "2019"})
    storage.create_edge("works_at", alice.id, company.id, {"role": "Engineer"})
    storage.create_edge("works_at", bob.id, company.id, {"role": "Designer"})
    
    return {
        "alice": alice,
        "bob": bob,
        "charlie": charlie,
        "company": company
    }


def test_vertex_traversal(storage, sample_graph):
    """Test basic vertex traversal"""
    alice = sample_graph["alice"]
    
    traversal = GraphTraversal(storage)
    friends = traversal.V([alice.id]).out("knows").to_list()
    
    assert len(friends) == 1
    assert friends[0].properties["name"] == "Bob"


def test_has_filter(storage, sample_graph):
    """Test has() filtering"""
    traversal = GraphTraversal(storage)
    young_people = traversal.V().has("age", 25).to_list()
    
    assert len(young_people) == 1
    assert young_people[0].properties["name"] == "Bob"


def test_chained_traversal(storage, sample_graph):
    """Test chained traversal operations"""
    alice = sample_graph["alice"]
    
    traversal = GraphTraversal(storage)
    colleagues = (traversal.V([alice.id])
                          .out("works_at")
                          .in_("works_at")
                          .has("name")
                          .to_list())
    
    # Should include Alice and Bob (both work at TechCorp)
    assert len(colleagues) >= 1
    names = [c.properties["name"] for c in colleagues]
    assert "Bob" in names


def test_shortest_path(storage, sample_graph):
    """Test shortest path finding"""
    alice = sample_graph["alice"]
    charlie = sample_graph["charlie"]
    
    traversal = GraphTraversal(storage)
    path = traversal.V([alice.id]).shortest_path(charlie.id, "knows")
    
    assert path is not None
    assert len(path) == 3  # Alice -> Bob -> Charlie
    assert path[0].id == alice.id
    assert path[-1].id == charlie.id


def test_count(storage, sample_graph):
    """Test count operation"""
    traversal = GraphTraversal(storage)
    
    # Use filter function to check label attribute (not property)
    person_count = traversal.V().filter(lambda n: n.label == "person").count()
    assert person_count == 3  # Alice, Bob, Charlie
    
    # Alternative: count all nodes and check manually
    all_nodes = traversal.V().to_list()
    persons = [n for n in all_nodes if n.label == "person"]
    assert len(persons) == 3


def test_dedup(storage, sample_graph):
    """Test deduplication"""
    traversal = GraphTraversal(storage)
    
    # This might create duplicates, dedup should remove them
    results = (traversal.V()
                       .filter(lambda n: n.label == "person")
                       .out("works_at")
                       .in_("works_at")
                       .dedup()
                       .to_list())
    
    # Check that there are no duplicate IDs
    ids = [r.id for r in results]
    assert len(ids) == len(set(ids))
