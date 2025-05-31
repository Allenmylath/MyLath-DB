# tests/test_api.py
import pytest
import json
from mylath.mylath.storage.redis_storage import RedisStorage
from mylath.mylath.api.graph_api import GraphAPI


@pytest.fixture
def storage():
    storage = RedisStorage(db=15)
    yield storage
    storage.redis.flushdb()


@pytest.fixture
def client(storage):
    """Create a test client"""
    api = GraphAPI(storage)
    api.app.config['TESTING'] = True
    with api.app.test_client() as client:
        yield client


def test_create_node_api(client):
    """Test node creation via API"""
    response = client.post('/nodes', 
                          json={"label": "person", "properties": {"name": "Alice"}})
    
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data["label"] == "person"
    assert data["properties"]["name"] == "Alice"
    assert "id" in data


def test_get_node_api(client):
    """Test node retrieval via API"""
    # Create node first
    create_response = client.post('/nodes',
                                 json={"label": "person", "properties": {"name": "Bob"}})
    create_data = json.loads(create_response.data)
    node_id = create_data["id"]
    
    # Get node
    response = client.get(f'/nodes/{node_id}')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert data["id"] == node_id
    assert data["properties"]["name"] == "Bob"


def test_create_edge_api(client):
    """Test edge creation via API"""
    # Create nodes first
    node1_response = client.post('/nodes', json={"label": "person", "properties": {"name": "Alice"}})
    node2_response = client.post('/nodes', json={"label": "person", "properties": {"name": "Bob"}})
    
    node1_id = json.loads(node1_response.data)["id"]
    node2_id = json.loads(node2_response.data)["id"]
    
    # Create edge
    response = client.post('/edges', json={
        "label": "knows",
        "from_node": node1_id,
        "to_node": node2_id,
        "properties": {"since": "2020"}
    })
    
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data["label"] == "knows"
    assert data["from_node"] == node1_id
    assert data["to_node"] == node2_id


def test_vector_search_api(client):
    """Test vector operations via API"""
    # Add vector
    add_response = client.post('/vectors', json={
        "data": [0.1, 0.2, 0.3, 0.4],
        "metadata": {"type": "test"},
        "properties": {"source": "doc1"}
    })
    
    assert add_response.status_code == 201
    
    # Search vectors
    search_response = client.post('/vectors/search', json={
        "query_vector": [0.15, 0.25, 0.35, 0.45],
        "k": 5,
        "metric": "cosine"
    })
    
    assert search_response.status_code == 200
    data = json.loads(search_response.data)
    assert len(data) >= 1
    assert "vector" in data[0]
    assert "score" in data[0]
