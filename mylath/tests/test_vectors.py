# tests/test_vectors.py
import pytest
import numpy as np
from mylath.mylath.storage.redis_storage import RedisStorage
from mylath.mylath.vector.vector_core import VectorCore


@pytest.fixture
def storage():
    storage = RedisStorage(db=15)
    yield storage
    storage.redis.flushdb()


@pytest.fixture
def vector_core(storage):
    return VectorCore(storage)


def test_add_vector(vector_core):
    """Test adding a vector"""
    data = [0.1, 0.2, 0.3, 0.4]
    metadata = {"type": "test"}
    properties = {"source": "test_doc"}
    
    vector = vector_core.add_vector(data, metadata, properties)
    
    assert vector.id is not None
    assert vector.data == data
    assert vector.metadata == metadata
    assert vector.properties == properties


def test_get_vector(vector_core):
    """Test retrieving a vector"""
    data = [0.1, 0.2, 0.3, 0.4]
    original = vector_core.add_vector(data)
    
    retrieved = vector_core.get_vector(original.id)
    
    assert retrieved is not None
    assert retrieved.id == original.id
    assert retrieved.data == data


def test_search_vectors(vector_core):
    """Test vector similarity search"""
    # Add some vectors
    vec1 = vector_core.add_vector([0.1, 0.2, 0.3, 0.4], properties={"type": "doc"})
    vec2 = vector_core.add_vector([0.2, 0.3, 0.4, 0.5], properties={"type": "doc"})
    vec3 = vector_core.add_vector([0.8, 0.9, 0.1, 0.2], properties={"type": "image"})
    
    # Search for similar vectors
    query = [0.15, 0.25, 0.35, 0.45]
    results = vector_core.search_vectors(query, k=2)
    
    assert len(results) == 2
    
    # Results should be sorted by similarity (highest first)
    assert results[0][1] >= results[1][1]  # First result has higher score
    
    # The most similar should be vec1 or vec2 (not vec3)
    top_result_id = results[0][0].id
    assert top_result_id in [vec1.id, vec2.id]


def test_search_with_filters(vector_core):
    """Test vector search with filters"""
    # Add vectors with different types
    doc_vec = vector_core.add_vector([0.1, 0.2, 0.3, 0.4], properties={"type": "doc"})
    img_vec = vector_core.add_vector([0.2, 0.3, 0.4, 0.5], properties={"type": "image"})
    
    # Search with filter
    query = [0.15, 0.25, 0.35, 0.45]
    doc_results = vector_core.search_vectors(query, k=10, filters={"type": "doc"})
    
    assert len(doc_results) == 1
    assert doc_results[0][0].id == doc_vec.id


def test_delete_vector(vector_core):
    """Test vector deletion"""
    vector = vector_core.add_vector([0.1, 0.2, 0.3, 0.4])
    
    success = vector_core.delete_vector(vector.id)
    assert success
    
    retrieved = vector_core.get_vector(vector.id)
    assert retrieved is None
