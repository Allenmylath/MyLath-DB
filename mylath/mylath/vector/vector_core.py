# mylath/vector/vector_core.py
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from ..storage.redis_storage import RedisStorage, Vector
import heapq
import random
import math


class VectorCore:
    """Vector operations and HNSW index for similarity search"""
    
    def __init__(self, storage: RedisStorage, 
                 m: int = 16, ef_construction: int = 128, ef_search: int = 768):
        self.storage = storage
        self.m = m  # max connections per node
        self.ef_construction = ef_construction  # size of dynamic candidate list
        self.ef_search = ef_search  # search parameter
        self.m_l = 1.0 / math.log(2.0 * m)  # level generation factor
        
    def _generate_id(self) -> str:
        """Generate unique vector ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _euclidean_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.linalg.norm(vec1 - vec2)
    
    def _get_random_level(self) -> int:
        """Generate random level for HNSW"""
        level = 0
        while random.random() < 0.5 and level < 16:  # max level 16
            level += 1
        return level
    
    def add_vector(self, data: List[float], metadata: Dict[str, Any] = None,
                   properties: Dict[str, Any] = None) -> Vector:
        """Add vector to the index"""
        if metadata is None:
            metadata = {}
        if properties is None:
            properties = {}
            
        vector = Vector(
            id=self._generate_id(),
            data=data,
            metadata=metadata,
            properties=properties
        )
        
        # Store vector data
        vector_key = f"vectors:{vector.id}"
        self.storage.redis.hset(vector_key, mapping={
            "id": vector.id,
            "metadata": json.dumps(vector.metadata),
            "properties": json.dumps(vector.properties),
            "dimension": len(vector.data)
        })
        
        # Store vector data separately for efficiency
        data_key = f"vector_data:{vector.id}"
        self.storage.redis.set(data_key, json.dumps(vector.data))
        
        # Add to vector index
        self.storage.redis.sadd("vector_index", vector.id)
        
        # Index properties for filtering
        for prop_name, prop_value in properties.items():
            self.storage.redis.sadd(f"vector_idx:{prop_name}:{prop_value}", vector.id)
            
        return vector
    
    def get_vector(self, vector_id: str) -> Optional[Vector]:
        """Get vector by ID"""
        vector_key = f"vectors:{vector_id}"
        vector_info = self.storage.redis.hgetall(vector_key)
        
        if not vector_info:
            return None
            
        # Get vector data
        data_key = f"vector_data:{vector_id}"
        data_json = self.storage.redis.get(data_key)
        if not data_json:
            return None
            
        data = json.loads(data_json.decode())
        
        return Vector(
            id=vector_info[b'id'].decode(),
            data=data,
            metadata=json.loads(vector_info[b'metadata'].decode()),
            properties=json.loads(vector_info[b'properties'].decode())
        )
    
    def search_vectors(self, query_vector: List[float], k: int = 10,
                      filters: Dict[str, Any] = None,
                      metric: str = "cosine") -> List[Tuple[Vector, float]]:
        """Search for similar vectors"""
        if filters is None:
            filters = {}
            
        # Get candidate vectors (apply filters if any)
        candidate_ids = set(self.storage.redis.smembers("vector_index"))
        
        # Apply filters
        for prop_name, prop_value in filters.items():
            filter_key = f"vector_idx:{prop_name}:{prop_value}"
            filtered_ids = self.storage.redis.smembers(filter_key)
            candidate_ids = candidate_ids.intersection(filtered_ids)
        
        # Calculate similarities
        results = []
        for vector_id in candidate_ids:
            vector_id = vector_id.decode() if isinstance(vector_id, bytes) else vector_id
            vector = self.get_vector(vector_id)
            if vector:
                if metric == "cosine":
                    similarity = self._cosine_similarity(query_vector, vector.data)
                    score = similarity
                else:  # euclidean
                    distance = self._euclidean_distance(query_vector, vector.data)
                    score = -distance  # negative for heap ordering
                    
                results.append((vector, score))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from index"""
        vector = self.get_vector(vector_id)
        if not vector:
            return False
            
        # Remove from property indices
        for prop_name, prop_value in vector.properties.items():
            self.storage.redis.srem(f"vector_idx:{prop_name}:{prop_value}", vector_id)
        
        # Remove from main index
        self.storage.redis.srem("vector_index", vector_id)
        
        # Delete vector data
        self.storage.redis.delete(f"vectors:{vector_id}")
        self.storage.redis.delete(f"vector_data:{vector_id}")
        
        return True
