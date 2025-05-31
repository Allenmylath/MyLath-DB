# mylath/mylath/vector/vector_core.py
"""
Updated Vector Core that uses Redis native vector search by default
Falls back to Python implementation if Redis Stack is not available
"""

import redis
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import uuid
import warnings


@dataclass
class Vector:
    id: str
    data: List[float]
    metadata: Dict[str, Any]
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class VectorCore:
    """
    Intelligent Vector Core that automatically uses the best available backend:
    1. Redis Stack/RediSearch (fastest - native vector search)
    2. Python fallback (slower but always works)
    """
    
    def __init__(self, storage, 
                 index_name: str = "mylath_vectors",
                 vector_dim: int = None,
                 prefer_redis_search: bool = True):
        self.storage = storage
        self.redis = storage.redis
        self.index_name = index_name
        self.vector_dim = vector_dim
        self.vector_key_prefix = "vec:"
        self.metadata_key_prefix = "vec_meta:"
        
        # Determine which backend to use
        self.use_redis_search = False
        if prefer_redis_search:
            self.use_redis_search = self._check_redis_search_available()
        
        if self.use_redis_search:
            print("✅ Using Redis native vector search (high performance)")
            self._ensure_vector_index()
        else:
            print("⚠️  Using Python vector search (Redis Stack not available)")
            self._ensure_python_indices()
    
    def _check_redis_search_available(self) -> bool:
        """Check if Redis Search module is available"""
        try:
            # Try to execute a simple FT command
            self.redis.execute_command("FT._LIST")
            return True
        except redis.ResponseError as e:
            if "unknown command" in str(e).lower():
                return False
            return True  # Other errors might be OK
        except Exception:
            return False
    
    def _ensure_vector_index(self):
        """Create Redis vector search index if it doesn't exist"""
        try:
            # Check if index exists
            try:
                self.redis.execute_command("FT.INFO", self.index_name)
                return  # Index already exists
            except redis.ResponseError:
                pass  # Index doesn't exist, create it
            
            # Create vector search index when first vector is added
            if self.vector_dim:
                self._create_redis_index(self.vector_dim)
                
        except Exception as e:
            print(f"Warning: Could not create vector index: {e}")
            # Fall back to Python search
            self.use_redis_search = False
            self._ensure_python_indices()
    
    def _create_redis_index(self, dimension: int):
        """Create Redis vector search index"""
        try:
            index_definition = [
                "FT.CREATE", self.index_name,
                "ON", "HASH",
                "PREFIX", "1", self.vector_key_prefix,
                "SCHEMA",
                "vector", "VECTOR", "FLAT", "6",
                "TYPE", "FLOAT32",
                "DIM", str(dimension),
                "DISTANCE_METRIC", "COSINE",
                "metadata", "TEXT",
                "properties", "TEXT"
            ]
            
            self.redis.execute_command(*index_definition)
            print(f"Created Redis vector index with dimension {dimension}")
            
        except redis.ResponseError as e:
            if "Index already exists" not in str(e):
                print(f"Error creating Redis vector index: {e}")
                # Fall back to Python search
                self.use_redis_search = False
                self._ensure_python_indices()
    
    def _ensure_python_indices(self):
        """Ensure Python-based indices exist"""
        # Create basic sets for Python-based search
        pass
    
    def _generate_id(self) -> str:
        """Generate unique vector ID"""
        return str(uuid.uuid4())
    
    def add_vector(self, data: List[float], 
                   metadata: Dict[str, Any] = None,
                   properties: Dict[str, Any] = None) -> Vector:
        """Add vector using the best available backend"""
        if metadata is None:
            metadata = {}
        if properties is None:
            properties = {}
        
        vector_id = self._generate_id()
        vector = Vector(
            id=vector_id,
            data=data,
            metadata=metadata,
            properties=properties
        )
        
        # Set vector dimension if not set
        if not self.vector_dim:
            self.vector_dim = len(data)
            if self.use_redis_search:
                self._create_redis_index(self.vector_dim)
        
        if self.use_redis_search:
            self._add_vector_redis_search(vector)
        else:
            self._add_vector_python(vector)
        
        return vector
    
    def _add_vector_redis_search(self, vector: Vector):
        """Add vector using Redis native search"""
        vector_bytes = np.array(vector.data, dtype=np.float32).tobytes()
        vector_key = f"{self.vector_key_prefix}{vector.id}"
        
        pipeline = self.redis.pipeline()
        
        # Store for Redis Search
        pipeline.hset(vector_key, mapping={
            "id": vector.id,
            "vector": vector_bytes,
            "metadata": json.dumps(vector.metadata),
            "properties": json.dumps(vector.properties),
            "dimension": len(vector.data)
        })
        
        # Also store metadata separately for easy access
        metadata_key = f"{self.metadata_key_prefix}{vector.id}"
        pipeline.hset(metadata_key, mapping={
            "id": vector.id,
            "metadata": json.dumps(vector.metadata),
            "properties": json.dumps(vector.properties),
            "dimension": len(vector.data)
        })
        
        # General management
        pipeline.sadd("vector_index", vector.id)
        
        # Property indices for filtering
        for prop_name, prop_value in vector.properties.items():
            pipeline.sadd(f"vector_prop:{prop_name}:{prop_value}", vector.id)
        
        pipeline.execute()
    
    def _add_vector_python(self, vector: Vector):
        """Add vector using Python backend"""
        # Store vector data
        vector_key = f"vectors:{vector.id}"
        self.redis.hset(vector_key, mapping={
            "id": vector.id,
            "metadata": json.dumps(vector.metadata),
            "properties": json.dumps(vector.properties),
            "dimension": len(vector.data)
        })
        
        # Store vector data separately
        data_key = f"vector_data:{vector.id}"
        self.redis.set(data_key, json.dumps(vector.data))
        
        # Add to index
        self.redis.sadd("vector_index", vector.id)
        
        # Property indices
        for prop_name, prop_value in vector.properties.items():
            self.redis.sadd(f"vector_idx:{prop_name}:{prop_value}", vector.id)
    
    def get_vector(self, vector_id: str) -> Optional[Vector]:
        """Get vector by ID"""
        if self.use_redis_search:
            return self._get_vector_redis_search(vector_id)
        else:
            return self._get_vector_python(vector_id)
    
    def _get_vector_redis_search(self, vector_id: str) -> Optional[Vector]:
        """Get vector using Redis search backend"""
        vector_key = f"{self.vector_key_prefix}{vector_id}"
        vector_data = self.redis.hgetall(vector_key)
        
        if not vector_data:
            return None
        
        # Reconstruct vector from bytes
        vector_bytes = vector_data[b'vector']
        data = np.frombuffer(vector_bytes, dtype=np.float32).tolist()
        
        return Vector(
            id=vector_data[b'id'].decode(),
            data=data,
            metadata=json.loads(vector_data[b'metadata'].decode()),
            properties=json.loads(vector_data[b'properties'].decode())
        )
    
    def _get_vector_python(self, vector_id: str) -> Optional[Vector]:
        """Get vector using Python backend"""
        vector_key = f"vectors:{vector_id}"
        vector_info = self.redis.hgetall(vector_key)
        
        if not vector_info:
            return None
        
        # Get vector data
        data_key = f"vector_data:{vector_id}"
        data_json = self.redis.get(data_key)
        if not data_json:
            return None
        
        data = json.loads(data_json.decode())
        
        return Vector(
            id=vector_info[b'id'].decode(),
            data=data,
            metadata=json.loads(vector_info[b'metadata'].decode()),
            properties=json.loads(vector_info[b'properties'].decode())
        )
    
    def search_vectors(self, query_vector: List[float], 
                      k: int = 10,
                      filters: Dict[str, Any] = None,
                      metric: str = "cosine",
                      score_threshold: float = None) -> List[Tuple[Vector, float]]:
        """Search for similar vectors using the best available backend"""
        if filters is None:
            filters = {}
        
        if self.use_redis_search:
            return self._search_vectors_redis_search(query_vector, k, filters, score_threshold)
        else:
            return self._search_vectors_python(query_vector, k, filters, metric)
    
    def _search_vectors_redis_search(self, query_vector: List[float], 
                                   k: int, filters: Dict[str, Any],
                                   score_threshold: float = None) -> List[Tuple[Vector, float]]:
        """Search using Redis native vector search - MUCH FASTER"""
        try:
            # Convert query vector to bytes
            query_bytes = np.array(query_vector, dtype=np.float32).tobytes()
            
            # Build base search query
            search_params = [
                "FT.SEARCH", self.index_name,
                f"*=>[KNN {k} @vector $query_vector AS distance]",
                "PARAMS", "2", "query_vector", query_bytes,
                "SORTBY", "distance",
                "DIALECT", "2",
                "RETURN", "4", "id", "metadata", "properties", "distance"
            ]
            
            # Apply filters if any
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    # Create filter for both metadata and properties
                    filter_parts.append(f"(@metadata:*{value}*)|(@properties:*{value}*)")
                
                if filter_parts:
                    filter_query = " ".join(filter_parts)
                    search_params[2] = f"({filter_query})=>[KNN {k} @vector $query_vector AS distance]"
            
            # Execute Redis search
            result = self.redis.execute_command(*search_params)
            
            # Parse results
            vectors_with_scores = []
            
            if len(result) > 1:
                total_results = result[0]
                results = result[1:]
                
                # Process results in pairs (key, fields)
                for i in range(0, len(results), 2):
                    if i + 1 < len(results):
                        vector_key = results[i].decode()
                        fields = results[i + 1]
                        
                        # Parse fields
                        field_dict = {}
                        for j in range(0, len(fields), 2):
                            if j + 1 < len(fields):
                                field_name = fields[j].decode()
                                field_value = fields[j + 1].decode()
                                field_dict[field_name] = field_value
                        
                        # Get the full vector
                        vector_id = field_dict.get('id')
                        if vector_id:
                            vector = self.get_vector(vector_id)
                            if vector:
                                distance = float(field_dict.get('distance', 1.0))
                                # Convert distance to similarity score
                                similarity = max(0.0, 1.0 - distance)
                                
                                # Apply score threshold if specified
                                if score_threshold is None or similarity >= score_threshold:
                                    vectors_with_scores.append((vector, similarity))
            
            return vectors_with_scores[:k]
            
        except Exception as e:
            print(f"Redis vector search failed: {e}")
            print("Falling back to Python search...")
            return self._search_vectors_python(query_vector, k, filters, "cosine")
    
    def _search_vectors_python(self, query_vector: List[float], 
                             k: int, filters: Dict[str, Any],
                             metric: str) -> List[Tuple[Vector, float]]:
        """Fallback Python-based vector search"""
        # Get candidate vectors
        candidate_ids = set(self.redis.smembers("vector_index"))
        
        # Apply filters
        if filters:
            for prop_name, prop_value in filters.items():
                filter_key = f"vector_idx:{prop_name}:{prop_value}"
                filtered_ids = self.redis.smembers(filter_key)
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
    
    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from index"""
        if self.use_redis_search:
            return self._delete_vector_redis_search(vector_id)
        else:
            return self._delete_vector_python(vector_id)
    
    def _delete_vector_redis_search(self, vector_id: str) -> bool:
        """Delete vector using Redis search backend"""
        vector = self.get_vector(vector_id)
        if not vector:
            return False
        
        pipeline = self.redis.pipeline()
        
        # Remove from property indices
        for prop_name, prop_value in vector.properties.items():
            pipeline.srem(f"vector_prop:{prop_name}:{prop_value}", vector_id)
        
        # Remove from main sets
        pipeline.srem("vector_index", vector_id)
        
        # Delete vector data
        pipeline.delete(f"{self.vector_key_prefix}{vector_id}")
        pipeline.delete(f"{self.metadata_key_prefix}{vector_id}")
        
        pipeline.execute()
        return True
    
    def _delete_vector_python(self, vector_id: str) -> bool:
        """Delete vector using Python backend"""
        vector = self.get_vector(vector_id)
        if not vector:
            return False
        
        # Remove from property indices
        for prop_name, prop_value in vector.properties.items():
            self.redis.srem(f"vector_idx:{prop_name}:{prop_value}", vector_id)
        
        # Remove from main index
        self.redis.srem("vector_index", vector_id)
        
        # Delete vector data
        self.redis.delete(f"vectors:{vector_id}")
        self.redis.delete(f"vector_data:{vector_id}")
        
        return True
    
    def similarity_search_by_id(self, vector_id: str, k: int = 10,
                               filters: Dict[str, Any] = None) -> List[Tuple[Vector, float]]:
        """Find vectors similar to a given vector ID"""
        vector = self.get_vector(vector_id)
        if not vector:
            return []
        
        results = self.search_vectors(vector.data, k + 1, filters)
        # Remove the original vector from results
        return [(v, s) for v, s in results if v.id != vector_id][:k]
    
    def get_vector_count(self) -> int:
        """Get total number of vectors"""
        return self.redis.scard("vector_index")
    
    def list_vectors(self, limit: int = 100, offset: int = 0) -> List[Vector]:
        """List vectors with pagination"""
        vector_ids = list(self.redis.smembers("vector_index"))
        paginated_ids = vector_ids[offset:offset + limit]
        
        vectors = []
        for vector_id_bytes in paginated_ids:
            vector_id = vector_id_bytes.decode() if isinstance(vector_id_bytes, bytes) else vector_id_bytes
            vector = self.get_vector(vector_id)
            if vector:
                vectors.append(vector)
        
        return vectors
    
    def update_vector_metadata(self, vector_id: str, 
                              metadata: Dict[str, Any] = None,
                              properties: Dict[str, Any] = None) -> bool:
        """Update vector metadata without changing the vector data"""
        vector = self.get_vector(vector_id)
        if not vector:
            return False
        
        # Update metadata and properties
        if metadata:
            vector.metadata.update(metadata)
        if properties:
            vector.properties.update(properties)
        
        # Update in Redis based on backend
        if self.use_redis_search:
            return self._update_vector_metadata_redis_search(vector)
        else:
            return self._update_vector_metadata_python(vector)
    
    def _update_vector_metadata_redis_search(self, vector: Vector) -> bool:
        """Update metadata using Redis search backend"""
        vector_key = f"{self.vector_key_prefix}{vector.id}"
        metadata_key = f"{self.metadata_key_prefix}{vector.id}"
        
        pipeline = self.redis.pipeline()
        
        # Update in Redis
        pipeline.hset(vector_key, "metadata", json.dumps(vector.metadata))
        pipeline.hset(vector_key, "properties", json.dumps(vector.properties))
        pipeline.hset(metadata_key, "metadata", json.dumps(vector.metadata))
        pipeline.hset(metadata_key, "properties", json.dumps(vector.properties))
        
        pipeline.execute()
        return True
    
    def _update_vector_metadata_python(self, vector: Vector) -> bool:
        """Update metadata using Python backend"""
        vector_key = f"vectors:{vector.id}"
        
        self.redis.hset(vector_key, "metadata", json.dumps(vector.metadata))
        self.redis.hset(vector_key, "properties", json.dumps(vector.properties))
        
        return True
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the current vector search backend"""
        info = {
            "backend": "Redis Search" if self.use_redis_search else "Python",
            "vector_count": self.get_vector_count(),
            "vector_dimension": self.vector_dim,
            "index_name": self.index_name if self.use_redis_search else "N/A"
        }
        
        if self.use_redis_search:
            try:
                redis_info = self.redis.execute_command("FT.INFO", self.index_name)
                # Parse basic info
                info["redis_search_available"] = True
                info["redis_search_info"] = "Available"
            except Exception as e:
                info["redis_search_error"] = str(e)
        else:
            info["redis_search_available"] = False
            info["fallback_reason"] = "Redis Stack/RediSearch not available"
        
        return info
    
    def benchmark_search(self, num_queries: int = 100) -> Dict[str, Any]:
        """Benchmark vector search performance"""
        import time
        import random
        
        if self.vector_dim is None or self.get_vector_count() == 0:
            return {"error": "No vectors available for benchmarking"}
        
        # Generate random queries
        search_times = []
        
        for _ in range(num_queries):
            query_vector = np.random.random(self.vector_dim).tolist()
            
            start_time = time.time()
            results = self.search_vectors(query_vector, k=10)
            search_time = time.time() - start_time
            
            search_times.append(search_time)
        
        avg_time = sum(search_times) / len(search_times)
        min_time = min(search_times)
        max_time = max(search_times)
        
        return {
            "backend": "Redis Search" if self.use_redis_search else "Python",
            "num_queries": num_queries,
            "vector_count": self.get_vector_count(),
            "avg_search_time_ms": avg_time * 1000,
            "min_search_time_ms": min_time * 1000,
            "max_search_time_ms": max_time * 1000,
            "searches_per_second": 1.0 / avg_time,
            "performance_tier": "High" if self.use_redis_search else "Medium"
        }
    
    def force_backend(self, use_redis_search: bool = True):
        """Force switch to a specific backend (for testing)"""
        if use_redis_search and not self._check_redis_search_available():
            print("⚠️  Cannot switch to Redis Search: not available")
            return False
        
        self.use_redis_search = use_redis_search
        
        if self.use_redis_search:
            print("✅ Switched to Redis native vector search")
            self._ensure_vector_index()
        else:
            print("✅ Switched to Python vector search")
            self._ensure_python_indices()
        
        return True


