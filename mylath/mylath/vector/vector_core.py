# mylath/mylath/vector/vector_core.py
"""
Redis Vector Database Implementation
Following official Redis vector database patterns from:
https://redis.io/docs/latest/develop/get-started/vector-database/
"""

import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from ..storage.redis_storage import RedisStorage, Vector
import uuid
import struct

class VectorCore:
    """
    Redis Vector Database implementation following official Redis patterns
    Uses Redis Stack's native vector search capabilities with automatic fallback
    """
    
    def __init__(self, storage: RedisStorage, 
                 m: int = 16, ef_construction: int = 128, ef_search: int = 768):
        self.storage = storage
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Test Redis Stack availability using official methods
        self.redis_stack_available = self._test_redis_stack()
        self.index_name = "idx:vectors"  # Following Redis naming conventions
        
        if self.redis_stack_available:
            print("🚀 Using Redis Stack native vector search")
            self._ensure_vector_index()
        else:
            print("⚙️  Using Python fallback for vector search")
    
    def _test_redis_stack(self) -> bool:
        """Test Redis Stack using the official approach"""
        try:
            # Method 1: Check loaded modules
            modules = self.storage.redis.execute_command("MODULE LIST")
            search_module_found = False
            
            for module in modules:
                if isinstance(module, dict):
                    name = module.get(b'name', b'')
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    if name.lower() == 'search':
                        search_module_found = True
                        print("✅ RediSearch module detected")
                        break
            
            if not search_module_found:
                print("❌ RediSearch module not found")
                return False
            
            # Method 2: Test FT._LIST command (official Redis way)
            try:
                self.storage.redis.execute_command("FT._LIST")
                print("✅ Redis vector search commands available")
                return True
            except Exception as e:
                print(f"❌ Redis search commands failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Redis Stack test failed: {e}")
            return False
    
    def _ensure_vector_index(self):
        """Create vector index using official Redis patterns"""
        if not self.redis_stack_available:
            return
            
        try:
            # Check if index exists using FT.INFO
            self.storage.redis.execute_command("FT.INFO", self.index_name)
            print(f"✅ Vector index '{self.index_name}' exists")
        except:
            # Create index using official Redis vector schema
            try:
                # Official Redis vector index creation
                # Based on: https://redis.io/docs/latest/develop/get-started/vector-database/
                schema = [
                    "vector", "VECTOR", "HNSW", "6",
                    "TYPE", "FLOAT32",
                    "DIM", "128",
                    "DISTANCE_METRIC", "COSINE",
                    "type", "TAG",
                    "title", "TEXT",
                    "score", "NUMERIC"
                ]
                
                # Create index with prefix
                self.storage.redis.execute_command(
                    "FT.CREATE", self.index_name,
                    "ON", "HASH",
                    "PREFIX", "1", "doc:",
                    "SCHEMA", *schema
                )
                print(f"✅ Created vector index '{self.index_name}' using Redis official schema")
                
            except Exception as e:
                print(f"❌ Failed to create vector index: {e}")
                print("   Falling back to Python-based vector search")
                self.redis_stack_available = False
    
    def _generate_id(self) -> str:
        """Generate unique vector ID"""
        return str(uuid.uuid4())
    
    def _vector_to_bytes(self, vector: List[float]) -> bytes:
        """Convert vector to bytes using official Redis format"""
        return struct.pack(f'{len(vector)}f', *vector)
    
    def _bytes_to_vector(self, data: bytes) -> List[float]:
        """Convert bytes back to vector using official Redis format"""
        return list(struct.unpack(f'{len(data)//4}f', data))
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity for Python fallback"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def add_vector(self, data: List[float], metadata: Dict[str, Any] = None,
                   properties: Dict[str, Any] = None) -> Vector:
        """Add vector using official Redis vector database approach"""
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
        
        if self.redis_stack_available:
            return self._add_vector_redis_official(vector)
        else:
            return self._add_vector_python(vector)
    
    def _add_vector_redis_official(self, vector: Vector) -> Vector:
        """Add vector using official Redis Stack approach"""
        try:
            # Store using official Redis vector format
            # Key pattern: doc:{id} (matches index prefix)
            doc_key = f"doc:{vector.id}"
            
            # Prepare data in Redis format
            doc_data = {
                "vector": self._vector_to_bytes(vector.data),
                "type": vector.metadata.get("type", ""),
                "title": vector.properties.get("title", ""),
                "score": float(vector.properties.get("score", 0.0)),
                "metadata": json.dumps(vector.metadata),
                "properties": json.dumps(vector.properties),
                "dimension": len(vector.data)
            }
            
            # Store in Redis
            self.storage.redis.hset(doc_key, mapping=doc_data)
            return vector
            
        except Exception as e:
            print(f"Redis Stack add failed: {e}, falling back to Python")
            return self._add_vector_python(vector)
    
    def _add_vector_python(self, vector: Vector) -> Vector:
        """Python fallback storage"""
        vector_key = f"vectors:{vector.id}"
        self.storage.redis.hset(vector_key, mapping={
            "id": vector.id,
            "metadata": json.dumps(vector.metadata),
            "properties": json.dumps(vector.properties),
            "dimension": len(vector.data)
        })
        
        data_key = f"vector_data:{vector.id}"
        self.storage.redis.set(data_key, json.dumps(vector.data))
        self.storage.redis.sadd("vector_index", vector.id)
        
        # Index for filtering
        for prop_name, prop_value in vector.properties.items():
            self.storage.redis.sadd(f"vector_idx:{prop_name}:{prop_value}", vector.id)
        for meta_name, meta_value in vector.metadata.items():
            self.storage.redis.sadd(f"vector_idx:{meta_name}:{meta_value}", vector.id)
            
        return vector
    
    def get_vector(self, vector_id: str) -> Optional[Vector]:
        """Get vector by ID"""
        if self.redis_stack_available:
            vector = self._get_vector_redis_official(vector_id)
            if vector:
                return vector
        
        return self._get_vector_python(vector_id)
    
    def _get_vector_redis_official(self, vector_id: str) -> Optional[Vector]:
        """Get vector using official Redis format"""
        try:
            doc_key = f"doc:{vector_id}"
            doc_data = self.storage.redis.hgetall(doc_key)
            
            if not doc_data:
                return None
            
            # Reconstruct vector using official format
            vector_bytes = doc_data[b'vector']
            vector_data = self._bytes_to_vector(vector_bytes)
            
            return Vector(
                id=vector_id,
                data=vector_data,
                metadata=json.loads(doc_data[b'metadata'].decode()),
                properties=json.loads(doc_data[b'properties'].decode())
            )
            
        except Exception as e:
            print(f"Redis official get failed: {e}")
            return None
    
    def _get_vector_python(self, vector_id: str) -> Optional[Vector]:
        """Python fallback get"""
        vector_key = f"vectors:{vector_id}"
        vector_info = self.storage.redis.hgetall(vector_key)
        
        if not vector_info:
            return None
            
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
        """Search vectors using official Redis approach"""
        if filters is None:
            filters = {}
        
        if self.redis_stack_available:
            try:
                return self._search_vectors_redis_official(query_vector, k, filters, metric)
            except Exception as e:
                print(f"Redis search failed: {e}, falling back to Python")
                return self._search_vectors_python(query_vector, k, filters, metric)
        else:
            return self._search_vectors_python(query_vector, k, filters, metric)
    
    def _search_vectors_redis_official(self, query_vector: List[float], k: int,
                                     filters: Dict[str, Any], metric: str) -> List[Tuple[Vector, float]]:
        """Official Redis vector search using FT.SEARCH"""
        try:
            # Convert query vector to Redis format
            query_blob = self._vector_to_bytes(query_vector)
            
            # Build query using official Redis syntax
            # Based on Redis docs: https://redis.io/docs/latest/develop/get-started/vector-database/
            filter_parts = []
            for key, value in filters.items():
                if key == "type":
                    filter_parts.append(f"@type:{{{value}}}")
                elif key == "score":
                    try:
                        score_val = float(value)
                        filter_parts.append(f"@score:[{score_val} +inf]")
                    except:
                        pass
            
            base_query = " ".join(filter_parts) if filter_parts else "*"
            
            # Official Redis vector search query
            vector_query = f"({base_query})=>[KNN {k} @vector $query_vector AS vector_score]"
            
            # Execute search using official Redis command
            result = self.storage.redis.execute_command(
                "FT.SEARCH", self.index_name,
                vector_query,
                "PARAMS", "2", "query_vector", query_blob,
                "SORTBY", "vector_score",
                "RETURN", "3", "vector_score", "metadata", "properties",
                "DIALECT", "2"
            )
            
            # Parse results (Redis returns: [count, doc1_id, doc1_fields, doc2_id, doc2_fields, ...])
            vector_results = []
            if len(result) > 1:
                count = result[0]
                for i in range(1, len(result), 2):
                    if i + 1 < len(result):
                        doc_id = result[i].decode() if isinstance(result[i], bytes) else result[i]
                        vector_id = doc_id.split(":")[-1]  # Remove doc: prefix
                        
                        # Get the vector
                        vector = self.get_vector(vector_id)
                        if vector:
                            # Extract score from fields
                            fields = result[i + 1]
                            score = 0.0
                            try:
                                if len(fields) >= 2:
                                    score = float(fields[1])  # vector_score is usually at index 1
                                    # Convert distance to similarity for cosine
                                    if metric == "cosine":
                                        score = max(0, 1 - score)
                            except:
                                score = 0.0
                            
                            vector_results.append((vector, score))
            
            return vector_results
            
        except Exception as e:
            print(f"Redis official search error: {e}")
            raise e
    
    def _search_vectors_python(self, query_vector: List[float], k: int,
                             filters: Dict[str, Any], metric: str) -> List[Tuple[Vector, float]]:
        """Python fallback search"""
        candidate_ids = set(self.storage.redis.smembers("vector_index"))
        
        # Apply filters
        for prop_name, prop_value in filters.items():
            filter_key = f"vector_idx:{prop_name}:{prop_value}"
            filtered_ids = self.storage.redis.smembers(filter_key)
            candidate_ids = candidate_ids.intersection(filtered_ids)
        
        results = []
        for vector_id_bytes in candidate_ids:
            vector_id = vector_id_bytes.decode() if isinstance(vector_id_bytes, bytes) else vector_id_bytes
            vector = self.get_vector(vector_id)
            if vector:
                similarity = self._cosine_similarity(query_vector, vector.data)
                results.append((vector, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from index"""
        success1 = False
        success2 = False
        
        if self.redis_stack_available:
            try:
                doc_key = f"doc:{vector_id}"
                success1 = bool(self.storage.redis.delete(doc_key))
            except:
                pass
        
        # Python cleanup
        vector = self.get_vector(vector_id)
        if vector:
            for prop_name, prop_value in vector.properties.items():
                self.storage.redis.srem(f"vector_idx:{prop_name}:{prop_value}", vector_id)
            for meta_name, meta_value in vector.metadata.items():
                self.storage.redis.srem(f"vector_idx:{meta_name}:{meta_value}", vector_id)
            
            self.storage.redis.srem("vector_index", vector_id)
            self.storage.redis.delete(f"vectors:{vector_id}")
            self.storage.redis.delete(f"vector_data:{vector_id}")
            success2 = True
        
        return success1 or success2
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get backend information"""
        return {
            "redis_stack_available": self.redis_stack_available,
            "backend": "Redis Stack (Official)" if self.redis_stack_available else "Python",
            "index_name": self.index_name if self.redis_stack_available else None,
            "approach": "Redis Official Vector Database" if self.redis_stack_available else "Python Fallback"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        stats = {}
        
        if self.redis_stack_available:
            try:
                # Get index info using official command
                index_info = self.storage.redis.execute_command("FT.INFO", self.index_name)
                stats["redis_official_vectors"] = len(self.storage.redis.keys("doc:*"))
                stats["index_info"] = "Redis Stack Official Index"
            except:
                stats["redis_official_vectors"] = 0
        
        stats["python_vectors"] = len(self.storage.redis.keys("vectors:*"))
        stats["vector_index_size"] = self.storage.redis.scard("vector_index") or 0
        
        return stats