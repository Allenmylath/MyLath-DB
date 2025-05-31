#!/usr/bin/env python3
# mylath/mylath/vector/vector_core.py
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from ..storage.redis_storage import RedisStorage, Vector # Assuming Vector is defined here or in a common types file
import heapq # Not used in current RediSearch/fallback logic, but kept if planned for HNSW Python impl.
import random # Not used in current RediSearch/fallback logic
import math

# Redis Search specific imports
from redis.commands.search.field import VectorField, TagField, TextField, NumericField # Added NumericField for completeness
from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import redis.exceptions
import uuid # For _generate_id

class VectorCore:
    """Vector operations and index management for similarity search, with Redis Stack support."""

    def __init__(self, storage: RedisStorage,
                 m: int = 16, ef_construction: int = 128, ef_search: int = 768, # HNSW params for Redis
                 default_vector_dimension: int = 128): # Default dimension
        self.storage = storage
        self.m = m  # HNSW M parameter (max outgoing connections)
        self.ef_construction = ef_construction  # HNSW efConstruction parameter
        self.ef_search = ef_search # HNSW efSearch parameter (used at query time if not overridden)
        # self.m_l = 1.0 / math.log(2.0 * m) # Level generation factor, not directly used by Redis Search HNSW
        
        self.index_name = "mylath_vectors"
        self.vector_dimension = default_vector_dimension # Set from param, can be updated if first vector has different dim
        self.redis_stack_enabled = False
        self.index_initialized_properly = False # Flag to track successful index setup

        # Attempt to initialize Redis Search index on startup
        self._initialize_redis_search_index()

    def _initialize_redis_search_index(self):
        """Attempts to initialize the Redis Search index.
        Checks for the RediSearch module and creates the vector index if it doesn't exist.
        """
        redisearch_loaded = False
        try:
            # Check if RediSearch module is loaded in Redis
            # redis-py .info('modules') typically returns a list of dicts with string keys
            modules_info_list = self.storage.redis.info('modules')
            
            if isinstance(modules_info_list, list):
                for module_details in modules_info_list:
                    if isinstance(module_details, dict) and module_details.get('name') == 'search':
                        redisearch_loaded = True
                        print(f"✅ Redis Search module (name: '{module_details.get('name')}', version: {module_details.get('ver')}) detected.")
                        break
            else:
                print(f"⚠️ Warning: Unexpected format for Redis modules info: {type(modules_info_list)}. Expected a list.")

            if not redisearch_loaded:
                print("ℹ️ Redis Search module not detected in Redis. Falling back to Python vector search.")
                self.redis_stack_enabled = False
                return # Exit early, fallback will be used

        except redis.exceptions.RedisError as e_info_mod:
            print(f"⚠️ Warning: Could not get Redis module info due to RedisError: {e_info_mod}. Falling back.")
            self.redis_stack_enabled = False
            return
        except Exception as e_info_mod_generic: # Catch other potential issues during module check
            print(f"⚠️ Warning: An unexpected error occurred while checking Redis modules: {e_info_mod_generic}. Falling back.")
            self.redis_stack_enabled = False
            return

        # If Redis Search module is loaded, proceed to define and create the index
        self.redis_stack_enabled = True # Tentatively enable, confirm after index setup

        try:
            # Define the schema for the RediSearch index.
            # Fields are named to allow easy mapping from metadata/properties.
            schema = (
                TagField("metadata_type", as_name="metadata_type"),
                TextField("metadata_source", as_name="metadata_source"),
                TextField("properties_title", as_name="properties_title"),
                # Consider NumericField for score if range queries are needed, TagField for exact match
                TagField("properties_score_tag", as_name="properties_score_tag"), # For score as a category/tag
                # NumericField("properties_score_num", as_name="properties_score_num"), # Example if score is numeric
                VectorField("vector_data", "HNSW", {
                    "TYPE": "FLOAT32",
                    "DIM": self.vector_dimension, # Dimension must match the vectors
                    "DISTANCE_METRIC": "COSINE",
                    "M": self.m,
                    "EF_CONSTRUCTION": self.ef_construction
                    # EF_RUNTIME can be set here or per query
                }, as_name="vector_data")
            )

            # Index definition: targeting HASHes prefixed with 'vec:'
            idx_definition = IndexDefinition(
                prefix=[f"vec:"], # Note: ensure this prefix matches keys used in add_vector
                index_type=IndexType.HASH # We are indexing Redis Hashes
            )

            # Check if the index already exists
            try:
                self.storage.redis.ft(self.index_name).info()
                print(f"✅ Redis Search index '{self.index_name}' already exists and is accessible.")
                self.index_initialized_properly = True
            except redis.exceptions.ResponseError as e:
                if "Unknown Index Name" in str(e) or "unknown command `FT.INFO`" in str(e).lower(): # Handle varying error messages
                    # Index does not exist, create it
                    print(f"ℹ️ Redis Search index '{self.index_name}' not found. Attempting to create...")
                    self.storage.redis.ft(self.index_name).create_index(
                        fields=schema,
                        definition=idx_definition
                    )
                    print(f"✅ Redis Search index '{self.index_name}' created successfully.")
                    self.index_initialized_properly = True
                else:
                    # Other Redis response errors during index check/creation
                    print(f"❌ Error checking/creating Redis Search index '{self.index_name}': {e}")
                    self.redis_stack_enabled = False # Disable if index setup failed
                    self.index_initialized_properly = False
        
        except redis.exceptions.RedisError as e_idx_setup: # Catch Redis errors during schema/index definition
            print(f"❌ RedisError during Redis Search index setup: {e_idx_setup}. Falling back.")
            self.redis_stack_enabled = False
            self.index_initialized_properly = False
        except Exception as e_generic_idx_setup: # Catch any other unexpected errors
            print(f"❌ An unexpected error occurred during Redis Search index setup: {e_generic_idx_setup}. Falling back.")
            self.redis_stack_enabled = False
            self.index_initialized_properly = False
        
        # Final check
        if not self.index_initialized_properly:
             print("ℹ️ MyLath will use Python-based vector search due to issues with Redis Search index setup.")


    def _generate_id(self) -> str:
        """Generate a unique vector ID using UUID."""
        return str(uuid.uuid4())

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors (Python fallback)."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        return 0.0 if norm1 == 0 or norm2 == 0 else dot_product / (norm1 * norm2)

    def add_vector(self, data: List[float], metadata: Dict[str, Any] = None,
                   properties: Dict[str, Any] = None) -> Vector:
        """Add a vector. Uses Redis Search if enabled, otherwise Python fallback."""
        if metadata is None: metadata = {}
        if properties is None: properties = {}

        current_dim = len(data)
        if self.vector_dimension != current_dim:
            if not self.index_initialized_properly : # Only allow re-init if not successfully initialized
                print(f"ℹ️ Updating vector dimension from {self.vector_dimension} to {current_dim} and re-attempting index initialization.")
                self.vector_dimension = current_dim
                self._initialize_redis_search_index() # Re-attempt with new dimension
            else: # Index already exists with a different dimension
                # This is a critical mismatch. For now, log and force fallback for this vector.
                # A more robust solution might involve multiple indexes or schema evolution (complex).
                print(f"❌ CRITICAL: Dimension mismatch! Vector dim {current_dim} vs Index dim {self.vector_dimension}. "
                      f"Cannot add to Redis Search index. Consider re-indexing or using a matching dimension.")
                # Forcing fallback for this specific add, or raise error:
                # For simplicity, we'll let it try and potentially fail if redis_stack_enabled is still true,
                # or ensure it falls back if self.redis_stack_enabled is (conditionally) false.
                # Better: handle this by not adding to a mismatched index.
                # If we force a fallback here, it would be:
                # temp_force_fallback = True
                # However, current logic relies on self.redis_stack_enabled & self.index_initialized_properly

        vector_obj = Vector(
            id=self._generate_id(),
            data=data,
            metadata=metadata,
            properties=properties
        )

        if self.redis_stack_enabled and self.index_initialized_properly:
            try:
                vector_key = f"vec:{vector_obj.id}" # Matches prefix in IndexDefinition
                vector_bytes = np.array(vector_obj.data).astype(np.float32).tobytes()

                hash_data = {
                    "id": vector_obj.id, # Store original ID for retrieval
                    "vector_data": vector_bytes,
                    # "dimension": len(vector_obj.data) # Storing dimension in hash is optional if fixed by schema
                }
                for k, v in metadata.items():
                    hash_data[f"metadata_{k}"] = str(v) # Ensure keys match schema
                for k, v in properties.items():
                    # Match schema: properties_score_tag example
                    if k == "score": # Specific handling for 'score' to match schema
                         hash_data[f"properties_score_tag"] = str(v)
                    # else: # Generic handling for other properties if schema supports them
                    hash_data[f"properties_{k}"] = str(v)


                self.storage.redis.hset(vector_key, mapping=hash_data)
                # RediSearch automatically indexes the hash based on schema and prefix.
            except Exception as e_add_rs:
                print(f"❌ Error adding vector {vector_obj.id} to Redis Search: {e_add_rs}. Data not added to Redis Search.")
                # Optionally, could attempt Python fallback storage here for this specific vector
        else:
            # Python fallback storage
            vector_key = f"vectors:{vector_obj.id}"
            self.storage.redis.hset(vector_key, mapping={
                "id": vector_obj.id,
                "metadata": json.dumps(vector_obj.metadata),
                "properties": json.dumps(vector_obj.properties),
                "dimension": len(vector_obj.data)
            })
            data_key = f"vector_data:{vector_obj.id}"
            self.storage.redis.set(data_key, json.dumps(vector_obj.data))
            self.storage.redis.sadd("vector_index", vector_obj.id) # Index of all vector IDs for fallback
            for prop_name, prop_value in {**metadata, **properties}.items(): # Index all metadata/properties
                self.storage.redis.sadd(f"vector_idx:{prop_name}:{prop_value}", vector_obj.id)
        return vector_obj

    def get_vector(self, vector_id: str) -> Optional[Vector]:
        """Retrieve a vector by its ID from appropriate storage."""
        # Try fetching from Redis Search structure first if it was supposed to be used
        # Note: This doesn't know if a *specific* vector used fallback, so it tries main path first
        if self.redis_stack_enabled and self.index_initialized_properly: # Check if RS was intended
            vector_key_rs = f"vec:{vector_id}"
            vector_info_rs = self.storage.redis.hgetall(vector_key_rs)
            if vector_info_rs: # Found in Redis Search hash structure
                try:
                    # Decode vector data bytes back to a list of floats
                    vector_data_bytes = vector_info_rs.get(b'vector_data')
                    if not vector_data_bytes: return None # Should not happen if correctly stored
                    data = np.frombuffer(vector_data_bytes, dtype=np.float32).tolist()

                    metadata_reconstructed = {}
                    properties_reconstructed = {}
                    for k_bytes, v_bytes in vector_info_rs.items():
                        key_str = k_bytes.decode()
                        value_str = v_bytes.decode()
                        if key_str.startswith("metadata_"):
                            metadata_reconstructed[key_str[len("metadata_"):]] = value_str
                        elif key_str.startswith("properties_"):
                            prop_original_key = key_str[len("properties_"):]
                            if prop_original_key == "score_tag": # map back from schema name
                                properties_reconstructed['score'] = value_str 
                            else:
                                properties_reconstructed[prop_original_key] = value_str
                    
                    return Vector(
                        id=vector_info_rs.get(b'id', b'N/A').decode(), # Use .get for safety
                        data=data,
                        metadata=metadata_reconstructed,
                        properties=properties_reconstructed
                    )
                except Exception as e_get_rs:
                    print(f"Error reconstructing vector from Redis Search hash for ID {vector_id}: {e_get_rs}")
                    # Fall through to try Python fallback keys if reconstruction fails

        # Try Python fallback storage if not found or RS not enabled/initialized
        vector_key_py = f"vectors:{vector_id}"
        vector_info_py = self.storage.redis.hgetall(vector_key_py)
        if not vector_info_py:
            return None # Not found in either storage

        try:
            data_key_py = f"vector_data:{vector_id}"
            data_json = self.storage.redis.get(data_key_py)
            if not data_json: return None # Data component missing

            return Vector(
                id=vector_info_py[b'id'].decode(),
                data=json.loads(data_json.decode()),
                metadata=json.loads(vector_info_py[b'metadata'].decode()),
                properties=json.loads(vector_info_py[b'properties'].decode())
            )
        except Exception as e_get_py:
            print(f"Error reconstructing vector from Python fallback storage for ID {vector_id}: {e_get_py}")
            return None


    def search_vectors(self, query_vector: List[float], k: int = 10,
                       filters: Dict[str, Any] = None,
                       metric: str = "cosine") -> List[Tuple[Vector, float]]:
        """Search for similar vectors. Uses RediSearch or Python fallback."""
        if filters is None: filters = {}

        if self.redis_stack_enabled and self.index_initialized_properly:
            try:
                query_vector_bytes = np.array(query_vector).astype(np.float32).tobytes()
                
                # Build filter string for RediSearch query
                # Example: @metadata_type:{document} @properties_score_tag:{0.5}
                filter_expressions = []
                for key, value in filters.items():
                    # This needs to map to actual indexed field names
                    if key == "type": # maps to metadata_type
                        filter_expressions.append(f"@metadata_type:{{{self._escape_tag_value(str(value))}}}")
                    elif key == "score": # maps to properties_score_tag
                         filter_expressions.append(f"@properties_score_tag:{{{self._escape_tag_value(str(value))}}}")
                    # Add more elif for other filterable fields based on schema
                
                filter_str = " ".join(filter_expressions) if filter_expressions else "*" # Use * for no filters

                # KNN query part
                # Note: EF_RUNTIME can be added to params for query-time HNSW tuning
                knn_query_part = f"KNN {k} @vector_data $query_vector"
                
                # Combine filters and KNN
                # If filter_str is "*", RediSearch treats it as "match all" before KNN
                # If specific filters, it's "filter AND KNN" effectively
                final_query_str = f"({filter_str})=>[{knn_query_part}]" if filter_str != "*" else knn_query_part

                # Define query parameters
                query_params = {"query_vector": query_vector_bytes}
                # if self.ef_search: # Optionally add ef_search at query time
                #     query_params["EF_RUNTIME"] = self.ef_search


                # Construct RediSearch Query object
                # Request all fields needed to reconstruct the Vector object, plus score
                # Must match fields stored in the hash and defined in schema for retrieval
                fields_to_return = [
                    "id", "vector_data", "__vector_score__", # Core fields
                    "metadata_type", "metadata_source",     # Metadata fields from schema
                    "properties_title", "properties_score_tag" # Properties fields from schema
                ]
                
                q = Query(final_query_str).return_fields(*fields_to_return).dialect(2) # Dialect 2 for KNN
                q.sort_by("__vector_score__") # Ascending for cosine distance (0=identical), or handle similarity
                q.paging(0, k) # Get top K results

                search_results_raw = self.storage.redis.ft(self.index_name).search(q, query_params=query_params)
                
                results_tuples = []
                for doc in search_results_raw.docs:
                    # RediSearch returns 1-score for COSINE distance, convert back to similarity if needed
                    # For COSINE distance, lower is better. If __vector_score__ is distance:
                    similarity_score = 1 - float(doc.__vector_score__) # If score is distance (0=best)
                    # If score is already similarity (1=best), then: similarity_score = float(doc.__vector_score__)

                    # Reconstruct Vector object
                    # This part is simplified; a more robust reconstruction is in get_vector
                    # For search, often ID and score are enough, full object is for convenience
                    # Assuming 'id' field in hash is the original vector ID
                    vector_id_from_doc = getattr(doc, 'id', None)
                    if not vector_id_from_doc: continue # Skip if no ID

                    # Fetch full vector object using get_vector for consistency in data structure
                    # This might add slight overhead but ensures correct object reconstruction
                    vec_obj = self.get_vector(vector_id_from_doc)
                    if vec_obj:
                         results_tuples.append((vec_obj, similarity_score))
                    else:
                        print(f"Warning: Could not retrieve full vector for ID {vector_id_from_doc} during search.")

                return results_tuples

            except redis.exceptions.ResponseError as e_search_rs:
                print(f"❌ RediSearch query failed: {e_search_rs}. Falling back to Python search for this query.")
                # Fall through to Python fallback for this specific query, don't disable redis_stack_enabled globally here
                # unless the error is persistent/config-related.
            except Exception as e_search_generic:
                print(f"❌ Unexpected error during RediSearch: {e_search_generic}. Falling back.")


        # Python fallback search (if RS disabled, or failed for this query)
        print("ℹ️ Performing search using Python fallback.")
        candidate_ids_bytes = self.storage.redis.smembers("vector_index")
        candidate_ids = {bid.decode() for bid in candidate_ids_bytes}

        # Apply filters for Python fallback
        if filters:
            for prop_name, prop_value in filters.items():
                filter_key = f"vector_idx:{prop_name}:{prop_value}"
                ids_for_filter_bytes = self.storage.redis.smembers(filter_key)
                ids_for_filter = {bid.decode() for bid in ids_for_filter_bytes}
                candidate_ids.intersection_update(ids_for_filter)
        
        # Calculate similarities for remaining candidates
        top_k_heap = [] # Min-heap to store (similarity, vector_id)
        for vector_id_str in candidate_ids:
            vector = self.get_vector(vector_id_str) # Use the consistent get_vector
            if vector:
                # Assuming metric is cosine for fallback, as Redis Search uses COSINE
                similarity = self._cosine_similarity(query_vector, vector.data)
                
                if len(top_k_heap) < k:
                    heapq.heappush(top_k_heap, (similarity, vector))
                elif similarity > top_k_heap[0][0]: # If current is better than smallest in heap
                    heapq.heapreplace(top_k_heap, (similarity, vector))
        
        # Sort by similarity descending and return
        sorted_results = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
        return [(vec, sim) for sim, vec in sorted_results] # Return (Vector, score)

    def _escape_tag_value(self, value: str) -> str:
        """Escapes characters for RediSearch TAG field values if necessary."""
        # Common characters to escape in TAGs: punctuation like ,, ., <, >, {, }, [, ], ", ', :, ;, !, @, #, $, %, ^, &, *, (, ), -, +, =, ~
        # Whitespace also needs care; often better to replace with a placeholder or ensure tags are single words.
        # For simplicity, this example might not be exhaustive.
        # RediSearch TAGs are typically split by separators. If value contains separators, it's treated as multiple tags.
        # If you want to treat the whole value as one tag, replace problematic characters.
        chars_to_escape = r"[,.<>{}\[\]\"':;!@#$%^&*()\-+=~ ]" # Added space
        import re
        return re.sub(chars_to_escape, r"\\\g", value)