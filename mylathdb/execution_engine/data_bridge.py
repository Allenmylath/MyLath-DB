# mylathdb/execution_engine/data_bridge.py

"""
MyLathDB Data Bridge
Handles data conversion between Redis entities and GraphBLAS matrices
Based on FalkorDB's data synchronization patterns
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

from .exceptions import MyLathDBDataError
from .utils import extract_node_id, safe_get_nested

logger = logging.getLogger(__name__)

@dataclass
class EntityMapping:
    """Mapping between entity IDs and matrix indices"""
    entity_to_index: Dict[str, int] = field(default_factory=dict)
    index_to_entity: Dict[int, str] = field(default_factory=dict)
    next_index: int = 0
    
    def get_or_create_index(self, entity_id: str) -> int:
        """Get existing index or create new one for entity"""
        if entity_id not in self.entity_to_index:
            index = self.next_index
            self.entity_to_index[entity_id] = index
            self.index_to_entity[index] = entity_id
            self.next_index += 1
            return index
        return self.entity_to_index[entity_id]
    
    def get_entity_id(self, index: int) -> Optional[str]:
        """Get entity ID for matrix index"""
        return self.index_to_entity.get(index)

class DataBridge:
    """
    Data bridge between Redis and GraphBLAS
    
    Handles:
    - Converting Redis node/edge data to GraphBLAS vectors/matrices
    - Converting GraphBLAS results back to entity data
    - Maintaining entity ID to matrix index mappings
    - Synchronizing changes between systems
    """
    
    def __init__(self, redis_executor, graphblas_executor):
        """Initialize data bridge with executor references"""
        self.redis_executor = redis_executor
        self.graphblas_executor = graphblas_executor
        
        # Entity mappings
        self.node_mapping = EntityMapping()
        self.edge_mapping = EntityMapping()
        
        # Relationship type mappings
        self.relationship_types: Dict[str, int] = {}
        self.label_types: Dict[str, int] = {}
        
        # Synchronization tracking
        self.pending_node_updates: Set[str] = set()
        self.pending_edge_updates: Set[str] = set()
        self.last_sync_timestamp = 0
    
    def sync_redis_to_graphblas(self):
        """
        Synchronize Redis entity data to GraphBLAS matrices
        Based on FalkorDB's Delta Matrix sync patterns
        """
        logger.info("Synchronizing Redis data to GraphBLAS matrices")
        
        try:
            # Sync nodes to vectors/matrices
            self._sync_nodes_to_matrices()
            
            # Sync edges to adjacency matrices
            self._sync_edges_to_matrices()
            
            # Clear pending updates
            self.pending_node_updates.clear()
            self.pending_edge_updates.clear()
            self.last_sync_timestamp = time.time()
            
            logger.info("Redis to GraphBLAS synchronization completed")
            
        except Exception as e:
            raise MyLathDBDataError(f"Redis to GraphBLAS sync failed: {e}")
    
    def _sync_nodes_to_matrices(self):
        """Sync Redis nodes to GraphBLAS label matrices"""
        if not self.redis_executor.redis:
            return
        
        # Scan all node keys
        node_keys = []
        for key in self.redis_executor.redis.scan_iter(match="node:*"):
            if key != self.redis_executor.storage.NEXT_NODE_ID_KEY:
                node_keys.append(key)
        
        logger.debug(f"Syncing {len(node_keys)} nodes to GraphBLAS")
        
        # Process each node
        for node_key in node_keys:
            node_id = node_key.split(':')[1]
            
            # Get node data
            node_data = self.redis_executor._get_node_data(node_id)
            if not node_data:
                continue
            
            # Get matrix index for this node
            matrix_index = self.node_mapping.get_or_create_index(node_id)
            
            # Update label matrices
            labels = node_data.get('_labels', [])
            for label in labels:
                self._update_label_matrix(label, matrix_index, True)
    
    def _sync_edges_to_matrices(self):
        """Sync Redis edges to GraphBLAS adjacency matrices"""
        if not self.redis_executor.redis:
            return
        
        # Get all relationship types
        rel_types = set()
        for key in self.redis_executor.redis.scan_iter(match="rel:*"):
            rel_type = key.split(':')[1]
            rel_types.add(rel_type)
        
        logger.debug(f"Syncing edges for {len(rel_types)} relationship types")
        
        # Process each relationship type
        for rel_type in rel_types:
            self._sync_relationship_edges(rel_type)
    
    def _sync_relationship_edges(self, rel_type: str):
        """Sync edges of a specific relationship type"""
        rel_key = self.redis_executor.storage.RELATIONSHIP_EDGES_KEY.format(rel_type=rel_type)
        edge_ids = self.redis_executor.redis.smembers(rel_key)
        
        # Get or create relation matrix
        relation_matrix = self.graphblas_executor._get_relation_matrix([rel_type], "outgoing")
        
        # Process each edge
        for edge_id in edge_ids:
            try:
                # Get edge endpoints
                endpoints_key = self.redis_executor.storage.EDGE_ENDPOINTS_KEY.format(edge_id=edge_id)
                endpoints_value = self.redis_executor.redis.get(endpoints_key)
                
                if endpoints_value:
                    src_id, dest_id, edge_rel_type = endpoints_value.split('|')
                    
                    if edge_rel_type == rel_type:
                        # Get matrix indices
                        src_index = self.node_mapping.get_or_create_index(src_id)
                        dest_index = self.node_mapping.get_or_create_index(dest_id)
                        
                        # Set matrix entry
                        if (src_index < relation_matrix.nrows and 
                            dest_index < relation_matrix.ncols):
                            relation_matrix[src_index, dest_index] = True
                            
                            # Also update main adjacency matrix
                            adj_matrix = self.graphblas_executor.graph.adjacency_matrix
                            if (src_index < adj_matrix.nrows and 
                                dest_index < adj_matrix.ncols):
                                adj_matrix[src_index, dest_index] = True
                
            except Exception as e:
                logger.warning(f"Failed to sync edge {edge_id}: {e}")
    
    def _update_label_matrix(self, label: str, node_index: int, value: bool):
        """Update label matrix for a specific label and node"""
        
        # Get or create label matrix
        if label not in self.graphblas_executor.graph.label_matrices:
            try:
                import graphblas as gb
                n = self.graphblas_executor.graph.node_capacity
                self.graphblas_executor.graph.label_matrices[label] = gb.Matrix.new(
                    gb.dtypes.BOOL, nrows=n, ncols=n
                )
            except Exception as e:
                logger.error(f"Failed to create label matrix for {label}: {e}")
                return
        
        label_matrix = self.graphblas_executor.graph.label_matrices[label]
        
        # Update matrix (diagonal entry for node labels)
        try:
            if node_index < label_matrix.nrows:
                label_matrix[node_index, node_index] = value
        except Exception as e:
            logger.error(f"Failed to update label matrix for {label}[{node_index}]: {e}")
    
    def convert_redis_results_to_vectors(self, redis_results: List[Dict[str, Any]], 
                                       variable_name: str):
        """Convert Redis scan results to GraphBLAS vectors"""
        try:
            import graphblas as gb
            
            n = self.graphblas_executor.graph.node_capacity
            vector = gb.Vector.new(gb.dtypes.BOOL, size=n)
            
            # Set vector entries for each result
            for result in redis_results:
                node_data = result.get(variable_name)
                if node_data:
                    node_id = extract_node_id(node_data)
                    if node_id:
                        matrix_index = self.node_mapping.get_or_create_index(node_id)
                        if matrix_index < vector.size:
                            vector[matrix_index] = True
            
            return vector
            
        except Exception as e:
            logger.error(f"Failed to convert Redis results to vector: {e}")
            return None
    
    def convert_vector_to_redis_results(self, vector, variable_name: str) -> List[Dict[str, Any]]:
        """Convert GraphBLAS vector back to Redis entity results"""
        results = []
        
        try:
            # Get non-zero indices from vector
            indices, values = vector.to_coo()
            
            # Convert each index back to entity data
            for index, value in zip(indices, values):
                if value:  # Non-zero entry
                    entity_id = self.node_mapping.get_entity_id(index)
                    if entity_id:
                        # Fetch full entity data from Redis
                        entity_data = self.redis_executor._get_node_data(entity_id)
                        if entity_data:
                            results.append({variable_name: entity_data})
            
        except Exception as e:
            logger.error(f"Failed to convert vector to Redis results: {e}")
        
        return results
    
    def get_node_index(self, node_id: str) -> Optional[int]:
        """Get matrix index for node ID"""
        return self.node_mapping.entity_to_index.get(node_id)
    
    def get_node_id(self, index: int) -> Optional[str]:
        """Get node ID for matrix index"""
        return self.node_mapping.get_entity_id(index)
    
    def mark_node_updated(self, node_id: str):
        """Mark node as needing sync"""
        self.pending_node_updates.add(node_id)
    
    def mark_edge_updated(self, edge_id: str):
        """Mark edge as needing sync"""
        self.pending_edge_updates.add(edge_id)
    
    def has_pending_updates(self) -> bool:
        """Check if there are pending updates to sync"""
        return len(self.pending_node_updates) > 0 or len(self.pending_edge_updates) > 0


class EntityManager:
    """
    Entity manager for coordinating entity operations
    Based on FalkorDB's entity management patterns
    """
    
    def __init__(self, data_bridge: DataBridge):
        """Initialize entity manager"""
        self.data_bridge = data_bridge
        self.redis_executor = data_bridge.redis_executor
        self.graphblas_executor = data_bridge.graphblas_executor
    
    def create_node(self, node_data: Dict[str, Any]) -> str:
        """Create a new node in both Redis and GraphBLAS"""
        
        # Generate node ID if not provided
        node_id = node_data.get('id') or node_data.get('_id')
        if not node_id:
            node_id = str(self.redis_executor.redis.incr(
                self.redis_executor.storage.NEXT_NODE_ID_KEY
            ))
        
        # Store in Redis
        self._create_node_in_redis(node_id, node_data)
        
        # Mark for GraphBLAS sync
        self.data_bridge.mark_node_updated(node_id)
        
        return node_id
    
    def create_edge(self, src_id: str, dest_id: str, rel_type: str, 
                   edge_data: Dict[str, Any] = None) -> str:
        """Create a new edge in both Redis and GraphBLAS"""
        
        # Generate edge ID
        edge_id = str(self.redis_executor.redis.incr(
            self.redis_executor.storage.NEXT_EDGE_ID_KEY
        ))
        
        # Store in Redis
        self._create_edge_in_redis(edge_id, src_id, dest_id, rel_type, edge_data or {})
        
        # Mark for GraphBLAS sync
        self.data_bridge.mark_edge_updated(edge_id)
        
        return edge_id
    
    def _create_node_in_redis(self, node_id: str, node_data: Dict[str, Any]):
        """Create node in Redis storage"""
        
        # Store node properties
        node_key = self.redis_executor.storage.NODE_KEY_PATTERN.format(node_id=node_id)
        properties = {k: v for k, v in node_data.items() 
                     if not k.startswith('_') and k != 'id'}
        
        if properties:
            self.redis_executor.redis.hset(node_key, mapping=properties)
        
        # Store node labels
        labels = node_data.get('_labels', [])
        if labels:
            labels_key = self.redis_executor.storage.NODE_LABELS_KEY.format(node_id=node_id)
            self.redis_executor.redis.sadd(labels_key, *labels)
            
            # Add to label indexes
            for label in labels:
                label_key = self.redis_executor.storage.LABEL_NODES_KEY.format(label=label)
                self.redis_executor.redis.sadd(label_key, node_id)
        
        # Create property indexes
        for prop_key, prop_value in properties.items():
            prop_index_key = self.redis_executor.storage.PROPERTY_INDEX_KEY.format(
                property=prop_key, value=prop_value
            )
            self.redis_executor.redis.sadd(prop_index_key, node_id)
    
    def _create_edge_in_redis(self, edge_id: str, src_id: str, dest_id: str, 
                             rel_type: str, edge_data: Dict[str, Any]):
        """Create edge in Redis storage"""
        
        # Store edge properties
        if edge_data:
            edge_key = self.redis_executor.storage.EDGE_KEY_PATTERN.format(edge_id=edge_id)
            self.redis_executor.redis.hset(edge_key, mapping=edge_data)
        
        # Store edge endpoints
        endpoints_key = self.redis_executor.storage.EDGE_ENDPOINTS_KEY.format(edge_id=edge_id)
        endpoints_value = f"{src_id}|{dest_id}|{rel_type}"
        self.redis_executor.redis.set(endpoints_key, endpoints_value)
        
        # Create relationship indexes
        out_key = self.redis_executor.storage.OUTGOING_EDGES_KEY.format(
            node_id=src_id, rel_type=rel_type
        )
        self.redis_executor.redis.sadd(out_key, edge_id)
        
        in_key = self.redis_executor.storage.INCOMING_EDGES_KEY.format(
            node_id=dest_id, rel_type=rel_type
        )
        self.redis_executor.redis.sadd(in_key, edge_id)
        
        rel_key = self.redis_executor.storage.RELATIONSHIP_EDGES_KEY.format(rel_type=rel_type)
        self.redis_executor.redis.sadd(rel_key, edge_id)
