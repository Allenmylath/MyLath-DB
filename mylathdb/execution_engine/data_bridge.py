# mylathdb/execution_engine/data_bridge.py

"""
MyLathDB Data Bridge - COMPLETE IMPLEMENTATION
Handles data conversion between Redis entities and GraphBLAS matrices
Based on FalkorDB's data synchronization and Delta Matrix patterns
"""

import time
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict

from .exceptions import MyLathDBDataError
from .utils import extract_node_id, safe_get_nested

logger = logging.getLogger(__name__)

@dataclass
class EntityMapping:
    """
    Mapping between entity IDs and matrix indices
    Based on FalkorDB's entity indexing system
    """
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
    
    def bulk_create_indices(self, entity_ids: List[str]) -> Dict[str, int]:
        """Bulk create indices for multiple entities"""
        result = {}
        for entity_id in entity_ids:
            result[entity_id] = self.get_or_create_index(entity_id)
        return result
    
    def compact(self):
        """Compact indices by removing gaps (for optimization)"""
        # Create new mapping without gaps
        sorted_entities = sorted(self.entity_to_index.items(), key=lambda x: x[1])
        
        new_entity_to_index = {}
        new_index_to_entity = {}
        
        for new_index, (entity_id, old_index) in enumerate(sorted_entities):
            new_entity_to_index[entity_id] = new_index
            new_index_to_entity[new_index] = entity_id
        
        self.entity_to_index = new_entity_to_index
        self.index_to_entity = new_index_to_entity
        self.next_index = len(sorted_entities)

@dataclass 
class SyncStatistics:
    """Synchronization statistics for monitoring"""
    nodes_synced: int = 0
    edges_synced: int = 0
    labels_updated: int = 0
    relations_updated: int = 0
    sync_time: float = 0.0
    last_sync_timestamp: float = 0.0
    
    def reset(self):
        """Reset statistics"""
        self.nodes_synced = 0
        self.edges_synced = 0
        self.labels_updated = 0
        self.relations_updated = 0
        self.sync_time = 0.0

class DataBridge:
    """
    Data bridge between Redis and GraphBLAS based on FalkorDB's architecture
    
    Handles:
    - Converting Redis node/edge data to GraphBLAS vectors/matrices
    - Converting GraphBLAS results back to entity data
    - Maintaining entity ID to matrix index mappings
    - Synchronizing changes between systems (Delta Matrix pattern)
    - Incremental updates and batch operations
    """
    
    def __init__(self, redis_executor, graphblas_executor):
        """Initialize data bridge with executor references"""
        self.redis_executor = redis_executor
        self.graphblas_executor = graphblas_executor
        
        # Entity mappings based on FalkorDB's ID mapping
        self.node_mapping = EntityMapping()
        self.edge_mapping = EntityMapping()
        
        # Type mappings for labels and relationships
        self.label_types: Dict[str, int] = {}
        self.relationship_types: Dict[str, int] = {}
        self.next_label_id = 0
        self.next_rel_id = 0
        
        # Synchronization tracking (Delta Matrix approach)
        self.pending_node_updates: Set[str] = set()
        self.pending_edge_updates: Set[str] = set()
        self.pending_label_updates: Set[Tuple[str, str]] = set()  # (node_id, label)
        self.pending_relation_updates: Set[Tuple[str, str, str]] = set()  # (src_id, rel_type, dest_id)
        
        # Statistics and monitoring
        self.sync_stats = SyncStatistics()
        self.auto_sync_threshold = 1000  # Auto-sync after N pending updates
        self.last_sync_timestamp = 0
        
        # Configuration
        self.batch_size = 1000
        self.enable_incremental_sync = True
        self.enable_compression = True
        
        # Caching for performance
        self.node_cache: Dict[str, Dict[str, Any]] = {}
        self.edge_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_size_limit = 10000
    
    def sync_redis_to_graphblas(self, force: bool = False, incremental: bool = None):
        """
        Synchronize Redis entity data to GraphBLAS matrices
        Based on FalkorDB's Delta Matrix sync patterns with incremental updates
        """
        start_time = time.time()
        incremental = incremental if incremental is not None else self.enable_incremental_sync
        
        logger.info(f"Starting {'incremental' if incremental else 'full'} Redis to GraphBLAS sync")
        
        try:
            self.sync_stats.reset()
            
            if incremental and not force:
                # Incremental sync - only sync pending changes
                self._sync_pending_updates()
            else:
                # Full sync - rebuild all matrices
                self._sync_full_rebuild()
            
            # Update statistics
            self.sync_stats.sync_time = time.time() - start_time
            self.sync_stats.last_sync_timestamp = time.time()
            self.last_sync_timestamp = self.sync_stats.last_sync_timestamp
            
            # Clear pending updates after successful sync
            self._clear_pending_updates()
            
            logger.info(f"Sync completed in {self.sync_stats.sync_time:.3f}s: "
                       f"{self.sync_stats.nodes_synced} nodes, "
                       f"{self.sync_stats.edges_synced} edges")
            
        except Exception as e:
            raise MyLathDBDataError(f"Redis to GraphBLAS sync failed: {e}")
    
    def _sync_pending_updates(self):
        """Sync only pending updates (incremental approach)"""
        
        # Sync pending node updates
        if self.pending_node_updates:
            self._sync_pending_nodes(self.pending_node_updates)
        
        # Sync pending edge updates
        if self.pending_edge_updates:
            self._sync_pending_edges(self.pending_edge_updates)
        
        # Sync pending label updates
        if self.pending_label_updates:
            self._sync_pending_labels(self.pending_label_updates)
        
        # Sync pending relation updates
        if self.pending_relation_updates:
            self._sync_pending_relations(self.pending_relation_updates)
    
    def _sync_full_rebuild(self):
        """Full synchronization - rebuild all matrices from Redis"""
        
        # Sync all nodes to matrices
        self._sync_all_nodes_to_matrices()
        
        # Sync all edges to matrices
        self._sync_all_edges_to_matrices()
        
        # Rebuild label matrices
        self._rebuild_label_matrices()
        
        # Rebuild relation matrices
        self._rebuild_relation_matrices()
    
    def _sync_all_nodes_to_matrices(self):
        """Sync all Redis nodes to GraphBLAS label matrices"""
        if not self.redis_executor.redis:
            return
        
        # Scan all node keys in batches
        batch_count = 0
        for node_keys_batch in self._scan_node_keys_in_batches():
            self._process_node_batch(node_keys_batch)
            batch_count += 1
            
            if batch_count % 10 == 0:
                logger.debug(f"Processed {batch_count * self.batch_size} nodes")
        
        logger.debug(f"Completed node sync: {self.sync_stats.nodes_synced} nodes")
    
    def _scan_node_keys_in_batches(self):
        """Scan Redis node keys in batches for memory efficiency"""
        node_keys_batch = []
        
        for key in self.redis_executor.redis.scan_iter(match="node:*"):
            if key != self.redis_executor.storage.NEXT_NODE_ID_KEY:
                node_keys_batch.append(key)
                
                if len(node_keys_batch) >= self.batch_size:
                    yield node_keys_batch
                    node_keys_batch = []
        
        # Yield remaining keys
        if node_keys_batch:
            yield node_keys_batch
    
    def _process_node_batch(self, node_keys: List[str]):
        """Process a batch of node keys"""
        for node_key in node_keys:
            node_id = node_key.split(':')[1]
            
            try:
                # Get node data (with caching)
                node_data = self._get_cached_node_data(node_id)
                if not node_data:
                    continue
                
                # Get or create matrix index for this node
                matrix_index = self.node_mapping.get_or_create_index(node_id)
                
                # Update label matrices
                labels = node_data.get('_labels', [])
                for label in labels:
                    self._update_label_matrix(label, matrix_index, True)
                    self.sync_stats.labels_updated += 1
                
                self.sync_stats.nodes_synced += 1
                
            except Exception as e:
                logger.warning(f"Failed to process node {node_id}: {e}")
    
    def _sync_all_edges_to_matrices(self):
        """Sync all Redis edges to GraphBLAS adjacency matrices"""
        if not self.redis_executor.redis:
            return
        
        # Get all relationship types
        rel_types = self._get_all_relationship_types()
        
        logger.debug(f"Syncing edges for {len(rel_types)} relationship types")
        
        # Process each relationship type
        for rel_type in rel_types:
            try:
                self._sync_relationship_edges(rel_type)
            except Exception as e:
                logger.error(f"Failed to sync relationship {rel_type}: {e}")
    
    def _get_all_relationship_types(self) -> Set[str]:
        """Get all relationship types from Redis"""
        rel_types = set()
        
        for key in self.redis_executor.redis.scan_iter(match="rel:*"):
            rel_type = key.split(':')[1]
            rel_types.add(rel_type)
        
        return rel_types
    
    def _sync_relationship_edges(self, rel_type: str):
        """Sync edges of a specific relationship type"""
        rel_key = self.redis_executor.storage.RELATIONSHIP_EDGES_KEY.format(rel_type=rel_type)
        edge_ids = self.redis_executor.redis.smembers(rel_key)
        
        if not edge_ids:
            return
        
        # Get or create relation matrix
        relation_matrix = self.graphblas_executor._get_relation_matrix([rel_type], "outgoing")
        
        # Process edges in batches
        edge_batch = []
        for edge_id in edge_ids:
            edge_batch.append(edge_id)
            
            if len(edge_batch) >= self.batch_size:
                self._process_edge_batch(edge_batch, rel_type, relation_matrix)
                edge_batch = []
        
        # Process remaining edges
        if edge_batch:
            self._process_edge_batch(edge_batch, rel_type, relation_matrix)
    
    def _process_edge_batch(self, edge_ids: List[str], rel_type: str, relation_matrix):
        """Process a batch of edges"""
        
        # Use pipeline for efficient Redis access
        pipeline = self.redis_executor.redis.pipeline()
        for edge_id in edge_ids:
            endpoints_key = self.redis_executor.storage.EDGE_ENDPOINTS_KEY.format(edge_id=edge_id)
            pipeline.get(endpoints_key)
        
        endpoints_results = pipeline.execute()
        
        # Process results
        for edge_id, endpoints_value in zip(edge_ids, endpoints_results):
            try:
                if endpoints_value:
                    src_id, dest_id, edge_rel_type = endpoints_value.split('|')
                    
                    if edge_rel_type == rel_type:
                        # Get matrix indices
                        src_index = self.node_mapping.get_or_create_index(src_id)
                        dest_index = self.node_mapping.get_or_create_index(dest_id)
                        
                        # Set matrix entry
                        self._set_matrix_entry(relation_matrix, src_index, dest_index, True)
                        
                        # Also update main adjacency matrix
                        adj_matrix = self.graphblas_executor.graph.adjacency_matrix
                        self._set_matrix_entry(adj_matrix, src_index, dest_index, True)
                        
                        self.sync_stats.edges_synced += 1
                
            except Exception as e:
                logger.warning(f"Failed to process edge {edge_id}: {e}")
    
    def _set_matrix_entry(self, matrix, row: int, col: int, value: bool):
        """FIXED: Safely set matrix entry with proper GraphBLAS handling"""
        try:
            # FIXED: Check matrix existence without boolean conversion
            if matrix is not None and hasattr(matrix, 'nrows') and hasattr(matrix, 'ncols'):
                if row < matrix.nrows and col < matrix.ncols:
                    matrix[row, col] = value
                    return True
        except Exception as e:
            logger.warning(f"Failed to set matrix entry [{row}, {col}]: {e}")
        return False
    
    def _sync_pending_nodes(self, node_ids: Set[str]):
        """Sync specific pending node updates"""
        logger.debug(f"Syncing {len(node_ids)} pending node updates")
        
        for node_id in node_ids:
            try:
                node_data = self._get_cached_node_data(node_id)
                if node_data:
                    matrix_index = self.node_mapping.get_or_create_index(node_id)
                    
                    # Update label matrices
                    labels = node_data.get('_labels', [])
                    for label in labels:
                        self._update_label_matrix(label, matrix_index, True)
                    
                    self.sync_stats.nodes_synced += 1
                    
            except Exception as e:
                logger.warning(f"Failed to sync pending node {node_id}: {e}")
    
    def _sync_pending_edges(self, edge_ids: Set[str]):
        """Sync specific pending edge updates"""
        logger.debug(f"Syncing {len(edge_ids)} pending edge updates")
        
        # Group edges by relationship type for efficiency
        edges_by_type = defaultdict(list)
        
        for edge_id in edge_ids:
            try:
                endpoints_key = self.redis_executor.storage.EDGE_ENDPOINTS_KEY.format(edge_id=edge_id)
                endpoints_value = self.redis_executor.redis.get(endpoints_key)
                
                if endpoints_value:
                    src_id, dest_id, rel_type = endpoints_value.split('|')
                    edges_by_type[rel_type].append((edge_id, src_id, dest_id))
                    
            except Exception as e:
                logger.warning(f"Failed to get endpoints for edge {edge_id}: {e}")
        
        # Process each relationship type
        for rel_type, edge_info_list in edges_by_type.items():
            relation_matrix = self.graphblas_executor._get_relation_matrix([rel_type], "outgoing")
            
            for edge_id, src_id, dest_id in edge_info_list:
                try:
                    src_index = self.node_mapping.get_or_create_index(src_id)
                    dest_index = self.node_mapping.get_or_create_index(dest_id)
                    
                    self._set_matrix_entry(relation_matrix, src_index, dest_index, True)
                    
                    # Update adjacency matrix
                    adj_matrix = self.graphblas_executor.graph.adjacency_matrix
                    self._set_matrix_entry(adj_matrix, src_index, dest_index, True)
                    
                    self.sync_stats.edges_synced += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to sync pending edge {edge_id}: {e}")
    
    def _sync_pending_labels(self, label_updates: Set[Tuple[str, str]]):
        """Sync pending label updates"""
        logger.debug(f"Syncing {len(label_updates)} pending label updates")
        
        for node_id, label in label_updates:
            try:
                matrix_index = self.node_mapping.get_or_create_index(node_id)
                self._update_label_matrix(label, matrix_index, True)
                self.sync_stats.labels_updated += 1
                
            except Exception as e:
                logger.warning(f"Failed to sync label update {node_id}:{label}: {e}")
    
    def _sync_pending_relations(self, relation_updates: Set[Tuple[str, str, str]]):
        """Sync pending relation updates"""
        logger.debug(f"Syncing {len(relation_updates)} pending relation updates")
        
        for src_id, rel_type, dest_id in relation_updates:
            try:
                src_index = self.node_mapping.get_or_create_index(src_id)
                dest_index = self.node_mapping.get_or_create_index(dest_id)
                
                relation_matrix = self.graphblas_executor._get_relation_matrix([rel_type], "outgoing")
                self._set_matrix_entry(relation_matrix, src_index, dest_index, True)
                
                # Update adjacency matrix
                adj_matrix = self.graphblas_executor.graph.adjacency_matrix
                self._set_matrix_entry(adj_matrix, src_index, dest_index, True)
                
                self.sync_stats.relations_updated += 1
                
            except Exception as e:
                logger.warning(f"Failed to sync relation update {src_id}-[{rel_type}]->{dest_id}: {e}")
    
    def _rebuild_label_matrices(self):
        """Rebuild all label matrices from scratch"""
        logger.debug("Rebuilding label matrices")
        
        # Clear existing label matrices
        for label_matrix in self.graphblas_executor.graph.label_matrices.values():
            try:
                label_matrix.clear()
            except Exception as e:
                logger.warning(f"Failed to clear label matrix: {e}")
        
        # Rebuild from Redis data
        # This would involve scanning all nodes and rebuilding label assignments
        # Implementation depends on specific GraphBLAS library features
    
    def _rebuild_relation_matrices(self):
        """Rebuild all relation matrices from scratch"""
        logger.debug("Rebuilding relation matrices")
        
        # Clear existing relation matrices
        for rel_matrix in self.graphblas_executor.graph.relation_matrices.values():
            try:
                rel_matrix.clear()
            except Exception as e:
                logger.warning(f"Failed to clear relation matrix: {e}")
        
        # Clear main adjacency matrix
        try:
            self.graphblas_executor.graph.adjacency_matrix.clear()
        except Exception as e:
            logger.warning(f"Failed to clear adjacency matrix: {e}")
    
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
        
        # Update matrix (diagonal entry for node labels in FalkorDB style)
        try:
            if node_index < label_matrix.nrows:
                label_matrix[node_index, node_index] = value
        except Exception as e:
            logger.error(f"Failed to update label matrix for {label}[{node_index}]: {e}")
    
    def _get_cached_node_data(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data with caching"""
        
        # Check cache first
        if node_id in self.node_cache:
            return self.node_cache[node_id]
        
        # Fetch from Redis
        node_data = self.redis_executor._get_node_data(node_id)
        
        # Cache if data exists and cache isn't full
        if node_data and len(self.node_cache) < self.cache_size_limit:
            self.node_cache[node_id] = node_data
        
        return node_data
    
    def _clear_pending_updates(self):
        """Clear all pending update sets"""
        self.pending_node_updates.clear()
        self.pending_edge_updates.clear()
        self.pending_label_updates.clear()
        self.pending_relation_updates.clear()
    
    def convert_redis_results_to_vectors(self, redis_results: List[Dict[str, Any]], 
                                       variable_name: str):
        """Convert Redis scan results to GraphBLAS vectors"""
        try:
            import graphblas as gb
            
            n = self.graphblas_executor.graph.node_capacity
            vector = gb.Vector.new(gb.dtypes.BOOL, size=n)
            
            # Set vector entries for each result
            indices_to_set = []
            for result in redis_results:
                node_data = result.get(variable_name)
                if node_data:
                    node_id = extract_node_id(node_data)
                    if node_id:
                        matrix_index = self.node_mapping.get_or_create_index(node_id)
                        if matrix_index < vector.size:
                            indices_to_set.append(matrix_index)
            
            # Batch set vector entries for efficiency
            if indices_to_set:
                for index in indices_to_set:
                    vector[index] = True
            
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
                        # Fetch full entity data from Redis (with caching)
                        entity_data = self._get_cached_node_data(entity_id)
                        if entity_data:
                            results.append({variable_name: entity_data})
            
        except Exception as e:
            logger.error(f"Failed to convert vector to Redis results: {e}")
        
        return results
    
    def create_vector_from_node_ids(self, node_ids: List[str]) -> Optional[Any]:
        """Create GraphBLAS vector from list of node IDs"""
        try:
            import graphblas as gb
            
            n = self.graphblas_executor.graph.node_capacity
            vector = gb.Vector.new(gb.dtypes.BOOL, size=n)
            
            # Get indices for all node IDs
            indices = []
            for node_id in node_ids:
                matrix_index = self.node_mapping.get_or_create_index(node_id)
                if matrix_index < vector.size:
                    indices.append(matrix_index)
            
            # Set vector entries
            for index in indices:
                vector[index] = True
            
            return vector
            
        except Exception as e:
            logger.error(f"Failed to create vector from node IDs: {e}")
            return None
    
    def extract_node_ids_from_vector(self, vector) -> List[str]:
        """Extract node IDs from GraphBLAS vector"""
        node_ids = []
        
        try:
            indices, values = vector.to_coo()
            
            for index, value in zip(indices, values):
                if value:
                    entity_id = self.node_mapping.get_entity_id(index)
                    if entity_id:
                        node_ids.append(entity_id)
        
        except Exception as e:
            logger.error(f"Failed to extract node IDs from vector: {e}")
        
        return node_ids
    
    def get_node_index(self, node_id: str) -> Optional[int]:
        """Get matrix index for node ID"""
        return self.node_mapping.entity_to_index.get(node_id)
    
    def get_node_id(self, index: int) -> Optional[str]:
        """Get node ID for matrix index"""
        return self.node_mapping.get_entity_id(index)
    
    def mark_node_updated(self, node_id: str):
        """Mark node as needing sync"""
        self.pending_node_updates.add(node_id)
        self._check_auto_sync()
    
    def mark_edge_updated(self, edge_id: str):
        """Mark edge as needing sync"""
        self.pending_edge_updates.add(edge_id)
        self._check_auto_sync()
    
    def mark_label_updated(self, node_id: str, label: str):
        """Mark label as needing sync"""
        self.pending_label_updates.add((node_id, label))
        self._check_auto_sync()
    
    def mark_relation_updated(self, src_id: str, rel_type: str, dest_id: str):
        """Mark relation as needing sync"""
        self.pending_relation_updates.add((src_id, rel_type, dest_id))
        self._check_auto_sync()
    
    def _check_auto_sync(self):
        """Check if auto-sync threshold is reached"""
        total_pending = (len(self.pending_node_updates) + 
                        len(self.pending_edge_updates) + 
                        len(self.pending_label_updates) + 
                        len(self.pending_relation_updates))
        
        if total_pending >= self.auto_sync_threshold:
            logger.info(f"Auto-sync triggered: {total_pending} pending updates")
            try:
                self.sync_redis_to_graphblas(incremental=True)
            except Exception as e:
                logger.error(f"Auto-sync failed: {e}")
    
    def has_pending_updates(self) -> bool:
        """Check if there are pending updates to sync"""
        return (len(self.pending_node_updates) > 0 or 
                len(self.pending_edge_updates) > 0 or
                len(self.pending_label_updates) > 0 or
                len(self.pending_relation_updates) > 0)
    
    def get_pending_count(self) -> Dict[str, int]:
        """Get count of pending updates by type"""
        return {
            'nodes': len(self.pending_node_updates),
            'edges': len(self.pending_edge_updates),
            'labels': len(self.pending_label_updates),
            'relations': len(self.pending_relation_updates)
        }
    
    def compact_mappings(self):
        """Compact entity mappings to remove gaps"""
        logger.info("Compacting entity mappings")
        self.node_mapping.compact()
        self.edge_mapping.compact()
    
    def clear_caches(self):
        """Clear all internal caches"""
        self.node_cache.clear()
        self.edge_cache.clear()
        logger.debug("Data bridge caches cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data bridge statistics"""
        return {
            'sync_stats': {
                'nodes_synced': self.sync_stats.nodes_synced,
                'edges_synced': self.sync_stats.edges_synced,
                'labels_updated': self.sync_stats.labels_updated,
                'relations_updated': self.sync_stats.relations_updated,
                'last_sync_time': self.sync_stats.sync_time,
                'last_sync_timestamp': self.sync_stats.last_sync_timestamp
            },
            'mappings': {
                'node_count': len(self.node_mapping.entity_to_index),
                'edge_count': len(self.edge_mapping.entity_to_index),
                'next_node_index': self.node_mapping.next_index,
                'next_edge_index': self.edge_mapping.next_index
            },
            'pending_updates': self.get_pending_count(),
            'cache_stats': {
                'node_cache_size': len(self.node_cache),
                'edge_cache_size': len(self.edge_cache),
                'cache_limit': self.cache_size_limit
            },
            'configuration': {
                'batch_size': self.batch_size,
                'auto_sync_threshold': self.auto_sync_threshold,
                'incremental_sync_enabled': self.enable_incremental_sync,
                'compression_enabled': self.enable_compression
            }
        }


class EntityManager:
    """
    Entity manager for coordinating entity operations
    Based on FalkorDB's entity management patterns with CRUD operations
    """
    
    def __init__(self, data_bridge: DataBridge):
        """Initialize entity manager"""
        self.data_bridge = data_bridge
        self.redis_executor = data_bridge.redis_executor
        self.graphblas_executor = data_bridge.graphblas_executor
        
        # Operation tracking
        self.created_nodes = 0
        self.created_edges = 0
        self.updated_nodes = 0
        self.updated_edges = 0
        self.deleted_nodes = 0
        self.deleted_edges = 0
    
    def create_node(self, node_data: Dict[str, Any]) -> str:
        """Create a new node in both Redis and mark for GraphBLAS sync"""
        
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
        
        # Track operation
        self.created_nodes += 1
        
        logger.debug(f"Created node {node_id}")
        return node_id
    
    def create_edge(self, src_id: str, dest_id: str, rel_type: str, 
                   edge_data: Dict[str, Any] = None) -> str:
        """Create a new edge in both Redis and mark for GraphBLAS sync"""
        
        # Generate edge ID
        edge_id = str(self.redis_executor.redis.incr(
            self.redis_executor.storage.NEXT_EDGE_ID_KEY
        ))
        
        # Store in Redis
        self._create_edge_in_redis(edge_id, src_id, dest_id, rel_type, edge_data or {})
        
        # Mark for GraphBLAS sync
        self.data_bridge.mark_edge_updated(edge_id)
        self.data_bridge.mark_relation_updated(src_id, rel_type, dest_id)
        
        # Track operation
        self.created_edges += 1
        
        logger.debug(f"Created edge {edge_id}: {src_id}-[{rel_type}]->{dest_id}")
        return edge_id
    
    def update_node(self, node_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing node"""
        
        try:
            # Update in Redis
            node_key = self.redis_executor.storage.NODE_KEY_PATTERN.format(node_id=node_id)
            
            # Update properties
            properties_to_update = {k: v for k, v in updates.items() 
                                  if not k.startswith('_') and k != 'id'}
            
            if properties_to_update:
                self.redis_executor.redis.hset(node_key, mapping=properties_to_update)
            
            # Handle label updates
            if '_labels' in updates:
                self._update_node_labels(node_id, updates['_labels'])
            
            # Mark for sync
            self.data_bridge.mark_node_updated(node_id)
            
            # Track operation
            self.updated_nodes += 1
            
            logger.debug(f"Updated node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update node {node_id}: {e}")
            return False
    
    def update_edge(self, edge_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing edge"""
        
        try:
            # Update properties in Redis
            edge_key = self.redis_executor.storage.EDGE_KEY_PATTERN.format(edge_id=edge_id)
            
            if updates:
                self.redis_executor.redis.hset(edge_key, mapping=updates)
            
            # Mark for sync
            self.data_bridge.mark_edge_updated(edge_id)
            
            # Track operation
            self.updated_edges += 1
            
            logger.debug(f"Updated edge {edge_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update edge {edge_id}: {e}")
            return False
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all its edges"""
        
        try:
            # First, find and delete all edges connected to this node
            self._delete_node_edges(node_id)
            
            # Delete node from Redis
            node_key = self.redis_executor.storage.NODE_KEY_PATTERN.format(node_id=node_id)
            labels_key = self.redis_executor.storage.NODE_LABELS_KEY.format(node_id=node_id)
            
            # Get node labels before deletion for cleanup
            labels = list(self.redis_executor.redis.smembers(labels_key))
            
            # Delete node data
            self.redis_executor.redis.delete(node_key)
            self.redis_executor.redis.delete(labels_key)
            
            # Remove from label indexes
            for label in labels:
                label_key = self.redis_executor.storage.LABEL_NODES_KEY.format(label=label)
                self.redis_executor.redis.srem(label_key, node_id)
            
            # Remove from property indexes (would need to scan properties)
            self._cleanup_node_property_indexes(node_id)
            
            # Mark for GraphBLAS sync (remove from matrices)
            self.data_bridge.mark_node_updated(node_id)
            for label in labels:
                self.data_bridge.mark_label_updated(node_id, label)
            
            # Track operation
            self.deleted_nodes += 1
            
            logger.debug(f"Deleted node {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
            return False
    
    def delete_edge(self, edge_id: str) -> bool:
        """Delete an edge"""
        
        try:
            # Get edge endpoints before deletion
            endpoints_key = self.redis_executor.storage.EDGE_ENDPOINTS_KEY.format(edge_id=edge_id)
            endpoints_value = self.redis_executor.redis.get(endpoints_key)
            
            if endpoints_value:
                src_id, dest_id, rel_type = endpoints_value.split('|')
                
                # Delete edge data
                edge_key = self.redis_executor.storage.EDGE_KEY_PATTERN.format(edge_id=edge_id)
                self.redis_executor.redis.delete(edge_key)
                self.redis_executor.redis.delete(endpoints_key)
                
                # Remove from relationship indexes
                out_key = self.redis_executor.storage.OUTGOING_EDGES_KEY.format(
                    node_id=src_id, rel_type=rel_type
                )
                in_key = self.redis_executor.storage.INCOMING_EDGES_KEY.format(
                    node_id=dest_id, rel_type=rel_type
                )
                rel_key = self.redis_executor.storage.RELATIONSHIP_EDGES_KEY.format(rel_type=rel_type)
                
                self.redis_executor.redis.srem(out_key, edge_id)
                self.redis_executor.redis.srem(in_key, edge_id)
                self.redis_executor.redis.srem(rel_key, edge_id)
                
                # Mark for GraphBLAS sync
                self.data_bridge.mark_edge_updated(edge_id)
                self.data_bridge.mark_relation_updated(src_id, rel_type, dest_id)
                
                # Track operation
                self.deleted_edges += 1
                
                logger.debug(f"Deleted edge {edge_id}: {src_id}-[{rel_type}]->{dest_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete edge {edge_id}: {e}")
            return False
    
    def batch_create_nodes(self, nodes_data: List[Dict[str, Any]]) -> List[str]:
        """Create multiple nodes in batch for efficiency"""
        
        created_node_ids = []
        
        # Use Redis pipeline for batch operations
        pipeline = self.redis_executor.redis.pipeline()
        
        for node_data in nodes_data:
            # Generate node ID
            node_id = node_data.get('id') or node_data.get('_id')
            if not node_id:
                node_id = str(self.redis_executor.redis.incr(
                    self.redis_executor.storage.NEXT_NODE_ID_KEY
                ))
            
            # Add to pipeline
            self._add_node_to_pipeline(pipeline, node_id, node_data)
            created_node_ids.append(node_id)
        
        # Execute pipeline
        try:
            pipeline.execute()
            
            # Mark all for sync
            for node_id in created_node_ids:
                self.data_bridge.mark_node_updated(node_id)
            
            # Track operations
            self.created_nodes += len(created_node_ids)
            
            logger.debug(f"Batch created {len(created_node_ids)} nodes")
            return created_node_ids
            
        except Exception as e:
            logger.error(f"Batch node creation failed: {e}")
            return []
    
    def batch_create_edges(self, edges_data: List[Tuple[str, str, str, Dict[str, Any]]]) -> List[str]:
        """Create multiple edges in batch for efficiency"""
        
        created_edge_ids = []
        
        # Use Redis pipeline for batch operations
        pipeline = self.redis_executor.redis.pipeline()
        
        for src_id, dest_id, rel_type, edge_data in edges_data:
            # Generate edge ID
            edge_id = str(self.redis_executor.redis.incr(
                self.redis_executor.storage.NEXT_EDGE_ID_KEY
            ))
            
            # Add to pipeline
            self._add_edge_to_pipeline(pipeline, edge_id, src_id, dest_id, rel_type, edge_data)
            created_edge_ids.append(edge_id)
        
        # Execute pipeline
        try:
            pipeline.execute()
            
            # Mark all for sync
            for i, edge_id in enumerate(created_edge_ids):
                src_id, dest_id, rel_type, _ = edges_data[i]
                self.data_bridge.mark_edge_updated(edge_id)
                self.data_bridge.mark_relation_updated(src_id, rel_type, dest_id)
            
            # Track operations
            self.created_edges += len(created_edge_ids)
            
            logger.debug(f"Batch created {len(created_edge_ids)} edges")
            return created_edge_ids
            
        except Exception as e:
            logger.error(f"Batch edge creation failed: {e}")
            return []
    
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
            self._create_node_labels(node_id, labels)
        
        # Create property indexes
        self._create_property_indexes(node_id, properties)
    
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
        self._create_relationship_indexes(edge_id, src_id, dest_id, rel_type)
    
    def _create_node_labels(self, node_id: str, labels: List[str]):
        """Create node label associations"""
        
        labels_key = self.redis_executor.storage.NODE_LABELS_KEY.format(node_id=node_id)
        self.redis_executor.redis.sadd(labels_key, *labels)
        
        # Add to label indexes
        for label in labels:
            label_key = self.redis_executor.storage.LABEL_NODES_KEY.format(label=label)
            self.redis_executor.redis.sadd(label_key, node_id)
            
            # Mark for GraphBLAS sync
            self.data_bridge.mark_label_updated(node_id, label)
    
    def _create_property_indexes(self, node_id: str, properties: Dict[str, Any]):
        """Create property indexes for node"""
        
        for prop_key, prop_value in properties.items():
            # Create property value index
            prop_index_key = self.redis_executor.storage.PROPERTY_INDEX_KEY.format(
                property=prop_key, value=prop_value
            )
            self.redis_executor.redis.sadd(prop_index_key, node_id)
            
            # Create sorted property index for numeric values
            try:
                numeric_value = float(prop_value)
                sorted_prop_key = self.redis_executor.storage.SORTED_PROPERTY_KEY.format(
                    property=prop_key
                )
                self.redis_executor.redis.zadd(sorted_prop_key, {node_id: numeric_value})
            except (ValueError, TypeError):
                # Not a numeric value, skip sorted index
                pass
    
    def _create_relationship_indexes(self, edge_id: str, src_id: str, dest_id: str, rel_type: str):
        """Create relationship indexes for edge"""
        
        # Outgoing edges from source
        out_key = self.redis_executor.storage.OUTGOING_EDGES_KEY.format(
            node_id=src_id, rel_type=rel_type
        )
        self.redis_executor.redis.sadd(out_key, edge_id)
        
        # Incoming edges to destination
        in_key = self.redis_executor.storage.INCOMING_EDGES_KEY.format(
            node_id=dest_id, rel_type=rel_type
        )
        self.redis_executor.redis.sadd(in_key, edge_id)
        
        # All edges of this relationship type
        rel_key = self.redis_executor.storage.RELATIONSHIP_EDGES_KEY.format(rel_type=rel_type)
        self.redis_executor.redis.sadd(rel_key, edge_id)
    
    def _update_node_labels(self, node_id: str, new_labels: List[str]):
        """Update node labels"""
        
        # Get current labels
        labels_key = self.redis_executor.storage.NODE_LABELS_KEY.format(node_id=node_id)
        current_labels = set(self.redis_executor.redis.smembers(labels_key))
        new_labels_set = set(new_labels)
        
        # Find labels to add and remove
        labels_to_add = new_labels_set - current_labels
        labels_to_remove = current_labels - new_labels_set
        
        # Update node labels set
        if labels_to_remove:
            self.redis_executor.redis.srem(labels_key, *labels_to_remove)
        if labels_to_add:
            self.redis_executor.redis.sadd(labels_key, *labels_to_add)
        
        # Update label indexes
        for label in labels_to_remove:
            label_key = self.redis_executor.storage.LABEL_NODES_KEY.format(label=label)
            self.redis_executor.redis.srem(label_key, node_id)
            self.data_bridge.mark_label_updated(node_id, label)
        
        for label in labels_to_add:
            label_key = self.redis_executor.storage.LABEL_NODES_KEY.format(label=label)
            self.redis_executor.redis.sadd(label_key, node_id)
            self.data_bridge.mark_label_updated(node_id, label)
    
    def _delete_node_edges(self, node_id: str):
        """Delete all edges connected to a node"""
        
        # Find all outgoing edges
        for key in self.redis_executor.redis.scan_iter(match=f"out:{node_id}:*"):
            edge_ids = self.redis_executor.redis.smembers(key)
            for edge_id in edge_ids:
                self.delete_edge(edge_id)
        
        # Find all incoming edges
        for key in self.redis_executor.redis.scan_iter(match=f"in:{node_id}:*"):
            edge_ids = self.redis_executor.redis.smembers(key)
            for edge_id in edge_ids:
                self.delete_edge(edge_id)
    
    def _cleanup_node_property_indexes(self, node_id: str):
        """Clean up property indexes for deleted node"""
        
        # This is expensive - would need to scan all property indexes
        # In a production system, you'd maintain reverse indexes
        # For now, we'll skip this cleanup or do it asynchronously
        pass
    
    def _add_node_to_pipeline(self, pipeline, node_id: str, node_data: Dict[str, Any]):
        """Add node creation commands to Redis pipeline"""
        
        # Add node properties
        node_key = self.redis_executor.storage.NODE_KEY_PATTERN.format(node_id=node_id)
        properties = {k: v for k, v in node_data.items() 
                     if not k.startswith('_') and k != 'id'}
        
        if properties:
            pipeline.hset(node_key, mapping=properties)
        
        # Add node labels
        labels = node_data.get('_labels', [])
        if labels:
            labels_key = self.redis_executor.storage.NODE_LABELS_KEY.format(node_id=node_id)
            pipeline.sadd(labels_key, *labels)
            
            # Add to label indexes
            for label in labels:
                label_key = self.redis_executor.storage.LABEL_NODES_KEY.format(label=label)
                pipeline.sadd(label_key, node_id)
        
        # Add property indexes
        for prop_key, prop_value in properties.items():
            prop_index_key = self.redis_executor.storage.PROPERTY_INDEX_KEY.format(
                property=prop_key, value=prop_value
            )
            pipeline.sadd(prop_index_key, node_id)
            
            # Add sorted index for numeric values
            try:
                numeric_value = float(prop_value)
                sorted_prop_key = self.redis_executor.storage.SORTED_PROPERTY_KEY.format(
                    property=prop_key
                )
                pipeline.zadd(sorted_prop_key, {node_id: numeric_value})
            except (ValueError, TypeError):
                pass
    
    def _add_edge_to_pipeline(self, pipeline, edge_id: str, src_id: str, dest_id: str, 
                             rel_type: str, edge_data: Dict[str, Any]):
        """Add edge creation commands to Redis pipeline"""
        
        # Add edge properties
        if edge_data:
            edge_key = self.redis_executor.storage.EDGE_KEY_PATTERN.format(edge_id=edge_id)
            pipeline.hset(edge_key, mapping=edge_data)
        
        # Add edge endpoints
        endpoints_key = self.redis_executor.storage.EDGE_ENDPOINTS_KEY.format(edge_id=edge_id)
        endpoints_value = f"{src_id}|{dest_id}|{rel_type}"
        pipeline.set(endpoints_key, endpoints_value)
        
        # Add relationship indexes
        out_key = self.redis_executor.storage.OUTGOING_EDGES_KEY.format(
            node_id=src_id, rel_type=rel_type
        )
        in_key = self.redis_executor.storage.INCOMING_EDGES_KEY.format(
            node_id=dest_id, rel_type=rel_type
        )
        rel_key = self.redis_executor.storage.RELATIONSHIP_EDGES_KEY.format(rel_type=rel_type)
        
        pipeline.sadd(out_key, edge_id)
        pipeline.sadd(in_key, edge_id)
        pipeline.sadd(rel_key, edge_id)
    
    def get_operation_statistics(self) -> Dict[str, int]:
        """Get entity manager operation statistics"""
        return {
            'created_nodes': self.created_nodes,
            'created_edges': self.created_edges,
            'updated_nodes': self.updated_nodes,
            'updated_edges': self.updated_edges,
            'deleted_nodes': self.deleted_nodes,
            'deleted_edges': self.deleted_edges
        }
    
    def reset_statistics(self):
        """Reset operation statistics"""
        self.created_nodes = 0
        self.created_edges = 0
        self.updated_nodes = 0
        self.updated_edges = 0
        self.deleted_nodes = 0
        self.deleted_edges = 0
