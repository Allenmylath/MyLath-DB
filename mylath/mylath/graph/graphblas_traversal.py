# mylath/graph/graphblas_traversal.py
"""
GraphBLAS-accelerated traversal engine for MyLath
Provides massive speedup for complex graph queries
"""

from typing import List, Dict, Any, Callable, Optional, Set, Union
import time

try:
    import graphblas as gb
    from graphblas import Matrix, Vector, Scalar
    from graphblas import semiring, binary, unary
    GRAPHBLAS_AVAILABLE = True
except ImportError:
    GRAPHBLAS_AVAILABLE = False

from .traversal import GraphTraversal
from ..storage.graphblas_storage import GraphBLASStorage


class GraphBLASTraversal(GraphTraversal):
    """
    High-performance graph traversal using GraphBLAS matrix operations
    Can be 10-1000x faster than traditional traversals
    """
    
    def __init__(self, storage: GraphBLASStorage):
        if not GRAPHBLAS_AVAILABLE:
            raise ImportError("GraphBLAS not available")
            
        super().__init__(storage)
        self.graphblas_storage = storage
        self._use_graphblas = True  # Flag to enable/disable GraphBLAS
        self._current_node_set = None  # Track current nodes as Vector
        
    def V(self, node_ids: List[str] = None) -> 'GraphBLASTraversal':
        """Start traversal from vertices with GraphBLAS acceleration"""
        if node_ids:
            # Convert node IDs to indices
            indices = []
            for node_id in node_ids:
                if node_id in self.graphblas_storage.node_id_to_index:
                    indices.append(self.graphblas_storage.node_id_to_index[node_id])
            
            # Create GraphBLAS vector
            self._current_node_set = Vector.new(gb.BOOL, self.graphblas_storage.max_nodes)
            for idx in indices:
                self._current_node_set[idx] = True
                
            # Also maintain traditional representation for compatibility
            nodes = [self.storage.get_node(nid) for nid in node_ids]
            nodes = [n for n in nodes if n is not None]
        else:
            # Get all nodes - use GraphBLAS for efficiency
            self._current_node_set = Vector.new(gb.BOOL, self.graphblas_storage.max_nodes)
            for i in range(self.graphblas_storage.next_index):
                self._current_node_set[i] = True
            
            # Traditional path for compatibility
            nodes = []
            for i in range(self.graphblas_storage.next_index):
                if i in self.graphblas_storage.index_to_node_id:
                    node_id = self.graphblas_storage.index_to_node_id[i]
                    node = self.storage.get_node(node_id)
                    if node:
                        nodes.append(node)
        
        self.current_step.nodes = nodes
        return self
    
    def out(self, label: str = None) -> 'GraphBLASTraversal':
        """
        Follow outgoing edges using GraphBLAS matrix multiplication
        Massive speedup for multi-hop traversals
        """
        if not self._use_graphblas or self._current_node_set is None:
            return super().out(label)  # Fallback to traditional
        
        start_time = time.time()
        
        # Choose appropriate matrix
        if label and label in self.graphblas_storage.label_matrices:
            adj_matrix = self.graphblas_storage.label_matrices[label]
        else:
            adj_matrix = self.graphblas_storage.adjacency_matrix
        
        # Matrix-vector multiplication: next = A^T @ current
        next_nodes = Vector.new(gb.BOOL, self.graphblas_storage.max_nodes)
        next_nodes << adj_matrix.T.mxv(self._current_node_set, semiring.any_pair)
        
        self._current_node_set = next_nodes
        
        # Convert back to Node objects for compatibility
        result_nodes = []
        indices, _ = next_nodes.to_values()
        for idx in indices:
            if idx in self.graphblas_storage.index_to_node_id:
                node_id = self.graphblas_storage.index_to_node_id[idx]
                node = self.storage.get_node(node_id)
                if node:
                    result_nodes.append(node)
        
        self.current_step.nodes = result_nodes
        
        # Update metrics
        self.graphblas_storage.metrics.traversal_time += time.time() - start_time
        self.graphblas_storage.metrics.matrix_operations += 1
        
        return self
    
    def in_(self, label: str = None) -> 'GraphBLASTraversal':
        """Follow incoming edges using GraphBLAS"""
        if not self._use_graphblas or self._current_node_set is None:
            return super().in_(label)
        
        start_time = time.time()
        
        # Choose appropriate matrix
        if label and label in self.graphblas_storage.label_matrices:
            adj_matrix = self.graphblas_storage.label_matrices[label]
        else:
            adj_matrix = self.graphblas_storage.adjacency_matrix
        
        # Matrix-vector multiplication: next = A @ current
        next_nodes = Vector.new(gb.BOOL, self.graphblas_storage.max_nodes)
        next_nodes << adj_matrix.mxv(self._current_node_set, semiring.any_pair)
        
        self._current_node_set = next_nodes
        
        # Convert back to Node objects
        result_nodes = []
        indices, _ = next_nodes.to_values()
        for idx in indices:
            if idx in self.graphblas_storage.index_to_node_id:
                node_id = self.graphblas_storage.index_to_node_id[idx]
                node = self.storage.get_node(node_id)
                if node:
                    result_nodes.append(node)
        
        self.current_step.nodes = result_nodes
        
        self.graphblas_storage.metrics.traversal_time += time.time() - start_time
        self.graphblas_storage.metrics.matrix_operations += 1
        
        return self
    
    def k_hop(self, k: int, label: str = None) -> 'GraphBLASTraversal':
        """
        Extremely fast k-hop traversal using matrix powers
        This is where GraphBLAS really shines!
        """
        if not self._use_graphblas or self._current_node_set is None:
            # Fallback to iterative approach
            result = self
            for _ in range(k):
                result = result.out(label)
            return result
        
        start_time = time.time()
        
        # Choose appropriate matrix
        if label and label in self.graphblas_storage.label_matrices:
            adj_matrix = self.graphblas_storage.label_matrices[label]
        else:
            adj_matrix = self.graphblas_storage.adjacency_matrix
        
        # Compute k-hop reachability efficiently
        current = self._current_node_set.dup()
        reachable = Vector.new(
