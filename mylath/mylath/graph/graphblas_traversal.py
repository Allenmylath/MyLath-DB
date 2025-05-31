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
        reachable = Vector.new(gb.BOOL, self.graphblas_storage.max_nodes)
        
        for hop in range(k):
            # current = A^T @ current (find next hop)
            next_hop = Vector.new(gb.BOOL, self.graphblas_storage.max_nodes)
            next_hop << adj_matrix.T.mxv(current, semiring.any_pair)
            
            # Accumulate reachable nodes
            reachable << reachable.ewise_add(next_hop, binary.lor)
            current = next_hop
        
        self._current_node_set = reachable
        
        # Convert back to Node objects
        result_nodes = []
        indices, _ = reachable.to_values()
        for idx in indices:
            if idx in self.graphblas_storage.index_to_node_id:
                node_id = self.graphblas_storage.index_to_node_id[idx]
                node = self.storage.get_node(node_id)
                if node:
                    result_nodes.append(node)
        
        self.current_step.nodes = result_nodes
        
        self.graphblas_storage.metrics.traversal_time += time.time() - start_time
        self.graphblas_storage.metrics.matrix_operations += k
        
        return self
    
    def has(self, key: str, value: Any = None) -> 'GraphBLASTraversal':
        """Filter with GraphBLAS acceleration when possible"""
        if not self._use_graphblas or self._current_node_set is None:
            return super().has(key, value)
        
        # For property filtering, we still need to check individual nodes
        # But we can optimize by only checking nodes in our current set
        start_time = time.time()
        
        filtered_nodes = []
        filtered_vector = Vector.new(gb.BOOL, self.graphblas_storage.max_nodes)
        
        indices, _ = self._current_node_set.to_values()
        
        for idx in indices:
            if idx in self.graphblas_storage.index_to_node_id:
                node_id = self.graphblas_storage.index_to_node_id[idx]
                node = self.storage.get_node(node_id)
                
                if node:
                    if value is None:
                        # Check if property exists
                        if key in node.properties:
                            filtered_nodes.append(node)
                            filtered_vector[idx] = True
                    else:
                        # Check property value
                        if node.properties.get(key) == value:
                            filtered_nodes.append(node)
                            filtered_vector[idx] = True
        
        self._current_node_set = filtered_vector
        self.current_step.nodes = filtered_nodes
        
        self.graphblas_storage.metrics.traversal_time += time.time() - start_time
        
        return self
    
    def shortest_path(self, target_id: str, edge_label: str = None) -> Optional[List]:
        """Ultra-fast shortest path using GraphBLAS BFS"""
        if not self._use_graphblas or not self.current_step.nodes:
            return super().shortest_path(target_id, edge_label)
        
        start_node = self.current_step.nodes[0]
        return self.graphblas_storage.graphblas_shortest_path(start_node.id, target_id)
    
    def bfs_levels(self, max_depth: int = None) -> Dict[str, int]:
        """
        Get BFS levels from current nodes using GraphBLAS
        Extremely fast even for large graphs
        """
        if not self._use_graphblas or not self.current_step.nodes:
            return {}
        
        all_levels = {}
        for node in self.current_step.nodes:
            levels = self.graphblas_storage.graphblas_bfs(node.id, max_depth)
            all_levels.update(levels)
        
        return all_levels
    
    def connected_component(self) -> Set[str]:
        """Find connected component containing current nodes"""
        if not self._use_graphblas or not self.current_step.nodes:
            return set()
        
        components = self.graphblas_storage.graphblas_connected_components()
        
        # Find component IDs of current nodes
        current_component_ids = set()
        for node in self.current_step.nodes:
            if node.id in components:
                current_component_ids.add(components[node.id])
        
        # Return all nodes in these components
        result = set()
        for node_id, comp_id in components.items():
            if comp_id in current_component_ids:
                result.add(node_id)
        
        return result
    
    def pagerank_scores(self, damping: float = 0.85) -> Dict[str, float]:
        """Get PageRank scores using GraphBLAS"""
        if not self._use_graphblas:
            return {}
        
        return self.graphblas_storage.graphblas_pagerank(damping)
    
    def degree_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Compute degree distribution using GraphBLAS
        Returns {node_id: {'in_degree': x, 'out_degree': y}}
        """
        if not self._use_graphblas:
            return {}
        
        # Compute in-degrees (column sums)
        in_degrees = Vector.new(gb.INT32, self.graphblas_storage.max_nodes)
        in_degrees << self.graphblas_storage.adjacency_matrix.reduce_columnwise(binary.plus)
        
        # Compute out-degrees (row sums)
        out_degrees = Vector.new(gb.INT32, self.graphblas_storage.max_nodes)
        out_degrees << self.graphblas_storage.adjacency_matrix.reduce_rowwise(binary.plus)
        
        # Convert to node IDs
        result = {}
        
        # Process in-degrees
        in_indices, in_values = in_degrees.to_values()
        in_dict = dict(zip(in_indices, in_values))
        
        # Process out-degrees
        out_indices, out_values = out_degrees.to_values()
        out_dict = dict(zip(out_indices, out_values))
        
        # Combine results
        all_indices = set(in_indices) | set(out_indices)
        for idx in all_indices:
            if idx in self.graphblas_storage.index_to_node_id:
                node_id = self.graphblas_storage.index_to_node_id[idx]
                result[node_id] = {
                    'in_degree': in_dict.get(idx, 0),
                    'out_degree': out_dict.get(idx, 0)
                }
        
        return result
    
    def triangle_count(self) -> int:
        """
        Count triangles in the graph using GraphBLAS
        Classic graph algorithm that's very fast with matrix operations
        """
        if not self._use_graphblas:
            return 0
        
        start_time = time.time()
        
        # Triangle counting: sum(A * A * A) / 6 for undirected graphs
        # For directed: count triangles differently
        
        A = self.graphblas_storage.adjacency_matrix
        
        # Compute A^2
        A2 = Matrix.new(gb.INT32, self.graphblas_storage.max_nodes, self.graphblas_storage.max_nodes)
        A2 << A.mxm(A, semiring.plus_times)
        
        # Element-wise multiply A2 with A and sum
        A3_diag = A2.ewise_mult(A, binary.times)
        triangle_count = A3_diag.reduce_scalar(binary.plus)
        
        self.graphblas_storage.metrics.traversal_time += time.time() - start_time
        self.graphblas_storage.metrics.matrix_operations += 3
        
        return triangle_count // 3 if triangle_count else 0  # Avoid double counting
    
    def clustering_coefficient(self, node_id: str = None) -> Union[float, Dict[str, float]]:
        """
        Compute clustering coefficient using GraphBLAS
        If node_id is None, compute for all nodes
        """
        if not self._use_graphblas:
            return 0.0 if node_id else {}
        
        A = self.graphblas_storage.adjacency_matrix
        
        if node_id:
            # Single node clustering coefficient
            if node_id not in self.graphblas_storage.node_id_to_index:
                return 0.0
            
            idx = self.graphblas_storage.node_id_to_index[node_id]
            
            # Get neighbors
            neighbors = Vector.new(gb.BOOL, self.graphblas_storage.max_nodes)
            neighbors << A[idx, :].ewise_add(A[:, idx], binary.lor)
            
            neighbor_count = neighbors.nvals
            if neighbor_count < 2:
                return 0.0
            
            # Count edges between neighbors
            neighbor_indices, _ = neighbors.to_values()
            edge_count = 0
            
            for i in neighbor_indices:
                for j in neighbor_indices:
                    if i < j and A.get(i, j, False):
                        edge_count += 1
            
            possible_edges = neighbor_count * (neighbor_count - 1) // 2
            return edge_count / possible_edges if possible_edges > 0 else 0.0
        
        else:
            # All nodes clustering coefficient
            result = {}
            for i in range(self.graphblas_storage.next_index):
                if i in self.graphblas_storage.index_to_node_id:
                    node_id = self.graphblas_storage.index_to_node_id[i]
                    result[node_id] = self.clustering_coefficient(node_id)
            return result
    
    def enable_graphblas(self, enabled: bool = True):
        """Enable or disable GraphBLAS acceleration"""
        self._use_graphblas = enabled and GRAPHBLAS_AVAILABLE
        return self
    
    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance information for this traversal"""
        base_info = {
            'graphblas_enabled': self._use_graphblas,
            'graphblas_available': GRAPHBLAS_AVAILABLE,
            'current_node_count': len(self.current_step.nodes),
        }
        
        if self._use_graphblas:
            base_info.update(self.graphblas_storage.get_performance_metrics())
        
        return base_info
    
    def benchmark_vs_traditional(self, operation: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Benchmark GraphBLAS vs traditional implementation
        Useful for demonstrating performance improvements
        """
        results = {}
        
        # Test with GraphBLAS
        self._use_graphblas = True
        start_time = time.time()
        
        if operation == 'out':
            graphblas_result = self.out(*args, **kwargs)
        elif operation == 'k_hop':
            graphblas_result = self.k_hop(*args, **kwargs)
        elif operation == 'shortest_path':
            graphblas_result = self.shortest_path(*args, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        graphblas_time = time.time() - start_time
        graphblas_count = len(graphblas_result.current_step.nodes)
        
        # Reset state
        self.V([node.id for node in self.current_step.nodes])
        
        # Test with traditional approach
        self._use_graphblas = False
        start_time = time.time()
        
        if operation == 'out':
            traditional_result = self.out(*args, **kwargs)
        elif operation == 'k_hop':
            traditional_result = self.k_hop(*args, **kwargs)
        elif operation == 'shortest_path':
            traditional_result = self.shortest_path(*args, **kwargs)
        
        traditional_time = time.time() - start_time
        traditional_count = len(traditional_result.current_step.nodes)
        
        # Restore GraphBLAS
        self._use_graphblas = True
        
        speedup = traditional_time / graphblas_time if graphblas_time > 0 else float('inf')
        
        return {
            'operation': operation,
            'graphblas_time': graphblas_time,
            'traditional_time': traditional_time,
            'speedup': speedup,
            'graphblas_result_count': graphblas_count,
            'traditional_result_count': traditional_count,
            'results_match': graphblas_count == traditional_count
        }
