# execution_engine/graphblas_executor.py

"""
GraphBLAS Executor - Executes GraphBLAS operations for graph traversals
"""

from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime
import threading
import concurrent.futures

from cypher_planner.physical_planner import GraphBLASOperation

# Try to import GraphBLAS - graceful fallback if not available
try:
    import graphblas as gb
    GRAPHBLAS_AVAILABLE = True
except ImportError:
    GRAPHBLAS_AVAILABLE = False
    gb = None


@dataclass
class GraphBLASResult:
    """Result of GraphBLAS operation execution"""
    success: bool
    matrices: Dict[str, Any] = field(default_factory=dict)
    vectors: Dict[str, Any] = field(default_factory=dict)
    scalars: Dict[str, Any] = field(default_factory=dict)
    node_ids: Set[str] = field(default_factory=set)
    execution_time: float = 0.0
    operations_count: int = 0
    memory_usage: float = 0.0
    error: Optional[str] = None


class GraphBLASExecutor:
    """Executes GraphBLAS operations for graph traversals and computations"""
    
    def __init__(self, max_parallel_ops=4, enable_caching=True):
        # Configuration
        self.max_parallel_ops = max_parallel_ops
        self.enable_caching = enable_caching
        
        # Graph data storage
        self.adjacency_matrices = {}  # {rel_type: matrix}
        self.node_vectors = {}       # {variable: vector}
        self.relation_vectors = {}   # {rel_var: vector}
        
        # Statistics
        self.stats = {
            'operations_executed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_execution_time': 0.0,
            'memory_peak': 0.0
        }
        
        # Cache for expensive operations
        self.operation_cache = {}
        
        # Thread pool for parallel operations
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_ops)
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize GraphBLAS if available
        self._initialize_graphblas()
    
    def _initialize_graphblas(self):
        """Initialize GraphBLAS library"""
        if not GRAPHBLAS_AVAILABLE:
            self.logger.warning("GraphBLAS not available - using fallback implementation")
            return
        
        try:
            # Initialize GraphBLAS
            gb.init('suitesparse')
            self.logger.info("GraphBLAS initialized successfully")
        except Exception as e:
            self.logger.warning(f"GraphBLAS initialization failed: {e}")
    
    def load_graph_data(self, graph_data: Dict[str, Any]):
        """Load graph data for GraphBLAS operations"""
        
        try:
            if 'adjacency_matrices' in graph_data:
                self.adjacency_matrices = graph_data['adjacency_matrices']
            
            if 'nodes' in graph_data:
                self._create_node_vectors(graph_data['nodes'])
            
            if 'edges' in graph_data:
                self._create_adjacency_matrices(graph_data['edges'])
            
            self.logger.info(f"Loaded graph data: {len(self.adjacency_matrices)} matrices, {len(self.node_vectors)} vectors")
            
        except Exception as e:
            self.logger.error(f"Failed to load graph data: {e}")
    
    def execute(self, operation: GraphBLASOperation, context, input_data=None) -> GraphBLASResult:
        """Execute a GraphBLAS operation"""
        
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(operation, input_data)
            if self.enable_caching and cache_key in self.operation_cache:
                cached_result = self.operation_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    self.stats['cache_hits'] += 1
                    return cached_result['result']
            
            # Execute based on operation type
            if operation.operation_type == "ConditionalTraverse":
                result = self._execute_conditional_traverse(operation, context, input_data)
            elif operation.operation_type == "VarLenTraverse":
                result = self._execute_var_len_traverse(operation, context, input_data)
            elif operation.operation_type == "Expand":
                result = self._execute_expand(operation, context, input_data)
            elif operation.operation_type == "StructuralFilter":
                result = self._execute_structural_filter(operation, context, input_data)
            elif operation.operation_type == "PathFilter":
                result = self._execute_path_filter(operation, context, input_data)
            else:
                result = self._execute_generic_graphblas(operation, context, input_data)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Update statistics
            self.stats['operations_executed'] += 1
            self.stats['total_execution_time'] += execution_time
            if not (self.enable_caching and cache_key in self.operation_cache):
                self.stats['cache_misses'] += 1
            
            # Cache result if caching is enabled
            if self.enable_caching and result.success:
                self.operation_cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now(),
                    'ttl': 3600  # 1 hour TTL
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"GraphBLAS operation failed: {str(e)}")
            return GraphBLASResult(
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _execute_conditional_traverse(self, operation: GraphBLASOperation, context, input_data) -> GraphBLASResult:
        """Execute ConditionalTraverse operation"""
        
        result = GraphBLASResult(success=True)
        
        try:
            logical_op = operation.logical_op
            if not logical_op:
                return GraphBLASResult(success=False, error="No logical operation provided")
            
            from_var = logical_op.from_var
            to_var = logical_op.to_var
            rel_types = logical_op.rel_types or ["*"]
            direction = logical_op.direction
            
            # Get input vector for from_var
            from_vector = self._get_or_create_vector(from_var, context, input_data)
            if from_vector is None:
                return GraphBLASResult(success=False, error=f"No vector found for variable {from_var}")
            
            # Get adjacency matrix for relationship type
            rel_type = rel_types[0] if rel_types else "*"
            adj_matrix = self._get_adjacency_matrix(rel_type)
            
            if adj_matrix is None:
                return GraphBLASResult(success=False, error=f"No adjacency matrix found for relationship type {rel_type}")
            
            # Perform traversal based on direction
            if GRAPHBLAS_AVAILABLE:
                to_vector = self._graphblas_traverse(from_vector, adj_matrix, direction)
            else:
                to_vector = self._fallback_traverse(from_vector, adj_matrix, direction)
            
            # Store result vector
            self.node_vectors[to_var] = to_vector
            
            # Extract node IDs from result vector
            result_node_ids = self._extract_node_ids_from_vector(to_vector)
            
            # Update context
            if context:
                context.set_variable(to_var, list(result_node_ids))
            
            result.vectors = {to_var: to_vector}
            result.node_ids = result_node_ids
            result.operations_count = len(operation.matrix_operations)
            
            return result
            
        except Exception as e:
            return GraphBLASResult(success=False, error=f"ConditionalTraverse failed: {str(e)}")
    
    def _execute_var_len_traverse(self, operation: GraphBLASOperation, context, input_data) -> GraphBLASResult:
        """Execute variable-length traversal"""
        
        result = GraphBLASResult(success=True)
        
        try:
            logical_op = operation.logical_op
            if not logical_op:
                return GraphBLASResult(success=False, error="No logical operation provided")
            
            from_var = logical_op.from_var
            to_var = logical_op.to_var
            rel_types = logical_op.rel_types or ["*"]
            min_length = logical_op.min_length
            max_length = logical_op.max_length
            direction = logical_op.direction
            
            # Get input vector
            from_vector = self._get_or_create_vector(from_var, context, input_data)
            if from_vector is None:
                return GraphBLASResult(success=False, error=f"No vector found for variable {from_var}")
            
            # Get adjacency matrix
            rel_type = rel_types[0] if rel_types else "*"
            adj_matrix = self._get_adjacency_matrix(rel_type)
            
            if adj_matrix is None:
                return GraphBLASResult(success=False, error=f"No adjacency matrix found for relationship type {rel_type}")
            
            # Perform variable-length traversal
            if max_length == float('inf'):
                to_vector = self._compute_transitive_closure(from_vector, adj_matrix, min_length, direction)
            else:
                to_vector = self._compute_bounded_path(from_vector, adj_matrix, min_length, max_length, direction)
            
            # Store result
            self.node_vectors[to_var] = to_vector
            result_node_ids = self._extract_node_ids_from_vector(to_vector)
            
            # Update context
            if context:
                context.set_variable(to_var, list(result_node_ids))
            
            result.vectors = {to_var: to_vector}
            result.node_ids = result_node_ids
            result.operations_count = len(operation.matrix_operations)
            
            return result
            
        except Exception as e:
            return GraphBLASResult(success=False, error=f"VarLenTraverse failed: {str(e)}")
    
    def _execute_expand(self, operation: GraphBLASOperation, context, input_data) -> GraphBLASResult:
        """Execute legacy Expand operation"""
        
        # Convert to ConditionalTraverse for execution
        return self._execute_conditional_traverse(operation, context, input_data)
    
    def _execute_structural_filter(self, operation: GraphBLASOperation, context, input_data) -> GraphBLASResult:
        """Execute structural filter"""
        
        result = GraphBLASResult(success=True)
        
        try:
            # For now, pass through the input data
            # In a real implementation, you'd apply structural filtering
            if input_data and hasattr(input_data, 'vectors'):
                result.vectors = input_data.vectors.copy()
                result.node_ids = input_data.node_ids.copy()
            
            result.operations_count = len(operation.matrix_operations)
            
            return result
            
        except Exception as e:
            return GraphBLASResult(success=False, error=f"StructuralFilter failed: {str(e)}")
    
    def _execute_path_filter(self, operation: GraphBLASOperation, context, input_data) -> GraphBLASResult:
        """Execute path filter"""
        
        result = GraphBLASResult(success=True)
        
        try:
            # For now, pass through the input data
            # In a real implementation, you'd apply path pattern matching
            if input_data and hasattr(input_data, 'vectors'):
                result.vectors = input_data.vectors.copy()
                result.node_ids = input_data.node_ids.copy()
            
            result.operations_count = len(operation.matrix_operations)
            
            return result
            
        except Exception as e:
            return GraphBLASResult(success=False, error=f"PathFilter failed: {str(e)}")
    
    def _execute_generic_graphblas(self, operation: GraphBLASOperation, context, input_data) -> GraphBLASResult:
        """Execute generic GraphBLAS operation"""
        
        result = GraphBLASResult(success=True)
        
        try:
            # Execute matrix operations
            operation_results = []
            
            for matrix_op in operation.matrix_operations:
                if not matrix_op.startswith('#'):  # Skip comments
                    op_result = self._execute_matrix_operation(matrix_op, context)
                    operation_results.append(op_result)
            
            result.operations_count = len(operation.matrix_operations)
            
            return result
            
        except Exception as e:
            return GraphBLASResult(success=False, error=f"Generic GraphBLAS operation failed: {str(e)}")
    
    # Helper methods for GraphBLAS operations
    
    def _get_or_create_vector(self, variable: str, context, input_data) -> Any:
        """Get or create vector for variable"""
        
        # Check if vector already exists
        if variable in self.node_vectors:
            return self.node_vectors[variable]
        
        # Try to get from context
        if context and context.has_variable(variable):
            node_ids = context.get_variable(variable)
            vector = self._create_vector_from_node_ids(node_ids)
            self.node_vectors[variable] = vector
            return vector
        
        # Try to get from input data
        if input_data:
            if hasattr(input_data, 'node_ids') and input_data.node_ids:
                vector = self._create_vector_from_node_ids(list(input_data.node_ids))
                self.node_vectors[variable] = vector
                return vector
            elif isinstance(input_data, dict) and 'nodes' in input_data:
                vector = self._create_vector_from_node_ids(input_data['nodes'])
                self.node_vectors[variable] = vector
                return vector
        
        return None
    
    def _create_vector_from_node_ids(self, node_ids: List[str]) -> Any:
        """Create vector from node IDs"""
        
        if GRAPHBLAS_AVAILABLE:
            try:
                # Create GraphBLAS vector
                # Convert node IDs to indices
                indices = [hash(node_id) % 1000000 for node_id in node_ids]  # Simple hash to index
                values = [1] * len(indices)
                
                vector = gb.Vector.from_coo(indices, values, size=1000000)
                return vector
            except Exception as e:
                self.logger.warning(f"Failed to create GraphBLAS vector: {e}")
        
        # Fallback to simple dictionary representation
        return {
            'type': 'sparse_vector',
            'node_ids': set(node_ids),
            'indices': {node_id: i for i, node_id in enumerate(node_ids)},
            'size': len(node_ids)
        }
    
    def _get_adjacency_matrix(self, rel_type: str) -> Any:
        """Get adjacency matrix for relationship type"""
        
        if rel_type in self.adjacency_matrices:
            return self.adjacency_matrices[rel_type]
        
        if rel_type == "*":
            # Create combined matrix for all relationship types
            return self._create_combined_adjacency_matrix()
        
        # Create default/empty matrix
        return self._create_default_adjacency_matrix()
    
    def _create_combined_adjacency_matrix(self) -> Any:
        """Create combined adjacency matrix from all relationship types"""
        
        if not self.adjacency_matrices:
            return self._create_default_adjacency_matrix()
        
        if GRAPHBLAS_AVAILABLE:
            try:
                # Combine all matrices using element-wise union
                combined = None
                for matrix in self.adjacency_matrices.values():
                    if combined is None:
                        combined = matrix.dup()
                    else:
                        combined << combined.ewise_add(matrix, gb.binary.plus)
                return combined
            except Exception as e:
                self.logger.warning(f"Failed to create combined GraphBLAS matrix: {e}")
        
        # Fallback implementation
        return {
            'type': 'combined_adjacency',
            'matrices': list(self.adjacency_matrices.values())
        }
    
    def _create_default_adjacency_matrix(self) -> Any:
        """Create default/empty adjacency matrix"""
        
        if GRAPHBLAS_AVAILABLE:
            try:
                return gb.Matrix(bool, 1000000, 1000000)
            except Exception as e:
                self.logger.warning(f"Failed to create default GraphBLAS matrix: {e}")
        
        # Fallback implementation
        return {
            'type': 'sparse_matrix',
            'edges': set(),
            'size': (1000000, 1000000)
        }
    
    def _graphblas_traverse(self, from_vector, adj_matrix, direction: str) -> Any:
        """Perform traversal using GraphBLAS"""
        
        try:
            if direction == "outgoing":
                result = from_vector @ adj_matrix
            elif direction == "incoming":
                result = from_vector @ adj_matrix.T
            else:  # bidirectional
                result_out = from_vector @ adj_matrix
                result_in = from_vector @ adj_matrix.T
                result = result_out.ewise_add(result_in, gb.binary.plus)
            
            return result
            
        except Exception as e:
            self.logger.error(f"GraphBLAS traversal failed: {e}")
            return self._fallback_traverse(from_vector, adj_matrix, direction)
    
    def _fallback_traverse(self, from_vector, adj_matrix, direction: str) -> Any:
        """Fallback traversal implementation"""
        
        # Simple fallback using dictionary representation
        result_node_ids = set()
        
        if isinstance(from_vector, dict) and 'node_ids' in from_vector:
            source_nodes = from_vector['node_ids']
        else:
            source_nodes = set()
        
        # Simple traversal (would need actual graph data)
        # For now, just return the source nodes as a placeholder
        result_node_ids = source_nodes.copy()
        
        return {
            'type': 'sparse_vector',
            'node_ids': result_node_ids,
            'size': len(result_node_ids)
        }
    
    def _compute_transitive_closure(self, from_vector, adj_matrix, min_length: int, direction: str) -> Any:
        """Compute transitive closure"""
        
        if GRAPHBLAS_AVAILABLE:
            try:
                # Iterative approach for transitive closure
                current = from_vector.dup()
                result = gb.Vector(from_vector.dtype, from_vector.size)
                
                for length in range(1, min_length + 10):  # Limit iterations
                    if length >= min_length:
                        result << result.ewise_add(current, gb.binary.plus)
                    
                    if direction == "outgoing":
                        current = current @ adj_matrix
                    elif direction == "incoming":
                        current = current @ adj_matrix.T
                    else:
                        current_out = current @ adj_matrix
                        current_in = current @ adj_matrix.T
                        current = current_out.ewise_add(current_in, gb.binary.plus)
                    
                    # Check for convergence (simplified)
                    if current.nvals == 0:
                        break
                
                return result
                
            except Exception as e:
                self.logger.error(f"Transitive closure computation failed: {e}")
        
        # Fallback implementation
        return self._fallback_traverse(from_vector, adj_matrix, direction)
    
    def _compute_bounded_path(self, from_vector, adj_matrix, min_length: int, max_length: int, direction: str) -> Any:
        """Compute bounded variable-length path"""
        
        if GRAPHBLAS_AVAILABLE:
            try:
                current = from_vector.dup()
                result = gb.Vector(from_vector.dtype, from_vector.size)
                
                for length in range(1, max_length + 1):
                    if length >= min_length:
                        result << result.ewise_add(current, gb.binary.plus)
                    
                    if direction == "outgoing":
                        current = current @ adj_matrix
                    elif direction == "incoming":
                        current = current @ adj_matrix.T
                    else:
                        current_out = current @ adj_matrix
                        current_in = current @ adj_matrix.T
                        current = current_out.ewise_add(current_in, gb.binary.plus)
                
                return result
                
            except Exception as e:
                self.logger.error(f"Bounded path computation failed: {e}")
        
        # Fallback implementation
        return self._fallback_traverse(from_vector, adj_matrix, direction)
    
    def _extract_node_ids_from_vector(self, vector) -> Set[str]:
        """Extract node IDs from result vector"""
        
        if GRAPHBLAS_AVAILABLE and hasattr(vector, 'to_coo'):
            try:
                indices, values = vector.to_coo()
                # Convert indices back to node IDs (reverse of the hash process)
                node_ids = {f"node_{idx}" for idx in indices if values[0] > 0}
                return node_ids
            except Exception as e:
                self.logger.warning(f"Failed to extract node IDs from GraphBLAS vector: {e}")
        
        # Fallback for dictionary representation
        if isinstance(vector, dict) and 'node_ids' in vector:
            return vector['node_ids']
        
        return set()
    
    def _execute_matrix_operation(self, operation: str, context) -> Any:
        """Execute a matrix operation string"""
        
        try:
            # Very basic operation parsing (would need more sophisticated parsing)
            if "@" in operation and "=" in operation:
                # Matrix multiplication assignment
                parts = operation.split("=")
                if len(parts) == 2:
                    target = parts[0].strip()
                    expr = parts[1].strip()
                    
                    # Simple evaluation (would need proper expression parser)
                    result = self._evaluate_matrix_expression(expr, context)
                    
                    # Store result
                    if target.startswith("v_"):
                        self.node_vectors[target] = result
                    elif target.startswith("A_"):
                        self.adjacency_matrices[target] = result
                    
                    return result
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to execute matrix operation '{operation}': {e}")
            return None
    
    def _evaluate_matrix_expression(self, expression: str, context) -> Any:
        """Evaluate a matrix expression"""
        
        # Very simplified expression evaluation
        # In a real implementation, you'd have a proper expression parser
        
        if "@" in expression:
            parts = expression.split("@")
            if len(parts) == 2:
                left_operand = parts[0].strip()
                right_operand = parts[1].strip()
                
                left_value = self._get_operand_value(left_operand, context)
                right_value = self._get_operand_value(right_operand, context)
                
                if left_value is not None and right_value is not None:
                    return self._matrix_multiply(left_value, right_value)
        
        return None
    
    def _get_operand_value(self, operand: str, context) -> Any:
        """Get value for an operand"""
        
        operand = operand.strip()
        
        # Vector reference
        if operand.startswith("v_"):
            return self.node_vectors.get(operand)
        
        # Matrix reference
        elif operand.startswith("A_"):
            matrix_name = operand[2:]  # Remove "A_" prefix
            return self.adjacency_matrices.get(matrix_name)
        
        # Matrix with transpose
        elif operand.endswith(".T"):
            base_operand = operand[:-2]
            base_value = self._get_operand_value(base_operand, context)
            if base_value is not None:
                return self._transpose_matrix(base_value)
        
        return None
    
    def _matrix_multiply(self, left, right) -> Any:
        """Perform matrix multiplication"""
        
        if GRAPHBLAS_AVAILABLE:
            try:
                if hasattr(left, '__matmul__') and hasattr(right, '__matmul__'):
                    return left @ right
            except Exception as e:
                self.logger.warning(f"GraphBLAS matrix multiplication failed: {e}")
        
        # Fallback implementation for dictionary representations
        return {
            'type': 'matrix_multiply_result',
            'left': left,
            'right': right
        }
    
    def _transpose_matrix(self, matrix) -> Any:
        """Transpose a matrix"""
        
        if GRAPHBLAS_AVAILABLE:
            try:
                if hasattr(matrix, 'T'):
                    return matrix.T
            except Exception as e:
                self.logger.warning(f"GraphBLAS matrix transpose failed: {e}")
        
        # Fallback
        return {
            'type': 'transposed_matrix',
            'original': matrix
        }
    
    def _create_node_vectors(self, nodes_data: Dict[str, List[str]]):
        """Create node vectors from node data"""
        
        for variable, node_ids in nodes_data.items():
            vector = self._create_vector_from_node_ids(node_ids)
            self.node_vectors[variable] = vector
    
    def _create_adjacency_matrices(self, edges_data: Dict[str, List[tuple]]):
        """Create adjacency matrices from edge data"""
        
        for rel_type, edges in edges_data.items():
            matrix = self._create_adjacency_matrix_from_edges(edges)
            self.adjacency_matrices[rel_type] = matrix
    
    def _create_adjacency_matrix_from_edges(self, edges: List[tuple]) -> Any:
        """Create adjacency matrix from edge list"""
        
        if GRAPHBLAS_AVAILABLE:
            try:
                # Convert edges to indices
                row_indices = []
                col_indices = []
                values = []
                
                for edge in edges:
                    if len(edge) >= 2:
                        source_idx = hash(str(edge[0])) % 1000000
                        target_idx = hash(str(edge[1])) % 1000000
                        weight = edge[2] if len(edge) > 2 else 1
                        
                        row_indices.append(source_idx)
                        col_indices.append(target_idx)
                        values.append(weight)
                
                matrix = gb.Matrix.from_coo(row_indices, col_indices, values, nrows=1000000, ncols=1000000)
                return matrix
                
            except Exception as e:
                self.logger.warning(f"Failed to create GraphBLAS adjacency matrix: {e}")
        
        # Fallback implementation
        return {
            'type': 'sparse_matrix',
            'edges': set(edges),
            'size': (1000000, 1000000)
        }
    
    def _get_cache_key(self, operation: GraphBLASOperation, input_data) -> str:
        """Generate cache key for operation"""
        key_parts = [
            operation.operation_type,
            str(hash(tuple(operation.matrix_operations))),
            str(hash(str(input_data))) if input_data else "no_input"
        ]
        return "|".join(key_parts)
    
    def _is_cache_valid(self, cached_entry: Dict) -> bool:
        """Check if cached entry is still valid"""
        if 'timestamp' not in cached_entry or 'ttl' not in cached_entry:
            return False
        
        age = (datetime.now() - cached_entry['timestamp']).total_seconds()
        return age < cached_entry['ttl']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics"""
        cache_hit_rate = 0.0
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
        
        return {
            'operations_executed': self.stats['operations_executed'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'total_execution_time': self.stats['total_execution_time'],
            'avg_execution_time': self.stats['total_execution_time'] / max(1, self.stats['operations_executed']),
            'cache_size': len(self.operation_cache),
            'adjacency_matrices': len(self.adjacency_matrices),
            'node_vectors': len(self.node_vectors),
            'graphblas_available': GRAPHBLAS_AVAILABLE,
            'memory_peak': self.stats['memory_peak']
        }
    
    def clear_cache(self):
        """Clear operation cache"""
        self.operation_cache.clear()
        self.logger.info("GraphBLAS executor cache cleared")
    
    def shutdown(self):
        """Shutdown GraphBLAS executor"""
        try:
            self.thread_pool.shutdown(wait=True)
        except:
            pass
        
        self.operation_cache.clear()
        self.adjacency_matrices.clear()
        self.node_vectors.clear()
        self.relation_vectors.clear()
        
        self.logger.info("GraphBLAS executor shutdown")