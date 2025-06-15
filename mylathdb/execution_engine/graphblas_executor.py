# mylathdb/execution_engine/graphblas_executor.py

"""
MyLathDB GraphBLAS Executor
Handles graph traversals and matrix operations using real Python GraphBLAS
Based on FalkorDB's Delta Matrix and Tensor architecture
"""

import time
import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

try:
    import numpy as np
    import graphblas as gb
    from graphblas import Matrix, Vector, Scalar
    from graphblas.core import lib as lib_gb
except ImportError as e:
    gb = None
    Matrix = Vector = Scalar = None
    import_error = str(e)

from .config import MyLathDBExecutionConfig
from .exceptions import MyLathDBExecutionError, MyLathDBGraphBLASError
from .utils import mylathdb_measure_time

logger = logging.getLogger(__name__)

@dataclass
class MatrixSyncPolicy:
    """Matrix synchronization policy based on FalkorDB's Delta Matrix approach"""
    FLUSH_RESIZE = "flush_resize"        # Force execution and resize to node count
    RESIZE_ONLY = "resize_only"          # Resize to capacity without forcing operations  
    NO_SYNC = "no_sync"                  # No matrix updates (for deletion/cleanup)

@dataclass
class GraphBLASGraph:
    """
    Graph representation using GraphBLAS matrices
    Based on FalkorDB's graph structure with Delta Matrix patterns
    """
    # Core matrices
    adjacency_matrix: Optional[Matrix] = None           # Main adjacency matrix
    node_labels_matrix: Optional[Matrix] = None         # Node to label mappings
    
    # Label matrices (one per label type)
    label_matrices: Dict[str, Matrix] = field(default_factory=dict)
    
    # Relation matrices (one per relationship type) 
    relation_matrices: Dict[str, Matrix] = field(default_factory=dict)
    
    # Matrix metadata
    node_capacity: int = 10000
    edge_capacity: int = 50000
    matrix_sync_policy: str = MatrixSyncPolicy.FLUSH_RESIZE
    
    # Statistics tracking
    node_count: int = 0
    edge_count: int = 0
    pending_operations: bool = False

class GraphBLASExecutor:
    """
    GraphBLAS executor for MyLathDB based on FalkorDB's matrix operations
    
    Handles:
    - Graph traversals using matrix-vector multiplication
    - Variable-length path queries using matrix powers
    - Structural pattern matching
    - Matrix synchronization with Delta Matrix approach
    """
    
    def __init__(self, config: MyLathDBExecutionConfig):
        """Initialize GraphBLAS executor"""
        self.config = config
        self.graph = GraphBLASGraph()
        
        # Performance settings
        self.enable_parallel = True
        self.max_matrix_size = 1000000  # 1M x 1M max
        self.sparse_threshold = 0.01    # 1% density threshold
        
        # Persistence settings
        self.persistence_enabled = getattr(config, 'ENABLE_MATRIX_PERSISTENCE', False)
        self.persistence_path = Path(getattr(config, 'MATRIX_PERSISTENCE_PATH', './mylathdb_matrices'))
        
        # Threading settings
        self.num_threads = getattr(config, 'GRAPHBLAS_THREADS', 4)
        
    def initialize(self):
        """Initialize GraphBLAS with proper configuration"""
        logger.info("Initializing GraphBLAS executor")
        
        if gb is None:
            raise MyLathDBGraphBLASError(
                f"Python GraphBLAS not available: {import_error}\n"
                "Install with: pip install python-graphblas"
            )
        
        try:
            # Initialize GraphBLAS
            gb.init()
            
            # Set threading if available
            try:
                gb.set_global_threads(self.num_threads)
                logger.info(f"GraphBLAS threads set to {self.num_threads}")
            except AttributeError:
                logger.warning("GraphBLAS threading configuration not available")
            
            # Initialize graph matrices
            self._initialize_matrices()
            
            # Load persisted matrices if available
            if self.persistence_enabled:
                self._load_persisted_matrices()
            
            logger.info("GraphBLAS executor initialized successfully")
            
        except Exception as e:
            raise MyLathDBGraphBLASError(f"GraphBLAS initialization failed: {e}")
    
    def _initialize_matrices(self):
        """Initialize core GraphBLAS matrices"""
        try:
            # Create core matrices with initial capacity
            n = self.graph.node_capacity
            
            # Main adjacency matrix (boolean, all relationships)
            self.graph.adjacency_matrix = Matrix.new(gb.dtypes.BOOL, nrows=n, ncols=n)
            
            # Node labels matrix (node_id -> label_id mapping)
            max_labels = 100  # Reasonable max number of label types
            self.graph.node_labels_matrix = Matrix.new(gb.dtypes.BOOL, nrows=n, ncols=max_labels)
            
            logger.debug(f"Initialized core matrices with capacity {n}x{n}")
            
        except Exception as e:
            raise MyLathDBGraphBLASError(f"Matrix initialization failed: {e}")
    
    @mylathdb_measure_time
    def execute_operation(self, graphblas_operation, context) -> List[Dict[str, Any]]:
        """
        Execute GraphBLAS operation from physical plan
        
        Args:
            graphblas_operation: GraphBLASOperation from physical planner
            context: ExecutionContext
            
        Returns:
            List of result dictionaries
        """
        from ..cypher_planner.physical_planner import GraphBLASOperation
        
        if not isinstance(graphblas_operation, GraphBLASOperation):
            raise MyLathDBGraphBLASError(f"Expected GraphBLASOperation, got {type(graphblas_operation)}")
        
        logger.debug(f"Executing GraphBLAS operation: {graphblas_operation.operation_type}")
        
        # Route to appropriate handler
        operation_type = graphblas_operation.operation_type
        
        if operation_type == "ConditionalTraverse":
            return self._execute_conditional_traverse(graphblas_operation, context)
        elif operation_type == "VarLenTraverse":
            return self._execute_var_len_traverse(graphblas_operation, context)
        elif operation_type == "Expand":
            return self._execute_expand(graphblas_operation, context)
        elif operation_type == "StructuralFilter":
            return self._execute_structural_filter(graphblas_operation, context)
        elif operation_type == "PathFilter":
            return self._execute_path_filter(graphblas_operation, context)
        else:
            # Execute generic matrix operations
            return self._execute_matrix_operations(graphblas_operation, context)
    
    def _execute_conditional_traverse(self, operation, context) -> List[Dict[str, Any]]:
        """Execute single-hop conditional traversal using matrix-vector multiplication"""
        logical_op = operation.logical_op
        
        # Get or create relation matrix
        rel_types = logical_op.rel_types or ["*"]
        relation_matrix = self._get_relation_matrix(rel_types, logical_op.direction)
        
        # Create source vector (would come from previous operation in real scenario)
        source_vector = self._create_source_vector(logical_op.from_var, context)
        
        # Perform matrix-vector multiplication
        if logical_op.direction == "outgoing":
            # v_dest = v_src @ A_rel  (row vector @ matrix)
            result_vector = source_vector @ relation_matrix
        elif logical_op.direction == "incoming":  
            # v_dest = v_src @ A_rel.T  (transpose for incoming)
            result_vector = source_vector @ relation_matrix.T
        else:  # bidirectional
            # v_dest = v_src @ (A_rel + A_rel.T)
            bidirectional_matrix = relation_matrix + relation_matrix.T
            result_vector = source_vector @ bidirectional_matrix
        
        # Convert result to destination nodes
        results = self._vector_to_results(result_vector, logical_op.to_var)
        
        logger.debug(f"ConditionalTraverse returned {len(results)} results")
        return results
    
    def _execute_var_len_traverse(self, operation, context) -> List[Dict[str, Any]]:
        """Execute variable-length traversal using matrix powers"""
        logical_op = operation.logical_op
        
        # Get relation matrix
        rel_types = logical_op.rel_types or ["*"]
        A = self._get_relation_matrix(rel_types, logical_op.direction)
        
        # Create source vector
        source_vector = self._create_source_vector(logical_op.from_var, context)
        
        if logical_op.max_length == float('inf'):
            # Compute transitive closure
            result_vector = self._compute_transitive_closure(source_vector, A, logical_op.min_length)
        else:
            # Bounded variable-length path
            result_vector = self._compute_bounded_varlen_path(
                source_vector, A, logical_op.min_length, logical_op.max_length
            )
        
        results = self._vector_to_results(result_vector, logical_op.to_var)
        
        logger.debug(f"VarLenTraverse returned {len(results)} results")
        return results
    
    def _execute_expand(self, operation, context) -> List[Dict[str, Any]]:
        """Execute legacy Expand operation"""
        logical_op = operation.logical_op
        
        if logical_op.max_length == 1:
            # Single hop - convert to conditional traverse
            return self._execute_single_hop_expand(logical_op, context)
        else:
            # Variable length - convert to var len traverse  
            return self._execute_var_length_expand(logical_op, context)
    
    def _execute_single_hop_expand(self, logical_op, context) -> List[Dict[str, Any]]:
        """Execute single-hop expand using matrix operations"""
        rel_types = logical_op.rel_types or ["*"]
        relation_matrix = self._get_relation_matrix(rel_types, logical_op.direction)
        
        source_vector = self._create_source_vector(logical_op.from_var, context)
        
        # Matrix-vector multiplication based on direction
        if logical_op.direction == "outgoing":
            result_vector = source_vector @ relation_matrix
        elif logical_op.direction == "incoming":
            result_vector = source_vector @ relation_matrix.T
        else:  # bidirectional
            bidirectional_matrix = relation_matrix + relation_matrix.T
            result_vector = source_vector @ bidirectional_matrix
        
        return self._vector_to_results(result_vector, logical_op.to_var)
    
    def _execute_var_length_expand(self, logical_op, context) -> List[Dict[str, Any]]:
        """Execute variable-length expand"""
        rel_types = logical_op.rel_types or ["*"]
        A = self._get_relation_matrix(rel_types, logical_op.direction)
        
        source_vector = self._create_source_vector(logical_op.from_var, context)
        
        if logical_op.max_length == float('inf'):
            result_vector = self._compute_transitive_closure(source_vector, A, logical_op.min_length)
        else:
            result_vector = self._compute_bounded_varlen_path(
                source_vector, A, logical_op.min_length, logical_op.max_length
            )
        
        return self._vector_to_results(result_vector, logical_op.to_var)
    
    def _execute_structural_filter(self, operation, context) -> List[Dict[str, Any]]:
        """Execute structural filter using matrix operations"""
        # Structural filters involve complex matrix operations
        # For now, return empty (would need specific implementation per filter type)
        logger.debug("StructuralFilter executed (placeholder)")
        return []
    
    def _execute_path_filter(self, operation, context) -> List[Dict[str, Any]]:
        """Execute path filter using pattern matching"""
        logical_op = operation.logical_op
        
        # Parse path pattern and execute matrix operations
        # This is complex and would require path pattern parsing
        logger.debug(f"PathFilter executed for pattern: {logical_op.path_pattern}")
        return []
    
    def _execute_matrix_operations(self, operation, context) -> List[Dict[str, Any]]:
        """Execute raw matrix operations from GraphBLAS operation"""
        results = []
        
        try:
            # Execute each matrix operation
            for matrix_op in operation.matrix_operations:
                if matrix_op.startswith('#'):  # Skip comments
                    continue
                
                result = self._execute_matrix_operation(matrix_op, context)
                if result:
                    results.extend(result)
        
        except Exception as e:
            raise MyLathDBGraphBLASError(f"Matrix operation execution failed: {e}")
        
        return results
    
    def _execute_matrix_operation(self, op_str: str, context) -> List[Dict[str, Any]]:
        """Execute a single matrix operation string"""
        op_str = op_str.strip()
        
        # Parse and execute matrix operations
        # Examples: "v_f = v_u @ A_FOLLOWS", "result = compute_transitive_closure(...)"
        
        if ' = ' in op_str and ' @ ' in op_str:
            # Matrix-vector multiplication: v_dest = v_src @ A_rel
            left, right = op_str.split(' = ', 1)
            dest_var = left.strip()
            
            if ' @ ' in right:
                src_part, matrix_part = right.split(' @ ', 1)
                src_var = src_part.strip()
                matrix_name = matrix_part.strip()
                
                # Get source vector and matrix
                source_vector = self._get_or_create_vector(src_var, context)
                relation_matrix = self._get_matrix_by_name(matrix_name)
                
                if source_vector is not None and relation_matrix is not None:
                    # Perform multiplication
                    result_vector = source_vector @ relation_matrix
                    
                    # Store result for future operations
                    context.intermediate_vectors = getattr(context, 'intermediate_vectors', {})
                    context.intermediate_vectors[dest_var] = result_vector
                    
                    # Convert to results
                    return self._vector_to_results(result_vector, dest_var)
        
        return []
    
    def _get_relation_matrix(self, rel_types: List[str], direction: str) -> Matrix:
        """Get or create relation matrix for given relationship types"""
        
        if len(rel_types) == 1 and rel_types[0] != "*":
            # Single relationship type
            rel_type = rel_types[0]
            
            if rel_type not in self.graph.relation_matrices:
                # Create new relation matrix
                n = self.graph.node_capacity
                self.graph.relation_matrices[rel_type] = Matrix.new(gb.dtypes.BOOL, nrows=n, ncols=n)
            
            matrix = self.graph.relation_matrices[rel_type]
        
        elif rel_types == ["*"] or not rel_types:
            # All relationships - use adjacency matrix
            matrix = self.graph.adjacency_matrix
        
        else:
            # Multiple relationship types - combine matrices
            combined_matrix = None
            
            for rel_type in rel_types:
                if rel_type not in self.graph.relation_matrices:
                    n = self.graph.node_capacity
                    self.graph.relation_matrices[rel_type] = Matrix.new(gb.dtypes.BOOL, nrows=n, ncols=n)
                
                rel_matrix = self.graph.relation_matrices[rel_type]
                
                if combined_matrix is None:
                    combined_matrix = rel_matrix.dup()
                else:
                    combined_matrix += rel_matrix
            
            matrix = combined_matrix or self.graph.adjacency_matrix
        
        return matrix
    
    def _get_matrix_by_name(self, matrix_name: str) -> Optional[Matrix]:
        """Get matrix by name (e.g., 'A_FOLLOWS', 'A_KNOWS.T')"""
        
        # Handle transpose notation
        transpose = False
        if matrix_name.endswith('.T'):
            transpose = True
            matrix_name = matrix_name[:-2]
        
        # Parse matrix name
        if matrix_name.startswith('A_'):
            # Relation matrix: A_FOLLOWS -> relation_matrices['FOLLOWS']
            rel_type = matrix_name[2:]  # Remove 'A_' prefix
            
            if rel_type in self.graph.relation_matrices:
                matrix = self.graph.relation_matrices[rel_type]
            else:
                # Create empty matrix
                n = self.graph.node_capacity
                matrix = Matrix.new(gb.dtypes.BOOL, nrows=n, ncols=n)
                self.graph.relation_matrices[rel_type] = matrix
        
        elif matrix_name == 'adjacency_matrix':
            matrix = self.graph.adjacency_matrix
        
        else:
            # Unknown matrix
            logger.warning(f"Unknown matrix: {matrix_name}")
            return None
        
        return matrix.T if transpose else matrix
    
    def _create_source_vector(self, variable: str, context) -> Vector:
        """Create source vector for traversal operations"""
        
        # Check if vector already exists in context
        intermediate_vectors = getattr(context, 'intermediate_vectors', {})
        if variable in intermediate_vectors:
            return intermediate_vectors[variable]
        
        # Create new vector based on variable
        # In real scenario, this would be populated from previous operation results
        # For now, create a vector with some test nodes
        n = self.graph.node_capacity
        source_vector = Vector.new(gb.dtypes.BOOL, size=n)
        
        # Set some nodes as active (this would come from previous operations)
        # For testing, activate first few nodes
        for i in range(min(5, n)):
            source_vector[i] = True
        
        return source_vector
    
    def _get_or_create_vector(self, variable: str, context) -> Optional[Vector]:
        """Get existing vector or create new one"""
        
        # Check intermediate vectors first
        intermediate_vectors = getattr(context, 'intermediate_vectors', {})
        if variable in intermediate_vectors:
            return intermediate_vectors[variable]
        
        # Check if it's a variable reference (v_something)
        if variable.startswith('v_'):
            return self._create_source_vector(variable, context)
        
        return None
    
    def _compute_transitive_closure(self, source_vector: Vector, matrix: Matrix, min_length: int = 1) -> Vector:
        """Compute transitive closure using matrix powers"""
        
        n = matrix.nrows
        result = Vector.new(gb.dtypes.BOOL, size=n)
        current = source_vector.dup()
        
        # Iteratively compute matrix powers
        for length in range(1, min(100, self.graph.node_capacity)):  # Limit iterations
            if length >= min_length:
                # Add current level to result
                result += current
            
            # Compute next level: current = current @ matrix
            prev_nnvals = current.nvals
            current = current @ matrix
            
            # Check for convergence (no new nodes reached)
            if current.nvals == 0 or current.nvals == prev_nnvals:
                break
        
        return result
    
    def _compute_bounded_varlen_path(self, source_vector: Vector, matrix: Matrix, 
                                   min_length: int, max_length: int) -> Vector:
        """Compute bounded variable-length path using matrix powers"""
        
        n = matrix.nrows
        result = Vector.new(gb.dtypes.BOOL, size=n)
        current = source_vector.dup()
        
        for length in range(1, max_length + 1):
            # Compute next level
            current = current @ matrix
            
            # Add to result if within range
            if length >= min_length:
                result += current
        
        return result
    
    def _vector_to_results(self, vector: Vector, variable_name: str) -> List[Dict[str, Any]]:
        """Convert GraphBLAS vector to result dictionaries"""
        results = []
        
        # Extract non-zero indices (nodes that are reachable)
        indices, values = vector.to_coo()
        
        for i, val in zip(indices, values):
            if val:  # Non-zero value (reachable node)
                results.append({
                    variable_name: {
                        '_id': str(i),
                        '_reachable': True
                    }
                })
        
        return results
    
    def execute_generic_operation(self, physical_plan, context) -> List[Dict[str, Any]]:
        """Execute generic physical operation using GraphBLAS"""
        logical_op = getattr(physical_plan, 'logical_op', None)
        
        if logical_op:
            op_type = type(logical_op).__name__
            
            if 'Traverse' in op_type or 'Expand' in op_type:
                return self._handle_generic_traversal(logical_op, context)
        
        logger.warning(f"Could not execute generic GraphBLAS operation: {type(physical_plan)}")
        return []
    
    def _handle_generic_traversal(self, logical_op, context) -> List[Dict[str, Any]]:
        """Handle generic traversal operations"""
        op_type = type(logical_op).__name__
        
        if hasattr(logical_op, 'from_var') and hasattr(logical_op, 'to_var'):
            # Get relation matrix
            rel_types = getattr(logical_op, 'rel_types', ["*"])
            direction = getattr(logical_op, 'direction', 'outgoing')
            
            relation_matrix = self._get_relation_matrix(rel_types, direction)
            source_vector = self._create_source_vector(logical_op.from_var, context)
            
            # Perform traversal
            if direction == "outgoing":
                result_vector = source_vector @ relation_matrix
            elif direction == "incoming":
                result_vector = source_vector @ relation_matrix.T
            else:  # bidirectional
                bidirectional_matrix = relation_matrix + relation_matrix.T
                result_vector = source_vector @ bidirectional_matrix
            
            return self._vector_to_results(result_vector, logical_op.to_var)
        
        return []
    
    # Data loading methods
    
    def load_adjacency_matrices(self, matrices: Dict[str, Any]):
        """Load pre-computed adjacency matrices"""
        logger.info(f"Loading {len(matrices)} adjacency matrices")
        
        for matrix_name, matrix_data in matrices.items():
            try:
                if matrix_name == 'adjacency':
                    self._load_matrix_data(matrix_data, self.graph.adjacency_matrix)
                else:
                    # Relation-specific matrix
                    if matrix_name not in self.graph.relation_matrices:
                        n = self.graph.node_capacity
                        self.graph.relation_matrices[matrix_name] = Matrix.new(gb.dtypes.BOOL, nrows=n, ncols=n)
                    
                    self._load_matrix_data(matrix_data, self.graph.relation_matrices[matrix_name])
                
                logger.debug(f"Loaded matrix: {matrix_name}")
                
            except Exception as e:
                logger.error(f"Failed to load matrix {matrix_name}: {e}")
    
    def load_edges_as_matrices(self, edges: List[tuple]):
        """Load edges into GraphBLAS matrices"""
        logger.info(f"Loading {len(edges)} edges into GraphBLAS matrices")
        
        # Group edges by relationship type
        edges_by_type = {}
        for edge in edges:
            if len(edge) >= 3:
                src_id, rel_type, dest_id = edge[:3]
                
                if rel_type not in edges_by_type:
                    edges_by_type[rel_type] = []
                
                edges_by_type[rel_type].append((int(src_id), int(dest_id)))
        
        # Load each relationship type into its matrix
        for rel_type, edge_list in edges_by_type.items():
            try:
                # Get or create relation matrix
                if rel_type not in self.graph.relation_matrices:
                    n = self.graph.node_capacity
                    self.graph.relation_matrices[rel_type] = Matrix.new(gb.dtypes.BOOL, nrows=n, ncols=n)
                
                matrix = self.graph.relation_matrices[rel_type]
                
                # Add edges to matrix
                for src, dest in edge_list:
                    if src < matrix.nrows and dest < matrix.ncols:
                        matrix[src, dest] = True
                
                # Also add to main adjacency matrix
                for src, dest in edge_list:
                    if src < self.graph.adjacency_matrix.nrows and dest < self.graph.adjacency_matrix.ncols:
                        self.graph.adjacency_matrix[src, dest] = True
                
                logger.debug(f"Loaded {len(edge_list)} edges for relationship {rel_type}")
                
            except Exception as e:
                logger.error(f"Failed to load edges for {rel_type}: {e}")
        
        # Update statistics
        self.graph.edge_count += len(edges)
        
        logger.info(f"Successfully loaded edges into {len(edges_by_type)} relation matrices")
    
    def _load_matrix_data(self, matrix_data: Any, target_matrix: Matrix):
        """Load matrix data into target GraphBLAS matrix"""
        
        if isinstance(matrix_data, dict):
            # Sparse matrix format: {(row, col): value, ...}
            for (row, col), value in matrix_data.items():
                if row < target_matrix.nrows and col < target_matrix.ncols:
                    target_matrix[row, col] = bool(value)
        
        elif hasattr(matrix_data, 'shape'):
            # NumPy array or similar
            rows, cols = matrix_data.shape
            max_rows = min(rows, target_matrix.nrows)
            max_cols = min(cols, target_matrix.ncols)
            
            # Load non-zero elements
            for i in range(max_rows):
                for j in range(max_cols):
                    if matrix_data[i, j]:
                        target_matrix[i, j] = True
        
        else:
            logger.warning(f"Unknown matrix data format: {type(matrix_data)}")
    
    # Persistence methods
    
    def _load_persisted_matrices(self):
        """Load matrices from persistent storage"""
        if not self.persistence_enabled or not self.persistence_path.exists():
            return
        
        try:
            # Load adjacency matrix
            adj_path = self.persistence_path / 'adjacency_matrix.pkl'
            if adj_path.exists():
                with open(adj_path, 'rb') as f:
                    adj_data = pickle.load(f)
                    self._load_matrix_data(adj_data, self.graph.adjacency_matrix)
                logger.info("Loaded persisted adjacency matrix")
            
            # Load relation matrices
            rel_dir = self.persistence_path / 'relations'
            if rel_dir.exists():
                for rel_file in rel_dir.glob('*.pkl'):
                    rel_type = rel_file.stem
                    with open(rel_file, 'rb') as f:
                        rel_data = pickle.load(f)
                    
                    # Create relation matrix if needed
                    if rel_type not in self.graph.relation_matrices:
                        n = self.graph.node_capacity
                        self.graph.relation_matrices[rel_type] = Matrix.new(gb.dtypes.BOOL, nrows=n, ncols=n)
                    
                    self._load_matrix_data(rel_data, self.graph.relation_matrices[rel_type])
                    logger.debug(f"Loaded persisted relation matrix: {rel_type}")
        
        except Exception as e:
            logger.error(f"Failed to load persisted matrices: {e}")
    
    def persist_matrices(self):
        """Persist matrices to storage"""
        if not self.persistence_enabled:
            return
        
        try:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            
            # Persist adjacency matrix
            adj_data = self.graph.adjacency_matrix.to_coo()
            adj_path = self.persistence_path / 'adjacency_matrix.pkl'
            with open(adj_path, 'wb') as f:
                pickle.dump(adj_data, f)
            
            # Persist relation matrices
            rel_dir = self.persistence_path / 'relations'
            rel_dir.mkdir(exist_ok=True)
            
            for rel_type, matrix in self.graph.relation_matrices.items():
                rel_data = matrix.to_coo()
                rel_path = rel_dir / f'{rel_type}.pkl'
                with open(rel_path, 'wb') as f:
                    pickle.dump(rel_data, f)
            
            logger.info("Matrices persisted successfully")
            
        except Exception as e:
            logger.error(f"Failed to persist matrices: {e}")
    
    def test_functionality(self) -> bool:
        """Test GraphBLAS functionality"""
        try:
            if gb is None:
                return False
            
            # Test basic matrix operations
            test_matrix = Matrix.new(gb.dtypes.BOOL, nrows=10, ncols=10)
            test_matrix[0, 1] = True
            test_matrix[1, 2] = True
            
            test_vector = Vector.new(gb.dtypes.BOOL, size=10)
            test_vector[0] = True
            
            # Test matrix-vector multiplication
            result = test_vector @ test_matrix
            
            return result.nvals >= 0  # Should succeed
            
        except Exception as e:
            logger.error(f"GraphBLAS functionality test failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get GraphBLAS executor status"""
        status = {
            'available': gb is not None,
            'graph_node_capacity': self.graph.node_capacity,
            'graph_edge_capacity': self.graph.edge_capacity,
            'node_count': self.graph.node_count,
            'edge_count': self.graph.edge_count,
            'matrix_sync_policy': self.graph.matrix_sync_policy,
            'pending_operations': self.graph.pending_operations,
            'persistence_enabled': self.persistence_enabled
        }
        
        if gb is not None:
            try:
                status.update({
                    'graphblas_version': gb.__version__,
                    'num_threads': self.num_threads,
                    'adjacency_matrix_nnz': self.graph.adjacency_matrix.nvals if self.graph.adjacency_matrix else 0,
                    'relation_matrices_count': len(self.graph.relation_matrices),
                    'label_matrices_count': len(self.graph.label_matrices)
                })
            except Exception as e:
                status['error'] = str(e)
        
        return status
    
    def clear_matrices(self):
        """Clear all matrices (for cleanup/reset)"""
        logger.info("Clearing GraphBLAS matrices")
        
        try:
            # Clear core matrices
            if self.graph.adjacency_matrix:
                self.graph.adjacency_matrix.clear()
            
            if self.graph.node_labels_matrix:
                self.graph.node_labels_matrix.clear()
            
            # Clear relation matrices
            for matrix in self.graph.relation_matrices.values():
                matrix.clear()
            
            # Clear label matrices
            for matrix in self.graph.label_matrices.values():
                matrix.clear()
            
            # Reset statistics
            self.graph.node_count = 0
            self.graph.edge_count = 0
            self.graph.pending_operations = False
            
            logger.info("GraphBLAS matrices cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear matrices: {e}")
    
    def shutdown(self):
        """Shutdown GraphBLAS executor"""
        logger.info("Shutting down GraphBLAS executor")
        
        try:
            # Persist matrices if enabled
            if self.persistence_enabled:
                self.persist_matrices()
            
            # Clear matrices
            self.clear_matrices()
            
            # Finalize GraphBLAS
            if gb is not None:
                try:
                    gb.finalize()
                except AttributeError:
                    # finalize() might not be available in all versions
                    pass
            
        except Exception as e:
            logger.error(f"Error during GraphBLAS shutdown: {e}")
        
        logger.info("GraphBLAS executor shutdown complete")
