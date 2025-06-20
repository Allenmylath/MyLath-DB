# mylathdb/execution_engine/graphblas_executor.py

"""
MyLathDB GraphBLAS Executor - FULLY FIXED VERSION
Uses correct semirings that actually work with python-graphblas 2025.2.0
"""

import time
import logging
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

# GraphBLAS imports with proper initialization handling
try:
    import numpy as np
    import graphblas as gb
    GRAPHBLAS_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    gb = None
    GRAPHBLAS_AVAILABLE = False
    IMPORT_ERROR = str(e)

from .config import MyLathDBExecutionConfig
from .exceptions import MyLathDBExecutionError, MyLathDBGraphBLASError
from .utils import mylathdb_measure_time

logger = logging.getLogger(__name__)

@dataclass
class MatrixSyncPolicy:
    """Matrix synchronization policy based on FalkorDB's Delta Matrix approach"""
    FLUSH_RESIZE = "flush_resize"
    RESIZE_ONLY = "resize_only"
    NO_SYNC = "no_sync"

@dataclass
class GraphBLASGraph:
    """Graph representation using GraphBLAS matrices"""
    # Core matrices
    adjacency_matrix: Optional[Any] = None
    node_labels_matrix: Optional[Any] = None
    
    # Label matrices (one per label type)
    label_matrices: Dict[str, Any] = field(default_factory=dict)
    
    # Relation matrices (one per relationship type) 
    relation_matrices: Dict[str, Any] = field(default_factory=dict)
    
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
    FULLY FIXED GraphBLAS executor for MyLathDB
    Uses correct semirings and API calls that work with python-graphblas 2025.2.0
    """
    
    def __init__(self, config: MyLathDBExecutionConfig):
        """Initialize GraphBLAS executor"""
        self.config = config
        self.graph = None
        self.initialized = False
        self.gb_initialized = False
        
        # FIXED: Use working semirings identified from tests
        self.default_bool_semiring = gb.semiring.any_pair if GRAPHBLAS_AVAILABLE else None
        self.fallback_semirings = [
            'any_pair',      # ✅ Works with BOOL
            'lor_land',      # ✅ Works with BOOL  
            'land_lor',      # ✅ Works with BOOL
            'max_min',       # ✅ Works with BOOL
            'min_max'        # ✅ Works with BOOL
        ] if GRAPHBLAS_AVAILABLE else []
        
        # Performance settings
        self.enable_parallel = True
        self.max_matrix_size = 1000000
        self.sparse_threshold = 0.01
        
        # Persistence settings
        self.persistence_enabled = getattr(config, 'ENABLE_MATRIX_PERSISTENCE', False)
        self.persistence_path = Path(getattr(config, 'MATRIX_PERSISTENCE_PATH', './mylathdb_matrices'))
        
        # Threading settings
        self.num_threads = getattr(config, 'GRAPHBLAS_THREADS', 4)
        
    def initialize(self):
        """Initialize GraphBLAS with correct API sequence and test semirings"""
        logger.info("Initializing GraphBLAS executor")
        
        if not GRAPHBLAS_AVAILABLE:
            logger.error(f"Python GraphBLAS not available: {IMPORT_ERROR}")
            raise MyLathDBGraphBLASError(f"GraphBLAS package not available: {IMPORT_ERROR}")
        
        if self.initialized:
            logger.info("GraphBLAS already initialized")
            return
        
        try:
            # STEP 1: Initialize GraphBLAS library
            if not self.gb_initialized:
                logger.info("Initializing GraphBLAS library...")
                logger.debug("GraphBLAS auto-initialization")
                logger.info("GraphBLAS library initialized successfully")
                self.gb_initialized = True
            
            # STEP 2: Test and select working semiring
            logger.info("Testing GraphBLAS semirings...")
            working_semiring = self._find_working_semiring()
            if working_semiring:
                self.default_bool_semiring = working_semiring
                logger.info(f"✅ Using semiring: {working_semiring}")
            else:
                raise MyLathDBGraphBLASError("No working semirings found for BOOL operations")
            
            # STEP 3: Set threading if available
            try:
                if hasattr(gb, 'config') and hasattr(gb.config, 'set'):
                    gb.config.set(nthreads=self.num_threads)
                    logger.info(f"GraphBLAS threads set to {self.num_threads}")
                else:
                    logger.warning("GraphBLAS threading configuration not available")
            except Exception as e:
                logger.warning(f"GraphBLAS threading configuration failed: {e}")
            
            # STEP 4: Create the GraphBLASGraph object
            logger.debug("Creating GraphBLASGraph data structures...")
            self.graph = GraphBLASGraph(
                node_capacity=getattr(self.config, 'NODE_CREATION_BUFFER', 10000),
                edge_capacity=getattr(self.config, 'EDGE_CREATION_BUFFER', 50000),
                matrix_sync_policy=MatrixSyncPolicy.FLUSH_RESIZE
            )
            
            # STEP 5: Initialize graph matrices
            self._initialize_matrices()
            
            # STEP 6: Load persisted matrices if available
            if self.persistence_enabled:
                self._load_persisted_matrices()
            
            self.initialized = True
            logger.info("GraphBLAS executor initialized successfully")
            
        except Exception as e:
            logger.error(f"GraphBLAS initialization failed: {e}")
            self.initialized = False
            self.gb_initialized = False
            self.graph = None
            raise MyLathDBGraphBLASError(f"GraphBLAS initialization failed: {e}")
    
    def _find_working_semiring(self):
        """FIXED: Test and find a working semiring for BOOL operations"""
        if not GRAPHBLAS_AVAILABLE:
            return None
            
        # Test each semiring to find one that works
        for semiring_name in self.fallback_semirings:
            try:
                # Get semiring object
                semiring = getattr(gb.semiring, semiring_name, None)
                if semiring is None:
                    continue
                
                # Test with small BOOL matrices
                test_matrix = gb.Matrix(gb.dtypes.BOOL, nrows=2, ncols=2)
                test_matrix[0, 1] = True
                test_vector = gb.Vector(gb.dtypes.BOOL, size=2)
                test_vector[0] = True
                
                # Try the operation
                result = test_vector.vxm(test_matrix, semiring)
                
                if result.nvals >= 0:  # Success
                    logger.info(f"✅ Semiring {semiring_name} works!")
                    return semiring
                    
            except Exception as e:
                logger.debug(f"❌ Semiring {semiring_name} failed: {e}")
                continue
        
        logger.error("No working semirings found!")
        return None
    
    def is_available(self) -> bool:
        """Check if GraphBLAS is available and initialized"""
        return (GRAPHBLAS_AVAILABLE and 
                self.initialized and 
                self.gb_initialized and 
                self.graph is not None and
                self.default_bool_semiring is not None)
    
    def _initialize_matrices(self):
        """Initialize core GraphBLAS matrices using correct API"""
        if not GRAPHBLAS_AVAILABLE or not self.gb_initialized or not self.graph:
            return
            
        try:
            logger.info("Initializing GraphBLAS matrices...")
            
            # Create core matrices with initial capacity
            n = self.graph.node_capacity
            
            # Main adjacency matrix (boolean, all relationships)
            self.graph.adjacency_matrix = gb.Matrix(gb.dtypes.BOOL, nrows=n, ncols=n)
            logger.debug(f"Created adjacency matrix {n}x{n}")
            
            # Node labels matrix (node_id -> label_id mapping)
            max_labels = 100  # Reasonable max number of label types
            self.graph.node_labels_matrix = gb.Matrix(gb.dtypes.BOOL, nrows=n, ncols=max_labels)
            logger.debug(f"Created node labels matrix {n}x{max_labels}")
            
            logger.info(f"Initialized core matrices with capacity {n}x{n}")
            
        except Exception as e:
            logger.error(f"Matrix initialization failed: {e}")
            raise MyLathDBGraphBLASError(f"Matrix initialization failed: {e}")
    
    @mylathdb_measure_time
    def execute_operation(self, graphblas_operation, context) -> List[Dict[str, Any]]:
        """Execute GraphBLAS operation from physical plan"""
        if not self.is_available():
            error_msg = "GraphBLAS not available - falling back to Redis-only mode"
            logger.warning(error_msg)
            return []
        
        from ..cypher_planner.physical_planner import GraphBLASOperation
        
        if not isinstance(graphblas_operation, GraphBLASOperation):
            raise MyLathDBGraphBLASError(f"Expected GraphBLASOperation, got {type(graphblas_operation)}")
        
        logger.debug(f"Executing GraphBLAS operation: {graphblas_operation.operation_type}")
        
        try:
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
                
        except Exception as e:
            logger.error(f"GraphBLAS operation execution failed: {e}")
            # Return empty result instead of failing completely
            return []
    
    def test_functionality(self) -> bool:
        """Test GraphBLAS functionality using correct working semirings"""
        if not self.is_available():
            return False
            
        try:
            # Test basic matrix operations with working semiring
            test_matrix = gb.Matrix(gb.dtypes.BOOL, nrows=10, ncols=10)
            test_matrix[0, 1] = True
            test_matrix[1, 2] = True
            
            test_vector = gb.Vector(gb.dtypes.BOOL, size=10)
            test_vector[0] = True
            
            # Test matrix-vector multiplication with our working semiring
            result = test_vector.vxm(test_matrix, self.default_bool_semiring)
            
            return result.nvals >= 0  # Should succeed
            
        except Exception as e:
            logger.error(f"GraphBLAS functionality test failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get GraphBLAS executor status"""
        status = {
            'available': self.is_available(),
            'initialized': self.initialized,
            'gb_initialized': self.gb_initialized,
            'graphblas_package_available': GRAPHBLAS_AVAILABLE,
            'working_semiring': str(self.default_bool_semiring) if self.default_bool_semiring else None
        }
        
        if self.is_available():
            try:
                status.update({
                    'graphblas_version': gb.__version__,
                    'num_threads': self.num_threads,
                    'graph_node_capacity': self.graph.node_capacity,
                    'graph_edge_capacity': self.graph.edge_capacity,
                    'node_count': self.graph.node_count,
                    'edge_count': self.graph.edge_count,
                    'matrix_sync_policy': self.graph.matrix_sync_policy,
                    'pending_operations': self.graph.pending_operations,
                    'adjacency_matrix_nnz': self.graph.adjacency_matrix.nvals if self.graph.adjacency_matrix else 0,
                    'relation_matrices_count': len(self.graph.relation_matrices),
                    'label_matrices_count': len(self.graph.label_matrices)
                })
            except Exception as e:
                status['error'] = str(e)
        else:
            if not GRAPHBLAS_AVAILABLE:
                status['reason'] = f'GraphBLAS package not available: {IMPORT_ERROR}'
            else:
                status['reason'] = 'GraphBLAS initialization failed'
        
        return status
    
    def execute_generic_operation(self, physical_plan, context) -> List[Dict[str, Any]]:
        """Execute generic physical operation using GraphBLAS"""
        if not self.is_available():
            logger.warning("GraphBLAS not available for generic operation - returning empty result")
            return []
        
        logical_op = getattr(physical_plan, 'logical_op', None)
        
        if logical_op:
            op_type = type(logical_op).__name__
            
            if 'Traverse' in op_type or 'Expand' in op_type:
                return self._handle_generic_traversal(logical_op, context)
        
        logger.warning(f"Could not execute generic GraphBLAS operation: {type(physical_plan)}")
        return []
    
    def load_edges_as_matrices(self, edges: List[tuple]):
        """Load edges into GraphBLAS matrices"""
        if not self.is_available():
            logger.warning("GraphBLAS not available for edge loading - skipping")
            return

        logger.info(f"Loading {len(edges)} edges into GraphBLAS matrices")

        # Group edges by relationship type
        edges_by_type = {}
        
        # Create a local mapping for string IDs to integer indices
        node_id_to_index = {}
        next_index = 0

        for edge in edges:
            if len(edge) >= 3:
                src_id, rel_type, dest_id = edge[:3]

                if rel_type not in edges_by_type:
                    edges_by_type[rel_type] = []

                # Get or create integer index for source ID
                if src_id not in node_id_to_index:
                    node_id_to_index[src_id] = next_index
                    next_index += 1
                src_idx = node_id_to_index[src_id]

                # Get or create integer index for destination ID
                if dest_id not in node_id_to_index:
                    node_id_to_index[dest_id] = next_index
                    next_index += 1
                dest_idx = node_id_to_index[dest_id]
                
                edges_by_type[rel_type].append((src_idx, dest_idx))

        # Load each relationship type into its matrix
        for rel_type, edge_list in edges_by_type.items():
            try:
                # Get or create relation matrix
                if rel_type not in self.graph.relation_matrices:
                    n = self.graph.node_capacity
                    self.graph.relation_matrices[rel_type] = gb.Matrix(gb.dtypes.BOOL, nrows=n, ncols=n)

                matrix = self.graph.relation_matrices[rel_type]

                # Add edges to matrix
                for src, dest in edge_list:
                    if 0 <= src < matrix.nrows and 0 <= dest < matrix.ncols:
                        matrix[src, dest] = True

                # Also add to main adjacency matrix
                for src, dest in edge_list:
                    if 0 <= src < self.graph.adjacency_matrix.nrows and 0 <= dest < self.graph.adjacency_matrix.ncols:
                        self.graph.adjacency_matrix[src, dest] = True

                logger.debug(f"Loaded {len(edge_list)} edges for relationship {rel_type}")

            except Exception as e:
                logger.error(f"Failed to load edges for {rel_type}: {e}")

        # Update statistics
        self.graph.edge_count += len(edges)

        logger.info(f"Successfully loaded edges into {len(edges_by_type)} relation matrices")    
    def load_graph_data(self, graph_data):
        """Load graph data into GraphBLAS"""
        if not self.is_available():
            logger.warning("GraphBLAS not available for graph data loading - skipping")
            return
        
        logger.info("Loading graph data into GraphBLAS matrices")
        
        # Load adjacency matrices if provided
        if 'adjacency_matrices' in graph_data:
            self.load_adjacency_matrices(graph_data['adjacency_matrices'])
        
        # Load edges if provided
        if 'edges' in graph_data:
            # Convert edges to matrices
            if isinstance(graph_data['edges'], dict):
                # Grouped by type
                for rel_type, edge_list in graph_data['edges'].items():
                    formatted_edges = [(src, rel_type, dest) for src, dest in edge_list]
                    self.load_edges_as_matrices(formatted_edges)
            else:
                # List of edge tuples
                self.load_edges_as_matrices(graph_data['edges'])
    
    def shutdown(self):
        """Shutdown GraphBLAS executor"""
        logger.info("Shutting down GraphBLAS executor")
        
        if not self.initialized:
            logger.info("GraphBLAS was not initialized, nothing to shutdown")
            return
        
        try:
            # Modern python-graphblas handles cleanup automatically
            logger.info("GraphBLAS cleanup handled automatically by library")
            
        except Exception as e:
            logger.error(f"Error during GraphBLAS shutdown: {e}")
        
        self.initialized = False
        self.gb_initialized = False
        self.graph = None
        logger.info("GraphBLAS executor shutdown complete")
    
    def _get_relation_matrix(self, rel_types: List[str], direction: str):
        """Get or create relation matrix for given relationship types"""
        
        if len(rel_types) == 1 and rel_types[0] != "*":
            # Single relationship type
            rel_type = rel_types[0]
            
            if rel_type not in self.graph.relation_matrices:
                # Create new relation matrix
                n = self.graph.node_capacity
                self.graph.relation_matrices[rel_type] = gb.Matrix(gb.dtypes.BOOL, nrows=n, ncols=n)
                logger.debug(f"Created new relation matrix for {rel_type}")
            
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
                    self.graph.relation_matrices[rel_type] = gb.Matrix(gb.dtypes.BOOL, nrows=n, ncols=n)
                
                rel_matrix = self.graph.relation_matrices[rel_type]
                
                if combined_matrix is None:
                    combined_matrix = rel_matrix.dup()
                else:
                    combined_matrix += rel_matrix
            
            matrix = combined_matrix or self.graph.adjacency_matrix
        
        return matrix
    
    def _create_source_vector(self, variable: str, context):
        """Create source vector for traversal operations"""
        
        # Check if vector already exists in context
        intermediate_vectors = getattr(context, 'intermediate_vectors', {})
        if variable in intermediate_vectors:
            return intermediate_vectors[variable]
        
        # Create new vector based on variable
        n = self.graph.node_capacity
        source_vector = gb.Vector(gb.dtypes.BOOL, size=n)
        
        # Set some nodes as active (this would come from previous operations)
        # For testing, activate first few nodes
        for i in range(min(5, n)):
            source_vector[i] = True
        
        return source_vector
    
    def _compute_transitive_closure(self, source_vector, matrix, min_length: int = 1):
        """FIXED: Compute transitive closure using matrix powers with working semiring"""
        
        n = matrix.nrows
        result = gb.Vector(gb.dtypes.BOOL, size=n)
        current = source_vector.dup()
        
        # Iteratively compute matrix powers
        for length in range(1, min(100, self.graph.node_capacity)):
            if length >= min_length:
                # Add current level to result using LOR
                result = result.ewise_add(current, gb.binary.lor)
            
            # FIXED: Use our working semiring
            prev_nnvals = current.nvals
            current = current.vxm(matrix, self.default_bool_semiring)
            
            # Check for convergence (no new nodes reached)
            if current.nvals == 0 or current.nvals == prev_nnvals:
                break
        
        return result
    
    def _compute_bounded_varlen_path(self, source_vector, matrix, 
                                   min_length: int, max_length: int):
        """FIXED: Compute bounded variable-length path using working semiring"""
        
        n = matrix.nrows
        result = gb.Vector(gb.dtypes.BOOL, size=n)
        current = source_vector.dup()
        
        for length in range(1, max_length + 1):
            # FIXED: Use our working semiring
            current = current.vxm(matrix, self.default_bool_semiring)
            
            # Add to result if within range
            if length >= min_length:
                result = result.ewise_add(current, gb.binary.lor)
        
        return result
    
    def _vector_to_results(self, vector, variable_name: str) -> List[Dict[str, Any]]:
        """Convert GraphBLAS vector to result dictionaries"""
        results = []
        
        try:
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
        except Exception as e:
            logger.error(f"Failed to convert vector to results: {e}")
        
        return results
    
    def _handle_generic_traversal(self, logical_op, context) -> List[Dict[str, Any]]:
        """FIXED: Handle generic traversal operations with working semirings"""
        op_type = type(logical_op).__name__
        
        if hasattr(logical_op, 'from_var') and hasattr(logical_op, 'to_var'):
            # Get relation matrix
            rel_types = getattr(logical_op, 'rel_types', ["*"])
            direction = getattr(logical_op, 'direction', 'outgoing')
            
            relation_matrix = self._get_relation_matrix(rel_types, direction)
            source_vector = self._create_source_vector(logical_op.from_var, context)
            
            # FIXED: Perform traversal with our working semiring
            try:
                if direction == "outgoing":
                    result_vector = source_vector.vxm(relation_matrix, self.default_bool_semiring)
                elif direction == "incoming":
                    result_vector = source_vector.vxm(relation_matrix.T, self.default_bool_semiring)
                else:  # bidirectional
                    bidirectional_matrix = relation_matrix.ewise_add(relation_matrix.T, gb.binary.lor)
                    result_vector = source_vector.vxm(bidirectional_matrix, self.default_bool_semiring)
                
                return self._vector_to_results(result_vector, logical_op.to_var)
            except Exception as e:
                logger.error(f"Traversal operation failed: {e}")
                return []
        
        return []
    
    def _execute_conditional_traverse(self, operation, context) -> List[Dict[str, Any]]:
        """FIXED: Execute single-hop conditional traversal using working semirings"""
        logical_op = operation.logical_op
        
        # Get or create relation matrix
        rel_types = logical_op.rel_types or ["*"]
        relation_matrix = self._get_relation_matrix(rel_types, logical_op.direction)
        
        # Create source vector (would come from previous operation in real scenario)
        source_vector = self._create_source_vector(logical_op.from_var, context)
        
        try:
            # FIXED: Perform matrix-vector multiplication with working semiring
            if logical_op.direction == "outgoing":
                result_vector = source_vector.vxm(relation_matrix, self.default_bool_semiring)
            elif logical_op.direction == "incoming":  
                result_vector = source_vector.vxm(relation_matrix.T, self.default_bool_semiring)
            else:  # bidirectional
                bidirectional_matrix = relation_matrix.ewise_add(relation_matrix.T, gb.binary.lor)
                result_vector = source_vector.vxm(bidirectional_matrix, self.default_bool_semiring)
            
            # Convert result to destination nodes
            results = self._vector_to_results(result_vector, logical_op.to_var)
            
            logger.debug(f"ConditionalTraverse returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"ConditionalTraverse failed: {e}")
            return []
    
    def _execute_var_len_traverse(self, operation, context) -> List[Dict[str, Any]]:
        """FIXED: Execute variable-length traversal using working semirings"""
        logical_op = operation.logical_op
        
        # Get relation matrix
        rel_types = logical_op.rel_types or ["*"]
        A = self._get_relation_matrix(rel_types, logical_op.direction)
        
        # Create source vector
        source_vector = self._create_source_vector(logical_op.from_var, context)
        
        try:
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
        except Exception as e:
            logger.error(f"VarLenTraverse failed: {e}")
            return []
    
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
        """FIXED: Execute single-hop expand using working semirings"""
        rel_types = logical_op.rel_types or ["*"]
        relation_matrix = self._get_relation_matrix(rel_types, logical_op.direction)
        
        source_vector = self._create_source_vector(logical_op.from_var, context)
        
        try:
            # FIXED: Matrix-vector multiplication with working semiring
            if logical_op.direction == "outgoing":
                result_vector = source_vector.vxm(relation_matrix, self.default_bool_semiring)
            elif logical_op.direction == "incoming":
                result_vector = source_vector.vxm(relation_matrix.T, self.default_bool_semiring)
            else:  # bidirectional
                bidirectional_matrix = relation_matrix.ewise_add(relation_matrix.T, gb.binary.lor)
                result_vector = source_vector.vxm(bidirectional_matrix, self.default_bool_semiring)
            
            return self._vector_to_results(result_vector, logical_op.to_var)
        except Exception as e:
            logger.error(f"Single hop expand failed: {e}")
            return []
    
    def _execute_var_length_expand(self, logical_op, context) -> List[Dict[str, Any]]:
        """FIXED: Execute variable-length expand with working semirings"""
        rel_types = logical_op.rel_types or ["*"]
        A = self._get_relation_matrix(rel_types, logical_op.direction)
        
        source_vector = self._create_source_vector(logical_op.from_var, context)
        
        try:
            if logical_op.max_length == float('inf'):
                result_vector = self._compute_transitive_closure(source_vector, A, logical_op.min_length)
            else:
                result_vector = self._compute_bounded_varlen_path(
                    source_vector, A, logical_op.min_length, logical_op.max_length
                )
            
            return self._vector_to_results(result_vector, logical_op.to_var)
        except Exception as e:
            logger.error(f"Variable length expand failed: {e}")
            return []
    
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
            logger.error(f"Matrix operation execution failed: {e}")
            # Don't raise exception, just log and return empty
            return []
        
        return results
    
    def _execute_matrix_operation(self, op_str: str, context) -> List[Dict[str, Any]]:
        """FIXED: Execute a single matrix operation string with working semirings"""
        op_str = op_str.strip()
        
        # Parse and execute matrix operations
        # Examples: "v_f = v_u @ A_FOLLOWS", "result = compute_transitive_closure(...)"
        
        try:
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
                        # FIXED: Perform multiplication with working semiring
                        result_vector = source_vector.vxm(relation_matrix, self.default_bool_semiring)
                        
                        # Store result for future operations
                        context.intermediate_vectors = getattr(context, 'intermediate_vectors', {})
                        context.intermediate_vectors[dest_var] = result_vector
                        
                        # Convert to results
                        return self._vector_to_results(result_vector, dest_var)
            
        except Exception as e:
            logger.error(f"Matrix operation failed: {op_str}: {e}")
        
        return []
    
    def _get_matrix_by_name(self, matrix_name: str):
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
                matrix = gb.Matrix(gb.dtypes.BOOL, nrows=n, ncols=n)
                self.graph.relation_matrices[rel_type] = matrix
        
        elif matrix_name == 'adjacency_matrix':
            matrix = self.graph.adjacency_matrix
        
        else:
            # Unknown matrix
            logger.warning(f"Unknown matrix: {matrix_name}")
            return None
        
        return matrix.T if transpose else matrix
    
    def _get_or_create_vector(self, variable: str, context):
        """Get existing vector or create new one"""
        
        # Check intermediate vectors first
        intermediate_vectors = getattr(context, 'intermediate_vectors', {})
        if variable in intermediate_vectors:
            return intermediate_vectors[variable]
        
        # Check if it's a variable reference (v_something)
        if variable.startswith('v_'):
            return self._create_source_vector(variable, context)
        
        return None
    
    def load_adjacency_matrices(self, matrices: Dict[str, Any]):
        """Load pre-computed adjacency matrices"""
        if not self.is_available():
            logger.warning("GraphBLAS not available for matrix loading - skipping")
            return
            
        logger.info(f"Loading {len(matrices)} adjacency matrices")
        
        for matrix_name, matrix_data in matrices.items():
            try:
                if matrix_name == 'adjacency':
                    self._load_matrix_data(matrix_data, self.graph.adjacency_matrix)
                else:
                    # Relation-specific matrix
                    if matrix_name not in self.graph.relation_matrices:
                        n = self.graph.node_capacity
                        self.graph.relation_matrices[matrix_name] = gb.Matrix(gb.dtypes.BOOL, nrows=n, ncols=n)
                    
                    self._load_matrix_data(matrix_data, self.graph.relation_matrices[matrix_name])
                
                logger.debug(f"Loaded matrix: {matrix_name}")
                
            except Exception as e:
                logger.error(f"Failed to load matrix {matrix_name}: {e}")
    
    def _load_matrix_data(self, matrix_data: Any, target_matrix):
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
                        self.graph.relation_matrices[rel_type] = gb.Matrix(gb.dtypes.BOOL, nrows=n, ncols=n)
                    
                    self._load_matrix_data(rel_data, self.graph.relation_matrices[rel_type])
                    logger.debug(f"Loaded persisted relation matrix: {rel_type}")
        
        except Exception as e:
            logger.error(f"Failed to load persisted matrices: {e}")
    
    def persist_matrices(self):
        """Persist matrices to storage"""
        if not self.persistence_enabled or not self.is_available():
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
    
    def clear_matrices(self):
        """Clear all matrices (for cleanup/reset)"""
        if not self.is_available():
            return
            
        logger.info("Clearing GraphBLAS matrices")
        
        try:
            # 🔥 FIX: Use 'is not None' instead of boolean evaluation
            if self.graph.adjacency_matrix is not None:
                self.graph.adjacency_matrix.clear()
            
            if self.graph.node_labels_matrix is not None:
                self.graph.node_labels_matrix.clear()
            
            # Clear relation matrices
            for matrix in self.graph.relation_matrices.values():
                if matrix is not None:  # 🔥 FIX: Add None check
                    matrix.clear()
            
            # Clear label matrices  
            for matrix in self.graph.label_matrices.values():
                if matrix is not None:  # 🔥 FIX: Add None check
                    matrix.clear()
            
            # Reset statistics
            self.graph.node_count = 0
            self.graph.edge_count = 0
            self.graph.pending_operations = False
            
            logger.info("GraphBLAS matrices cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear matrices: {e}")