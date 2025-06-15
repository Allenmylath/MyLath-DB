# mylathdb/execution_engine/engine.py

"""
MyLathDB Core Execution Engine
Main execution engine that coordinates Redis + GraphBLAS execution
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from .config import MyLathDBExecutionConfig
from .redis_executor import RedisExecutor
from .graphblas_executor import GraphBLASExecutor
from .coordinator import ExecutionCoordinator
from .data_bridge import DataBridge
from .result_formatter import ResultFormatter
from .exceptions import MyLathDBExecutionError, MyLathDBTimeoutError
from .utils import mylathdb_measure_time

logger = logging.getLogger(__name__)

@dataclass
class ExecutionContext:
    """Context for query execution"""
    parameters: Dict[str, Any] = field(default_factory=dict)
    graph_data: Optional[Dict[str, Any]] = None
    execution_id: str = field(default_factory=lambda: f"exec_{int(time.time() * 1000)}")
    max_execution_time: float = 300.0  # 5 minutes default
    enable_parallel: bool = True
    cache_results: bool = True
    
@dataclass 
class ExecutionResult:
    """Result of executing a physical plan"""
    success: bool
    data: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    operations_executed: int = 0
    redis_operations: int = 0
    graphblas_operations: int = 0
    coordinator_operations: int = 0
    cache_hits: int = 0
    error: Optional[str] = None
    execution_id: Optional[str] = None
    execution_plan_summary: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    @property
    def row_count(self) -> int:
        """Number of result rows"""
        return len(self.data)
    
    def add_warning(self, message: str):
        """Add a warning message"""
        self.warnings.append(message)
        logger.warning(f"Execution warning: {message}")

class ExecutionEngine:
    """
    Main MyLathDB Execution Engine
    
    Coordinates execution between Redis (entities/properties) and 
    GraphBLAS (graph traversals) based on FalkorDB architecture.
    """
    
    def __init__(self, config: MyLathDBExecutionConfig = None):
        """
        Initialize MyLathDB execution engine
        
        Args:
            config: Configuration object
        """
        self.config = config or MyLathDBExecutionConfig()
        
        # Initialize components based on FalkorDB architecture
        self.redis_executor = RedisExecutor(self.config)
        self.graphblas_executor = GraphBLASExecutor(self.config)
        self.coordinator = ExecutionCoordinator(self.config)
        self.data_bridge = DataBridge(self.redis_executor, self.graphblas_executor)
        self.result_formatter = ResultFormatter()
        
        # Execution statistics
        self.total_queries_executed = 0
        self.total_execution_time = 0.0
        self.cache = {} if self.config.ENABLE_CACHING else None
        
        # Initialize systems
        self._initialize_systems()
        
        logger.info("MyLathDB Execution Engine initialized successfully")
    
    def _initialize_systems(self):
        """Initialize Redis and GraphBLAS systems"""
        try:
            # Initialize Redis
            self.redis_executor.initialize()
            
            # Initialize GraphBLAS
            self.graphblas_executor.initialize()
            
            # Test connectivity
            self._test_connectivity()
            
        except Exception as e:
            raise MyLathDBExecutionError(f"Failed to initialize execution systems: {e}")
    
    def _test_connectivity(self):
        """Test Redis and GraphBLAS connectivity"""
        # Test Redis
        if not self.redis_executor.test_connection():
            raise MyLathDBExecutionError("Redis connection failed")
        
        # Test GraphBLAS
        if not self.graphblas_executor.test_functionality():
            raise MyLathDBExecutionError("GraphBLAS initialization failed")
        
        logger.info("All execution systems connected successfully")
    
    @mylathdb_measure_time
    def execute(self, physical_plan, parameters: Dict[str, Any] = None,
                graph_data: Dict[str, Any] = None, **kwargs) -> ExecutionResult:
        """
        Execute a physical plan
        
        Args:
            physical_plan: Physical execution plan from PhysicalPlanner
            parameters: Query parameters (e.g., {"age": 25})
            graph_data: Optional graph data for GraphBLAS operations
            **kwargs: Additional execution options
            
        Returns:
            ExecutionResult with query results and execution statistics
        """
        # Create execution context
        context = ExecutionContext(
            parameters=parameters or {},
            graph_data=graph_data,
            max_execution_time=kwargs.get('max_execution_time', self.config.MAX_EXECUTION_TIME),
            enable_parallel=kwargs.get('enable_parallel', True),
            cache_results=kwargs.get('cache_results', self.config.ENABLE_CACHING)
        )
        
        start_time = time.time()
        execution_result = ExecutionResult(
            success=False,
            execution_id=context.execution_id
        )
        
        try:
            logger.info(f"Starting execution {context.execution_id}")
            
            # Check cache first
            if context.cache_results and self.cache:
                cache_key = self._generate_cache_key(physical_plan, parameters)
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    execution_result = cached_result.copy()
                    execution_result.cache_hits = 1
                    execution_result.execution_time = time.time() - start_time
                    logger.info(f"Cache hit for execution {context.execution_id}")
                    return execution_result
            
            # Load graph data if provided
            if graph_data:
                self._load_graph_data(graph_data)
            
            # Execute the physical plan
            result_data = self._execute_physical_plan(physical_plan, context, execution_result)
            
            # Format results
            formatted_data = self.result_formatter.format_results(result_data, physical_plan)
            
            # Build successful result
            execution_result.success = True
            execution_result.data = formatted_data
            execution_result.execution_time = time.time() - start_time
            execution_result.execution_plan_summary = self._generate_plan_summary(physical_plan)
            
            # Cache result if enabled
            if context.cache_results and self.cache:
                self.cache[cache_key] = execution_result
            
            # Update statistics
            self.total_queries_executed += 1
            self.total_execution_time += execution_result.execution_time
            
            logger.info(f"Execution {context.execution_id} completed successfully in {execution_result.execution_time:.3f}s")
            
        except MyLathDBTimeoutError as e:
            execution_result.error = f"Execution timeout: {str(e)}"
            execution_result.execution_time = time.time() - start_time
            logger.error(f"Execution {context.execution_id} timed out")
            
        except Exception as e:
            execution_result.error = f"Execution failed: {str(e)}"
            execution_result.execution_time = time.time() - start_time
            logger.error(f"Execution {context.execution_id} failed: {e}", exc_info=True)
        
        return execution_result
    
    def _execute_physical_plan(self, physical_plan, context: ExecutionContext, 
                              execution_result: ExecutionResult) -> List[Dict[str, Any]]:
        """
        Execute the physical plan using appropriate executors
        
        Based on FalkorDB's execution model with Redis + GraphBLAS coordination
        """
        from ..cypher_planner.physical_planner import (
            RedisOperation, GraphBLASOperation, CoordinatorOperation, PhysicalOperation
        )
        
        # Route to appropriate executor based on operation type
        if isinstance(physical_plan, RedisOperation):
            execution_result.redis_operations += 1
            return self.redis_executor.execute_operation(physical_plan, context)
            
        elif isinstance(physical_plan, GraphBLASOperation):
            execution_result.graphblas_operations += 1
            return self.graphblas_executor.execute_operation(physical_plan, context)
            
        elif isinstance(physical_plan, CoordinatorOperation):
            execution_result.coordinator_operations += 1
            return self.coordinator.execute_operation(physical_plan, context)
            
        else:
            # Generic physical operation - determine best executor
            return self._execute_generic_operation(physical_plan, context, execution_result)
    
    def _execute_generic_operation(self, physical_plan, context: ExecutionContext,
                                  execution_result: ExecutionResult) -> List[Dict[str, Any]]:
        """Execute generic physical operation by routing to appropriate executor"""
        
        # Check execution target hint
        target = getattr(physical_plan, 'target', 'mixed')
        
        if target == 'redis':
            execution_result.redis_operations += 1
            return self.redis_executor.execute_generic_operation(physical_plan, context)
        elif target == 'graphblas':
            execution_result.graphblas_operations += 1
            return self.graphblas_executor.execute_generic_operation(physical_plan, context)
        else:
            # Mixed operation - use coordinator
            execution_result.coordinator_operations += 1
            return self.coordinator.execute_generic_operation(physical_plan, context)
    
    def _load_graph_data(self, graph_data: Dict[str, Any]):
        """Load graph data into appropriate systems"""
        logger.info("Loading graph data into execution systems")
        
        # Load into Redis if node/edge data provided
        if 'nodes' in graph_data:
            self.redis_executor.load_nodes(graph_data['nodes'])
        
        if 'edges' in graph_data:
            self.redis_executor.load_edges(graph_data['edges'])
        
        # Load into GraphBLAS if matrix data provided
        if 'adjacency_matrices' in graph_data:
            self.graphblas_executor.load_adjacency_matrices(graph_data['adjacency_matrices'])
        
        if 'edges' in graph_data:
            # Also create matrices from edge data
            self.graphblas_executor.load_edges_as_matrices(graph_data['edges'])
    
    def _generate_cache_key(self, physical_plan, parameters: Dict[str, Any]) -> str:
        """Generate cache key for physical plan + parameters"""
        import hashlib
        
        # Simple cache key based on plan string representation and parameters
        plan_str = str(physical_plan)
        params_str = str(sorted(parameters.items())) if parameters else ""
        combined = f"{plan_str}|{params_str}"
        
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _generate_plan_summary(self, physical_plan) -> str:
        """Generate execution plan summary"""
        return f"Executed: {type(physical_plan).__name__}"
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution engine statistics"""
        return {
            'total_queries_executed': self.total_queries_executed,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': (
                self.total_execution_time / max(1, self.total_queries_executed)
            ),
            'cache_size': len(self.cache) if self.cache else 0,
            'redis_status': self.redis_executor.get_status(),
            'graphblas_status': self.graphblas_executor.get_status()
        }
    
    def clear_cache(self):
        """Clear execution cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Execution cache cleared")
    
    def shutdown(self):
        """Shutdown execution engine and cleanup resources"""
        logger.info("Shutting down MyLathDB execution engine")
        
        try:
            # Shutdown executors
            self.redis_executor.shutdown()
            self.graphblas_executor.shutdown()
            self.coordinator.shutdown()
            
            # Clear cache
            self.clear_cache()
            
            logger.info("MyLathDB execution engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise MyLathDBExecutionError(f"Shutdown failed: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
