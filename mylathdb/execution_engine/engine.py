# execution_engine/engine.py

"""
Main Execution Engine - Coordinates Redis + GraphBLAS execution
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time

# Import your physical plan types
from cypher_planner.physical_planner import (
    PhysicalOperation, RedisOperation, GraphBLASOperation, CoordinatorOperation
)

from .redis_executor import RedisExecutor, RedisResult
from .graphblas_executor import GraphBLASExecutor, GraphBLASResult
from .coordinator import ExecutionCoordinator, CoordinationResult
from .data_bridge import DataBridge, ResultSet
from .result_formatter import ResultFormatter


@dataclass
class ExecutionContext:
    """Execution context holding variables and intermediate results"""
    variables: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    redis_data: Dict[str, Any] = field(default_factory=dict)
    graphblas_data: Dict[str, Any] = field(default_factory=dict)
    execution_stats: Dict[str, float] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    
    def set_variable(self, name: str, value: Any):
        """Set a variable value"""
        self.variables[name] = value
    
    def get_variable(self, name: str, default=None):
        """Get variable value"""
        return self.variables.get(name, default)
    
    def has_variable(self, name: str) -> bool:
        """Check if variable exists"""
        return name in self.variables
    
    def set_result(self, operation_id: str, result: Any):
        """Store intermediate result"""
        self.intermediate_results[operation_id] = result
    
    def get_result(self, operation_id: str, default=None):
        """Get intermediate result"""
        return self.intermediate_results.get(operation_id, default)


@dataclass
class ExecutionResult:
    """Result of executing a physical plan"""
    success: bool
    data: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    operations_executed: int = 0
    error: Optional[str] = None
    context: Optional[ExecutionContext] = None
    redis_operations: int = 0
    graphblas_operations: int = 0
    coordinator_operations: int = 0
    
    def __post_init__(self):
        if self.context:
            self.execution_time = (datetime.now() - self.context.start_time).total_seconds()


class ExecutionEngine:
    """Main execution engine for physical plans"""
    
    def __init__(self, redis_client=None, enable_caching=True, 
                 max_parallel_ops=4, debug=False):
        
        # Core executors
        self.redis_executor = RedisExecutor(redis_client, enable_caching)
        self.graphblas_executor = GraphBLASExecutor(max_parallel_ops)
        self.coordinator = ExecutionCoordinator()
        self.data_bridge = DataBridge()
        self.formatter = ResultFormatter()
        
        # Configuration
        self.enable_caching = enable_caching
        self.max_parallel_ops = max_parallel_ops
        self.debug = debug
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
    
    def execute(self, physical_plan: PhysicalOperation, 
                graph_data: Dict[str, Any] = None,
                parameters: Dict[str, Any] = None,
                timeout: float = 30.0) -> ExecutionResult:
        """
        Execute a physical plan
        
        Args:
            physical_plan: Root physical operation to execute
            graph_data: Optional graph data for GraphBLAS operations
            parameters: Query parameters
            timeout: Execution timeout in seconds
            
        Returns:
            ExecutionResult with data and statistics
        """
        
        context = ExecutionContext()
        
        # Set parameters as variables
        if parameters:
            for name, value in parameters.items():
                context.set_variable(name, value)
        
        # Initialize graph data if provided
        if graph_data:
            self.graphblas_executor.load_graph_data(graph_data)
        
        try:
            start_time = time.time()
            
            # Execute the plan
            result_data = self._execute_operation(physical_plan, context)
            
            execution_time = time.time() - start_time
            
            # Format final results
            formatted_data = self.formatter.format_results(result_data, context)
            
            return ExecutionResult(
                success=True,
                data=formatted_data,
                execution_time=execution_time,
                operations_executed=self._count_operations(physical_plan),
                context=context,
                redis_operations=context.execution_stats.get('redis_ops', 0),
                graphblas_operations=context.execution_stats.get('graphblas_ops', 0),
                coordinator_operations=context.execution_stats.get('coordinator_ops', 0)
            )
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}", exc_info=True)
            
            return ExecutionResult(
                success=False,
                error=str(e),
                context=context
            )
    
    def _execute_operation(self, operation: PhysicalOperation, 
                          context: ExecutionContext) -> Any:
        """Execute a single physical operation"""
        
        operation_id = f"{operation.operation_type}_{id(operation)}"
        
        if self.debug:
            self.logger.debug(f"Executing {operation.operation_type} operation")
        
        try:
            # Execute children first (bottom-up execution)
            child_results = []
            for child in operation.children:
                child_result = self._execute_operation(child, context)
                child_results.append(child_result)
            
            # Execute current operation based on type
            if isinstance(operation, RedisOperation):
                result = self._execute_redis_operation(operation, context, child_results)
                context.execution_stats['redis_ops'] = context.execution_stats.get('redis_ops', 0) + 1
                
            elif isinstance(operation, GraphBLASOperation):
                result = self._execute_graphblas_operation(operation, context, child_results)
                context.execution_stats['graphblas_ops'] = context.execution_stats.get('graphblas_ops', 0) + 1
                
            elif isinstance(operation, CoordinatorOperation):
                result = self._execute_coordinator_operation(operation, context, child_results)
                context.execution_stats['coordinator_ops'] = context.execution_stats.get('coordinator_ops', 0) + 1
                
            else:
                # Generic operation
                result = self._execute_generic_operation(operation, context, child_results)
            
            # Store result in context
            context.set_result(operation_id, result)
            
            if self.debug:
                self.logger.debug(f"Operation {operation.operation_type} completed successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to execute {operation.operation_type}: {str(e)}")
            raise
    
    def _execute_redis_operation(self, operation: RedisOperation,
                                context: ExecutionContext,
                                child_results: List[Any]) -> RedisResult:
        """Execute Redis operation"""
        
        # Merge child results if any
        input_data = self.data_bridge.merge_redis_inputs(child_results) if child_results else None
        
        # Execute Redis commands
        redis_result = self.redis_executor.execute(operation, context, input_data)
        
        # Update context with Redis data
        if redis_result.success and redis_result.data:
            context.redis_data.update(redis_result.data)
        
        return redis_result
    
    def _execute_graphblas_operation(self, operation: GraphBLASOperation,
                                   context: ExecutionContext,
                                   child_results: List[Any]) -> GraphBLASResult:
        """Execute GraphBLAS operation"""
        
        # Convert Redis results to GraphBLAS format if needed
        graphblas_inputs = self.data_bridge.redis_to_graphblas(child_results, context)
        
        # Execute GraphBLAS operations
        gb_result = self.graphblas_executor.execute(operation, context, graphblas_inputs)
        
        # Update context with GraphBLAS data
        if gb_result.success and gb_result.matrices:
            context.graphblas_data.update(gb_result.matrices)
        
        return gb_result
    
    def _execute_coordinator_operation(self, operation: CoordinatorOperation,
                                     context: ExecutionContext,
                                     child_results: List[Any]) -> CoordinationResult:
        """Execute coordination operation"""
        
        return self.coordinator.execute(operation, context, child_results)
    
    def _execute_generic_operation(self, operation: PhysicalOperation,
                                 context: ExecutionContext,
                                 child_results: List[Any]) -> Any:
        """Execute generic operation"""
        
        if self.debug:
            self.logger.debug(f"Executing generic operation: {operation.operation_type}")
        
        # For generic operations, just pass through the child results
        if len(child_results) == 1:
            return child_results[0]
        elif len(child_results) > 1:
            return child_results
        else:
            return {"operation": operation.operation_type, "result": "completed"}
    
    def _count_operations(self, operation: PhysicalOperation) -> int:
        """Count total operations in plan"""
        count = 1
        for child in operation.children:
            count += self._count_operations(child)
        return count
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution engine statistics"""
        return {
            "redis_executor": self.redis_executor.get_statistics(),
            "graphblas_executor": self.graphblas_executor.get_statistics(),
            "coordinator": self.coordinator.get_statistics(),
            "caching_enabled": self.enable_caching,
            "max_parallel_ops": self.max_parallel_ops
        }
    
    def clear_cache(self):
        """Clear all caches"""
        if self.enable_caching:
            self.redis_executor.clear_cache()
            self.graphblas_executor.clear_cache()
    
    def shutdown(self):
        """Shutdown the execution engine"""
        self.redis_executor.shutdown()
        self.graphblas_executor.shutdown()
        self.coordinator.shutdown()