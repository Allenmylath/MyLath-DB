# mylathdb/execution_engine/coordinator.py

"""
MyLathDB Execution Coordinator
Coordinates execution between Redis and GraphBLAS based on FalkorDB patterns
"""

import logging
from typing import Dict, List, Any, Optional
from .config import MyLathDBExecutionConfig
from .exceptions import MyLathDBExecutionError
from .utils import mylathdb_measure_time

logger = logging.getLogger(__name__)

class ExecutionCoordinator:
    """
    Execution coordinator for MyLathDB
    
    Handles complex operations that require coordination between
    Redis (entities/properties) and GraphBLAS (graph traversals)
    """
    
    def __init__(self, config: MyLathDBExecutionConfig):
        """Initialize execution coordinator"""
        self.config = config
        self.redis_executor = None  # Will be set by engine
        self.graphblas_executor = None  # Will be set by engine
        
    def set_executors(self, redis_executor, graphblas_executor):
        """Set executor references (called by main engine)"""
        self.redis_executor = redis_executor
        self.graphblas_executor = graphblas_executor
        
        # CRITICAL FIX: Set data bridge reference for ID mapping
        self.data_bridge = getattr(redis_executor, 'data_bridge', None) or getattr(graphblas_executor, 'data_bridge', None)
    
    @mylathdb_measure_time
    def execute_operation(self, coordinator_operation, context) -> List[Dict[str, Any]]:
        """
        Execute coordinator operation from physical plan
        
        Args:
            coordinator_operation: CoordinatorOperation from physical planner
            context: ExecutionContext
            
        Returns:
            List of result dictionaries
        """
        from ..cypher_planner.physical_planner import CoordinatorOperation
        
        if not isinstance(coordinator_operation, CoordinatorOperation):
            raise MyLathDBExecutionError(f"Expected CoordinatorOperation, got {type(coordinator_operation)}")
        
        logger.debug(f"Executing coordinator operation: {coordinator_operation.operation_type}")
        
        # Route to appropriate coordination pattern
        operation_type = coordinator_operation.operation_type
        coordination_pattern = coordinator_operation.coordination_pattern
        
        if operation_type == "SemiApply":
            return self._execute_semi_apply(coordinator_operation, context)
        elif operation_type == "Apply":
            return self._execute_apply(coordinator_operation, context)
        elif operation_type == "Optional":
            return self._execute_optional(coordinator_operation, context)
        elif coordination_pattern == "semi_apply_exists_check":
            return self._execute_exists_check(coordinator_operation, context)
        elif coordination_pattern == "left_outer_join":
            return self._execute_left_outer_join(coordinator_operation, context)
        else:
            # Execute generic coordination
            return self._execute_generic_coordination(coordinator_operation, context)
    
    def _execute_semi_apply(self, operation, context) -> List[Dict[str, Any]]:
        """Execute SemiApply operation (EXISTS-style filtering)"""
        logical_op = operation.logical_op
        
        # Semi-apply requires left and right branch execution
        if len(operation.children) >= 2:
            left_child = operation.children[0]
            right_child = operation.children[1]
            
            # Execute left branch (main pattern)
            left_results = self._execute_child_operation(left_child, context)
            
            # For each left result, check if right branch exists
            filtered_results = []
            for record in left_results:
                # Create context with current record variables
                exists_context = self._create_exists_context(context, record)
                
                # Execute right branch
                right_results = self._execute_child_operation(right_child, exists_context)
                
                # Apply semi-apply logic
                exists = len(right_results) > 0
                if logical_op.anti:
                    exists = not exists  # Anti-semi-apply (NOT EXISTS)
                
                if exists:
                    filtered_results.append(record)
            
            return filtered_results
        
        return []
    
    def _execute_apply(self, operation, context) -> List[Dict[str, Any]]:
        """Execute Apply operation (correlated subquery)"""
        
        if len(operation.children) >= 2:
            left_child = operation.children[0]
            right_child = operation.children[1]
            
            # Execute left branch
            left_results = self._execute_child_operation(left_child, context)
            
            # For each left result, execute right branch and combine
            combined_results = []
            for record in left_results:
                # Create correlated context
                correlated_context = self._create_correlated_context(context, record)
                
                # Execute right branch
                right_results = self._execute_child_operation(right_child, correlated_context)
                
                # Combine results
                if right_results:
                    for right_record in right_results:
                        combined_record = {**record, **right_record}
                        combined_results.append(combined_record)
                else:
                    # No right results, include left only if not required
                    combined_results.append(record)
            
            return combined_results
        
        return []
    
    def _execute_optional(self, operation, context) -> List[Dict[str, Any]]:
        """Execute Optional operation (OPTIONAL MATCH)"""
        
        if len(operation.children) >= 2:
            required_child = operation.children[0]
            optional_child = operation.children[1]
            
            # Execute required part
            required_results = self._execute_child_operation(required_child, context)
            
            # For each required result, try to execute optional part
            final_results = []
            for record in required_results:
                # Create context with required variables
                optional_context = self._create_optional_context(context, record)
                
                # Execute optional part
                optional_results = self._execute_child_operation(optional_child, optional_context)
                
                if optional_results:
                    # Combine with optional results
                    for optional_record in optional_results:
                        combined_record = {**record, **optional_record}
                        final_results.append(combined_record)
                else:
                    # No optional results, include required with NULLs for optional variables
                    final_results.append(record)
            
            return final_results
        
        return []
    
    def _execute_exists_check(self, operation, context) -> List[Dict[str, Any]]:
        """Execute EXISTS pattern check coordination"""
        
        # Extract pattern from data transfer operations
        # This would parse the actual EXISTS pattern and execute accordingly
        
        # For now, return empty (would need pattern parsing implementation)
        logger.debug("EXISTS check coordination executed (placeholder)")
        return []
    
    def _execute_left_outer_join(self, operation, context) -> List[Dict[str, Any]]:
        """Execute left outer join coordination"""
        
        if len(operation.children) >= 2:
            left_child = operation.children[0]
            right_child = operation.children[1]
            
            # Execute both sides
            left_results = self._execute_child_operation(left_child, context)
            right_results = self._execute_child_operation(right_child, context)
            
            # Perform left outer join
            joined_results = []
            
            for left_record in left_results:
                matched = False
                
                for right_record in right_results:
                    # Check if records can be joined (share common variables)
                    if self._can_join_records(left_record, right_record):
                        combined_record = {**left_record, **right_record}
                        joined_results.append(combined_record)
                        matched = True
                
                # If no match found, include left record with NULLs
                if not matched:
                    joined_results.append(left_record)
            
            return joined_results
        
        return []
    
    def _execute_generic_coordination(self, operation, context) -> List[Dict[str, Any]]:
        """Execute generic coordination operation"""
        
        # Execute all children and combine results
        all_results = []
        
        for child in operation.children:
            child_results = self._execute_child_operation(child, context)
            all_results.extend(child_results)
        
        return all_results
    
    def _execute_child_operation(self, child_operation, context) -> List[Dict[str, Any]]:
        """Execute a child operation using appropriate executor"""
        from ..cypher_planner.physical_planner import (
            RedisOperation, GraphBLASOperation, CoordinatorOperation
        )
        
        if isinstance(child_operation, RedisOperation):
            return self.redis_executor.execute_operation(child_operation, context)
        elif isinstance(child_operation, GraphBLASOperation):
            return self.graphblas_executor.execute_operation(child_operation, context)
        elif isinstance(child_operation, CoordinatorOperation):
            return self.execute_operation(child_operation, context)
        else:
            # Generic operation - determine best executor
            target = getattr(child_operation, 'target', 'mixed')
            
            if target == 'redis':
                return self.redis_executor.execute_generic_operation(child_operation, context)
            elif target == 'graphblas':
                return self.graphblas_executor.execute_generic_operation(child_operation, context)
            else:
                return self.execute_generic_operation(child_operation, context)
    
    def execute_generic_operation(self, physical_plan, context) -> List[Dict[str, Any]]:
        """Execute generic physical operation with coordination"""
        
        # Try to execute with both executors and coordinate results
        redis_results = []
        graphblas_results = []
        
        try:
            # Try Redis execution first
            redis_results = self.redis_executor.execute_generic_operation(physical_plan, context)
        except Exception as e:
            logger.debug(f"Redis execution failed: {e}")
        
        try:
            # Try GraphBLAS execution
            graphblas_results = self.graphblas_executor.execute_generic_operation(physical_plan, context)
        except Exception as e:
            logger.debug(f"GraphBLAS execution failed: {e}")
        
        # Coordinate results
        if redis_results and graphblas_results:
            # Both succeeded - need to coordinate/join results
            return self._coordinate_mixed_results(redis_results, graphblas_results)
        elif redis_results:
            return redis_results
        elif graphblas_results:
            return graphblas_results
        else:
            return []
    
    def _coordinate_mixed_results(self, redis_results: List[Dict[str, Any]], 
                                 graphblas_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Coordinate results from both Redis and GraphBLAS"""
        
        # Simple coordination - merge results based on common variables
        coordinated = []
        
        for redis_record in redis_results:
            for graphblas_record in graphblas_results:
                if self._can_join_records(redis_record, graphblas_record):
                    combined = {**redis_record, **graphblas_record}
                    coordinated.append(combined)
        
        # If no joins possible, return all results separately
        if not coordinated:
            coordinated.extend(redis_results)
            coordinated.extend(graphblas_results)
        
        return coordinated
    
    def _can_join_records(self, record1: Dict[str, Any], record2: Dict[str, Any]) -> bool:
        """Check if two records can be joined based on common variables"""
        
        # Find common variable names
        common_vars = set(record1.keys()) & set(record2.keys())
        
        if not common_vars:
            return False
        
        # Check if values match for common variables
        for var in common_vars:
            val1 = record1[var]
            val2 = record2[var]
            
            # Extract IDs for comparison
            id1 = self._extract_entity_id(val1)
            id2 = self._extract_entity_id(val2)
            
            if id1 is not None and id2 is not None and id1 != id2:
                return False
        
        return True
    
    def _extract_entity_id(self, value: Any) -> Optional[str]:
        """Extract entity ID from value for comparison"""
        
        if isinstance(value, dict):
            return value.get('_id') or value.get('id')
        elif isinstance(value, str):
            return value
        
        return None
    
    def _create_exists_context(self, base_context, record: Dict[str, Any]):
        """Create context for EXISTS check with current record variables"""
        
        # Create new context with variables from current record
        exists_context = type(base_context)(
            parameters=base_context.parameters.copy(),
            graph_data=base_context.graph_data,
            execution_id=f"{base_context.execution_id}_exists",
            max_execution_time=base_context.max_execution_time
        )
        
        # Add record variables to context
        exists_context.current_record = record
        
        return exists_context
    
    def _create_correlated_context(self, base_context, record: Dict[str, Any]):
        """Create context for correlated subquery with current record variables"""
        
        correlated_context = type(base_context)(
            parameters=base_context.parameters.copy(),
            graph_data=base_context.graph_data,
            execution_id=f"{base_context.execution_id}_corr",
            max_execution_time=base_context.max_execution_time
        )
        
        # Add record variables to context
        correlated_context.current_record = record
        
        return correlated_context
    
    def _create_optional_context(self, base_context, record: Dict[str, Any]):
        """Create context for optional match with required variables"""
        
        optional_context = type(base_context)(
            parameters=base_context.parameters.copy(),
            graph_data=base_context.graph_data,
            execution_id=f"{base_context.execution_id}_opt",
            max_execution_time=base_context.max_execution_time
        )
        
        # Add required variables to context
        optional_context.required_record = record
        
        return optional_context
    def _execute_conditional_traverse_with_children(self, operation, child_results, context):
        """
        NEW METHOD: Execute ConditionalTraverse with source results from children
        This bridges Redis scan results to GraphBLAS traversal operations
        """
        print(f"🔍 Coordinator: ConditionalTraverse with {len(child_results)} source results")
        
        logical_op = getattr(operation, 'logical_op', None)
        if not logical_op:
            print("   ❌ No logical operation found")
            return child_results
        
        # If GraphBLAS is not available, return child results
        if not self.graphblas_executor or not self.graphblas_executor.is_available():
            print("   ⚠️  GraphBLAS not available, returning child results without traversal")
            return child_results
        
        try:
            # Extract source node IDs from child results
            source_node_ids = []
            for result in child_results:
                for var_name, entity in result.items():
                    if isinstance(entity, dict):
                        node_id = entity.get('_id') or entity.get('id')
                        if node_id:
                            source_node_ids.append(str(node_id))
            
            print(f"   📍 Found {len(source_node_ids)} source nodes: {source_node_ids}")
            
            if not source_node_ids:
                print("   ❌ No source node IDs found in child results")
                return []
            
            # Execute GraphBLAS traversal
            traversal_results = self._execute_graphblas_traversal(
                logical_op, source_node_ids, context
            )
            
            print(f"   🎯 Traversal returned {len(traversal_results)} results")
            return traversal_results
            
        except Exception as e:
            print(f"   ❌ ConditionalTraverse execution failed: {e}")
            import traceback
            traceback.print_exc()
            return child_results

    def _execute_graphblas_traversal(self, logical_op, source_node_ids, context):
        """
        Execute GraphBLAS traversal from source nodes
        """
        print(f"🔍 GraphBLAS traversal: {logical_op.from_var} → {logical_op.to_var}")
        
        try:
            # Get relation types and direction
            rel_types = getattr(logical_op, 'rel_types', ['*'])
            direction = getattr(logical_op, 'direction', 'outgoing')
            
            print(f"   📊 Relation types: {rel_types}, Direction: {direction}")
            
            # Get relation matrix from GraphBLAS executor
            relation_matrix = self.graphblas_executor._get_relation_matrix(rel_types, direction)
            if relation_matrix is None:
                print("   ❌ Could not get relation matrix")
                return []
            
            # Create source vector from node IDs
            source_vector = self._create_source_vector_from_ids(source_node_ids)
            if source_vector is None:
                print("   ❌ Could not create source vector")
                return []
            
            # Perform matrix-vector multiplication
            if direction == "outgoing":
                result_vector = source_vector.vxm(relation_matrix, self.graphblas_executor.default_bool_semiring)
            elif direction == "incoming":
                result_vector = source_vector.vxm(relation_matrix.T, self.graphblas_executor.default_bool_semiring)
            else:  # bidirectional
                import graphblas as gb
                bidirectional_matrix = relation_matrix.ewise_add(relation_matrix.T, gb.binary.lor)
                result_vector = source_vector.vxm(bidirectional_matrix, self.graphblas_executor.default_bool_semiring)
            
            # Convert result vector back to node data
            destination_results = self._convert_vector_to_node_results(
                result_vector, logical_op.to_var, context
            )
            
            print(f"   ✅ GraphBLAS traversal found {len(destination_results)} destinations")
            return destination_results
            
        except Exception as e:
            print(f"   ❌ GraphBLAS traversal failed: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _create_source_vector_from_ids(self, node_ids):
        """Create GraphBLAS vector from list of node IDs"""
        try:
            import graphblas as gb
            
            # Get node capacity from GraphBLAS executor
            n = self.graphblas_executor.graph.node_capacity
            source_vector = gb.Vector(gb.dtypes.BOOL, size=n)
            
            # Get data bridge for ID mapping
            data_bridge = getattr(self, 'data_bridge', None)
            if not data_bridge:
                # Try to get it from graphblas_executor
                data_bridge = getattr(self.graphblas_executor, 'data_bridge', None)
            
            if not data_bridge:
                print("   ❌ No data bridge available for ID mapping")
                return None
            
            # Map node IDs to matrix indices
            indices_set = []
            for node_id in node_ids:
                # Try to get existing mapping
                matrix_index = data_bridge.node_mapping.entity_to_index.get(str(node_id))
                if matrix_index is not None and matrix_index < n:
                    indices_set.append(matrix_index)
                    print(f"     📍 Mapped node {node_id} → index {matrix_index}")
            
            # Set vector entries
            for index in indices_set:
                source_vector[index] = True
            
            print(f"   ✅ Created source vector with {len(indices_set)} active nodes")
            return source_vector
            
        except Exception as e:
            print(f"   ❌ Source vector creation failed: {e}")
            return None

    def _convert_vector_to_node_results(self, result_vector, variable_name, context):
        """Convert GraphBLAS result vector back to node data results"""
        
        try:
            # Get non-zero indices from vector
            indices, values = result_vector.to_coo()
            
            # Get data bridge for reverse mapping
            data_bridge = getattr(self, 'data_bridge', None)
            if not data_bridge:
                data_bridge = getattr(self.graphblas_executor, 'data_bridge', None)
            
            if not data_bridge:
                print("   ❌ No data bridge available for reverse mapping")
                return []
            
            results = []
            
            for index, value in zip(indices, values):
                if value:  # Non-zero entry (reachable node)
                    # Get node ID from matrix index
                    node_id = data_bridge.node_mapping.get_entity_id(index)
                    if node_id:
                        # Fetch full node data from Redis
                        node_data = self.redis_executor._get_node_data_complete(node_id)
                        if node_data:
                            # Format result with variable name
                            result_record = {variable_name: node_data}
                            results.append(result_record)
                            print(f"     ✅ Found destination node: {node_data.get('name', node_id)}")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Vector to results conversion failed: {e}")
            return []

    def shutdown(self):
        """Shutdown coordinator"""
        logger.info("Execution coordinator shutdown complete")
    

