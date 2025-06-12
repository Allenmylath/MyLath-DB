# execution_engine/coordinator.py

"""
Execution Coordinator - Coordinates between Redis and GraphBLAS operations
"""

from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime

from cypher_planner.physical_planner import CoordinatorOperation


@dataclass
class CoordinationResult:
    """Result of coordination operation"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    coordination_type: str = ""
    execution_time: float = 0.0
    operations_coordinated: int = 0
    error: Optional[str] = None


class ExecutionCoordinator:
    """Coordinates execution between Redis and GraphBLAS systems"""
    
    def __init__(self):
        self.stats = {
            'coordinations_executed': 0,
            'data_transfers': 0,
            'total_execution_time': 0.0
        }
        self.logger = logging.getLogger(__name__)
    
    def execute(self, operation: CoordinatorOperation, context, child_results: List[Any]) -> CoordinationResult:
        """Execute coordination operation"""
        
        start_time = datetime.now()
        
        try:
            coordination_type = operation.coordination_pattern
            
            if coordination_type == "semi_apply_exists_check":
                result = self._execute_semi_apply(operation, context, child_results)
            elif coordination_type == "correlated_subquery":
                result = self._execute_correlated_subquery(operation, context, child_results)
            elif coordination_type == "left_outer_join":
                result = self._execute_left_outer_join(operation, context, child_results)
            else:
                result = self._execute_generic_coordination(operation, context, child_results)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            result.coordination_type = coordination_type
            
            # Update statistics
            self.stats['coordinations_executed'] += 1
            self.stats['total_execution_time'] += execution_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Coordination failed: {str(e)}")
            return CoordinationResult(
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _execute_semi_apply(self, operation: CoordinatorOperation, context, child_results: List[Any]) -> CoordinationResult:
        """Execute semi-apply pattern for EXISTS-style filtering"""
        
        result = CoordinationResult(success=True)
        
        try:
            # Get left and right branches
            if len(child_results) >= 2:
                left_result = child_results[0]
                right_result = child_results[1]
            else:
                left_result = child_results[0] if child_results else None
                right_result = None
            
            filtered_results = []
            
            # Get left branch data
            if hasattr(left_result, 'data') and 'nodes' in left_result.data:
                left_nodes = left_result.data['nodes']
            elif hasattr(left_result, 'node_ids'):
                left_nodes = list(left_result.node_ids)
            else:
                left_nodes = []
            
            # Apply EXISTS check for each left record
            for node_id in left_nodes:
                # Check if right branch produces results for this node
                exists = self._check_existence(node_id, right_result, context)
                
                # Apply anti-semi logic if needed
                logical_op = operation.logical_op
                anti = logical_op and hasattr(logical_op, 'anti') and logical_op.anti
                
                if (exists and not anti) or (not exists and anti):
                    filtered_results.append(node_id)
            
            result.data = {
                'nodes': filtered_results,
                'operation_type': 'semi_apply'
            }
            result.operations_coordinated = len(operation.data_transfer_ops)
            
            return result
            
        except Exception as e:
            return CoordinationResult(success=False, error=f"Semi-apply failed: {str(e)}")
    
    def _execute_correlated_subquery(self, operation: CoordinatorOperation, context, child_results: List[Any]) -> CoordinationResult:
        """Execute correlated subquery pattern"""
        
        result = CoordinationResult(success=True)
        
        try:
            # For correlated subqueries, we need to execute the right branch
            # for each record from the left branch
            
            if len(child_results) >= 1:
                left_result = child_results[0]
            else:
                left_result = None
            
            if not left_result:
                result.data = {'nodes': [], 'operation_type': 'correlated_subquery'}
                return result
            
            # Get left branch data
            if hasattr(left_result, 'data') and 'nodes' in left_result.data:
                left_nodes = left_result.data['nodes']
            elif hasattr(left_result, 'node_ids'):
                left_nodes = list(left_result.node_ids)
            else:
                left_nodes = []
            
            correlated_results = []
            
            for node_id in left_nodes:
                # Execute right branch with context of current node
                subquery_result = self._execute_correlated_branch(node_id, operation, context)
                if subquery_result:
                    correlated_results.extend(subquery_result)
            
            result.data = {
                'nodes': correlated_results,
                'operation_type': 'correlated_subquery'
            }
            result.operations_coordinated = len(left_nodes)
            
            return result
            
        except Exception as e:
            return CoordinationResult(success=False, error=f"Correlated subquery failed: {str(e)}")
    
    def _execute_left_outer_join(self, operation: CoordinatorOperation, context, child_results: List[Any]) -> CoordinationResult:
        """Execute left outer join for OPTIONAL MATCH"""
        
        result = CoordinationResult(success=True)
        
        try:
            if len(child_results) >= 2:
                left_result = child_results[0]
                right_result = child_results[1]
            else:
                left_result = child_results[0] if child_results else None
                right_result = None
            
            # Get left branch data (required)
            if hasattr(left_result, 'data') and 'nodes' in left_result.data:
                left_nodes = left_result.data['nodes']
                left_properties = left_result.data.get('properties', {})
            elif hasattr(left_result, 'node_ids'):
                left_nodes = list(left_result.node_ids)
                left_properties = getattr(left_result, 'properties', {})
            else:
                left_nodes = []
                left_properties = {}
            
            # Get right branch data (optional)
            if right_result:
                if hasattr(right_result, 'data') and 'nodes' in right_result.data:
                    right_nodes = set(right_result.data['nodes'])
                    right_properties = right_result.data.get('properties', {})
                elif hasattr(right_result, 'node_ids'):
                    right_nodes = right_result.node_ids
                    right_properties = getattr(right_result, 'properties', {})
                else:
                    right_nodes = set()
                    right_properties = {}
            else:
                right_nodes = set()
                right_properties = {}
            
            # Perform left outer join
            joined_results = []
            joined_properties = {}
            
            for node_id in left_nodes:
                joined_results.append(node_id)
                
                # Merge properties from both sides
                node_props = left_properties.get(node_id, {}).copy()
                if node_id in right_nodes and node_id in right_properties:
                    node_props.update(right_properties[node_id])
                
                joined_properties[node_id] = node_props
            
            result.data = {
                'nodes': joined_results,
                'properties': joined_properties,
                'operation_type': 'left_outer_join'
            }
            result.operations_coordinated = 2  # Left and right branches
            
            return result
            
        except Exception as e:
            return CoordinationResult(success=False, error=f"Left outer join failed: {str(e)}")
    
    def _execute_generic_coordination(self, operation: CoordinatorOperation, context, child_results: List[Any]) -> CoordinationResult:
        """Execute generic coordination operation"""
        
        result = CoordinationResult(success=True)
        
        try:
            # For generic coordination, merge all child results
            merged_data = self._merge_child_results(child_results)
            
            result.data = merged_data
            result.operations_coordinated = len(child_results)
            
            return result
            
        except Exception as e:
            return CoordinationResult(success=False, error=f"Generic coordination failed: {str(e)}")
    
    def _check_existence(self, node_id: str, right_result: Any, context) -> bool:
        """Check if a node exists in the right branch result"""
        
        if not right_result:
            return False
        
        # Check in various result formats
        if hasattr(right_result, 'node_ids'):
            return node_id in right_result.node_ids
        elif hasattr(right_result, 'data') and 'nodes' in right_result.data:
            return node_id in right_result.data['nodes']
        elif isinstance(right_result, dict) and 'nodes' in right_result:
            return node_id in right_result['nodes']
        
        return False
    
    def _execute_correlated_branch(self, node_id: str, operation: CoordinatorOperation, context) -> List[str]:
        """Execute correlated branch for a specific node"""
        
        # This is a simplified implementation
        # In a real system, you'd re-execute the right branch with the node context
        
        # For now, return the node itself as a placeholder
        return [node_id]
    
    def _merge_child_results(self, child_results: List[Any]) -> Dict[str, Any]:
        """Merge results from multiple child operations"""
        
        merged_nodes = []
        merged_properties = {}
        
        for child_result in child_results:
            if hasattr(child_result, 'data'):
                if 'nodes' in child_result.data:
                    merged_nodes.extend(child_result.data['nodes'])
                if 'properties' in child_result.data:
                    merged_properties.update(child_result.data['properties'])
            elif hasattr(child_result, 'node_ids'):
                merged_nodes.extend(list(child_result.node_ids))
                if hasattr(child_result, 'properties'):
                    merged_properties.update(child_result.properties)
        
        return {
            'nodes': list(set(merged_nodes)),  # Remove duplicates
            'properties': merged_properties,
            'operation_type': 'merged_results'
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            'coordinations_executed': self.stats['coordinations_executed'],
            'data_transfers': self.stats['data_transfers'],
            'total_execution_time': self.stats['total_execution_time'],
            'avg_execution_time': self.stats['total_execution_time'] / max(1, self.stats['coordinations_executed'])
        }
    
    def shutdown(self):
        """Shutdown coordinator"""
        self.logger.info("Execution coordinator shutdown")