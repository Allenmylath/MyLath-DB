# execution_engine/data_bridge.py

"""
Data Bridge - Handles data conversion between Redis and GraphBLAS formats
"""

from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
import logging


@dataclass
class ResultSet:
    """Unified result set format"""
    nodes: Set[str] = field(default_factory=set)
    edges: Set[tuple] = field(default_factory=set)
    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataBridge:
    """Bridges data between Redis and GraphBLAS execution contexts"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def merge_redis_inputs(self, redis_results: List[Any]) -> Optional[Any]:
        """Merge multiple Redis results into a single input"""
        
        if not redis_results:
            return None
        
        if len(redis_results) == 1:
            return redis_results[0]
        
        try:
            merged_nodes = set()
            merged_properties = {}
            
            for result in redis_results:
                if hasattr(result, 'node_ids'):
                    merged_nodes.update(result.node_ids)
                if hasattr(result, 'properties'):
                    merged_properties.update(result.properties)
                elif hasattr(result, 'data') and isinstance(result.data, dict):
                    if 'nodes' in result.data:
                        merged_nodes.update(result.data['nodes'])
                    if 'properties' in result.data:
                        merged_properties.update(result.data['properties'])
            
            # Create a merged result object
            class MergedRedisResult:
                def __init__(self):
                    self.node_ids = merged_nodes
                    self.properties = merged_properties
                    self.data = {
                        'nodes': list(merged_nodes),
                        'properties': merged_properties
                    }
            
            return MergedRedisResult()
            
        except Exception as e:
            self.logger.error(f"Failed to merge Redis inputs: {e}")
            return redis_results[0]  # Fallback to first result
    
    def redis_to_graphblas(self, redis_results: List[Any], context) -> Dict[str, Any]:
        """Convert Redis results to GraphBLAS input format"""
        
        try:
            graphblas_inputs = {
                'node_vectors': {},
                'adjacency_matrices': {},
                'node_ids': set(),
                'properties': {}
            }
            
            for result in redis_results:
                if hasattr(result, 'node_ids'):
                    graphblas_inputs['node_ids'].update(result.node_ids)
                if hasattr(result, 'properties'):
                    graphblas_inputs['properties'].update(result.properties)
                elif hasattr(result, 'data') and isinstance(result.data, dict):
                    if 'nodes' in result.data:
                        graphblas_inputs['node_ids'].update(result.data['nodes'])
                    if 'properties' in result.data:
                        graphblas_inputs['properties'].update(result.data['properties'])
            
            # Convert node sets to vector specifications
            if graphblas_inputs['node_ids']:
                graphblas_inputs['node_vectors']['input_nodes'] = list(graphblas_inputs['node_ids'])
            
            return graphblas_inputs
            
        except Exception as e:
            self.logger.error(f"Failed to convert Redis to GraphBLAS format: {e}")
            return {'node_vectors': {}, 'adjacency_matrices': {}, 'node_ids': set(), 'properties': {}}
    
    def graphblas_to_redis(self, graphblas_results: List[Any], context) -> Dict[str, Any]:
        """Convert GraphBLAS results to Redis format"""
        
        try:
            redis_format = {
                'nodes': [],
                'properties': {},
                'operation_type': 'graphblas_conversion'
            }
            
            for result in graphblas_results:
                if hasattr(result, 'node_ids'):
                    redis_format['nodes'].extend(list(result.node_ids))
                if hasattr(result, 'vectors'):
                    # Extract node IDs from vectors
                    for vector_name, vector in result.vectors.items():
                        if isinstance(vector, dict) and 'node_ids' in vector:
                            redis_format['nodes'].extend(list(vector['node_ids']))
            
            # Remove duplicates
            redis_format['nodes'] = list(set(redis_format['nodes']))
            
            return redis_format
            
        except Exception as e:
            self.logger.error(f"Failed to convert GraphBLAS to Redis format: {e}")
            return {'nodes': [], 'properties': {}, 'operation_type': 'conversion_failed'}
    
    def create_unified_result_set(self, redis_results: List[Any], 
                                 graphblas_results: List[Any],
                                 context) -> ResultSet:
        """Create unified result set from mixed results"""
        
        result_set = ResultSet()
        
        try:
            # Process Redis results
            for result in redis_results:
                if hasattr(result, 'node_ids'):
                    result_set.nodes.update(result.node_ids)
                if hasattr(result, 'properties'):
                    result_set.properties.update(result.properties)
                elif hasattr(result, 'data') and isinstance(result.data, dict):
                    if 'nodes' in result.data:
                        result_set.nodes.update(result.data['nodes'])
                    if 'properties' in result.data:
                        result_set.properties.update(result.data['properties'])
            
            # Process GraphBLAS results
            for result in graphblas_results:
                if hasattr(result, 'node_ids'):
                    result_set.nodes.update(result.node_ids)
                if hasattr(result, 'vectors'):
                    for vector_name, vector in result.vectors.items():
                        if isinstance(vector, dict) and 'node_ids' in vector:
                            result_set.nodes.update(vector['node_ids'])
            
            # Add metadata
            result_set.metadata = {
                'redis_results': len(redis_results),
                'graphblas_results': len(graphblas_results),
                'total_nodes': len(result_set.nodes),
                'total_properties': len(result_set.properties)
            }
            
            return result_set
            
        except Exception as e:
            self.logger.error(f"Failed to create unified result set: {e}")
            return result_set
    
    def extract_variables_from_context(self, context) -> Dict[str, Any]:
        """Extract variables from execution context"""
        
        variables = {}
        
        try:
            if hasattr(context, 'variables'):
                variables.update(context.variables)
            
            if hasattr(context, 'redis_data'):
                for key, value in context.redis_data.items():
                    if key not in variables:
                        variables[key] = value
            
            if hasattr(context, 'graphblas_data'):
                for key, value in context.graphblas_data.items():
                    if key not in variables:
                        variables[key] = value
            
            return variables
            
        except Exception as e:
            self.logger.error(f"Failed to extract variables from context: {e}")
            return {}
    
    def convert_to_output_format(self, result_set: ResultSet, 
                                output_format: str = "json") -> Any:
        """Convert result set to specified output format"""
        
        try:
            if output_format.lower() == "json":
                return {
                    'nodes': list(result_set.nodes),
                    'edges': list(result_set.edges),
                    'properties': result_set.properties,
                    'variables': result_set.variables,
                    'metadata': result_set.metadata
                }
            
            elif output_format.lower() == "table":
                # Create table-like structure
                rows = []
                for node_id in result_set.nodes:
                    row = {'node_id': node_id}
                    if node_id in result_set.properties:
                        row.update(result_set.properties[node_id])
                    rows.append(row)
                return rows
            
            elif output_format.lower() == "cypher_result":
                # Format similar to Neo4j result format
                records = []
                for node_id in result_set.nodes:
                    record = {
                        'node': {
                            'identity': node_id,
                            'labels': [],
                            'properties': result_set.properties.get(node_id, {})
                        }
                    }
                    records.append(record)
                
                return {
                    'records': records,
                    'summary': {
                        'result_available_after': 0,
                        'result_consumed_after': 0,
                        'statement_type': 'r',
                        'counters': {
                            'nodes_created': 0,
                            'nodes_deleted': 0,
                            'relationships_created': 0,
                            'relationships_deleted': 0,
                            'properties_set': 0,
                            'labels_added': 0,
                            'labels_removed': 0,
                            'indexes_added': 0,
                            'indexes_removed': 0,
                            'constraints_added': 0,
                            'constraints_removed': 0
                        }
                    }
                }
            
            else:
                # Default to returning the result set as-is
                return result_set
                
        except Exception as e:
            self.logger.error(f"Failed to convert to output format '{output_format}': {e}")
            return result_set
                    