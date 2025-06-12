# execution_engine/result_formatter.py

"""
Result Formatter - Formats execution results for different output requirements
"""

from typing import Dict, List, Any, Optional, Set, Union
import json
import logging
from datetime import datetime


class ResultFormatter:
    """Formats execution results into various output formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_results(self, result_data: Any, context, 
                      output_format: str = "cypher_records") -> List[Dict[str, Any]]:
        """Format execution results into the specified format"""
        
        try:
            if output_format == "cypher_records":
                return self._format_as_cypher_records(result_data, context)
            elif output_format == "table_rows":
                return self._format_as_table_rows(result_data, context)
            elif output_format == "graph_data":
                return self._format_as_graph_data(result_data, context)
            elif output_format == "json":
                return self._format_as_json(result_data, context)
            else:
                return self._format_as_cypher_records(result_data, context)
                
        except Exception as e:
            self.logger.error(f"Failed to format results: {e}")
            return [{"error": f"Formatting failed: {str(e)}"}]
    
    def _format_as_cypher_records(self, result_data: Any, context) -> List[Dict[str, Any]]:
        """Format results as Cypher-style records"""
        
        records = []
        
        try:
            # Extract data from various result formats
            nodes, properties, variables = self._extract_basic_data(result_data, context)
            
            if not nodes and not variables:
                return []
            
            # If we have variables from context, create records based on them
            if context and hasattr(context, 'variables') and context.variables:
                return self._create_records_from_variables(context.variables, properties)
            
            # Otherwise, create records from nodes
            for node_id in nodes:
                record = {}
                
                # Add node data
                if node_id in properties:
                    node_props = properties[node_id]
                    record['n'] = {
                        'identity': node_id,
                        'labels': self._extract_labels_from_properties(node_props),
                        'properties': {k: v for k, v in node_props.items() if not k.startswith('_')}
                    }
                else:
                    record['n'] = {
                        'identity': node_id,
                        'labels': [],
                        'properties': {}
                    }
                
                records.append(record)
            
            return records
            
        except Exception as e:
            self.logger.error(f"Failed to format as Cypher records: {e}")
            return [{"error": str(e)}]
    
    def _format_as_table_rows(self, result_data: Any, context) -> List[Dict[str, Any]]:
        """Format results as table rows"""
        
        rows = []
        
        try:
            nodes, properties, variables = self._extract_basic_data(result_data, context)
            
            # If we have variables, create columns for each variable
            if variables:
                for i, (var_name, var_value) in enumerate(variables.items()):
                    if isinstance(var_value, list):
                        for j, item in enumerate(var_value):
                            row = {var_name: item}
                            
                            # Add properties if available
                            if isinstance(item, str) and item in properties:
                                for prop_key, prop_value in properties[item].items():
                                    if not prop_key.startswith('_'):
                                        row[f"{var_name}.{prop_key}"] = prop_value
                            
                            rows.append(row)
                    else:
                        row = {var_name: var_value}
                        rows.append(row)
            
            # Otherwise, create rows from nodes
            elif nodes:
                for node_id in nodes:
                    row = {'node_id': node_id}
                    
                    if node_id in properties:
                        for prop_key, prop_value in properties[node_id].items():
                            if not prop_key.startswith('_'):
                                row[prop_key] = prop_value
                    
                    rows.append(row)
            
            return rows
            
        except Exception as e:
            self.logger.error(f"Failed to format as table rows: {e}")
            return [{"error": str(e)}]
    
    def _format_as_graph_data(self, result_data: Any, context) -> List[Dict[str, Any]]:
        """Format results as graph data (nodes and edges)"""
        
        try:
            nodes, properties, variables = self._extract_basic_data(result_data, context)
            
            # Create graph structure
            graph_nodes = []
            graph_edges = []
            
            # Add nodes
            for node_id in nodes:
                node = {
                    'id': node_id,
                    'properties': properties.get(node_id, {}),
                    'labels': self._extract_labels_from_properties(properties.get(node_id, {}))
                }
                graph_nodes.append(node)
            
            # Extract edges if available (simplified - would need relationship data)
            # For now, we don't have edge information in the basic result format
            
            return [{
                'nodes': graph_nodes,
                'edges': graph_edges,
                'metadata': {
                    'node_count': len(graph_nodes),
                    'edge_count': len(graph_edges)
                }
            }]
            
        except Exception as e:
            self.logger.error(f"Failed to format as graph data: {e}")
            return [{"error": str(e)}]
    
    def _format_as_json(self, result_data: Any, context) -> List[Dict[str, Any]]:
        """Format results as pure JSON"""
        
        try:
            nodes, properties, variables = self._extract_basic_data(result_data, context)
            
            result = {
                'nodes': list(nodes),
                'properties': properties,
                'variables': variables,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add context information if available
            if context:
                if hasattr(context, 'execution_stats'):
                    result['execution_stats'] = context.execution_stats
                
                if hasattr(context, 'redis_data'):
                    result['redis_data'] = context.redis_data
                
                if hasattr(context, 'graphblas_data'):
                    # Convert GraphBLAS data to serializable format
                    result['graphblas_data'] = self._serialize_graphblas_data(context.graphblas_data)
            
            return [result]
            
        except Exception as e:
            self.logger.error(f"Failed to format as JSON: {e}")
            return [{"error": str(e)}]
    
    def _extract_basic_data(self, result_data: Any, context) -> tuple:
        """Extract basic data components from result"""
        
        nodes = set()
        properties = {}
        variables = {}
        
        # Extract from result_data
        if hasattr(result_data, 'node_ids'):
            nodes.update(result_data.node_ids)
        if hasattr(result_data, 'properties'):
            properties.update(result_data.properties)
        if hasattr(result_data, 'data') and isinstance(result_data.data, dict):
            if 'nodes' in result_data.data:
                nodes.update(result_data.data['nodes'])
            if 'properties' in result_data.data:
                properties.update(result_data.data['properties'])
        elif isinstance(result_data, dict):
            if 'nodes' in result_data:
                nodes.update(result_data['nodes'])
            if 'properties' in result_data:
                properties.update(result_data['properties'])
        
        # Extract variables from context
        if context and hasattr(context, 'variables'):
            variables.update(context.variables)
        
        return nodes, properties, variables
    
    def _create_records_from_variables(self, variables: Dict[str, Any], 
                                     properties: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create records from context variables"""
        
        records = []
        
        # Find the maximum length among all variable values
        max_length = 1
        for value in variables.values():
            if isinstance(value, list):
                max_length = max(max_length, len(value))
        
        # Create records for each position
        for i in range(max_length):
            record = {}
            
            for var_name, var_value in variables.items():
                if isinstance(var_value, list):
                    if i < len(var_value):
                        item = var_value[i]
                        
                        # If item is a node ID and we have properties, create node object
                        if isinstance(item, str) and item in properties:
                            record[var_name] = {
                                'identity': item,
                                'labels': self._extract_labels_from_properties(properties[item]),
                                'properties': {k: v for k, v in properties[item].items() 
                                             if not k.startswith('_')}
                            }
                        else:
                            record[var_name] = item
                    else:
                        record[var_name] = None
                else:
                    # Single value - only add to first record
                    if i == 0:
                        if isinstance(var_value, str) and var_value in properties:
                            record[var_name] = {
                                'identity': var_value,
                                'labels': self._extract_labels_from_properties(properties[var_value]),
                                'properties': {k: v for k, v in properties[var_value].items() 
                                             if not k.startswith('_')}
                            }
                        else:
                            record[var_name] = var_value
                    else:
                        record[var_name] = None
            
            # Only add record if it has non-null values
            if any(v is not None for v in record.values()):
                records.append(record)
        
        return records
    
    def _extract_labels_from_properties(self, props: Dict[str, Any]) -> List[str]:
        """Extract labels from node properties"""
        
        labels = []
        
        # Look for label information in properties
        if '_labels' in props:
            if isinstance(props['_labels'], list):
                labels.extend(props['_labels'])
            elif isinstance(props['_labels'], str):
                labels.append(props['_labels'])
        
        if '_label' in props:
            if isinstance(props['_label'], str):
                labels.append(props['_label'])
        
        # Remove duplicates and return
        return list(set(labels))
    
    def _serialize_graphblas_data(self, graphblas_data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize GraphBLAS data to JSON-compatible format"""
        
        serialized = {}
        
        for key, value in graphblas_data.items():
            try:
                # Try to convert to basic types
                if hasattr(value, 'to_coo'):
                    # GraphBLAS matrix/vector
                    indices, values = value.to_coo()
                    serialized[key] = {
                        'type': 'graphblas_sparse',
                        'indices': indices.tolist() if hasattr(indices, 'tolist') else list(indices),
                        'values': values.tolist() if hasattr(values, 'tolist') else list(values),
                        'size': getattr(value, 'size', None)
                    }
                elif isinstance(value, dict):
                    serialized[key] = value
                elif hasattr(value, '__dict__'):
                    # Object with attributes
                    serialized[key] = {
                        'type': type(value).__name__,
                        'attributes': {k: v for k, v in value.__dict__.items() 
                                     if not k.startswith('_')}
                    }
                else:
                    # Try to serialize as-is
                    json.dumps(value)  # Test if serializable
                    serialized[key] = value
                    
            except Exception as e:
                # If serialization fails, store as string representation
                serialized[key] = {
                    'type': 'non_serializable',
                    'string_repr': str(value),
                    'error': str(e)
                }
        
        return serialized
    
    def format_summary_statistics(self, execution_result) -> Dict[str, Any]:
        """Format execution summary statistics"""
        
        try:
            summary = {
                'execution_time': getattr(execution_result, 'execution_time', 0.0),
                'operations_executed': getattr(execution_result, 'operations_executed', 0),
                'success': getattr(execution_result, 'success', False),
                'result_count': len(getattr(execution_result, 'data', [])),
                'redis_operations': getattr(execution_result, 'redis_operations', 0),
                'graphblas_operations': getattr(execution_result, 'graphblas_operations', 0),
                'coordinator_operations': getattr(execution_result, 'coordinator_operations', 0)
            }
            
            if hasattr(execution_result, 'error') and execution_result.error:
                summary['error'] = execution_result.error
            
            if hasattr(execution_result, 'context') and execution_result.context:
                context = execution_result.context
                if hasattr(context, 'execution_stats'):
                    summary['detailed_stats'] = context.execution_stats
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to format summary statistics: {e}")
            return {'error': f"Statistics formatting failed: {str(e)}"}
    
    def format_error_result(self, error: str, context=None) -> List[Dict[str, Any]]:
        """Format error result"""
        
        error_record = {
            'error': True,
            'message': error,
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            error_record['context'] = {
                'variables': getattr(context, 'variables', {}),
                'execution_stats': getattr(context, 'execution_stats', {})
            }
        
        return [error_record]
    
    def validate_output_format(self, format_name: str) -> bool:
        """Validate if output format is supported"""
        
        supported_formats = {
            'cypher_records',
            'table_rows', 
            'graph_data',
            'json'
        }
        
        return format_name.lower() in supported_formats