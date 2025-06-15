# mylathdb/execution_engine/result_formatter.py

"""
MyLathDB Result Formatter - COMPLETE IMPLEMENTATION
Formats execution results for different output requirements based on FalkorDB patterns
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import decimal

from .utils import extract_node_id, format_execution_time

logger = logging.getLogger(__name__)

@dataclass
class ResultSet:
    """Structured result set for MyLathDB queries"""
    columns: List[str] = field(default_factory=list)
    data: List[List[Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def row_count(self) -> int:
        """Number of result rows"""
        return len(self.data)
    
    def to_dict_records(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries"""
        return [dict(zip(self.columns, row)) for row in self.data]
    
    def to_json(self, indent: int = None) -> str:
        """Convert to JSON string"""
        return json.dumps({
            'columns': self.columns,
            'data': self._serialize_data_for_json(),
            'statistics': self.statistics,
            'metadata': self.metadata
        }, indent=indent, default=self._json_serializer)
    
    def _serialize_data_for_json(self) -> List[List[Any]]:
        """Serialize data for JSON output"""
        serialized_data = []
        for row in self.data:
            serialized_row = []
            for item in row:
                serialized_row.append(self._serialize_value(item))
            serialized_data.append(serialized_row)
        return serialized_data
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value for JSON"""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, decimal.Decimal):
            return float(value)
        else:
            return str(value)
    
    @staticmethod
    def _json_serializer(obj):
        """JSON serializer for non-standard types"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def to_csv(self, separator: str = ',') -> str:
        """Convert to CSV format"""
        lines = []
        
        # Add header
        lines.append(separator.join(f'"{col}"' for col in self.columns))
        
        # Add data rows
        for row in self.data:
            csv_row = []
            for item in row:
                # Format item for CSV
                if item is None:
                    csv_row.append('')
                elif isinstance(item, str):
                    # Escape quotes and wrap in quotes
                    escaped = item.replace('"', '""')
                    csv_row.append(f'"{escaped}"')
                elif isinstance(item, (dict, list)):
                    # Serialize complex objects as JSON strings
                    csv_row.append(f'"{json.dumps(item)}"')
                else:
                    csv_row.append(str(item))
            lines.append(separator.join(csv_row))
        
        return '\n'.join(lines)
    
    def to_table(self, max_width: int = 100) -> str:
        """Convert to formatted table string"""
        if not self.columns:
            return "No data"
        
        # Calculate column widths
        col_widths = []
        for i, col in enumerate(self.columns):
            width = len(col)
            for row in self.data:
                if i < len(row):
                    cell_str = str(row[i]) if row[i] is not None else ''
                    width = max(width, len(cell_str))
            # Limit column width
            col_widths.append(min(width, max_width // len(self.columns)))
        
        # Build table
        lines = []
        
        # Header
        header_line = ' | '.join(col.ljust(col_widths[i]) for i, col in enumerate(self.columns))
        lines.append(header_line)
        lines.append('-' * len(header_line))
        
        # Data rows
        for row in self.data:
            row_cells = []
            for i, cell in enumerate(row):
                if i >= len(col_widths):
                    break
                cell_str = str(cell) if cell is not None else ''
                # Truncate if too long
                if len(cell_str) > col_widths[i]:
                    cell_str = cell_str[:col_widths[i]-3] + '...'
                row_cells.append(cell_str.ljust(col_widths[i]))
            lines.append(' | '.join(row_cells))
        
        return '\n'.join(lines)

class ResultFormatter:
    """
    Result formatter for MyLathDB execution results
    Based on FalkorDB's result formatting patterns
    
    Handles:
    - Converting internal result format to standard formats
    - Applying projections and aliases from logical plans
    - Formatting different data types (nodes, edges, properties)
    - Creating tabular and structured output
    - Type coercion and data validation
    """
    
    def __init__(self):
        """Initialize result formatter"""
        self.default_format = "dict_records"
        self.max_string_length = 1000
        self.max_array_length = 100
        self.date_format = "%Y-%m-%d %H:%M:%S"
        
        # Type formatters
        self.type_formatters = {
            'node': self._format_node,
            'edge': self._format_edge,
            'relationship': self._format_edge,  # Alias for edge
            'path': self._format_path,
            'list': self._format_list,
            'map': self._format_map,
            'datetime': self._format_datetime,
            'duration': self._format_duration,
            'point': self._format_point,
        }
    
    def format_results(self, results: List[Dict[str, Any]], 
                      physical_plan, format_type: str = None) -> List[Dict[str, Any]]:
        """
        Format execution results based on physical plan projections
        
        Args:
            results: Raw execution results from executors
            physical_plan: Physical plan with projection information
            format_type: Output format type ('dict_records', 'table', 'json', etc.)
            
        Returns:
            Formatted results
        """
        if not results:
            return []
        
        try:
            # Apply projections if specified in plan
            projected_results = self._apply_projections(results, physical_plan)
            
            # Format data types according to Cypher/FalkorDB conventions
            formatted_results = self._format_data_types(projected_results)
            
            # Apply final output format
            final_results = self._apply_output_format(formatted_results, format_type or self.default_format)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Result formatting failed: {e}", exc_info=True)
            return results  # Return original results on formatting error
    
    def create_result_set(self, results: List[Dict[str, Any]], 
                         physical_plan, statistics: Dict[str, Any] = None) -> ResultSet:
        """
        Create a structured ResultSet from execution results
        
        Args:
            results: Execution results
            physical_plan: Physical plan for projection info
            statistics: Execution statistics
            
        Returns:
            Structured ResultSet object
        """
        if not results:
            return ResultSet(
                columns=[],
                data=[],
                statistics=statistics or {},
                metadata={'empty_result': True}
            )
        
        # Extract column names from projections or first result
        columns = self._extract_column_names(results, physical_plan)
        
        # Convert results to tabular format
        tabular_data = self._convert_to_tabular(results, columns)
        
        # Add metadata
        metadata = {
            'result_type': 'query_result',
            'row_count': len(tabular_data),
            'column_count': len(columns),
            'formatted_at': datetime.now().isoformat()
        }
        
        return ResultSet(
            columns=columns,
            data=tabular_data,
            statistics=statistics or {},
            metadata=metadata
        )
    
    def _apply_projections(self, results: List[Dict[str, Any]], physical_plan) -> List[Dict[str, Any]]:
        """Apply projections from physical plan"""
        
        # Extract projection information from plan
        projections = self._extract_projections(physical_plan)
        
        if not projections:
            return results
        
        projected_results = []
        
        for result in results:
            projected_record = {}
            
            for expr, alias in projections:
                try:
                    # Evaluate projection expression
                    value = self._evaluate_projection_expression(expr, result)
                    
                    # Use alias if provided, otherwise derive name from expression
                    key = alias if alias else self._derive_expression_name(expr)
                    projected_record[key] = value
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate projection {expr}: {e}")
                    # Use null value for failed projections
                    key = alias if alias else str(expr)
                    projected_record[key] = None
            
            projected_results.append(projected_record)
        
        return projected_results
    
    def _extract_projections(self, physical_plan) -> List[Tuple[Any, Optional[str]]]:
        """Extract projection expressions from physical plan"""
        projections = []
        
        # Look for Project operations in the plan tree
        def find_projections(op):
            if hasattr(op, 'operation_type') and op.operation_type == "Project":
                logical_op = getattr(op, 'logical_op', None)
                if logical_op and hasattr(logical_op, 'projections'):
                    projections.extend(logical_op.projections)
            
            # Recursively check children
            for child in getattr(op, 'children', []):
                find_projections(child)
        
        if physical_plan:
            find_projections(physical_plan)
        
        return projections
    
    def _evaluate_projection_expression(self, expr, result: Dict[str, Any]) -> Any:
        """
        Evaluate projection expression against result record
        Based on FalkorDB's expression evaluation patterns
        """
        
        from ..cypher_planner.ast_nodes import (
            PropertyExpression, VariableExpression, LiteralExpression,
            BinaryExpression, FunctionCall
        )
        
        if isinstance(expr, PropertyExpression):
            # Property access: variable.property
            entity = result.get(expr.variable)
            if entity and isinstance(entity, dict):
                return entity.get(expr.property_name)
            return None
            
        elif isinstance(expr, VariableExpression):
            # Simple variable reference
            return result.get(expr.name)
            
        elif isinstance(expr, LiteralExpression):
            # Literal value
            return expr.value
            
        elif isinstance(expr, BinaryExpression):
            # Binary operation: left operator right
            left_val = self._evaluate_projection_expression(expr.left, result)
            right_val = self._evaluate_projection_expression(expr.right, result)
            
            return self._apply_binary_operator(left_val, expr.operator, right_val)
            
        elif isinstance(expr, FunctionCall):
            # Function call
            args = [self._evaluate_projection_expression(arg, result) for arg in expr.arguments]
            return self._apply_function(expr.name, args)
            
        else:
            # Unknown expression type - convert to string
            logger.warning(f"Unknown expression type: {type(expr)}")
            return str(expr)
    
    def _apply_binary_operator(self, left: Any, operator: str, right: Any) -> Any:
        """Apply binary operator to values"""
        try:
            if operator == '+':
                return left + right
            elif operator == '-':
                return left - right
            elif operator == '*':
                return left * right
            elif operator == '/':
                return left / right if right != 0 else None
            elif operator == '%':
                return left % right if right != 0 else None
            elif operator == '=':
                return left == right
            elif operator == '!=':
                return left != right
            elif operator == '<':
                return left < right
            elif operator == '<=':
                return left <= right
            elif operator == '>':
                return left > right
            elif operator == '>=':
                return left >= right
            elif operator.upper() == 'AND':
                return bool(left) and bool(right)
            elif operator.upper() == 'OR':
                return bool(left) or bool(right)
            else:
                logger.warning(f"Unknown binary operator: {operator}")
                return None
        except Exception as e:
            logger.warning(f"Binary operation failed: {left} {operator} {right}: {e}")
            return None
    
    def _apply_function(self, func_name: str, args: List[Any]) -> Any:
        """Apply function to arguments"""
        func_name_upper = func_name.upper()
        
        try:
            if func_name_upper == 'COUNT':
                return len([arg for arg in args if arg is not None])
            elif func_name_upper == 'SIZE':
                if args and hasattr(args[0], '__len__'):
                    return len(args[0])
                return 0
            elif func_name_upper == 'LENGTH':
                if args and isinstance(args[0], str):
                    return len(args[0])
                return 0
            elif func_name_upper == 'UPPER':
                if args and isinstance(args[0], str):
                    return args[0].upper()
                return args[0] if args else None
            elif func_name_upper == 'LOWER':
                if args and isinstance(args[0], str):
                    return args[0].lower()
                return args[0] if args else None
            elif func_name_upper == 'TRIM':
                if args and isinstance(args[0], str):
                    return args[0].strip()
                return args[0] if args else None
            elif func_name_upper == 'TYPE':
                if args:
                    return self._get_cypher_type(args[0])
                return 'NULL'
            elif func_name_upper == 'ID':
                if args and isinstance(args[0], dict):
                    return args[0].get('_id') or args[0].get('id')
                return None
            elif func_name_upper == 'LABELS':
                if args and isinstance(args[0], dict):
                    return args[0].get('_labels', [])
                return []
            elif func_name_upper == 'KEYS':
                if args and isinstance(args[0], dict):
                    return [k for k in args[0].keys() if not k.startswith('_')]
                return []
            elif func_name_upper == 'PROPERTIES':
                if args and isinstance(args[0], dict):
                    return {k: v for k, v in args[0].items() if not k.startswith('_')}
                return {}
            elif func_name_upper == 'COALESCE':
                for arg in args:
                    if arg is not None:
                        return arg
                return None
            else:
                logger.warning(f"Unknown function: {func_name}")
                return None
                
        except Exception as e:
            logger.warning(f"Function {func_name} failed with args {args}: {e}")
            return None
    
    def _get_cypher_type(self, value: Any) -> str:
        """Get Cypher type name for value"""
        if value is None:
            return 'NULL'
        elif isinstance(value, bool):
            return 'Boolean'
        elif isinstance(value, int):
            return 'Integer'
        elif isinstance(value, float):
            return 'Float'
        elif isinstance(value, str):
            return 'String'
        elif isinstance(value, list):
            return 'List'
        elif isinstance(value, dict):
            # Check if it's a node or edge
            if '_labels' in value:
                return 'Node'
            elif '_type' in value or ('_source' in value and '_target' in value):
                return 'Relationship'
            else:
                return 'Map'
        else:
            return 'Unknown'
    
    def _derive_expression_name(self, expr) -> str:
        """Derive a name from an expression for use as a column name"""
        
        from ..cypher_planner.ast_nodes import (
            PropertyExpression, VariableExpression, LiteralExpression,
            BinaryExpression, FunctionCall
        )
        
        if isinstance(expr, PropertyExpression):
            return f"{expr.variable}.{expr.property_name}"
        elif isinstance(expr, VariableExpression):
            return expr.name
        elif isinstance(expr, LiteralExpression):
            return str(expr.value)
        elif isinstance(expr, FunctionCall):
            return f"{expr.name}(...)"
        elif isinstance(expr, BinaryExpression):
            return f"({self._derive_expression_name(expr.left)} {expr.operator} {self._derive_expression_name(expr.right)})"
        else:
            return str(expr)
    
    def _format_data_types(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format data types according to FalkorDB/Cypher conventions"""
        
        formatted_results = []
        
        for result in results:
            formatted_record = {}
            
            for key, value in result.items():
                formatted_record[key] = self._format_value(value)
            
            formatted_results.append(formatted_record)
        
        return formatted_results
    
    def _format_value(self, value: Any) -> Any:
        """Format a single value according to its detected type"""
        
        if value is None:
            return None
        
        # Detect and format based on type
        value_type = self._detect_value_type(value)
        
        if value_type in self.type_formatters:
            return self.type_formatters[value_type](value)
        else:
            return value
    
    def _detect_value_type(self, value: Any) -> str:
        """Detect the logical type of a value"""
        
        if isinstance(value, dict):
            if '_labels' in value:
                return 'node'
            elif '_type' in value or ('_source' in value and '_target' in value):
                return 'edge'
            elif all(isinstance(k, str) for k in value.keys()):
                return 'map'
        elif isinstance(value, list):
            return 'list'
        elif isinstance(value, datetime):
            return 'datetime'
        
        return 'scalar'
    
    def _format_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Format node according to FalkorDB conventions"""
        
        formatted_node = {
            'id': node.get('_id') or node.get('id'),
            'labels': node.get('_labels', []),
            'properties': {}
        }
        
        # Extract properties (non-internal keys)
        for key, value in node.items():
            if not key.startswith('_') and key != 'id':
                formatted_node['properties'][key] = self._format_value(value)
        
        return formatted_node
    
    def _format_edge(self, edge: Dict[str, Any]) -> Dict[str, Any]:
        """Format edge/relationship according to FalkorDB conventions"""
        
        formatted_edge = {
            'id': edge.get('_id') or edge.get('id'),
            'type': edge.get('_type') or edge.get('type', 'UNKNOWN'),
            'source': edge.get('_source') or edge.get('source'),
            'target': edge.get('_target') or edge.get('target'),
            'properties': {}
        }
        
        # Extract properties
        internal_keys = {'_id', 'id', '_type', 'type', '_source', 'source', '_target', 'target'}
        for key, value in edge.items():
            if key not in internal_keys:
                formatted_edge['properties'][key] = self._format_value(value)
        
        return formatted_edge
    
    def _format_path(self, path: List[Any]) -> Dict[str, Any]:
        """Format path according to FalkorDB conventions"""
        
        nodes = []
        relationships = []
        
        for i, element in enumerate(path):
            if i % 2 == 0:  # Even indices are nodes
                nodes.append(self._format_value(element))
            else:  # Odd indices are relationships
                relationships.append(self._format_value(element))
        
        return {
            'length': len(relationships),
            'nodes': nodes,
            'relationships': relationships
        }
    
    def _format_list(self, lst: List[Any]) -> List[Any]:
        """Format list with element formatting"""
        
        # Limit list size for performance
        if len(lst) > self.max_array_length:
            formatted_list = [self._format_value(item) for item in lst[:self.max_array_length]]
            formatted_list.append(f"... and {len(lst) - self.max_array_length} more items")
            return formatted_list
        else:
            return [self._format_value(item) for item in lst]
    
    def _format_map(self, mapping: Dict[str, Any]) -> Dict[str, Any]:
        """Format map/dictionary"""
        
        return {key: self._format_value(value) for key, value in mapping.items()}
    
    def _format_datetime(self, dt: datetime) -> str:
        """Format datetime according to ISO standard"""
        return dt.strftime(self.date_format)
    
    def _format_duration(self, duration: Any) -> str:
        """Format duration (placeholder for duration objects)"""
        return str(duration)
    
    def _format_point(self, point: Any) -> Dict[str, Any]:
        """Format spatial point (placeholder for point objects)"""
        if isinstance(point, dict):
            return point
        else:
            return {'point': str(point)}
    
    def _extract_column_names(self, results: List[Dict[str, Any]], physical_plan) -> List[str]:
        """Extract column names from projections or results"""
        
        # Try to get from projections first
        projections = self._extract_projections(physical_plan)
        if projections:
            columns = []
            for expr, alias in projections:
                if alias:
                    columns.append(alias)
                else:
                    columns.append(self._derive_expression_name(expr))
            return columns
        
        # Fallback to result keys
        if results:
            return list(results[0].keys())
        
        return []
    
    def _convert_to_tabular(self, results: List[Dict[str, Any]], columns: List[str]) -> List[List[Any]]:
        """Convert dictionary results to tabular format"""
        
        tabular_data = []
        
        for result in results:
            row = []
            for col in columns:
                value = result.get(col)
                # Convert complex objects to strings for tabular display
                if isinstance(value, (dict, list)) and not self._is_simple_dict(value):
                    row.append(self._summarize_complex_value(value))
                else:
                    row.append(value)
            tabular_data.append(row)
        
        return tabular_data
    
    def _is_simple_dict(self, value: Any) -> bool:
        """Check if dictionary is simple enough for direct display"""
        if not isinstance(value, dict):
            return False
        
        return len(value) <= 3 and all(
            isinstance(v, (str, int, float, bool, type(None))) 
            for v in value.values()
        )
    
    def _summarize_complex_value(self, value: Any) -> str:
        """Create a summary string for complex values"""
        
        if isinstance(value, dict):
            if '_labels' in value:
                # Node summary
                labels = value.get('_labels', [])
                label_str = ':'.join(labels) if labels else 'Node'
                return f"({label_str})"
            elif '_type' in value:
                # Edge summary
                return f"[:{value.get('_type', 'UNKNOWN')}]"
            else:
                # Generic map
                return f"{{...}} ({len(value)} keys)"
        elif isinstance(value, list):
            return f"[...] ({len(value)} items)"
        else:
            str_val = str(value)
            if len(str_val) > 50:
                return str_val[:47] + "..."
            return str_val
    
    def _apply_output_format(self, results: List[Dict[str, Any]], format_type: str) -> List[Dict[str, Any]]:
        """Apply final output formatting"""
        
        # For now, dict_records is the main format
        # Other formats would be handled by create_result_set()
        return results
