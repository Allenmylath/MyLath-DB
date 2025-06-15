# mylathdb/execution_engine/result_formatter.py

"""
MyLathDB Result Formatter
Formats execution results for different output requirements
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

from .utils import extract_node_id, format_execution_time

logger = logging.getLogger(__name__)

@dataclass
class ResultSet:
    """Structured result set for MyLathDB queries"""
    columns: List[str] = field(default_factory=list)
    data: List[List[Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def row_count(self) -> int:
        """Number of result rows"""
        return len(self.data)
    
    def to_dict_records(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries"""
        return [dict(zip(self.columns, row)) for row in self.data]
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({
            'columns': self.columns,
            'data': self.data,
            'statistics': self.statistics
        })

class ResultFormatter:
    """
    Result formatter for MyLathDB execution results
    
    Handles:
    - Converting internal result format to standard formats
    - Applying projections and aliases
    - Formatting different data types
    - Creating tabular output
    """
    
    def __init__(self):
        """Initialize result formatter"""
        self.default_format = "dict_records"
        self.max_string_length = 1000
        self.max_array_length = 100
    
    def format_results(self, results: List[Dict[str, Any]], 
                      physical_plan, format_type: str = None) -> List[Dict[str, Any]]:
        """
        Format execution results based on physical plan projections
        
        Args:
            results: Raw execution results
            physical_plan: Physical plan with projection information
            format_type: Output format type
            
        Returns:
            Formatted results
        """
        if not results:
            return []
        
        try:
            # Apply projections if specified in plan
            projected_results = self._apply_projections(results, physical_plan)
            
            # Format data types
            formatted_results = self._format_data_types(projected_results)
            
            # Apply result format
            final_results = self._apply_format(formatted_results, format_type or self.default_format)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Result formatting failed: {e}")
            return results  # Return original results on formatting error
    
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
                # Evaluate projection expression
                value = self._evaluate_projection_expression(expr, result)
                
                # Use alias if provided, otherwise derive name from expression
                key = alias if alias else self._derive_expression_name(expr)
                projected_record[key] = value
            
            projected_results.append(projected_record)
        
        return projected_results
    
    def _extract_projections(self, physical_plan) -> List[tuple]:
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
        
        find_projections(physical_plan)
        return projections
    
    def _evaluate_projection_expression(self, expr, result: Dict[str, Any]) -> Any:
        """Evaluate projection expression against result record"""
        
        from ..cypher_planner.ast_nodes import (
            PropertyExpression, VariableExpression, LiteralExpression,
            BinaryExpression, FunctionCall
        )
        
        if isinstance(expr, PropertyExpression):
            # Property access: variable.property
            entity = result.get(expr.variable)

