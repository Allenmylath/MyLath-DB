# Add at the top of filter_placement.py
from __future__ import annotations
from typing import List, Set, Dict, Optional, Tuple, Union
from enum import Enum
from .logical_operators import *
from .ast_nodes import Expression, BinaryExpression, PropertyExpression, VariableExpression, FunctionCall

class FilterType(Enum):
    PROPERTY = "property"
    STRUCTURAL = "structural" 
    PATH = "path"
    GENERAL = "general"

class FilterNode:
    """Represents a filter condition in the filter tree"""
    def __init__(self, condition: Expression, filter_type: FilterType):
        self.condition = condition
        self.filter_type = filter_type
        self.referenced_variables: Set[str] = set()
        self.cost_estimate: float = 0.0
        self._extract_variables()
    
    def _extract_variables(self):
        """Extract all variables referenced in this filter"""
        def extract_from_expression(expr):
            if isinstance(expr, PropertyExpression):
                self.referenced_variables.add(expr.variable)
            elif isinstance(expr, VariableExpression):
                self.referenced_variables.add(expr.name)
            elif isinstance(expr, BinaryExpression):
                extract_from_expression(expr.left)
                extract_from_expression(expr.right)
            elif isinstance(expr, FunctionCall):
                for arg in expr.arguments:
                    extract_from_expression(arg)
        
        extract_from_expression(self.condition)

class FilterPlacementEngine:
    """Advanced filter placement engine"""
    
    # Operations that should not have filters pushed below them
    FILTER_BLACKLIST = [
        "Merge", "Apply", "Optional", "SemiApply", 
        "ApplyMultiplexer", "ArgumentList"
    ]
    
    def __init__(self):
        pass
    
    def place_filters(self, root_op: LogicalOperator, filters: List[FilterNode]) -> LogicalOperator:
        """Place filters at optimal positions in the execution plan"""
        
        # Step 1: Analyze each filter and determine optimal placement
        filter_placements = []
        for filter_node in filters:
            placement = self._find_optimal_placement(root_op, filter_node)
            filter_placements.append((filter_node, placement))
        
        # Step 2: Sort filters by cost (cheaper filters first)
        filter_placements.sort(key=lambda x: x[0].cost_estimate)
        
        # Step 3: Place filters in order
        current_root = root_op
        for filter_node, placement_op in filter_placements:
            current_root = self._place_filter(current_root, filter_node, placement_op)
        
        return current_root
    
    def _find_optimal_placement(self, root_op: LogicalOperator, 
                               filter_node: FilterNode) -> LogicalOperator:
        """Find the optimal placement for a filter"""
        
        if not filter_node.referenced_variables:
            # Filter with no variable references (e.g., WHERE 1=1)
            return root_op
        
        # Find the earliest operation that's aware of all referenced variables
        optimal_op = self._find_earliest_aware_operation(root_op, filter_node.referenced_variables)
        
        if not optimal_op:
            raise ValueError(f"Cannot resolve variables {filter_node.referenced_variables} in plan")
        
        return optimal_op
    
    # FIX 1: Update cypher_planner/filter_placement.py
# Replace the _find_earliest_aware_operation method with this fixed version:

    def _find_earliest_aware_operation(self, root_op: LogicalOperator, 
                                    variables: Set[str]) -> Optional[LogicalOperator]:
        """Find earliest operation aware of all variables, respecting blacklist"""
        
        def search(op: LogicalOperator) -> Optional[LogicalOperator]:
            # Check if this operation is blacklisted
            if any(blacklisted in type(op).__name__ for blacklisted in self.FILTER_BLACKLIST):
                return op  # Don't go below blacklisted operations
            
            # Check if current operation is aware of all variables
            # FIXED: Also check modifies list for variables
            op_variables = set(op.awareness) | set(op.modifies)
            if not all(var in op_variables for var in variables):
                return None
            
            # Try to go deeper
            for child in op.children:
                deeper = search(child)
                if deeper and deeper != op:  # Found a deeper valid position
                    return deeper
            
            # This is the deepest valid position
            return op
        
        result = search(root_op)
        
        # FALLBACK: If no operation found, return root
        if result is None:
            print(f"Warning: Could not find operation aware of {variables}, using root")
            return root_op
        
        return result    
    def _place_filter(self, root_op: LogicalOperator, filter_node: FilterNode, 
                     target_op: LogicalOperator) -> LogicalOperator:
        """Place a filter operation at the specified target"""
        
        # Create appropriate filter operation based on type
        if filter_node.filter_type == FilterType.PROPERTY:
            filter_op = self._create_property_filter(filter_node)
        elif filter_node.filter_type == FilterType.STRUCTURAL:
            filter_op = StructuralFilter(filter_node.condition)
        elif filter_node.filter_type == FilterType.PATH:
            filter_op = PathFilter(str(filter_node.condition))
        else:
            filter_op = Filter(filter_node.condition, "general")
        
        # Insert filter operation above target
        if target_op.parent:
            # Replace target in parent's children
            parent = target_op.parent
            for i, child in enumerate(parent.children):
                if child == target_op:
                    parent.children[i] = filter_op
                    break
            filter_op.parent = parent
        else:
            # Target is root, so filter becomes new root
            root_op = filter_op
        
        filter_op.add_child(target_op)
        return root_op
    
    def _create_property_filter(self, filter_node: FilterNode) -> PropertyFilter:
        """Create a specialized property filter"""
        
        # Extract property access details from condition
        if isinstance(filter_node.condition, BinaryExpression):
            if isinstance(filter_node.condition.left, PropertyExpression):
                prop_expr = filter_node.condition.left
                return PropertyFilter(
                    prop_expr.variable,
                    prop_expr.property_name,
                    filter_node.condition.operator,
                    filter_node.condition.right.value
                )
        
        # Fallback to general filter
        return Filter(filter_node.condition, "property")
    
    def estimate_filter_cost(self, filter_node: FilterNode, 
                           cardinality_estimate: int = 1000) -> float:
        """Estimate the cost of applying a filter"""
        
        base_costs = {
            FilterType.PROPERTY: 0.1,    # Very cheap with indexes
            FilterType.STRUCTURAL: 1.0,  # Moderate cost
            FilterType.PATH: 10.0,       # Expensive path matching
            FilterType.GENERAL: 5.0      # Expensive general evaluation
        }
        
        base_cost = base_costs.get(filter_node.filter_type, 5.0)
        variable_cost = len(filter_node.referenced_variables) * 0.5
        
        return (base_cost + variable_cost) * cardinality_estimate

class FilterOptimizer:
    """Optimizes filter conditions and placement"""
    
    def __init__(self):
        self.placement_engine = FilterPlacementEngine()
    
    def optimize_filters(self, root_op: LogicalOperator, 
                        where_conditions: List[Expression]) -> LogicalOperator:
        """Optimize and place all filter conditions"""
        
        # Step 1: Analyze and classify filters
        filter_nodes = []
        for condition in where_conditions:
            filter_type = self._classify_filter(condition)
            filter_node = FilterNode(condition, filter_type)
            filter_node.cost_estimate = self.placement_engine.estimate_filter_cost(filter_node)
            filter_nodes.append(filter_node)
        
        # Step 2: Decompose complex filters
        decomposed_filters = self._decompose_filters(filter_nodes)
        
        # Step 3: Place filters optimally
        optimized_plan = self.placement_engine.place_filters(root_op, decomposed_filters)
        
        return optimized_plan
    
    def _classify_filter(self, condition: Expression) -> FilterType:
        """Classify filter based on its structure"""
        
        def analyze_expression(expr) -> FilterType:
            if isinstance(expr, PropertyExpression):
                return FilterType.PROPERTY
            elif isinstance(expr, BinaryExpression):
                if isinstance(expr.left, PropertyExpression):
                    return FilterType.PROPERTY
                elif "path" in str(expr).lower():
                    return FilterType.PATH
                else:
                    return FilterType.STRUCTURAL
            elif isinstance(expr, FunctionCall):
                if expr.name.lower() in ['exists', 'path_filter']:
                    return FilterType.PATH
                else:
                    return FilterType.GENERAL
            
            return FilterType.GENERAL
        
        return analyze_expression(condition)
    
    def _decompose_filters(self, filter_nodes: List[FilterNode]) -> List[FilterNode]:
        """Decompose complex filters into simpler ones"""
        
        decomposed = []
        
        for filter_node in filter_nodes:
            if isinstance(filter_node.condition, BinaryExpression):
                if filter_node.condition.operator.upper() == "AND":
                    # Split AND conditions into separate filters
                    left_filter = FilterNode(filter_node.condition.left, 
                                           self._classify_filter(filter_node.condition.left))
                    right_filter = FilterNode(filter_node.condition.right,
                                            self._classify_filter(filter_node.condition.right))
                    decomposed.extend([left_filter, right_filter])
                else:
                    decomposed.append(filter_node)
            else:
                decomposed.append(filter_node)
        
        return decomposed