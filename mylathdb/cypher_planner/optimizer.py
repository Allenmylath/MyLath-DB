# ==========================================
# cypher_planner/optimizer.py
# ==========================================

"""
Rule-Based Optimizer for Logical Plans
"""

from .logical_operators import *
from .ast_nodes import *
# STEP 4: Update cypher_planner/optimizer.py
# Add these imports and methods to your existing RuleBasedOptimizer class:

# ADD TO IMPORTS:
from .filter_placement import FilterOptimizer, FilterNode, FilterType

class RuleBasedOptimizer:
    """ENHANCED: Rule-based optimizer with advanced filter placement"""

    def __init__(self):
        self.filter_optimizer = FilterOptimizer()  # NEW

    def optimize(self, plan: LogicalOperator) -> LogicalOperator:
        """ENHANCED: Apply rule-based optimizations with filter placement"""

        # Apply optimization rules in order
        optimized_plan = plan
        optimized_plan = self._push_down_property_filters(optimized_plan)
        optimized_plan = self._combine_adjacent_filters(optimized_plan)
        optimized_plan = self._optimize_scan_operations(optimized_plan)
        
        # NEW: Apply advanced filter optimization
        optimized_plan = self._apply_advanced_filter_optimization(optimized_plan)
        
        # NEW: Convert deprecated operations to new ones
        optimized_plan = self._modernize_operations(optimized_plan)

        return optimized_plan

    def _apply_advanced_filter_optimization(self, plan: LogicalOperator) -> LogicalOperator:
        """NEW: Apply advanced filter placement optimization"""
        
        # Collect all filter operations
        filters = self._collect_filters(plan)
        
        if not filters:
            return plan
        
        # Remove filters from plan and collect their conditions
        plan_without_filters = self._remove_filters(plan)
        filter_conditions = [f.condition for f in filters]
        
        # Apply advanced filter optimization
        optimized_plan = self.filter_optimizer.optimize_filters(
            plan_without_filters, filter_conditions)
        
        return optimized_plan
    
    def _collect_filters(self, plan: LogicalOperator) -> List[LogicalOperator]:
        """NEW: Collect all filter operations from the plan"""
        filters = []
        
        def collect_recursive(op):
            if isinstance(op, (Filter, PropertyFilter, StructuralFilter, PathFilter)):
                filters.append(op)
            for child in op.children:
                collect_recursive(child)
        
        collect_recursive(plan)
        return filters
    
    def _remove_filters(self, plan: LogicalOperator) -> LogicalOperator:
        """NEW: Remove filter operations from plan tree"""
        
        def remove_recursive(op):
            # Remove filter children and reconnect
            new_children = []
            for child in op.children:
                if isinstance(child, (Filter, PropertyFilter, StructuralFilter, PathFilter)):
                    # Skip filter, but process its children
                    new_children.extend([remove_recursive(grandchild) for grandchild in child.children])
                else:
                    new_children.append(remove_recursive(child))
            
            op.children = new_children
            return op
        
        return remove_recursive(plan)
    
    def _modernize_operations(self, plan: LogicalOperator) -> LogicalOperator:
        """NEW: Convert deprecated operations to new enhanced versions"""
        
        def modernize_recursive(op):
            # Convert children first
            for i, child in enumerate(op.children):
                op.children[i] = modernize_recursive(child)
            
            # Convert deprecated operations
            if isinstance(op, NodeScan):
                if op.labels and len(op.labels) == 1:
                    # Convert to NodeByLabelScan
                    new_op = NodeByLabelScan(op.variable, op.labels[0], op.properties)
                    new_op.children = op.children
                    new_op.parent = op.parent
                    return new_op
                elif not op.labels and not op.properties:
                    # Convert to AllNodeScan
                    new_op = AllNodeScan(op.variable)
                    new_op.children = op.children
                    new_op.parent = op.parent
                    return new_op
            
            elif isinstance(op, Expand):
                if op.max_length > 1:
                    # Convert to ConditionalVarLenTraverse
                    new_op = ConditionalVarLenTraverse(
                        op.from_var, op.to_var, op.rel_var,
                        op.rel_types, op.min_length, op.max_length, op.direction
                    )
                else:
                    # Convert to ConditionalTraverse
                    new_op = ConditionalTraverse(
                        op.from_var, op.to_var, op.rel_var,
                        op.rel_types, op.direction
                    )
                new_op.children = op.children
                new_op.parent = op.parent
                return new_op
            
            return op
        
        return modernize_recursive(plan)

    # KEEP ALL EXISTING METHODS:
    def _push_down_property_filters(self, plan: LogicalOperator) -> LogicalOperator:
        """Push property filters down to scan operations when possible"""
        
        if isinstance(plan, Filter) and plan.filter_type == "property":
            # Check if we can push this filter into a scan operation
            if (len(plan.children) == 1 
                and isinstance(plan.children[0], (NodeScan, NodeByLabelScan))
                and isinstance(plan.condition, BinaryExpression)
                and isinstance(plan.condition.left, PropertyExpression)):
                
                # Extract the property condition
                prop_expr = plan.condition.left
                scan_op = plan.children[0]
                
                if prop_expr.variable == scan_op.variable:
                    # We can push this filter into the scan
                    scan_op.properties[prop_expr.property_name] = plan.condition.right.value
                    return scan_op

        # Recursively optimize children
        for i, child in enumerate(plan.children):
            plan.children[i] = self._push_down_property_filters(child)

        return plan

    def _combine_adjacent_filters(self, plan: LogicalOperator) -> LogicalOperator:
        """Combine adjacent filter operations"""

        if isinstance(plan, Filter) and len(plan.children) == 1:
            child = plan.children[0]
            if isinstance(child, Filter):
                # Combine the two filters with AND
                combined_condition = BinaryExpression(
                    plan.condition, "AND", child.condition
                )
                combined_filter = Filter(combined_condition, "general")
                combined_filter.children = child.children
                return self._combine_adjacent_filters(combined_filter)

        # Recursively optimize children
        for i, child in enumerate(plan.children):
            plan.children[i] = self._combine_adjacent_filters(child)

        return plan

    def _optimize_scan_operations(self, plan: LogicalOperator) -> LogicalOperator:
        """Optimize scan operations based on selectivity"""

        if isinstance(plan, (NodeScan, NodeByLabelScan)):
            # If we have both labels and properties, prioritize the most selective
            if hasattr(plan, 'labels') and hasattr(plan, 'properties'):
                if plan.labels and plan.properties:
                    plan.optimized = True

        # Recursively optimize children
        for i, child in enumerate(plan.children):
            plan.children[i] = self._optimize_scan_operations(child)

        return plan
