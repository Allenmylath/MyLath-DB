# ==========================================
# cypher_planner/optimizer.py
# ==========================================

"""
Rule-Based Optimizer for Logical Plans
"""

from .logical_operators import *
from .ast_nodes import *


class RuleBasedOptimizer:
    """Implements basic rule-based optimizations for the logical plan"""

    def optimize(self, plan: LogicalOperator) -> LogicalOperator:
        """Apply rule-based optimizations to the plan"""

        # Apply optimization rules in order
        optimized_plan = plan
        optimized_plan = self._push_down_property_filters(optimized_plan)
        optimized_plan = self._combine_adjacent_filters(optimized_plan)
        optimized_plan = self._optimize_scan_operations(optimized_plan)

        return optimized_plan

    def _push_down_property_filters(self, plan: LogicalOperator) -> LogicalOperator:
        """Push property filters down to scan operations when possible"""

        if isinstance(plan, Filter) and plan.filter_type == "property":
            # Check if we can push this filter into a scan operation
            if (
                len(plan.children) == 1
                and isinstance(plan.children[0], NodeScan)
                and isinstance(plan.condition, BinaryExpression)
                and isinstance(plan.condition.left, PropertyExpression)
            ):
                # Extract the property condition
                prop_expr = plan.condition.left
                if prop_expr.variable == plan.children[0].variable:
                    # We can push this filter into the NodeScan
                    scan = plan.children[0]
                    scan.properties[prop_expr.property_name] = (
                        plan.condition.right.value
                    )
                    return scan

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

        if isinstance(plan, NodeScan):
            # If we have both labels and properties, prioritize the most selective
            if plan.labels and plan.properties:
                # In a real system, you'd use statistics to determine selectivity
                # For now, we'll just mark it as optimized
                plan.optimized = True

        # Recursively optimize children
        for i, child in enumerate(plan.children):
            plan.children[i] = self._optimize_scan_operations(child)

        return plan
