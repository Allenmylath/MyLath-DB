# =============================================================================
# FIXED cypher_planner/logical_planner.py - Replace the existing logical_planner.py with this
# =============================================================================

"""
Logical Planner - FIXED VERSION
Converts AST to Logical Execution Plans with proper relationship handling
"""

from typing import Optional
from .ast_nodes import *
from .logical_operators import *


class LogicalPlanner:
    def __init__(self):
        self.variable_counter = 0

    def create_logical_plan(self, ast: Query) -> LogicalOperator:
        """Convert AST to logical execution plan"""

        # Start with data sourcing operations
        plan = self._create_data_source_plan(ast)

        # Add filtering
        if ast.where_clause:
            plan = self._add_filter_operations(plan, ast.where_clause)

        # Add projections
        if ast.return_clause:
            plan = self._add_projection_operations(plan, ast.return_clause)

        return plan

    def _create_data_source_plan(self, ast: Query) -> LogicalOperator:
        """Create the initial data sourcing part of the plan"""

        if not ast.match_clauses:
            return None

        # Handle all MATCH clauses
        plans = []
        for match_clause in ast.match_clauses:
            for pattern in match_clause.patterns:
                plan = self._create_pattern_plan(pattern)
                if plan:
                    plans.append(plan)

        # If multiple plans, join them
        if len(plans) == 1:
            return plans[0]
        elif len(plans) > 1:
            # Create joins for multiple patterns
            current_plan = plans[0]
            for plan in plans[1:]:
                # Find common variables for joining
                join_vars = self._find_common_variables(current_plan, plan)
                join_op = Join(join_vars)
                join_op.children = [current_plan, plan]
                current_plan = join_op
            return current_plan

        return None

    def _create_pattern_plan(self, pattern: Pattern) -> LogicalOperator:
        """Create logical plan for a single pattern - FIXED VERSION"""

        if not pattern.elements:
            return None

        # Find all nodes and relationships in the pattern
        nodes = []
        relationships = []

        for element in pattern.elements:
            if isinstance(element, NodePattern):
                nodes.append(element)
            elif isinstance(element, RelationshipPattern):
                relationships.append(element)

        if not nodes:
            return None

        # Start with the first node
        first_node = nodes[0]
        current_plan = NodeScan(
            variable=first_node.variable or f"_node_{self.variable_counter}",
            labels=first_node.labels,
            properties=first_node.properties,
        )

        if not first_node.variable:
            self.variable_counter += 1

        current_var = first_node.variable or current_plan.variable

        # Process relationships and subsequent nodes
        for i, rel in enumerate(relationships):
            if i + 1 < len(nodes):
                next_node = nodes[i + 1]
                next_var = next_node.variable or f"_node_{self.variable_counter}"

                if not next_node.variable:
                    self.variable_counter += 1

                # Create expand operation
                expand = Expand(
                    from_var=current_var,
                    to_var=next_var,
                    rel_var=rel.variable,
                    rel_types=rel.types,
                    direction=rel.direction,
                    min_length=rel.min_length or 1,
                    max_length=rel.max_length or 1,
                )

                expand.children = [current_plan]
                current_plan = expand
                current_var = next_var

                # Add property filters for the target node
                if next_node.properties:
                    for prop_key, prop_value in next_node.properties.items():
                        filter_condition = BinaryExpression(
                            PropertyExpression(next_var, prop_key),
                            "=",
                            LiteralExpression(prop_value),
                        )
                        filter_op = Filter(filter_condition, "property")
                        filter_op.children = [current_plan]
                        current_plan = filter_op

                # Add relationship property filters
                if rel.properties:
                    for prop_key, prop_value in rel.properties.items():
                        if rel.variable:
                            filter_condition = BinaryExpression(
                                PropertyExpression(rel.variable, prop_key),
                                "=",
                                LiteralExpression(prop_value),
                            )
                            filter_op = Filter(filter_condition, "property")
                            filter_op.children = [current_plan]
                            current_plan = filter_op

        return current_plan

    def _find_common_variables(
        self, plan1: LogicalOperator, plan2: LogicalOperator
    ) -> List[str]:
        """Find variables common to both plans"""
        vars1 = self._extract_variables(plan1)
        vars2 = self._extract_variables(plan2)
        return list(set(vars1) & set(vars2))

    def _extract_variables(self, plan: LogicalOperator) -> List[str]:
        """Extract all variables from a plan"""
        variables = []

        if hasattr(plan, "variable"):
            variables.append(plan.variable)
        if hasattr(plan, "from_var"):
            variables.append(plan.from_var)
        if hasattr(plan, "to_var"):
            variables.append(plan.to_var)
        if hasattr(plan, "rel_var") and plan.rel_var:
            variables.append(plan.rel_var)

        for child in plan.children:
            variables.extend(self._extract_variables(child))

        return variables

    def _add_filter_operations(
        self, plan: LogicalOperator, where_clause: WhereClause
    ) -> LogicalOperator:
        """Add filter operations to the plan"""

        filter_type = self._determine_filter_type(where_clause.condition)
        filter_op = Filter(where_clause.condition, filter_type)
        filter_op.children = [plan]

        return filter_op

    def _determine_filter_type(self, condition: Expression) -> str:
        """Determine if filter is property-based, structural, or general"""

        if isinstance(condition, BinaryExpression):
            if isinstance(condition.left, PropertyExpression):
                return "property"
            elif isinstance(condition.left, VariableExpression):
                return "structural"

        return "general"

    def _add_projection_operations(
        self, plan: LogicalOperator, return_clause: ReturnClause
    ) -> LogicalOperator:
        """Add projection operations to the plan"""

        projections = []
        for item in return_clause.items:
            projections.append((item.expression, item.alias))

        project_op = Project(projections)
        project_op.children = [plan]

        current_plan = project_op

        # Add ordering if specified
        if return_clause.order_by:
            sort_items = []
            for order_item in return_clause.order_by.items:
                sort_items.append((order_item.expression, order_item.ascending))

            order_op = OrderBy(sort_items)
            order_op.children = [current_plan]
            current_plan = order_op

        # Add limit/skip if specified
        if return_clause.limit is not None or return_clause.skip is not None:
            limit_op = Limit(
                count=return_clause.limit or float("inf"), skip=return_clause.skip or 0
            )
            limit_op.children = [current_plan]
            current_plan = limit_op

        return current_plan
