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
from .logical_operators import (
    NodeByLabelScan, AllNodeScan, PropertyScan,
    ConditionalTraverse, ConditionalVarLenTraverse,
    PropertyFilter, StructuralFilter, PathFilter,
    Apply, SemiApply, Optional
)


class LogicalPlanner:
    def __init__(self):
        self.variable_counter = 0
    def _handle_optional_match_clauses(self, ast: Query, plan: LogicalOperator) -> LogicalOperator:
        """Handle OPTIONAL MATCH clauses by creating Optional operations"""
    
        if not ast.optional_match_clauses:
            return plan
        
        current_plan = plan
        
        for optional_clause in ast.optional_match_clauses:
            # Create the optional pattern plan
            optional_pattern_plan = None
            
            for pattern in optional_clause.patterns:
                pattern_plan = self._create_pattern_plan(pattern)
                if pattern_plan:
                    if optional_pattern_plan:
                        # Join multiple patterns in OPTIONAL MATCH
                        join_vars = self._find_common_variables(optional_pattern_plan, pattern_plan)
                        join_op = Join(join_vars)
                        join_op.add_child(optional_pattern_plan)
                        join_op.add_child(pattern_plan)
                        optional_pattern_plan = join_op
                    else:
                        optional_pattern_plan = pattern_plan
            
            if optional_pattern_plan:
                # Create Optional operation
                optional_op = Optional()
                optional_op.add_child(current_plan)  # Left side (required)
                optional_op.add_child(optional_pattern_plan)  # Right side (optional)
                current_plan = optional_op
        
        return current_plan


    def create_logical_plan(self, ast: Query) -> LogicalOperator:
        """Convert AST to logical execution plan"""

        # Start with data sourcing operations
        plan = self._create_data_source_plan(ast)

        if ast.optional_match_clauses:
            plan = self._handle_optional_match_clauses(ast, plan)

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

        # Start with the first node - USE NEW OPERATORS
        first_node = nodes[0]
        current_var = first_node.variable or f"_node_{self.variable_counter}"
        
        # CHOOSE OPTIMAL SCAN OPERATION
        if first_node.labels and first_node.properties:
            # Use specialized label scan with properties
            current_plan = NodeByLabelScan(
                variable=current_var,
                label=first_node.labels[0],  # Use first label
                properties=first_node.properties
            )
        elif first_node.labels:
            # Use label scan
            current_plan = NodeByLabelScan(
                variable=current_var,
                label=first_node.labels[0],
                properties={}
            )
        elif first_node.properties:
            # Use property scan if only properties specified
            prop_key, prop_value = next(iter(first_node.properties.items()))
            current_plan = PropertyScan(current_var, prop_key, prop_value)
        else:
            # Fallback to all node scan
            current_plan = AllNodeScan(current_var)

        if not first_node.variable:
            self.variable_counter += 1

        # Process relationships and subsequent nodes
        for i, rel in enumerate(relationships):
            if i + 1 < len(nodes):
                next_node = nodes[i + 1]
                next_var = next_node.variable or f"_node_{self.variable_counter}"

                if not next_node.variable:
                    self.variable_counter += 1

                # CREATE APPROPRIATE TRAVERSAL OPERATION
                if rel.min_length and rel.max_length and rel.max_length > 1:
                    # Variable-length traversal
                    expand = ConditionalVarLenTraverse(
                        from_var=current_var,
                        to_var=next_var,
                        rel_var=rel.variable,
                        rel_types=rel.types,
                        direction=rel.direction,
                        min_length=rel.min_length or 1,
                        max_length=rel.max_length or 1,
                    )
                else:
                    # Single-hop traversal
                    expand = ConditionalTraverse(
                        from_var=current_var,
                        to_var=next_var,
                        rel_var=rel.variable,
                        rel_types=rel.types,
                        direction=rel.direction,
                    )

                expand.add_child(current_plan)  # Use new add_child method
                current_plan = expand
                current_var = next_var

                # Add property filters for the target node using NEW OPERATORS
                if next_node.properties:
                    for prop_key, prop_value in next_node.properties.items():
                        filter_op = PropertyFilter(next_var, prop_key, "=", prop_value)
                        filter_op.add_child(current_plan)
                        current_plan = filter_op

                # Add relationship property filters
                if rel.properties:
                    for prop_key, prop_value in rel.properties.items():
                        if rel.variable:
                            filter_op = PropertyFilter(rel.variable, prop_key, "=", prop_value)
                            filter_op.add_child(current_plan)
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

    def _add_filter_operations(self, plan: LogicalOperator, where_clause: WhereClause) -> LogicalOperator:
        """ENHANCED: Add filter operations using new filter types"""
        
        filter_type = self._determine_filter_type(where_clause.condition)
        
        # Create appropriate filter operation
        if filter_type == "property" and isinstance(where_clause.condition, BinaryExpression):
            if isinstance(where_clause.condition.left, PropertyExpression):
                prop_expr = where_clause.condition.left
                filter_op = PropertyFilter(
                    prop_expr.variable,
                    prop_expr.property_name,
                    where_clause.condition.operator,
                    where_clause.condition.right.value
                )
            else:
                filter_op = Filter(where_clause.condition, filter_type)
        elif filter_type == "structural":
            filter_op = StructuralFilter(where_clause.condition)
        elif filter_type == "path":
            filter_op = PathFilter(str(where_clause.condition))
        else:
            filter_op = Filter(where_clause.condition, filter_type)
        
        filter_op.add_child(plan)
        return filter_op    

    def _determine_filter_type(self, condition: Expression) -> str:
        """ENHANCED: Determine filter type with new classifications"""
        
        if isinstance(condition, BinaryExpression):
            if isinstance(condition.left, PropertyExpression):
                return "property"
            elif isinstance(condition.left, VariableExpression):
                return "structural"
            elif isinstance(condition, FunctionCall):
                if condition.name.lower() in ['exists', 'path']:
                    return "path"
        
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
