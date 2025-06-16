# mylathdb/cypher_planner/logical_planner.py - FIXED VERSION

"""
Logical Planner - FIXED VERSION
The issue is in _add_projection_operations - it's not properly linking the logical operation
"""

from typing import List, Dict, Any, Optional, Union, Set
from .ast_nodes import *
from .logical_operators import *


class LogicalPlanner:
    def __init__(self):
        self.variable_counter = 0

    def create_logical_plan(self, ast: Query) -> LogicalOperator:
        """Convert AST to logical execution plan"""
        
        # Start with data sourcing operations
        plan = self.create_data_source_plan(ast)
        
        # Handle OPTIONAL MATCH clauses
        if ast.optional_match_clauses and plan:
            plan = self._handle_optional_match_clauses(ast, plan)

        # Add filtering
        if ast.where_clause and plan:
            plan = self._add_filter_operations(plan, ast.where_clause)

        # Add projections - FIXED: This is where the bug was!
        if ast.return_clause and plan:
            plan = self._add_projection_operations_fixed(plan, ast.return_clause)

        return plan

    def _add_projection_operations_fixed(self, plan: LogicalOperator, return_clause: ReturnClause) -> LogicalOperator:
        """FIXED: Add projection operations to the plan with proper logical_op linking"""

        if not plan or not return_clause:
            return plan

        try:
            # Extract projections from return items
            projections = []
            for item in return_clause.items:
                projections.append((item.expression, item.alias))

            # FIXED: Create Project operation and properly link the logical_op
            project_op = Project(projections)
            
            # THE FIX: Set the logical_op reference so physical planner can access it
            project_op.logical_op = project_op  # Reference to itself for physical planner
            
            # Add child relationship
            if hasattr(project_op, 'add_child'):
                project_op.add_child(plan)
            else:
                project_op.children = [plan]

            current_plan = project_op

            # Add ordering if specified
            if return_clause.order_by:
                sort_items = []
                for order_item in return_clause.order_by.items:
                    sort_items.append((order_item.expression, order_item.ascending))

                order_op = OrderBy(sort_items)
                order_op.logical_op = order_op  # FIXED: Add logical_op reference
                
                if hasattr(order_op, 'add_child'):
                    order_op.add_child(current_plan)
                else:
                    order_op.children = [current_plan]
                current_plan = order_op

            # Add limit/skip if specified
            if return_clause.limit is not None or return_clause.skip is not None:
                limit_op = Limit(
                    count=return_clause.limit or float("inf"), 
                    skip=return_clause.skip or 0
                )
                limit_op.logical_op = limit_op  # FIXED: Add logical_op reference
                
                if hasattr(limit_op, 'add_child'):
                    limit_op.add_child(current_plan)
                else:
                    limit_op.children = [current_plan]
                current_plan = limit_op

            return current_plan
            
        except Exception as e:
            print(f"Warning: Could not add projection operations: {e}")
            return plan

    # Keep all other existing methods unchanged...
    def create_data_source_plan(self, ast: Query) -> LogicalOperator:
        """Create the initial data sourcing part of the plan"""

        if not ast.match_clauses:
            # No MATCH clauses - create a simple scan for RETURN-only queries
            if ast.return_clause:
                # Create a dummy scan for queries like "RETURN 1"
                return AllNodeScan("_dummy_")
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
                if join_vars:
                    join_op = Join(join_vars)
                    if hasattr(join_op, 'add_child'):
                        join_op.add_child(current_plan)
                        join_op.add_child(plan)
                    else:
                        join_op.children = [current_plan, plan]
                    current_plan = join_op
                else:
                    # No common variables - create a Cartesian product warning
                    print(f"Warning: No common variables between patterns, creating Cartesian product")
                    join_op = Join([])
                    if hasattr(join_op, 'add_child'):
                        join_op.add_child(current_plan)
                        join_op.add_child(plan)
                    else:
                        join_op.children = [current_plan, plan]
                    current_plan = join_op
            return current_plan

        return None

    def _handle_optional_match_clauses(self, ast: Query, plan: LogicalOperator) -> LogicalOperator:
        """Handle OPTIONAL MATCH clauses by creating Optional operations"""
        
        if not ast.optional_match_clauses or not plan:
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
                        if join_vars:
                            join_op = Join(join_vars)
                            if hasattr(join_op, 'add_child'):
                                join_op.add_child(optional_pattern_plan)
                                join_op.add_child(pattern_plan)
                            else:
                                join_op.children = [optional_pattern_plan, pattern_plan]
                            optional_pattern_plan = join_op
                        else:
                            optional_pattern_plan = pattern_plan
                    else:
                        optional_pattern_plan = pattern_plan
            
            if optional_pattern_plan:
                # Create Optional operation
                try:
                    optional_op = Optional()
                    if hasattr(optional_op, 'add_child'):
                        optional_op.add_child(current_plan)
                        optional_op.add_child(optional_pattern_plan)
                    else:
                        optional_op.children = [current_plan, optional_pattern_plan]
                    current_plan = optional_op
                except Exception as e:
                    print(f"Warning: Could not create Optional operation: {e}")
        
        return current_plan

    def _create_pattern_plan(self, pattern):
        """Create logical plan for a pattern"""
        
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
        try:
            if first_node.labels and first_node.properties:
                # Use specialized label scan with properties
                current_plan = NodeByLabelScan(
                    variable=current_var,
                    label=first_node.labels[0],
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
        except Exception as e:
            print(f"Warning: Could not create specialized scan, using basic NodeScan: {e}")
            current_plan = NodeScan(current_var, first_node.labels, first_node.properties)

        # FIXED: Also set logical_op reference for scan operations
        current_plan.logical_op = current_plan

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
                try:
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
                except Exception as e:
                    print(f"Warning: Could not create specialized traversal, using basic Expand: {e}")
                    expand = Expand(
                        from_var=current_var,
                        to_var=next_var,
                        rel_var=rel.variable,
                        rel_types=rel.types,
                        direction=rel.direction,
                        min_length=rel.min_length or 1,
                        max_length=rel.max_length or 1
                    )

                # FIXED: Set logical_op reference for traversal operations
                expand.logical_op = expand

                # Safely add child
                if hasattr(expand, 'add_child') and current_plan:
                    expand.add_child(current_plan)
                elif current_plan:
                    expand.children = [current_plan]
                
                current_plan = expand
                current_var = next_var

                # Add property filters for the target node using NEW OPERATORS
                if next_node.properties:
                    for prop_key, prop_value in next_node.properties.items():
                        try:
                            filter_op = PropertyFilter(next_var, prop_key, "=", prop_value)
                            filter_op.logical_op = filter_op  # FIXED: Add logical_op reference
                            
                            if hasattr(filter_op, 'add_child') and current_plan:
                                filter_op.add_child(current_plan)
                            elif current_plan:
                                filter_op.children = [current_plan]
                            current_plan = filter_op
                        except Exception as e:
                            print(f"Warning: Could not create PropertyFilter, using basic Filter: {e}")
                            from .ast_nodes import BinaryExpression, PropertyExpression, LiteralExpression
                            condition = BinaryExpression(
                                PropertyExpression(next_var, prop_key),
                                "=",
                                LiteralExpression(prop_value)
                            )
                            filter_op = Filter(condition, "property")
                            filter_op.logical_op = filter_op  # FIXED: Add logical_op reference
                            if current_plan:
                                filter_op.children = [current_plan]
                            current_plan = filter_op

                # Add relationship property filters
                if rel.properties:
                    for prop_key, prop_value in rel.properties.items():
                        if rel.variable:
                            try:
                                filter_op = PropertyFilter(rel.variable, prop_key, "=", prop_value)
                                filter_op.logical_op = filter_op  # FIXED: Add logical_op reference
                                
                                if hasattr(filter_op, 'add_child') and current_plan:
                                    filter_op.add_child(current_plan)
                                elif current_plan:
                                    filter_op.children = [current_plan]
                                current_plan = filter_op
                            except Exception as e:
                                print(f"Warning: Could not create PropertyFilter for relationship: {e}")

        return current_plan

    def _find_common_variables(self, plan1: LogicalOperator, plan2: LogicalOperator) -> list:
        """Find variables common to both plans"""
        if not plan1 or not plan2:
            return []
            
        try:
            vars1 = self._extract_variables(plan1)
            vars2 = self._extract_variables(plan2)
            common = list(set(vars1) & set(vars2))
            return common
        except Exception as e:
            print(f"Warning: Could not find common variables: {e}")
            return []

    def _extract_variables(self, plan: LogicalOperator) -> list:
        """Extract all variables from a plan"""
        if not plan:
            return []
            
        variables = []

        try:
            if hasattr(plan, "variable") and plan.variable:
                variables.append(plan.variable)
            if hasattr(plan, "from_var") and plan.from_var:
                variables.append(plan.from_var)
            if hasattr(plan, "to_var") and plan.to_var:
                variables.append(plan.to_var)
            if hasattr(plan, "rel_var") and plan.rel_var:
                variables.append(plan.rel_var)

            # Recursively extract from children
            if hasattr(plan, "children"):
                for child in plan.children:
                    if child:
                        variables.extend(self._extract_variables(child))
        except Exception as e:
            print(f"Warning: Could not extract variables from plan: {e}")

        return variables

    def _add_filter_operations(self, plan: LogicalOperator, where_clause: WhereClause) -> LogicalOperator:
        """Add filter operations using new filter types"""
        
        if not plan or not where_clause:
            return plan
            
        try:
            filter_type = self._determine_filter_type(where_clause.condition)
            
            # Create appropriate filter operation
            if filter_type == "property" and isinstance(where_clause.condition, BinaryExpression):
                if isinstance(where_clause.condition.left, PropertyExpression):
                    try:
                        prop_expr = where_clause.condition.left
                        filter_op = PropertyFilter(
                            prop_expr.variable,
                            prop_expr.property_name,
                            where_clause.condition.operator,
                            where_clause.condition.right.value
                        )
                    except Exception as e:
                        print(f"Warning: Could not create PropertyFilter: {e}")
                        filter_op = Filter(where_clause.condition, filter_type)
                else:
                    filter_op = Filter(where_clause.condition, filter_type)
            elif filter_type == "structural":
                try:
                    filter_op = StructuralFilter(where_clause.condition)
                except Exception as e:
                    print(f"Warning: Could not create StructuralFilter: {e}")
                    filter_op = Filter(where_clause.condition, filter_type)
            elif filter_type == "path":
                try:
                    filter_op = PathFilter(str(where_clause.condition))
                except Exception as e:
                    print(f"Warning: Could not create PathFilter: {e}")
                    filter_op = Filter(where_clause.condition, filter_type)
            else:
                filter_op = Filter(where_clause.condition, filter_type)
            
            # FIXED: Set logical_op reference for filter operations
            filter_op.logical_op = filter_op
            
            # Safely add child
            if hasattr(filter_op, 'add_child'):
                filter_op.add_child(plan)
            else:
                filter_op.children = [plan]
                
            return filter_op
        except Exception as e:
            print(f"Warning: Could not add filter operations: {e}")
            return plan

    def _determine_filter_type(self, condition: Expression) -> str:
        """Determine filter type with new classifications"""
        
        try:
            if isinstance(condition, BinaryExpression):
                if isinstance(condition.left, PropertyExpression):
                    return "property"
                elif isinstance(condition.left, VariableExpression):
                    return "structural"
            elif isinstance(condition, FunctionCall):
                if condition.name.lower() in ['exists', 'path']:
                    return "path"
        except Exception as e:
            print(f"Warning: Could not determine filter type: {e}")
        
        return "general"