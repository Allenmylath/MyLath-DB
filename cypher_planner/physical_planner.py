# =============================================================================
# FINAL FIX 2: Complete Physical Planner - Replace cypher_planner/physical_planner.py
# =============================================================================

"""
Physical Planner - COMPLETE VERSION
Converts logical plans to Redis + GraphBLAS operations
"""

from typing import List
from .logical_operators import *
from .ast_nodes import *


class PhysicalOperation:
    """Base class for physical operations"""

    def __init__(self, operation_type: str, target: str):
        self.operation_type = operation_type
        self.target = target  # "redis" or "graphblas"
        self.estimated_cost = 0.0
        self.children = []


class RedisOperation(PhysicalOperation):
    def __init__(self, operation_type: str, redis_commands: List[str]):
        super().__init__(operation_type, "redis")
        self.redis_commands = redis_commands


class GraphBLASOperation(PhysicalOperation):
    def __init__(self, operation_type: str, matrix_ops: List[str]):
        super().__init__(operation_type, "graphblas")
        self.matrix_operations = matrix_ops


class PhysicalPlanner:
    """Converts logical plan to physical execution plan with Redis/GraphBLAS specifics"""

    def create_physical_plan(self, logical_plan: LogicalOperator) -> PhysicalOperation:
        """Convert logical plan to physical execution plan"""
        return self._convert_operator(logical_plan)

    def _convert_operator(self, logical_op: LogicalOperator) -> PhysicalOperation:
        """Convert a single logical operator to physical operation(s)"""

        if isinstance(logical_op, NodeScan):
            return self._convert_node_scan(logical_op)
        elif isinstance(logical_op, Expand):
            return self._convert_expand(logical_op)
        elif isinstance(logical_op, Filter):
            return self._convert_filter(logical_op)
        elif isinstance(logical_op, Project):
            return self._convert_project(logical_op)
        elif isinstance(logical_op, OrderBy):
            return self._convert_order_by(logical_op)
        elif isinstance(logical_op, Limit):
            return self._convert_limit(logical_op)
        elif isinstance(logical_op, Join):
            return self._convert_join(logical_op)
        else:
            # Default conversion
            physical_op = PhysicalOperation(type(logical_op).__name__, "mixed")
            physical_op.children = [
                self._convert_operator(child) for child in logical_op.children
            ]
            return physical_op

    def _convert_node_scan(self, node_scan: NodeScan) -> RedisOperation:
        """Convert NodeScan to Redis operations"""

        redis_commands = []

        if node_scan.labels:
            # Use label indexes
            for label in node_scan.labels:
                redis_commands.append(f"SMEMBERS label:{label}")

        if node_scan.properties:
            # Use property indexes or filters
            for prop_key, prop_value in node_scan.properties.items():
                redis_commands.append(f"SMEMBERS prop:{prop_key}:{prop_value}")

        # If both labels and properties, intersect them
        if node_scan.labels and node_scan.properties:
            sets_to_intersect = [f"label:{label}" for label in node_scan.labels]
            sets_to_intersect.extend(
                [f"prop:{k}:{v}" for k, v in node_scan.properties.items()]
            )
            redis_commands.append(f"SINTER {' '.join(sets_to_intersect)}")

        if not redis_commands:
            redis_commands.append("SCAN 0 MATCH node:*")

        redis_op = RedisOperation("NodeScan", redis_commands)
        return redis_op

    def _convert_expand(self, expand: Expand) -> PhysicalOperation:
        """Convert Expand to GraphBLAS matrix operations"""

        matrix_ops = []

        if expand.max_length == 1:
            # Single hop - simple matrix-vector multiplication
            rel_type = expand.rel_types[0] if expand.rel_types else "*"

            if expand.direction == "outgoing":
                matrix_ops.append(
                    f"v_{expand.to_var} = v_{expand.from_var} @ A_{rel_type}"
                )
            elif expand.direction == "incoming":
                matrix_ops.append(
                    f"v_{expand.to_var} = v_{expand.from_var} @ A_{rel_type}.T"
                )
            else:  # bidirectional
                matrix_ops.append(
                    f"v_{expand.to_var} = v_{expand.from_var} @ (A_{rel_type} + A_{rel_type}.T)"
                )
        else:
            # Multi-hop - iterative matrix multiplication
            rel_type = expand.rel_types[0] if expand.rel_types else "*"
            matrix_ops.append(
                f"# Variable length path {expand.min_length}..{expand.max_length}"
            )

            if expand.max_length == float("inf"):
                matrix_ops.append(
                    f"result = compute_transitive_closure(v_{expand.from_var}, A_{rel_type}, {expand.min_length})"
                )
            else:
                matrix_ops.append(
                    f"result = compute_variable_path(v_{expand.from_var}, A_{rel_type}, {expand.min_length}, {expand.max_length})"
                )

        graphblas_op = GraphBLASOperation("Expand", matrix_ops)

        # Add children (converted recursively)
        for child in expand.children:
            graphblas_op.children.append(self._convert_operator(child))

        return graphblas_op

    def _convert_filter(self, filter_op: Filter) -> PhysicalOperation:
        """Convert Filter based on filter type"""

        if filter_op.filter_type == "property":
            # Property filter - Redis operation
            redis_commands = [f"# Apply property filter: {filter_op.condition}"]

            # Extract property access patterns for specific Redis commands
            if isinstance(filter_op.condition, BinaryExpression):
                if isinstance(filter_op.condition.left, PropertyExpression):
                    prop_expr = filter_op.condition.left
                    redis_commands.append(f"HGET node:{{id}} {prop_expr.property_name}")

            redis_op = RedisOperation("PropertyFilter", redis_commands)
        else:
            # General filter - might need both systems
            redis_op = PhysicalOperation("Filter", "mixed")

        # Add children
        for child in filter_op.children:
            redis_op.children.append(self._convert_operator(child))

        return redis_op

    def _convert_project(self, project: Project) -> RedisOperation:
        """Convert Project operation"""

        redis_commands = []

        for expr, alias in project.projections:
            if isinstance(expr, PropertyExpression):
                redis_commands.append(f"HGET node:{{id}} {expr.property_name}")
            elif isinstance(expr, VariableExpression):
                redis_commands.append(f"# Return variable: {expr.name}")

        if not redis_commands:
            redis_commands.append("# Project selected columns")

        redis_op = RedisOperation("Project", redis_commands)

        # Add children
        for child in project.children:
            redis_op.children.append(self._convert_operator(child))

        return redis_op

    def _convert_order_by(self, order_by: OrderBy) -> RedisOperation:
        """Convert OrderBy operation"""

        redis_commands = ["# Sort results"]
        for expr, ascending in order_by.sort_items:
            direction = "ASC" if ascending else "DESC"
            if isinstance(expr, PropertyExpression):
                redis_commands.append(
                    f"SORT BY {expr.variable}.{expr.property_name} {direction}"
                )

        redis_op = RedisOperation("OrderBy", redis_commands)

        # Add children
        for child in order_by.children:
            redis_op.children.append(self._convert_operator(child))

        return redis_op

    def _convert_limit(self, limit: Limit) -> RedisOperation:
        """Convert Limit operation"""

        redis_commands = []
        if limit.skip > 0:
            redis_commands.append(f"SKIP {limit.skip}")
        if limit.count != float("inf"):
            redis_commands.append(f"LIMIT {limit.count}")

        if not redis_commands:
            redis_commands.append("# No limit applied")

        redis_op = RedisOperation("Limit", redis_commands)

        # Add children
        for child in limit.children:
            redis_op.children.append(self._convert_operator(child))

        return redis_op

    def _convert_join(self, join: Join) -> PhysicalOperation:
        """Convert Join operation"""

        # Joins typically require coordination between Redis and GraphBLAS
        physical_op = PhysicalOperation("Join", "mixed")

        # Add children
        for child in join.children:
            physical_op.children.append(self._convert_operator(child))

        return physical_op


def print_physical_plan(physical_op: PhysicalOperation, indent: int = 0) -> None:
    """Pretty print the physical plan"""
    prefix = "  " * indent

    if isinstance(physical_op, RedisOperation):
        print(f"{prefix}[Redis] {physical_op.operation_type}")
        for cmd in physical_op.redis_commands:
            print(f"{prefix}  > {cmd}")
    elif isinstance(physical_op, GraphBLASOperation):
        print(f"{prefix}[GraphBLAS] {physical_op.operation_type}")
        for op in physical_op.matrix_operations:
            print(f"{prefix}  > {op}")
    else:
        print(f"{prefix}[{physical_op.target}] {physical_op.operation_type}")

    for child in physical_op.children:
        print_physical_plan(child, indent + 1)
