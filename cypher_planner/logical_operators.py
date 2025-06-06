"""
Logical Plan Operators for Cypher Execution Planning
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from .ast_nodes import Expression

# =============================================================================
# Base Logical Operator
# =============================================================================


class LogicalOperator(ABC):
    def __init__(self):
        self.children: List["LogicalOperator"] = []
        self.estimated_cardinality: Optional[int] = None
        self.estimated_cost: Optional[float] = None

    @abstractmethod
    def __str__(self) -> str:
        pass


# =============================================================================
# Data Source Operators
# =============================================================================


class NodeScan(LogicalOperator):
    def __init__(
        self, variable: str, labels: List[str] = None, properties: Dict[str, Any] = None
    ):
        super().__init__()
        self.variable = variable
        self.labels = labels or []
        self.properties = properties or {}
        self.execution_target = "redis"  # This operation targets Redis

    def __str__(self) -> str:
        label_str = f":{':'.join(self.labels)}" if self.labels else ""
        prop_str = f" {self.properties}" if self.properties else ""
        return f"NodeScan({self.variable}{label_str}{prop_str})"


class RelationshipScan(LogicalOperator):
    def __init__(
        self, variable: str, types: List[str] = None, properties: Dict[str, Any] = None
    ):
        super().__init__()
        self.variable = variable
        self.types = types or []
        self.properties = properties or {}
        self.execution_target = "redis"

    def __str__(self) -> str:
        type_str = f":{':'.join(self.types)}" if self.types else ""
        prop_str = f" {self.properties}" if self.properties else ""
        return f"RelationshipScan({self.variable}{type_str}{prop_str})"


# =============================================================================
# Graph Traversal Operators
# =============================================================================


class Expand(LogicalOperator):
    def __init__(
        self,
        from_var: str,
        to_var: str,
        rel_var: Optional[str] = None,
        rel_types: List[str] = None,
        direction: str = "outgoing",
        min_length: int = 1,
        max_length: int = 1,
    ):
        super().__init__()
        self.from_var = from_var
        self.to_var = to_var
        self.rel_var = rel_var
        self.rel_types = rel_types or []
        self.direction = direction
        self.min_length = min_length
        self.max_length = max_length
        self.execution_target = "graphblas"  # This operation targets GraphBLAS

    def __str__(self) -> str:
        rel_str = (
            f"[{self.rel_var}:{':'.join(self.rel_types)}]"
            if self.rel_var or self.rel_types
            else "[]"
        )
        direction_symbol = (
            "->"
            if self.direction == "outgoing"
            else "<-"
            if self.direction == "incoming"
            else "--"
        )
        length_str = (
            f"*{self.min_length}..{self.max_length}" if self.max_length > 1 else ""
        )
        return f"Expand({self.from_var})-{rel_str}{length_str}{direction_symbol}({self.to_var})"


# =============================================================================
# Filtering and Joining Operators
# =============================================================================


class Filter(LogicalOperator):
    def __init__(self, condition: Expression, filter_type: str = "general"):
        super().__init__()
        self.condition = condition
        self.filter_type = filter_type  # "property", "structural", "general"
        self.execution_target = "redis" if filter_type == "property" else "mixed"

    def __str__(self) -> str:
        return f"Filter({self.condition})"


class Join(LogicalOperator):
    def __init__(self, join_variables: List[str], join_type: str = "inner"):
        super().__init__()
        self.join_variables = join_variables
        self.join_type = join_type  # "inner", "left_outer"
        self.execution_target = "mixed"

    def __str__(self) -> str:
        return f"Join({', '.join(self.join_variables)}, type={self.join_type})"


# =============================================================================
# Data Processing Operators
# =============================================================================


class Project(LogicalOperator):
    def __init__(self, projections: List[tuple]):  # [(expression, alias), ...]
        super().__init__()
        self.projections = projections
        self.execution_target = "mixed"  # May need both Redis and GraphBLAS

    def __str__(self) -> str:
        proj_strs = []
        for expr, alias in self.projections:
            if alias:
                proj_strs.append(f"{expr} AS {alias}")
            else:
                proj_strs.append(str(expr))
        return f"Project({', '.join(proj_strs)})"


class Aggregate(LogicalOperator):
    def __init__(self, grouping_keys: List[str], aggregations: List[tuple]):
        super().__init__()
        self.grouping_keys = grouping_keys
        self.aggregations = aggregations  # [(function, expression, alias), ...]
        self.execution_target = "mixed"

    def __str__(self) -> str:
        return f"Aggregate(group_by={self.grouping_keys}, aggs={self.aggregations})"


# =============================================================================
# Result Processing Operators
# =============================================================================


class OrderBy(LogicalOperator):
    def __init__(self, sort_items: List[tuple]):  # [(expression, ascending), ...]
        super().__init__()
        self.sort_items = sort_items
        self.execution_target = "redis"

    def __str__(self) -> str:
        return f"OrderBy({self.sort_items})"


class Limit(LogicalOperator):
    def __init__(self, count: int, skip: int = 0):
        super().__init__()
        self.count = count
        self.skip = skip
        self.execution_target = "mixed"

    def __str__(self) -> str:
        return f"Limit(skip={self.skip}, limit={self.count})"


class Distinct(LogicalOperator):
    def __init__(self):
        super().__init__()
        self.execution_target = "mixed"

    def __str__(self) -> str:
        return "Distinct()"


# =============================================================================
# Utility Functions
# =============================================================================


def print_plan(operator: LogicalOperator, indent: int = 0) -> None:
    """Pretty print the logical plan"""
    print("  " * indent + str(operator))
    for child in operator.children:
        print_plan(child, indent + 1)


def analyze_plan_execution_targets(operator: LogicalOperator) -> Dict[str, int]:
    """Analyze which execution targets are used in the plan"""
    targets = {"redis": 0, "graphblas": 0, "mixed": 0}

    def count_targets(op):
        if hasattr(op, "execution_target"):
            targets[op.execution_target] = targets.get(op.execution_target, 0) + 1
        for child in op.children:
            count_targets(child)

    count_targets(operator)
    return targets
