

# Add these imports at the very top of the file
from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Union
from abc import ABC, abstractmethod
from .ast_nodes import Expression

class LogicalOperator(ABC):
    """Enhanced base class for all logical operators"""
    
    def __init__(self):
        self.children: List["LogicalOperator"] = []
        self.parent: Optional["LogicalOperator"] = None
        self.estimated_cardinality: Optional[int] = None
        self.estimated_cost: Optional[float] = None
        self.modifies: List[str] = []  # NEW: Variables this operation introduces
        self.awareness: Set[str] = set()  # NEW: Variables this operation is aware of
        self.execution_target: str = "mixed"

    @abstractmethod
    def __str__(self) -> str:
        pass

    def add_child(self, child: "LogicalOperator"):
        """Add a child operation and propagate awareness"""
        self.children.append(child)
        child.parent = self
        self._propagate_awareness(child)

    def _propagate_awareness(self, child: "LogicalOperator"):
        """Propagate awareness from child to parent"""
        self.awareness.update(child.awareness)
        self.awareness.update(child.modifies)

# =============================================================================
# Enhanced Scan Operations (REPLACING EXISTING)
# =============================================================================

class NodeByLabelScan(LogicalOperator):
    """NEW: Specialized node scan using label index"""
    def __init__(self, variable: str, label: str, properties: Dict[str, Any] = None):
        super().__init__()
        self.variable = variable
        self.label = label
        self.properties = properties or {}
        self.modifies = [variable]
        self.awareness.add(variable)
        self.execution_target = "redis"

    def __str__(self) -> str:
        prop_str = f" {self.properties}" if self.properties else ""
        return f"NodeByLabelScan({self.variable}:{self.label}{prop_str})"

class AllNodeScan(LogicalOperator):
    """NEW: Full node scan without label filtering"""
    def __init__(self, variable: str):
        super().__init__()
        self.variable = variable
        self.modifies = [variable]
        self.awareness.add(variable)
        self.execution_target = "redis"

    def __str__(self) -> str:
        return f"AllNodeScan({self.variable})"

class PropertyScan(LogicalOperator):
    """NEW: Scan nodes by property index"""
    def __init__(self, variable: str, property_key: str, property_value: Any):
        super().__init__()
        self.variable = variable
        self.property_key = property_key
        self.property_value = property_value
        self.modifies = [variable]
        self.awareness.add(variable)
        self.execution_target = "redis"

    def __str__(self) -> str:
        return f"PropertyScan({self.variable} WHERE {self.property_key} = {self.property_value})"

# Keep existing NodeScan but mark as deprecated
class NodeScan(LogicalOperator):
    """DEPRECATED: Use NodeByLabelScan or AllNodeScan instead"""
    def __init__(self, variable: str, labels: List[str] = None, properties: Dict[str, Any] = None):
        super().__init__()
        self.variable = variable
        self.labels = labels or []
        self.properties = properties or {}
        self.modifies = [variable]
        self.awareness.add(variable)
        self.execution_target = "redis"

    def __str__(self) -> str:
        label_str = f":{':'.join(self.labels)}" if self.labels else ""
        prop_str = f" {self.properties}" if self.properties else ""
        return f"NodeScan({self.variable}{label_str}{prop_str})"

# =============================================================================
# Enhanced Traversal Operations (REPLACING EXISTING)
# =============================================================================

class ConditionalTraverse(LogicalOperator):
    """NEW: Single-hop conditional traversal"""
    def __init__(self, from_var: str, to_var: str, rel_var: str = None, 
                 rel_types: List[str] = None, direction: str = "outgoing"):
        super().__init__()
        self.from_var = from_var
        self.to_var = to_var
        self.rel_var = rel_var
        self.rel_types = rel_types or []
        self.direction = direction
        self.modifies = [to_var]
        if rel_var:
            self.modifies.append(rel_var)
        self.awareness.update([from_var, to_var])
        if rel_var:
            self.awareness.add(rel_var)
        self.execution_target = "graphblas"

    def __str__(self) -> str:
        rel_str = f"[{self.rel_var}:{':'.join(self.rel_types)}]" if self.rel_var or self.rel_types else "[]"
        direction_symbol = "->" if self.direction == "outgoing" else "<-" if self.direction == "incoming" else "--"
        return f"ConditionalTraverse({self.from_var})-{rel_str}{direction_symbol}({self.to_var})"

class ConditionalVarLenTraverse(LogicalOperator):
    """NEW: Variable-length conditional traversal"""
    def __init__(self, from_var: str, to_var: str, rel_var: str = None,
                 rel_types: List[str] = None, min_length: int = 1, max_length: int = None,
                 direction: str = "outgoing"):
        super().__init__()
        self.from_var = from_var
        self.to_var = to_var
        self.rel_var = rel_var
        self.rel_types = rel_types or []
        self.min_length = min_length
        self.max_length = max_length or float('inf')
        self.direction = direction
        self.modifies = [to_var]
        if rel_var:
            self.modifies.append(rel_var)
        self.awareness.update([from_var, to_var])
        if rel_var:
            self.awareness.add(rel_var)
        self.execution_target = "graphblas"

    def __str__(self) -> str:
        rel_str = f"[{self.rel_var}:{':'.join(self.rel_types)}*{self.min_length}..{self.max_length}]"
        direction_symbol = "->" if self.direction == "outgoing" else "<-" if self.direction == "incoming" else "--"
        return f"ConditionalVarLenTraverse({self.from_var})-{rel_str}{direction_symbol}({self.to_var})"

# Keep existing Expand but mark as deprecated
class Expand(LogicalOperator):
    """DEPRECATED: Use ConditionalTraverse or ConditionalVarLenTraverse instead"""
    def __init__(self, from_var: str, to_var: str, rel_var: Optional[str] = None,
                 rel_types: List[str] = None, direction: str = "outgoing",
                 min_length: int = 1, max_length: int = 1):
        super().__init__()
        self.from_var = from_var
        self.to_var = to_var
        self.rel_var = rel_var
        self.rel_types = rel_types or []
        self.direction = direction
        self.min_length = min_length
        self.max_length = max_length
        self.modifies = [to_var]
        if rel_var:
            self.modifies.append(rel_var)
        self.awareness.update([from_var, to_var])
        if rel_var:
            self.awareness.add(rel_var)
        self.execution_target = "graphblas"

    def __str__(self) -> str:
        rel_str = f"[{self.rel_var}:{':'.join(self.rel_types)}]" if self.rel_var or self.rel_types else "[]"
        direction_symbol = "->" if self.direction == "outgoing" else "<-" if self.direction == "incoming" else "--"
        length_str = f"*{self.min_length}..{self.max_length}" if self.max_length > 1 else ""
        return f"Expand({self.from_var})-{rel_str}{length_str}{direction_symbol}({self.to_var})"

# =============================================================================
# Apply Family Operations (NEW)
# =============================================================================

class Apply(LogicalOperator):
    """NEW: Apply operation for correlated subqueries"""
    def __init__(self):
        super().__init__()
        self.execution_target = "mixed"

    def __str__(self) -> str:
        return "Apply"

class SemiApply(LogicalOperator):
    """NEW: Semi-apply for EXISTS-style filtering"""
    def __init__(self, anti: bool = False):
        super().__init__()
        self.anti = anti
        self.execution_target = "mixed"

    def __str__(self) -> str:
        return f"{'Anti' if self.anti else ''}SemiApply"

class ApplyMultiplexer(LogicalOperator):
    """NEW: Multiplexer for OR/AND operations in apply context"""
    def __init__(self, operation: str):  # "OR" or "AND"
        super().__init__()
        self.operation = operation
        self.execution_target = "mixed"

    def __str__(self) -> str:
        return f"ApplyMultiplexer({self.operation})"

class Optional(LogicalOperator):
    """NEW: Optional operation for OPTIONAL MATCH"""
    def __init__(self):
        super().__init__()
        self.execution_target = "mixed"

    def __str__(self) -> str:
        return "Optional"

# =============================================================================
# Enhanced Filter Operations (NEW)
# =============================================================================

class PropertyFilter(LogicalOperator):
    """NEW: Specialized filter for property conditions"""
    def __init__(self, variable: str, property_key: str, operator: str, value: Any):
        super().__init__()
        self.variable = variable
        self.property_key = property_key
        self.operator = operator
        self.value = value
        self.awareness.add(variable)
        self.execution_target = "redis"
        from .ast_nodes import BinaryExpression, PropertyExpression, LiteralExpression
        self.condition = BinaryExpression(
            PropertyExpression(variable, property_key),
            operator,
            LiteralExpression(value)
        )

    def __str__(self) -> str:
        return f"PropertyFilter({self.variable}.{self.property_key} {self.operator} {self.value})"

class StructuralFilter(LogicalOperator):
    """NEW: Filter for structural/topological conditions"""
    def __init__(self, condition: Expression):
        super().__init__()
        self.condition = condition
        self.execution_target = "graphblas"

    def __str__(self) -> str:
        return f"StructuralFilter({self.condition})"

class PathFilter(LogicalOperator):
    """NEW: Filter for path existence conditions"""
    def __init__(self, path_pattern: str, anti: bool = False):
        super().__init__()
        self.path_pattern = path_pattern
        self.anti = anti
        self.execution_target = "graphblas"

    def __str__(self) -> str:
        return f"PathFilter({'NOT ' if self.anti else ''}{self.path_pattern})"

# Keep existing Filter class for backward compatibility
class Filter(LogicalOperator):
    """ENHANCED: General filter with type classification"""
    def __init__(self, condition: Expression, filter_type: str = "general"):
        super().__init__()
        self.condition = condition
        self.filter_type = filter_type  # "property", "structural", "general"
        self.execution_target = "redis" if filter_type == "property" else "mixed"

    def __str__(self) -> str:
        return f"Filter({self.condition})"

# =============================================================================
# Keep all existing operations for backward compatibility
# =============================================================================

class RelationshipScan(LogicalOperator):
    def __init__(self, variable: str, types: List[str] = None, properties: Dict[str, Any] = None):
        super().__init__()
        self.variable = variable
        self.types = types or []
        self.properties = properties or {}
        self.modifies = [variable]
        self.awareness.add(variable)
        self.execution_target = "redis"

    def __str__(self) -> str:
        type_str = f":{':'.join(self.types)}" if self.types else ""
        prop_str = f" {self.properties}" if self.properties else ""
        return f"RelationshipScan({self.variable}{type_str}{prop_str})"

class Join(LogicalOperator):
    def __init__(self, join_variables: List[str], join_type: str = "inner"):
        super().__init__()
        self.join_variables = join_variables
        self.join_type = join_type
        self.awareness.update(join_variables)
        self.execution_target = "mixed"

    def __str__(self) -> str:
        return f"Join({', '.join(self.join_variables)}, type={self.join_type})"

class Project(LogicalOperator):
    def __init__(self, projections: List[tuple]):
        super().__init__()
        self.projections = projections
        self.execution_target = "mixed"

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
        self.aggregations = aggregations
        self.execution_target = "mixed"

    def __str__(self) -> str:
        return f"Aggregate(group_by={self.grouping_keys}, aggs={self.aggregations})"

class OrderBy(LogicalOperator):
    def __init__(self, sort_items: List[tuple]):
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
# Utility Functions (ENHANCED)
# =============================================================================

def print_plan(operator: LogicalOperator, indent: int = 0) -> None:
    """Enhanced pretty print with awareness information"""
    prefix = "  " * indent
    awareness_str = f" [aware: {', '.join(sorted(operator.awareness))}]" if operator.awareness else ""
    modifies_str = f" [modifies: {', '.join(operator.modifies)}]" if operator.modifies else ""
    
    print(f"{prefix}{operator}{awareness_str}{modifies_str}")
    for child in operator.children:
        print_plan(child, indent + 1)

def analyze_plan_execution_targets(operator: LogicalOperator) -> Dict[str, int]:
    """Enhanced analysis with new operation types"""
    targets = {"redis": 0, "graphblas": 0, "mixed": 0}

    def count_targets(op):
        if hasattr(op, "execution_target"):
            targets[op.execution_target] = targets.get(op.execution_target, 0) + 1
        for child in op.children:
            count_targets(child)

    count_targets(operator)
    return targets

def is_aware_of_variables(operator: LogicalOperator, variables: List[str]) -> bool:
    """NEW: Check if operator is aware of all specified variables"""
    return all(var in operator.awareness for var in variables)

def find_earliest_aware_operator(root: LogicalOperator, variables: List[str]) -> Optional[LogicalOperator]:
    """NEW: Find the earliest operator in the tree that's aware of all variables"""
    if not is_aware_of_variables(root, variables):
        return None
    
    # Try to go deeper
    for child in root.children:
        deeper = find_earliest_aware_operator(child, variables)
        if deeper:
            return deeper
    
    return root