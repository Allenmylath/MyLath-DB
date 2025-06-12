"""
AST Node Definitions for Cypher Parser
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# =============================================================================
# Base AST Node
# =============================================================================


class ASTNode(ABC):
    """Base class for all AST nodes"""

    pass


# =============================================================================
# Query Structure Nodes
# =============================================================================


@dataclass
class Query(ASTNode):
    match_clauses: List["MatchClause"] = field(default_factory=list)
    where_clause: Optional["WhereClause"] = None
    return_clause: Optional["ReturnClause"] = None
    optional_match_clauses: List["OptionalMatchClause"] = field(default_factory=list)
    with_clauses: List["WithClause"] = field(default_factory=list)


@dataclass
class MatchClause(ASTNode):
    patterns: List["Pattern"]


@dataclass
class OptionalMatchClause(ASTNode):
    patterns: List["Pattern"]


@dataclass
class Pattern(ASTNode):
    elements: List[Union["NodePattern", "RelationshipPattern"]]


@dataclass
class NodePattern(ASTNode):
    variable: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationshipPattern(ASTNode):
    variable: Optional[str] = None
    types: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    direction: str = "outgoing"  # "outgoing", "incoming", "bidirectional"
    min_length: Optional[int] = None
    max_length: Optional[int] = None


# =============================================================================
# Clause Nodes
# =============================================================================


@dataclass
class WhereClause(ASTNode):
    condition: "Expression"


@dataclass
class ReturnClause(ASTNode):
    items: List["ReturnItem"]
    distinct: bool = False
    order_by: Optional["OrderByClause"] = None
    skip: Optional[int] = None
    limit: Optional[int] = None


@dataclass
class ReturnItem(ASTNode):
    expression: "Expression"
    alias: Optional[str] = None


@dataclass
class OrderByClause(ASTNode):
    items: List["OrderByItem"]


@dataclass
class OrderByItem(ASTNode):
    expression: "Expression"
    ascending: bool = True


@dataclass
class WithClause(ASTNode):
    items: List["ReturnItem"]
    where_clause: Optional["WhereClause"] = None


# =============================================================================
# Expression Nodes
# =============================================================================


class Expression(ASTNode):
    pass


@dataclass
class PropertyExpression(Expression):
    variable: str
    property_name: str

    def __str__(self) -> str:
        return f"{self.variable}.{self.property_name}"


@dataclass
class VariableExpression(Expression):
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class LiteralExpression(Expression):
    value: Any

    def __str__(self) -> str:
        if isinstance(self.value, str):
            return f"'{self.value}'"
        return str(self.value)


@dataclass
class BinaryExpression(Expression):
    left: Expression
    operator: str
    right: Expression

    def __str__(self) -> str:
        return f"({self.left} {self.operator} {self.right})"


@dataclass
class FunctionCall(Expression):
    name: str
    arguments: List[Expression]

    def __str__(self) -> str:
        args = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.name}({args})"
