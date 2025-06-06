# =============================================================================
# FIXED cypher_planner/parser.py - Replace the existing parser.py with this
# =============================================================================

"""
Cypher Query Parser - FIXED VERSION
Converts Cypher query strings into Abstract Syntax Trees (AST)
"""

import re
from typing import List, Optional, Dict, Any
from .ast_nodes import *


class CypherParser:
    def __init__(self):
        self.tokens = []
        self.position = 0

    def parse(self, query: str) -> Query:
        """Parse a Cypher query into an AST"""
        self.tokens = self._tokenize(query)
        self.position = 0
        return self._parse_query()

    def _tokenize(self, query: str) -> List[str]:
        """Enhanced tokenizer for Cypher queries"""
        token_pattern = r"""
            (?P<KEYWORD>MATCH|WHERE|RETURN|OPTIONAL|WITH|ORDER\s+BY|SKIP|LIMIT|AS|AND|OR|NOT|DISTINCT)\s*|
            (?P<STRING>'[^']*'|"[^"]*")\s*|
            (?P<NUMBER>\d+(?:\.\d+)?)\s*|
            (?P<IDENTIFIER>[a-zA-Z_][a-zA-Z0-9_]*)\s*|
            (?P<OPERATOR><=|>=|<>|!=|=|<|>)\s*|
            (?P<ARROW><--|-->)\s*|
            (?P<DASH>--)\s*|
            (?P<VARIABLE_LENGTH>\*\d+\.\.\d+|\*\d+\.\.|\*\.\.\d+|\*)\s*|
            (?P<PUNCTUATION>[(){}\[\],:.-])\s*|
            (?P<WHITESPACE>\s+)
        """

        tokens = []
        for match in re.finditer(token_pattern, query, re.IGNORECASE | re.VERBOSE):
            kind = match.lastgroup
            value = match.group().strip()
            if kind != "WHITESPACE" and value:
                tokens.append(value)

        return tokens

    def _current_token(self) -> Optional[str]:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def _consume_token(self) -> Optional[str]:
        token = self._current_token()
        if token:
            self.position += 1
        return token

    def _peek_token(self, offset: int = 1) -> Optional[str]:
        pos = self.position + offset
        return self.tokens[pos] if pos < len(self.tokens) else None

    def _expect_token(self, expected: str) -> str:
        token = self._consume_token()
        if not token or token.upper() != expected.upper():
            raise ValueError(f"Expected '{expected}', got '{token}'")
        return token

    def _parse_query(self) -> Query:
        query = Query()

        while self._current_token():
            token = self._current_token().upper()

            if token == "MATCH":
                query.match_clauses.append(self._parse_match_clause())
            elif token == "OPTIONAL":
                if self._peek_token() and self._peek_token().upper() == "MATCH":
                    self._consume_token()  # consume 'OPTIONAL'
                    query.optional_match_clauses.append(
                        self._parse_optional_match_clause()
                    )
                else:
                    raise ValueError("Expected MATCH after OPTIONAL")
            elif token == "WHERE":
                query.where_clause = self._parse_where_clause()
            elif token == "RETURN":
                query.return_clause = self._parse_return_clause()
                break  # RETURN is typically the last clause
            elif token == "WITH":
                query.with_clauses.append(self._parse_with_clause())
            else:
                raise ValueError(f"Unexpected token: {token}")

        return query

    def _parse_match_clause(self) -> MatchClause:
        self._expect_token("MATCH")
        patterns = self._parse_patterns()
        return MatchClause(patterns)

    def _parse_optional_match_clause(self) -> OptionalMatchClause:
        self._expect_token("MATCH")
        patterns = self._parse_patterns()
        return OptionalMatchClause(patterns)

    def _parse_patterns(self) -> List[Pattern]:
        patterns = []
        patterns.append(self._parse_pattern())

        while self._current_token() == ",":
            self._consume_token()  # consume ','
            patterns.append(self._parse_pattern())

        return patterns

    def _parse_pattern(self) -> Pattern:
        elements = []

        # Parse first node
        if self._current_token() == "(":
            elements.append(self._parse_node_pattern())

        # Parse relationship and subsequent nodes
        while self._current_token() in ["<--", "-->", "--", "-", "["]:
            rel_pattern = self._parse_relationship_pattern()
            elements.append(rel_pattern)

            # Parse the target node
            if self._current_token() == "(":
                elements.append(self._parse_node_pattern())
            else:
                # If no target node, create anonymous one
                elements.append(NodePattern())

        return Pattern(elements)

    def _parse_node_pattern(self) -> NodePattern:
        self._expect_token("(")

        variable = None
        labels = []
        properties = {}

        # Parse variable
        if self._current_token() and self._current_token() not in [":", ")"]:
            variable = self._consume_token()

        # Parse labels
        while self._current_token() == ":":
            self._consume_token()  # consume ':'
            if self._current_token() and self._current_token() != ")":
                labels.append(self._consume_token())

        # Parse properties (simplified)
        if self._current_token() == "{":
            properties = self._parse_properties()

        self._expect_token(")")
        return NodePattern(variable, labels, properties)

    def _parse_relationship_pattern(self) -> RelationshipPattern:
        direction = "outgoing"
        rel_variable = None
        rel_types = []
        properties = {}
        min_length = None
        max_length = None

        # Handle incoming relationships
        if self._current_token() == "<--":
            direction = "incoming"
            self._consume_token()
            return RelationshipPattern(
                variable=rel_variable,
                types=rel_types,
                properties=properties,
                direction=direction,
                min_length=min_length,
                max_length=max_length,
            )

        # Handle outgoing relationships
        elif self._current_token() == "-->":
            direction = "outgoing"
            self._consume_token()
            return RelationshipPattern(
                variable=rel_variable,
                types=rel_types,
                properties=properties,
                direction=direction,
                min_length=min_length,
                max_length=max_length,
            )

        # Handle bidirectional relationships
        elif self._current_token() == "--":
            direction = "bidirectional"
            self._consume_token()
            return RelationshipPattern(
                variable=rel_variable,
                types=rel_types,
                properties=properties,
                direction=direction,
                min_length=min_length,
                max_length=max_length,
            )

        # Handle complex relationship patterns with brackets
        elif self._current_token() == "-":
            self._consume_token()  # consume first '-'

            # Check for relationship details in brackets
            if self._current_token() == "[":
                self._consume_token()  # consume '['

                # Parse variable
                if (
                    self._current_token()
                    and self._current_token() not in [":", "]", "*"]
                    and not self._current_token().startswith("*")
                ):
                    rel_variable = self._consume_token()

                # Parse types
                while self._current_token() == ":":
                    self._consume_token()  # consume ':'
                    if (
                        self._current_token()
                        and self._current_token() not in ["]", "*"]
                        and not self._current_token().startswith("*")
                    ):
                        rel_types.append(self._consume_token())

                # Parse variable length (FIXED)
                if self._current_token() and self._current_token().startswith("*"):
                    var_length = self._consume_token()  # consume the whole *1..3 token

                    # Parse the variable length specification
                    if var_length == "*":
                        min_length = 1
                        max_length = float("inf")
                    else:
                        # Remove the * and parse the range
                        range_part = var_length[1:]  # Remove *
                        if ".." in range_part:
                            parts = range_part.split("..")
                            min_length = int(parts[0]) if parts[0] else 1
                            max_length = int(parts[1]) if parts[1] else float("inf")
                        else:
                            # Single number like *3
                            min_length = 1
                            max_length = int(range_part)

                # Parse properties
                if self._current_token() == "{":
                    properties = self._parse_properties()

                self._expect_token("]")

            # Determine final direction based on what follows
            if self._current_token() == "-":
                self._consume_token()  # consume second '-'
                if self._current_token() == ">":
                    self._consume_token()  # consume '>'
                    direction = "outgoing"
                else:
                    direction = "bidirectional"
            elif self._current_token() == ">":
                self._consume_token()  # consume '>'
                direction = "outgoing"
            else:
                direction = "bidirectional"

        return RelationshipPattern(
            variable=rel_variable,
            types=rel_types,
            properties=properties,
            direction=direction,
            min_length=min_length,
            max_length=max_length,
        )

    def _parse_properties(self) -> Dict[str, Any]:
        self._expect_token("{")
        properties = {}

        while self._current_token() != "}":
            key = self._consume_token()
            self._expect_token(":")
            value = self._parse_literal()
            properties[key] = value

            if self._current_token() == ",":
                self._consume_token()

        self._expect_token("}")
        return properties

    def _parse_literal(self) -> Any:
        token = self._consume_token()
        if not token:
            return None

        # String literal
        if token.startswith(("'", '"')):
            return token[1:-1]

        # Number literal
        if token.replace(".", "").isdigit():
            return float(token) if "." in token else int(token)

        # Boolean or null
        if token.upper() == "TRUE":
            return True
        elif token.upper() == "FALSE":
            return False
        elif token.upper() == "NULL":
            return None

        # Default to string
        return token

    def _parse_where_clause(self) -> WhereClause:
        self._expect_token("WHERE")
        condition = self._parse_expression()
        return WhereClause(condition)

    def _parse_return_clause(self) -> ReturnClause:
        self._expect_token("RETURN")

        distinct = False
        if self._current_token() and self._current_token().upper() == "DISTINCT":
            distinct = True
            self._consume_token()

        items = self._parse_return_items()

        order_by = None
        skip = None
        limit = None

        # Parse optional clauses
        while self._current_token():
            token = self._current_token().upper()
            if token == "ORDER":
                order_by = self._parse_order_by()
            elif token == "SKIP":
                self._consume_token()
                skip = int(self._consume_token())
            elif token == "LIMIT":
                self._consume_token()
                limit = int(self._consume_token())
            else:
                break

        return ReturnClause(items, distinct, order_by, skip, limit)

    def _parse_return_items(self) -> List[ReturnItem]:
        items = []
        items.append(self._parse_return_item())

        while self._current_token() == ",":
            self._consume_token()
            items.append(self._parse_return_item())

        return items

    def _parse_return_item(self) -> ReturnItem:
        expression = self._parse_expression()
        alias = None

        if self._current_token() and self._current_token().upper() == "AS":
            self._consume_token()
            alias = self._consume_token()

        return ReturnItem(expression, alias)

    def _parse_order_by(self) -> OrderByClause:
        self._expect_token("ORDER")
        self._expect_token("BY")

        items = []
        items.append(self._parse_order_by_item())

        while self._current_token() == ",":
            self._consume_token()
            items.append(self._parse_order_by_item())

        return OrderByClause(items)

    def _parse_order_by_item(self) -> OrderByItem:
        expression = self._parse_expression()
        ascending = True

        if self._current_token() and self._current_token().upper() in ["ASC", "DESC"]:
            ascending = self._consume_token().upper() == "ASC"

        return OrderByItem(expression, ascending)

    def _parse_with_clause(self) -> WithClause:
        self._expect_token("WITH")
        items = self._parse_return_items()
        where_clause = None

        if self._current_token() and self._current_token().upper() == "WHERE":
            where_clause = self._parse_where_clause()

        return WithClause(items, where_clause)

    def _parse_expression(self) -> Expression:
        """Parse expressions (simplified version)"""
        return self._parse_or_expression()

    def _parse_or_expression(self) -> Expression:
        left = self._parse_and_expression()

        while self._current_token() and self._current_token().upper() == "OR":
            op = self._consume_token()
            right = self._parse_and_expression()
            left = BinaryExpression(left, op, right)

        return left

    def _parse_and_expression(self) -> Expression:
        left = self._parse_comparison_expression()

        while self._current_token() and self._current_token().upper() == "AND":
            op = self._consume_token()
            right = self._parse_comparison_expression()
            left = BinaryExpression(left, op, right)

        return left

    def _parse_comparison_expression(self) -> Expression:
        left = self._parse_primary_expression()

        if self._current_token() in ["=", "<>", "!=", "<", ">", "<=", ">="]:
            op = self._consume_token()
            right = self._parse_primary_expression()
            return BinaryExpression(left, op, right)

        return left

    def _parse_primary_expression(self) -> Expression:
        token = self._current_token()

        if not token:
            raise ValueError("Unexpected end of expression")

        # Property access (variable.property)
        if self._peek_token() == ".":
            variable = self._consume_token()
            self._consume_token()  # consume '.'
            property_name = self._consume_token()
            return PropertyExpression(variable, property_name)

        # Function call
        if self._peek_token() == "(":
            func_name = self._consume_token()
            self._consume_token()  # consume '('
            args = []

            if self._current_token() != ")":
                args.append(self._parse_expression())
                while self._current_token() == ",":
                    self._consume_token()
                    args.append(self._parse_expression())

            self._expect_token(")")
            return FunctionCall(func_name, args)

        # Literal
        if token.startswith(("'", '"')) or token.replace(".", "").isdigit():
            return LiteralExpression(self._parse_literal())

        # Variable
        return VariableExpression(self._consume_token())
