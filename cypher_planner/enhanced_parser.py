# cypher_planner/enhanced_parser.py

"""
Enhanced Cypher Parser with comprehensive error handling
Based on FalkorDB error handling patterns
"""

import re
from typing import List, Optional, Dict, Any, Set
from .ast_nodes import *
from .error_context import ErrorCode, set_error, set_warning, get_error_context, ErrorPosition
from .query_validator import QueryValidator

class EnhancedCypherParser:
    """Enhanced Cypher parser with comprehensive error handling"""
    
    def __init__(self):
        self.tokens = []
        self.position = 0
        self.query_text = ""
        self.validator = QueryValidator()
        self.variable_scope_stack = []  # Track variable scopes
        self.current_scope_variables = set()  # Variables in current scope
        
    def parse(self, query: str) -> Optional[Query]:
        """
        Parse a Cypher query with comprehensive error handling
        Returns None if parsing fails
        """
        if not query:
            set_error(ErrorCode.MISSING_TOKEN, "Empty query provided")
            return None
            
        # Initialize error context
        error_ctx = get_error_context()
        error_ctx.set_query(query)
        self.query_text = query
        
        # Pre-validation
        if not self.validator.validate_query(query):
            return None
            
        try:
            # Tokenization with error handling
            self.tokens = self._safe_tokenize(query)
            if error_ctx.has_errors:
                return None
                
            self.position = 0
            
            # Parse with recovery
            return self._parse_query_with_recovery()
            
        except Exception as e:
            set_error(ErrorCode.UNEXPECTED_TOKEN, 
                     f"Internal parser error: {str(e)}")
            return None
            
    def _safe_tokenize(self, query: str) -> List[str]:
        """Enhanced tokenizer with error reporting"""
        token_pattern = r"""
            (?P<KEYWORD>MATCH|WHERE|RETURN|OPTIONAL|WITH|ORDER\s+BY|SKIP|LIMIT|AS|AND|OR|NOT|DISTINCT|CONTAINS|EXISTS|CREATE|MERGE|DELETE|SET|REMOVE|UNWIND|FOREACH|CALL|YIELD|UNION|ASC|DESC|IS|NULL|TRUE|FALSE|IN|STARTS|ENDS)\s*|
            (?P<STRING>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")\s*|
            (?P<NUMBER>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*|
            (?P<IDENTIFIER>[a-zA-Z_][a-zA-Z0-9_]*)\s*|
            (?P<OPERATOR><=|>=|<>|!=|=~|=|<|>)\s*|
            (?P<ARROW><--|-->)\s*|
            (?P<DASH>--)\s*|
            (?P<VARIABLE_LENGTH>\*(?:\d+)?(?:\.\.(?:\d+)?)?)\s*|
            (?P<PUNCTUATION>[(){}\[\],:.-])\s*|
            (?P<PLUS_MINUS>[+-])\s*|
            (?P<MULT_DIV_MOD>[*/%])\s*|
            (?P<WHITESPACE>\s+)|
            (?P<UNKNOWN>.+?)
        """
        
        tokens = []
        line_num = 1
        line_start = 0
        
        for match in re.finditer(token_pattern, query, re.IGNORECASE | re.VERBOSE):
            kind = match.lastgroup
            value = match.group().strip()
            start_pos = match.start()
            
            # Update line tracking
            newlines = query[line_start:start_pos].count('\n')
            if newlines:
                line_num += newlines
                line_start = query.rfind('\n', 0, start_pos) + 1
                
            if kind == "WHITESPACE":
                continue
            elif kind == "UNKNOWN":
                # Handle unknown tokens
                error_pos = ErrorPosition(
                    line=line_num,
                    column=start_pos - line_start + 1,
                    position=start_pos,
                    length=len(value)
                )
                set_error(ErrorCode.UNEXPECTED_TOKEN, 
                         f"Unexpected character sequence: '{value}'",
                         position=error_pos,
                         suggestion="Check for typos or invalid characters")
                continue
            elif kind == "STRING":
                # Validate string literal
                if not self._validate_string_token(value, start_pos, line_num, line_start):
                    continue
            elif kind == "NUMBER":
                # Validate numeric literal
                if not self._validate_number_token(value, start_pos, line_num, line_start):
                    continue
                    
            if value:
                tokens.append(value)
                
        return tokens
        
    def _validate_string_token(self, token: str, pos: int, line: int, line_start: int) -> bool:
        """Validate string token"""
        if len(token) < 2:
            error_pos = ErrorPosition(line, pos - line_start + 1, pos, len(token))
            set_error(ErrorCode.UNEXPECTED_TOKEN, 
                     "Unterminated string literal",
                     position=error_pos)
            return False
            
        quote_char = token[0]
        if token[-1] != quote_char:
            error_pos = ErrorPosition(line, pos - line_start + 1, pos, len(token))
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Unterminated string literal starting with {quote_char}",
                     position=error_pos)
            return False
            
        return True
        
    def _validate_number_token(self, token: str, pos: int, line: int, line_start: int) -> bool:
        """Validate numeric token"""
        try:
            float(token)
            return True
        except ValueError:
            error_pos = ErrorPosition(line, pos - line_start + 1, pos, len(token))
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Invalid numeric literal: '{token}'",
                     position=error_pos)
            return False
            
    def _parse_query_with_recovery(self) -> Optional[Query]:
        """Parse query with error recovery"""
        query = Query()
        
        # Track variables for scope validation
        self.current_scope_variables = set()
        
        while self._current_token():
            try:
                token = self._current_token().upper()
                
                if token == "MATCH":
                    clause = self._safe_parse_match_clause()
                    if clause:
                        query.match_clauses.append(clause)
                elif token == "OPTIONAL":
                    clause = self._safe_parse_optional_match()
                    if clause:
                        query.optional_match_clauses.append(clause)
                elif token == "WHERE":
                    clause = self._safe_parse_where_clause()
                    if clause:
                        query.where_clause = clause
                elif token == "RETURN":
                    clause = self._safe_parse_return_clause()
                    if clause:
                        query.return_clause = clause
                        break  # RETURN is typically last
                elif token == "WITH":
                    clause = self._safe_parse_with_clause()
                    if clause:
                        query.with_clauses.append(clause)
                else:
                    set_error(ErrorCode.UNEXPECTED_TOKEN,
                             f"Unexpected token: '{token}'",
                             suggestion="Expected MATCH, WHERE, RETURN, etc.")
                    self._skip_to_next_clause()
                    
            except Exception as e:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         f"Error parsing clause: {str(e)}")
                self._skip_to_next_clause()
                
        # Validate the constructed query
        if not self._validate_query_semantics(query):
            return None
            
        return query if not get_error_context().has_errors else None
        
    def _safe_parse_match_clause(self) -> Optional[MatchClause]:
        """Parse MATCH clause with error handling"""
        if not self._expect_token("MATCH"):
            return None
            
        patterns = self._safe_parse_patterns()
        return MatchClause(patterns) if patterns else None
        
    def _safe_parse_optional_match(self) -> Optional[OptionalMatchClause]:
        """Parse OPTIONAL MATCH clause with error handling"""
        if not self._expect_token("OPTIONAL"):
            return None
            
        if not self._peek_token() or self._peek_token().upper() != "MATCH":
            set_error(ErrorCode.MISSING_TOKEN, 
                     "Expected MATCH after OPTIONAL",
                     suggestion="Use OPTIONAL MATCH")
            return None
            
        if not self._expect_token("MATCH"):
            return None
            
        patterns = self._safe_parse_patterns()
        return OptionalMatchClause(patterns) if patterns else None
        
    def _safe_parse_patterns(self) -> List[Pattern]:
        """Parse patterns with error recovery"""
        patterns = []
        
        pattern = self._safe_parse_pattern()
        if pattern:
            patterns.append(pattern)
            
            while self._current_token() == ",":
                self._consume_token()  # consume ','
                next_pattern = self._safe_parse_pattern()
                if next_pattern:
                    patterns.append(next_pattern)
                else:
                    break  # Stop on parse error
                    
        return patterns
        
    def _safe_parse_pattern(self) -> Optional[Pattern]:
        """Parse pattern with error handling"""
        elements = []
        
        # Parse first node
        if self._current_token() == "(":
            node = self._safe_parse_node_pattern()
            if node:
                elements.append(node)
            else:
                return None
        else:
            set_error(ErrorCode.INVALID_PATTERN,
                     "Pattern must start with a node ()",
                     suggestion="Start pattern with (variable:Label)")
            return None
            
        # Parse relationships and subsequent nodes
        while self._current_token() in ["<--", "-->", "--", "-", "<"]:
            # Parse relationship
            rel = self._safe_parse_relationship_pattern()
            if rel:
                elements.append(rel)
                
                # Parse target node
                if self._current_token() == "(":
                    node = self._safe_parse_node_pattern()
                    if node:
                        elements.append(node)
                    else:
                        set_error(ErrorCode.INVALID_PATTERN,
                                 "Relationship must connect to a node",
                                 suggestion="Add target node: ()-[]->()")
                        return None
                else:
                    # Create anonymous target node
                    elements.append(NodePattern())
            else:
                break
                
        return Pattern(elements)
        
    def _safe_parse_node_pattern(self) -> Optional[NodePattern]:
        """Parse node pattern with error handling"""
        if not self._expect_token("("):
            return None
            
        variable = None
        labels = []
        properties = {}
        
        # Parse variable
        if (self._current_token() and 
            self._current_token() not in [":", ")", "{"] and
            self._is_valid_identifier(self._current_token())):
            variable = self._consume_token()
            
            # Track variable usage
            if variable:
                self.current_scope_variables.add(variable)
                
        # Parse labels
        while self._current_token() == ":":
            self._consume_token()  # consume ':'
            if (self._current_token() and 
                self._current_token() not in [")", "{", ":"]):
                label = self._consume_token()
                if self._is_valid_identifier(label):
                    labels.append(label)
                else:
                    set_error(ErrorCode.INVALID_LABEL,
                             f"Invalid label name: '{label}'",
                             suggestion="Label names must be valid identifiers")
            else:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected label name after ':'",
                         suggestion="Provide label name or remove ':'")
                
        # Parse properties
        if self._current_token() == "{":
            properties = self._safe_parse_properties()
            if properties is None:
                return None
                
        if not self._expect_token(")"):
            return None
            
        return NodePattern(variable, labels, properties)
        
    def _safe_parse_relationship_pattern(self) -> Optional[RelationshipPattern]:
        """Parse relationship pattern with error handling"""
        direction = "outgoing"
        rel_variable = None
        rel_types = []
        properties = {}
        min_length = None
        max_length = None
        
        # Determine direction and consume direction tokens
        if self._current_token() == "<--":
            direction = "incoming"
            self._consume_token()
        elif self._current_token() == "-->":
            direction = "outgoing"
            self._consume_token()
        elif self._current_token() == "--":
            direction = "bidirectional"
            self._consume_token()
        elif self._current_token() == "<":
            # Complex incoming pattern <-[]-
            self._consume_token()
            if not self._expect_token("-"):
                return None
            direction = "incoming"
            
            # Parse relationship details in brackets if present
            if self._current_token() == "[":
                rel_details = self._safe_parse_relationship_brackets()
                if rel_details:
                    rel_variable, rel_types, properties, min_length, max_length = rel_details
                else:
                    return None
                    
            if not self._expect_token("-"):
                return None
                
        elif self._current_token() == "-":
            self._consume_token()
            
            # Parse relationship details in brackets if present
            if self._current_token() == "[":
                rel_details = self._safe_parse_relationship_brackets()
                if rel_details:
                    rel_variable, rel_types, properties, min_length, max_length = rel_details
                else:
                    return None
                    
            # Determine final direction
            if self._current_token() == "-":
                self._consume_token()
                if self._current_token() == ">":
                    self._consume_token()
                    direction = "outgoing"
                else:
                    direction = "bidirectional"
            elif self._current_token() == ">":
                self._consume_token()
                direction = "outgoing"
            else:
                direction = "bidirectional"
        else:
            set_error(ErrorCode.INVALID_PATTERN,
                     f"Expected relationship pattern, got '{self._current_token()}'",
                     suggestion="Use patterns like -[]-, -->, <--, etc.")
            return None
            
        # Track relationship variable if specified
        if rel_variable:
            self.current_scope_variables.add(rel_variable)
            
        return RelationshipPattern(
            variable=rel_variable,
            types=rel_types,
            properties=properties,
            direction=direction,
            min_length=min_length,
            max_length=max_length
        )
        
    def _safe_parse_relationship_brackets(self) -> Optional[tuple]:
        """Parse relationship details within brackets"""
        if not self._expect_token("["):
            return None
            
        rel_variable = None
        rel_types = []
        properties = {}
        min_length = None
        max_length = None
        
        # Parse variable
        if (self._current_token() and 
            self._current_token() not in [":", "]", "*"] and
            not self._current_token().startswith("*") and
            self._is_valid_identifier(self._current_token())):
            rel_variable = self._consume_token()
            
        # Parse types
        while self._current_token() == ":":
            self._consume_token()  # consume ':'
            if (self._current_token() and 
                self._current_token() not in ["]", "*", ":"] and
                not self._current_token().startswith("*")):
                rel_type = self._consume_token()
                if self._is_valid_identifier(rel_type):
                    rel_types.append(rel_type)
                else:
                    set_error(ErrorCode.INVALID_RELATIONSHIP,
                             f"Invalid relationship type: '{rel_type}'")
                    return None
                    
        # Parse variable length
        if self._current_token() and self._current_token().startswith("*"):
            var_length_result = self._safe_parse_variable_length()
            if var_length_result is None:
                return None
            min_length, max_length = var_length_result
            
        # Parse properties
        if self._current_token() == "{":
            properties = self._safe_parse_properties()
            if properties is None:
                return None
                
        if not self._expect_token("]"):
            return None
            
        return rel_variable, rel_types, properties, min_length, max_length
        
    def _safe_parse_variable_length(self) -> Optional[tuple]:
        """Parse variable length specification"""
        var_length = self._consume_token()
        
        if var_length == "*":
            return 1, float("inf")
            
        # Parse range: *1..3, *..3, *1.., etc.
        range_part = var_length[1:]  # Remove *
        
        if ".." not in range_part:
            # Single number like *3
            try:
                max_val = int(range_part)
                if max_val < 0:
                    set_error(ErrorCode.INVALID_VARIABLE_LENGTH,
                             "Variable length cannot be negative")
                    return None
                return 1, max_val
            except ValueError:
                set_error(ErrorCode.INVALID_VARIABLE_LENGTH,
                         f"Invalid variable length: '{var_length}'")
                return None
                
        # Range specification
        parts = range_part.split("..")
        if len(parts) != 2:
            set_error(ErrorCode.INVALID_VARIABLE_LENGTH,
                     f"Invalid variable length range: '{var_length}'")
            return None
            
        try:
            min_val = int(parts[0]) if parts[0] else 1
            max_val = int(parts[1]) if parts[1] else float("inf")
            
            if min_val < 0:
                set_error(ErrorCode.INVALID_VARIABLE_LENGTH,
                         "Variable length minimum cannot be negative")
                return None
                
            if max_val != float("inf") and min_val > max_val:
                set_error(ErrorCode.INVALID_VARIABLE_LENGTH,
                         "Variable length minimum cannot exceed maximum")
                return None
                
            return min_val, max_val
            
        except ValueError:
            set_error(ErrorCode.INVALID_VARIABLE_LENGTH,
                     f"Invalid variable length range: '{var_length}'")
            return None
            
    def _safe_parse_properties(self) -> Optional[Dict[str, Any]]:
        """Parse properties with error handling"""
        if not self._expect_token("{"):
            return None
            
        properties = {}
        
        if self._current_token() == "}":
            self._consume_token()
            return properties
            
        while True:
            # Parse property key
            if not self._current_token():
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected property name or '}'")
                return None
                
            key = self._consume_token()
            if not self._is_valid_identifier(key):
                set_error(ErrorCode.INVALID_PROPERTY,
                         f"Invalid property name: '{key}'")
                return None
                
            if not self._expect_token(":"):
                return None
                
            # Parse property value
            value = self._safe_parse_literal()
            if value is None:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected property value after ':'")
                return None
                
            properties[key] = value
            
            if self._current_token() == ",":
                self._consume_token()
                if self._current_token() == "}":
                    set_warning(ErrorCode.UNEXPECTED_TOKEN,
                               "Trailing comma in properties")
                    break
            elif self._current_token() == "}":
                break
            else:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected ',' or '}' in properties")
                return None
                
        if not self._expect_token("}"):
            return None
            
        return properties
        
    def _safe_parse_literal(self) -> Optional[Any]:
        """Parse literal with error handling"""
        token = self._current_token()
        if not token:
            return None
            
        # String literal
        if token.startswith(("'", '"')):
            value = self._consume_token()
            if len(value) >= 2 and value[0] == value[-1]:
                return value[1:-1]  # Remove quotes
            else:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "Malformed string literal")
                return None
                
        # Number literal
        if re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?
            value = self._consume_token()
            try:
                return float(value) if '.' in value or 'e' in value.lower() else int(value)
            except ValueError:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         f"Invalid number: '{value}'")
                return None
                
        # Boolean and null
        if token.upper() in ("TRUE", "FALSE", "NULL"):
            value = self._consume_token().upper()
            if value == "TRUE":
                return True
            elif value == "FALSE":
                return False
            else:  # NULL
                return None
                
        # Default to string
        return self._consume_token()
        
    def _safe_parse_where_clause(self) -> Optional[WhereClause]:
        """Parse WHERE clause with error handling"""
        if not self._expect_token("WHERE"):
            return None
            
        condition = self._safe_parse_expression()
        return WhereClause(condition) if condition else None
        
    def _safe_parse_expression(self) -> Optional[Expression]:
        """Parse expression with error handling"""
        try:
            return self._parse_or_expression()
        except Exception as e:
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Error parsing expression: {str(e)}")
            return None
            
    def _parse_or_expression(self) -> Optional[Expression]:
        """Parse OR expression"""
        left = self._parse_and_expression()
        if not left:
            return None
            
        while self._current_token() and self._current_token().upper() == "OR":
            op = self._consume_token()
            right = self._parse_and_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected expression after 'OR'")
                return None
            left = BinaryExpression(left, op, right)
            
        return left
        
    def _parse_and_expression(self) -> Optional[Expression]:
        """Parse AND expression"""
        left = self._parse_comparison_expression()
        if not left:
            return None
            
        while self._current_token() and self._current_token().upper() == "AND":
            op = self._consume_token()
            right = self._parse_comparison_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected expression after 'AND'")
                return None
            left = BinaryExpression(left, op, right)
            
        return left
        
    def _parse_comparison_expression(self) -> Optional[Expression]:
        """Parse comparison expression"""
        left = self._parse_primary_expression()
        if not left:
            return None
            
        if self._current_token() in ["=", "<>", "!=", "<", ">", "<=", ">=", "=~", "CONTAINS"]:
            op = self._consume_token()
            right = self._parse_primary_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         f"Expected expression after '{op}'")
                return None
            return BinaryExpression(left, op, right)
            
        return left
        
    def _parse_primary_expression(self) -> Optional[Expression]:
        """Parse primary expression"""
        token = self._current_token()
        if not token:
            return None
            
        # Handle EXISTS function specially
        if token.upper() == "EXISTS":
            return self._parse_exists_function()
            
        # Property access (variable.property)
        if self._peek_token() == ".":
            variable = self._consume_token()
            if not self._is_valid_identifier(variable):
                set_error(ErrorCode.INVALID_PROPERTY,
                         f"Invalid variable name: '{variable}'")
                return None
                
            # Check if variable is in scope
            if variable not in self.current_scope_variables:
                set_warning(ErrorCode.UNDEFINED_VARIABLE,
                           f"Variable '{variable}' may not be defined")
                           
            self._consume_token()  # consume '.'
            property_name = self._consume_token()
            if not property_name or not self._is_valid_identifier(property_name):
                set_error(ErrorCode.INVALID_PROPERTY,
                         "Expected property name after '.'")
                return None
                
            return PropertyExpression(variable, property_name)
            
        # Function call
        if self._peek_token() == "(":
            func_name = self._consume_token()
            if not self._is_valid_identifier(func_name):
                set_error(ErrorCode.UNKNOWN_FUNCTION,
                         f"Invalid function name: '{func_name}'")
                return None
                
            return self._parse_function_call(func_name)
            
        # Literal
        if (token.startswith(("'", '"')) or 
            re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?
            literal_value = self._safe_parse_literal()
            return LiteralExpression(literal_value) if literal_value is not None else None
            
        # Variable
        if self._is_valid_identifier(token):
            variable = self._consume_token()
            if variable not in self.current_scope_variables:
                set_warning(ErrorCode.UNDEFINED_VARIABLE,
                           f"Variable '{variable}' may not be defined")
            return VariableExpression(variable)
            
        set_error(ErrorCode.UNEXPECTED_TOKEN,
                 f"Unexpected token in expression: '{token}'")
        return None
        
    def _parse_exists_function(self) -> Optional[FunctionCall]:
        """Parse EXISTS function with pattern"""
        func_name = self._consume_token()  # consume 'EXISTS'
        
        if not self._expect_token("("):
            return None
            
        # Parse the pattern inside EXISTS
        pattern_tokens = []
        paren_count = 1
        
        while self._current_token() and paren_count > 0:
            token = self._consume_token()
            if token == "(":
                paren_count += 1
            elif token == ")":
                paren_count -= 1
                
            if paren_count > 0:
                pattern_tokens.append(token)
                
        if paren_count > 0:
            set_error(ErrorCode.UNBALANCED_PARENTHESES,
                     "Unmatched parentheses in EXISTS function")
            return None
            
        pattern_str = " ".join(pattern_tokens)
        args = [LiteralExpression(pattern_str)]
        
        return FunctionCall(func_name, args)
        
    def _parse_function_call(self, func_name: str) -> Optional[FunctionCall]:
        """Parse function call"""
        if not self._expect_token("("):
            return None
            
        args = []
        
        if self._current_token() != ")":
            arg = self._safe_parse_expression()
            if arg:
                args.append(arg)
                
                while self._current_token() == ",":
                    self._consume_token()
                    next_arg = self._safe_parse_expression()
                    if next_arg:
                        args.append(next_arg)
                    else:
                        set_error(ErrorCode.MISSING_TOKEN,
                                 "Expected argument after ','")
                        return None
                        
        if not self._expect_token(")"):
            return None
            
        return FunctionCall(func_name, args)
        
    def _safe_parse_return_clause(self) -> Optional[ReturnClause]:
        """Parse RETURN clause with error handling"""
        if not self._expect_token("RETURN"):
            return None
            
        distinct = False
        if self._current_token() and self._current_token().upper() == "DISTINCT":
            distinct = True
            self._consume_token()
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "RETURN clause must specify what to return")
            return None
            
        order_by = None
        skip = None
        limit = None
        
        # Parse optional clauses
        while self._current_token():
            token = self._current_token().upper()
            if token == "ORDER":
                order_by = self._safe_parse_order_by()
                if not order_by:
                    break
            elif token == "SKIP":
                skip = self._safe_parse_skip()
                if skip is None:
                    break
            elif token == "LIMIT":
                limit = self._safe_parse_limit()
                if limit is None:
                    break
            else:
                break
                
        return ReturnClause(items, distinct, order_by, skip, limit)
        
    def _safe_parse_return_items(self) -> List[ReturnItem]:
        """Parse return items with error handling"""
        items = []
        
        item = self._safe_parse_return_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_return_item()
                if next_item:
                    items.append(next_item)
                else:
                    set_error(ErrorCode.MISSING_TOKEN,
                             "Expected return item after ','")
                    break
                    
        return items
        
    def _safe_parse_return_item(self) -> Optional[ReturnItem]:
        """Parse return item with error handling"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        alias = None
        if self._current_token() and self._current_token().upper() == "AS":
            self._consume_token()
            alias = self._consume_token()
            if not alias or not self._is_valid_identifier(alias):
                set_error(ErrorCode.INVALID_LABEL,
                         "Expected valid alias after 'AS'")
                return None
                
        return ReturnItem(expression, alias)
        
    def _safe_parse_order_by(self) -> Optional[OrderByClause]:
        """Parse ORDER BY clause"""
        if not self._expect_token("ORDER"):
            return None
        if not self._expect_token("BY"):
            return None
            
        items = []
        item = self._safe_parse_order_by_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_order_by_item()
                if next_item:
                    items.append(next_item)
                else:
                    break
                    
        return OrderByClause(items) if items else None
        
    def _safe_parse_order_by_item(self) -> Optional[OrderByItem]:
        """Parse ORDER BY item"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        ascending = True
        if self._current_token() and self._current_token().upper() in ["ASC", "DESC"]:
            ascending = self._consume_token().upper() == "ASC"
            
        return OrderByItem(expression, ascending)
        
    def _safe_parse_skip(self) -> Optional[int]:
        """Parse SKIP value"""
        self._consume_token()  # consume 'SKIP'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value < 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "SKIP value cannot be negative")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected integer after SKIP, got '{token}'")
            return None
            
    def _safe_parse_limit(self) -> Optional[int]:
        """Parse LIMIT value"""
        self._consume_token()  # consume 'LIMIT'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value <= 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "LIMIT value must be positive")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected positive integer after LIMIT, got '{token}'")
            return None
            
    def _safe_parse_with_clause(self) -> Optional[WithClause]:
        """Parse WITH clause"""
        if not self._expect_token("WITH"):
            return None
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "WITH clause must specify items")
            return None
            
        where_clause = None
        if self._current_token() and self._current_token().upper() == "WHERE":
            where_clause = self._safe_parse_where_clause()
            
        return WithClause(items, where_clause)
        
    def _validate_query_semantics(self, query: Query) -> bool:
        """Validate semantic correctness of parsed query"""
        valid = True
        
        # Check that variables used in WHERE/RETURN are defined in MATCH
        defined_vars = set()
        
        # Collect variables from MATCH clauses
        for match_clause in query.match_clauses:
            for pattern in match_clause.patterns:
                for element in pattern.elements:
                    if isinstance(element, NodePattern) and element.variable:
                        defined_vars.add(element.variable)
                    elif isinstance(element, RelationshipPattern) and element.variable:
                        defined_vars.add(element.variable)
                        
        # Check WHERE clause references
        if query.where_clause:
            where_vars = self._extract_variables_from_expression(query.where_clause.condition)
            for var in where_vars:
                if var not in defined_vars:
                    set_error(ErrorCode.UNDEFINED_VARIABLE,
                             f"Variable '{var}' in WHERE clause is not defined in MATCH")
                    valid = False
                    
        # Check RETURN clause references  
        if query.return_clause:
            for item in query.return_clause.items:
                return_vars = self._extract_variables_from_expression(item.expression)
                for var in return_vars:
                    if var not in defined_vars:
                        set_error(ErrorCode.UNDEFINED_VARIABLE,
                                 f"Variable '{var}' in RETURN clause is not defined in MATCH")
                        valid = False
                        
        return valid
        
    def _extract_variables_from_expression(self, expr: Expression) -> Set[str]:
        """Extract all variable references from an expression"""
        variables = set()
        
        if isinstance(expr, VariableExpression):
            variables.add(expr.name)
        elif isinstance(expr, PropertyExpression):
            variables.add(expr.variable)
        elif isinstance(expr, BinaryExpression):
            variables.update(self._extract_variables_from_expression(expr.left))
            variables.update(self._extract_variables_from_expression(expr.right))
        elif isinstance(expr, FunctionCall):
            for arg in expr.arguments:
                variables.update(self._extract_variables_from_expression(arg))
                
        return variables
        
    # Utility methods
    def _current_token(self) -> Optional[str]:
        """Get current token"""
        return self.tokens[self.position] if self.position < len(self.tokens) else None
        
    def _consume_token(self) -> Optional[str]:
        """Consume and return current token"""
        token = self._current_token()
        if token:
            self.position += 1
            get_error_context().current_position = self.position
        return token
        
    def _peek_token(self, offset: int = 1) -> Optional[str]:
        """Peek at future token"""
        pos = self.position + offset
        return self.tokens[pos] if pos < len(self.tokens) else None
        
    def _expect_token(self, expected: str) -> bool:
        """Expect a specific token"""
        token = self._consume_token()
        if not token or token.upper() != expected.upper():
            set_error(ErrorCode.MISSING_TOKEN,
                     f"Expected '{expected}', got '{token or 'end of input'}'",
                     suggestion=f"Add '{expected}' to your query")
            return False
        return True
        
    def _is_valid_identifier(self, token: str) -> bool:
        """Check if token is a valid identifier"""
        if not token:
            return False
        return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*
        
    def _skip_to_next_clause(self):
        """Skip tokens until next clause for error recovery"""
        clause_keywords = {
            'MATCH', 'WHERE', 'RETURN', 'WITH', 'CREATE', 'MERGE', 
            'DELETE', 'SET', 'REMOVE', 'UNWIND', 'FOREACH', 'CALL'
        }
        
        while self._current_token():
            if self._current_token().upper() in clause_keywords:
                break
            self._consume_token()


# Create wrapper to maintain compatibility with existing code
class CypherParser(EnhancedCypherParser):
    """Backward-compatible wrapper for enhanced parser"""
    
    def parse(self, query: str) -> Query:
        """Parse query and raise exception on error for backward compatibility"""
        result = super().parse(query)
        if result is None:
            error_msg = get_error_context().format_errors()
            raise ValueError(f"Parse error:\n{error_msg}")
        return result, token):
            value = self._consume_token()
            try:
                return float(value) if '.' in value or 'e' in value.lower() else int(value)
            except ValueError:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         f"Invalid number: '{value}'")
                return None
                
        # Boolean and null
        if token.upper() in ("TRUE", "FALSE", "NULL"):
            value = self._consume_token().upper()
            if value == "TRUE":
                return True
            elif value == "FALSE":
                return False
            else:  # NULL
                return None
                
        # Default to string
        return self._consume_token()
        
    def _safe_parse_where_clause(self) -> Optional[WhereClause]:
        """Parse WHERE clause with error handling"""
        if not self._expect_token("WHERE"):
            return None
            
        condition = self._safe_parse_expression()
        return WhereClause(condition) if condition else None
        
    def _safe_parse_expression(self) -> Optional[Expression]:
        """Parse expression with error handling"""
        try:
            return self._parse_or_expression()
        except Exception as e:
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Error parsing expression: {str(e)}")
            return None
            
    def _parse_or_expression(self) -> Optional[Expression]:
        """Parse OR expression"""
        left = self._parse_and_expression()
        if not left:
            return None
            
        while self._current_token() and self._current_token().upper() == "OR":
            op = self._consume_token()
            right = self._parse_and_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected expression after 'OR'")
                return None
            left = BinaryExpression(left, op, right)
            
        return left
        
    def _parse_and_expression(self) -> Optional[Expression]:
        """Parse AND expression"""
        left = self._parse_comparison_expression()
        if not left:
            return None
            
        while self._current_token() and self._current_token().upper() == "AND":
            op = self._consume_token()
            right = self._parse_comparison_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected expression after 'AND'")
                return None
            left = BinaryExpression(left, op, right)
            
        return left
        
    def _parse_comparison_expression(self) -> Optional[Expression]:
        """Parse comparison expression"""
        left = self._parse_primary_expression()
        if not left:
            return None
            
        if self._current_token() in ["=", "<>", "!=", "<", ">", "<=", ">=", "=~", "CONTAINS"]:
            op = self._consume_token()
            right = self._parse_primary_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         f"Expected expression after '{op}'")
                return None
            return BinaryExpression(left, op, right)
            
        return left
        
    def _parse_primary_expression(self) -> Optional[Expression]:
        """Parse primary expression"""
        token = self._current_token()
        if not token:
            return None
            
        # Handle EXISTS function specially
        if token.upper() == "EXISTS":
            return self._parse_exists_function()
            
        # Property access (variable.property)
        if self._peek_token() == ".":
            variable = self._consume_token()
            if not self._is_valid_identifier(variable):
                set_error(ErrorCode.INVALID_PROPERTY,
                         f"Invalid variable name: '{variable}'")
                return None
                
            # Check if variable is in scope
            if variable not in self.current_scope_variables:
                set_warning(ErrorCode.UNDEFINED_VARIABLE,
                           f"Variable '{variable}' may not be defined")
                           
            self._consume_token()  # consume '.'
            property_name = self._consume_token()
            if not property_name or not self._is_valid_identifier(property_name):
                set_error(ErrorCode.INVALID_PROPERTY,
                         "Expected property name after '.'")
                return None
                
            return PropertyExpression(variable, property_name)
            
        # Function call
        if self._peek_token() == "(":
            func_name = self._consume_token()
            if not self._is_valid_identifier(func_name):
                set_error(ErrorCode.UNKNOWN_FUNCTION,
                         f"Invalid function name: '{func_name}'")
                return None
                
            return self._parse_function_call(func_name)
            
        # Literal
        if (token.startswith(("'", '"')) or 
            re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?, token) or
            token.upper() in ("TRUE", "FALSE", "NULL")):
            literal_value = self._safe_parse_literal()
            return LiteralExpression(literal_value) if literal_value is not None else None
            
        # Variable
        if self._is_valid_identifier(token):
            variable = self._consume_token()
            if variable not in self.current_scope_variables:
                set_warning(ErrorCode.UNDEFINED_VARIABLE,
                           f"Variable '{variable}' may not be defined")
            return VariableExpression(variable)
            
        set_error(ErrorCode.UNEXPECTED_TOKEN,
                 f"Unexpected token in expression: '{token}'")
        return None
        
    def _parse_exists_function(self) -> Optional[FunctionCall]:
        """Parse EXISTS function with pattern"""
        func_name = self._consume_token()  # consume 'EXISTS'
        
        if not self._expect_token("("):
            return None
            
        # Parse the pattern inside EXISTS
        pattern_tokens = []
        paren_count = 1
        
        while self._current_token() and paren_count > 0:
            token = self._consume_token()
            if token == "(":
                paren_count += 1
            elif token == ")":
                paren_count -= 1
                
            if paren_count > 0:
                pattern_tokens.append(token)
                
        if paren_count > 0:
            set_error(ErrorCode.UNBALANCED_PARENTHESES,
                     "Unmatched parentheses in EXISTS function")
            return None
            
        pattern_str = " ".join(pattern_tokens)
        args = [LiteralExpression(pattern_str)]
        
        return FunctionCall(func_name, args)
        
    def _parse_function_call(self, func_name: str) -> Optional[FunctionCall]:
        """Parse function call"""
        if not self._expect_token("("):
            return None
            
        args = []
        
        if self._current_token() != ")":
            arg = self._safe_parse_expression()
            if arg:
                args.append(arg)
                
                while self._current_token() == ",":
                    self._consume_token()
                    next_arg = self._safe_parse_expression()
                    if next_arg:
                        args.append(next_arg)
                    else:
                        set_error(ErrorCode.MISSING_TOKEN,
                                 "Expected argument after ','")
                        return None
                        
        if not self._expect_token(")"):
            return None
            
        return FunctionCall(func_name, args)
        
    def _safe_parse_return_clause(self) -> Optional[ReturnClause]:
        """Parse RETURN clause with error handling"""
        if not self._expect_token("RETURN"):
            return None
            
        distinct = False
        if self._current_token() and self._current_token().upper() == "DISTINCT":
            distinct = True
            self._consume_token()
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "RETURN clause must specify what to return")
            return None
            
        order_by = None
        skip = None
        limit = None
        
        # Parse optional clauses
        while self._current_token():
            token = self._current_token().upper()
            if token == "ORDER":
                order_by = self._safe_parse_order_by()
                if not order_by:
                    break
            elif token == "SKIP":
                skip = self._safe_parse_skip()
                if skip is None:
                    break
            elif token == "LIMIT":
                limit = self._safe_parse_limit()
                if limit is None:
                    break
            else:
                break
                
        return ReturnClause(items, distinct, order_by, skip, limit)
        
    def _safe_parse_return_items(self) -> List[ReturnItem]:
        """Parse return items with error handling"""
        items = []
        
        item = self._safe_parse_return_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_return_item()
                if next_item:
                    items.append(next_item)
                else:
                    set_error(ErrorCode.MISSING_TOKEN,
                             "Expected return item after ','")
                    break
                    
        return items
        
    def _safe_parse_return_item(self) -> Optional[ReturnItem]:
        """Parse return item with error handling"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        alias = None
        if self._current_token() and self._current_token().upper() == "AS":
            self._consume_token()
            alias = self._consume_token()
            if not alias or not self._is_valid_identifier(alias):
                set_error(ErrorCode.INVALID_LABEL,
                         "Expected valid alias after 'AS'")
                return None
                
        return ReturnItem(expression, alias)
        
    def _safe_parse_order_by(self) -> Optional[OrderByClause]:
        """Parse ORDER BY clause"""
        if not self._expect_token("ORDER"):
            return None
        if not self._expect_token("BY"):
            return None
            
        items = []
        item = self._safe_parse_order_by_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_order_by_item()
                if next_item:
                    items.append(next_item)
                else:
                    break
                    
        return OrderByClause(items) if items else None
        
    def _safe_parse_order_by_item(self) -> Optional[OrderByItem]:
        """Parse ORDER BY item"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        ascending = True
        if self._current_token() and self._current_token().upper() in ["ASC", "DESC"]:
            ascending = self._consume_token().upper() == "ASC"
            
        return OrderByItem(expression, ascending)
        
    def _safe_parse_skip(self) -> Optional[int]:
        """Parse SKIP value"""
        self._consume_token()  # consume 'SKIP'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value < 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "SKIP value cannot be negative")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected integer after SKIP, got '{token}'")
            return None
            
    def _safe_parse_limit(self) -> Optional[int]:
        """Parse LIMIT value"""
        self._consume_token()  # consume 'LIMIT'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value <= 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "LIMIT value must be positive")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected positive integer after LIMIT, got '{token}'")
            return None
            
    def _safe_parse_with_clause(self) -> Optional[WithClause]:
        """Parse WITH clause"""
        if not self._expect_token("WITH"):
            return None
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "WITH clause must specify items")
            return None
            
        where_clause = None
        if self._current_token() and self._current_token().upper() == "WHERE":
            where_clause = self._safe_parse_where_clause()
            
        return WithClause(items, where_clause)
        
    def _validate_query_semantics(self, query: Query) -> bool:
        """Validate semantic correctness of parsed query"""
        valid = True
        
        # Check that variables used in WHERE/RETURN are defined in MATCH
        defined_vars = set()
        
        # Collect variables from MATCH clauses
        for match_clause in query.match_clauses:
            for pattern in match_clause.patterns:
                for element in pattern.elements:
                    if isinstance(element, NodePattern) and element.variable:
                        defined_vars.add(element.variable)
                    elif isinstance(element, RelationshipPattern) and element.variable:
                        defined_vars.add(element.variable)
                        
        # Check WHERE clause references
        if query.where_clause:
            where_vars = self._extract_variables_from_expression(query.where_clause.condition)
            for var in where_vars:
                if var not in defined_vars:
                    set_error(ErrorCode.UNDEFINED_VARIABLE,
                             f"Variable '{var}' in WHERE clause is not defined in MATCH")
                    valid = False
                    
        # Check RETURN clause references  
        if query.return_clause:
            for item in query.return_clause.items:
                return_vars = self._extract_variables_from_expression(item.expression)
                for var in return_vars:
                    if var not in defined_vars:
                        set_error(ErrorCode.UNDEFINED_VARIABLE,
                                 f"Variable '{var}' in RETURN clause is not defined in MATCH")
                        valid = False
                        
        return valid
        
    def _extract_variables_from_expression(self, expr: Expression) -> Set[str]:
        """Extract all variable references from an expression"""
        variables = set()
        
        if isinstance(expr, VariableExpression):
            variables.add(expr.name)
        elif isinstance(expr, PropertyExpression):
            variables.add(expr.variable)
        elif isinstance(expr, BinaryExpression):
            variables.update(self._extract_variables_from_expression(expr.left))
            variables.update(self._extract_variables_from_expression(expr.right))
        elif isinstance(expr, FunctionCall):
            for arg in expr.arguments:
                variables.update(self._extract_variables_from_expression(arg))
                
        return variables
        
    # Utility methods
    def _current_token(self) -> Optional[str]:
        """Get current token"""
        return self.tokens[self.position] if self.position < len(self.tokens) else None
        
    def _consume_token(self) -> Optional[str]:
        """Consume and return current token"""
        token = self._current_token()
        if token:
            self.position += 1
            get_error_context().current_position = self.position
        return token
        
    def _peek_token(self, offset: int = 1) -> Optional[str]:
        """Peek at future token"""
        pos = self.position + offset
        return self.tokens[pos] if pos < len(self.tokens) else None
        
    def _expect_token(self, expected: str) -> bool:
        """Expect a specific token"""
        token = self._consume_token()
        if not token or token.upper() != expected.upper():
            set_error(ErrorCode.MISSING_TOKEN,
                     f"Expected '{expected}', got '{token or 'end of input'}'",
                     suggestion=f"Add '{expected}' to your query")
            return False
        return True
        
    def _is_valid_identifier(self, token: str) -> bool:
        """Check if token is a valid identifier"""
        if not token:
            return False
        return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*, token) is not None
        
    def _skip_to_next_clause(self):
        """Skip tokens until next clause for error recovery"""
        clause_keywords = {
            'MATCH', 'WHERE', 'RETURN', 'WITH', 'CREATE', 'MERGE', 
            'DELETE', 'SET', 'REMOVE', 'UNWIND', 'FOREACH', 'CALL'
        }
        
        while self._current_token():
            if self._current_token().upper() in clause_keywords:
                break
            self._consume_token()


# Create wrapper to maintain compatibility with existing code
class CypherParser(EnhancedCypherParser):
    """Backward-compatible wrapper for enhanced parser"""
    
    def parse(self, query: str) -> Query:
        """Parse query and raise exception on error for backward compatibility"""
        result = super().parse(query)
        if result is None:
            error_msg = get_error_context().format_errors()
            raise ValueError(f"Parse error:\n{error_msg}")
        return result, token) or
            token.upper() in ("TRUE", "FALSE", "NULL")):
            literal_value = self._safe_parse_literal()
            return LiteralExpression(literal_value) if literal_value is not None else None
            
        # Variable
        if self._is_valid_identifier(token):
            variable = self._consume_token()
            if variable not in self.current_scope_variables:
                set_warning(ErrorCode.UNDEFINED_VARIABLE,
                           f"Variable '{variable}' may not be defined")
            return VariableExpression(variable)
            
        set_error(ErrorCode.UNEXPECTED_TOKEN,
                 f"Unexpected token in expression: '{token}'")
        return None
        
    def _parse_exists_function(self) -> Optional[FunctionCall]:
        """Parse EXISTS function with pattern"""
        func_name = self._consume_token()  # consume 'EXISTS'
        
        if not self._expect_token("("):
            return None
            
        # Parse the pattern inside EXISTS
        pattern_tokens = []
        paren_count = 1
        
        while self._current_token() and paren_count > 0:
            token = self._consume_token()
            if token == "(":
                paren_count += 1
            elif token == ")":
                paren_count -= 1
                
            if paren_count > 0:
                pattern_tokens.append(token)
                
        if paren_count > 0:
            set_error(ErrorCode.UNBALANCED_PARENTHESES,
                     "Unmatched parentheses in EXISTS function")
            return None
            
        pattern_str = " ".join(pattern_tokens)
        args = [LiteralExpression(pattern_str)]
        
        return FunctionCall(func_name, args)
        
    def _parse_function_call(self, func_name: str) -> Optional[FunctionCall]:
        """Parse function call"""
        if not self._expect_token("("):
            return None
            
        args = []
        
        if self._current_token() != ")":
            arg = self._safe_parse_expression()
            if arg:
                args.append(arg)
                
                while self._current_token() == ",":
                    self._consume_token()
                    next_arg = self._safe_parse_expression()
                    if next_arg:
                        args.append(next_arg)
                    else:
                        set_error(ErrorCode.MISSING_TOKEN,
                                 "Expected argument after ','")
                        return None
                        
        if not self._expect_token(")"):
            return None
            
        return FunctionCall(func_name, args)
        
    def _safe_parse_return_clause(self) -> Optional[ReturnClause]:
        """Parse RETURN clause with error handling"""
        if not self._expect_token("RETURN"):
            return None
            
        distinct = False
        if self._current_token() and self._current_token().upper() == "DISTINCT":
            distinct = True
            self._consume_token()
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "RETURN clause must specify what to return")
            return None
            
        order_by = None
        skip = None
        limit = None
        
        # Parse optional clauses
        while self._current_token():
            token = self._current_token().upper()
            if token == "ORDER":
                order_by = self._safe_parse_order_by()
                if not order_by:
                    break
            elif token == "SKIP":
                skip = self._safe_parse_skip()
                if skip is None:
                    break
            elif token == "LIMIT":
                limit = self._safe_parse_limit()
                if limit is None:
                    break
            else:
                break
                
        return ReturnClause(items, distinct, order_by, skip, limit)
        
    def _safe_parse_return_items(self) -> List[ReturnItem]:
        """Parse return items with error handling"""
        items = []
        
        item = self._safe_parse_return_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_return_item()
                if next_item:
                    items.append(next_item)
                else:
                    set_error(ErrorCode.MISSING_TOKEN,
                             "Expected return item after ','")
                    break
                    
        return items
        
    def _safe_parse_return_item(self) -> Optional[ReturnItem]:
        """Parse return item with error handling"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        alias = None
        if self._current_token() and self._current_token().upper() == "AS":
            self._consume_token()
            alias = self._consume_token()
            if not alias or not self._is_valid_identifier(alias):
                set_error(ErrorCode.INVALID_LABEL,
                         "Expected valid alias after 'AS'")
                return None
                
        return ReturnItem(expression, alias)
        
    def _safe_parse_order_by(self) -> Optional[OrderByClause]:
        """Parse ORDER BY clause"""
        if not self._expect_token("ORDER"):
            return None
        if not self._expect_token("BY"):
            return None
            
        items = []
        item = self._safe_parse_order_by_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_order_by_item()
                if next_item:
                    items.append(next_item)
                else:
                    break
                    
        return OrderByClause(items) if items else None
        
    def _safe_parse_order_by_item(self) -> Optional[OrderByItem]:
        """Parse ORDER BY item"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        ascending = True
        if self._current_token() and self._current_token().upper() in ["ASC", "DESC"]:
            ascending = self._consume_token().upper() == "ASC"
            
        return OrderByItem(expression, ascending)
        
    def _safe_parse_skip(self) -> Optional[int]:
        """Parse SKIP value"""
        self._consume_token()  # consume 'SKIP'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value < 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "SKIP value cannot be negative")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected integer after SKIP, got '{token}'")
            return None
            
    def _safe_parse_limit(self) -> Optional[int]:
        """Parse LIMIT value"""
        self._consume_token()  # consume 'LIMIT'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value <= 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "LIMIT value must be positive")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected positive integer after LIMIT, got '{token}'")
            return None
            
    def _safe_parse_with_clause(self) -> Optional[WithClause]:
        """Parse WITH clause"""
        if not self._expect_token("WITH"):
            return None
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "WITH clause must specify items")
            return None
            
        where_clause = None
        if self._current_token() and self._current_token().upper() == "WHERE":
            where_clause = self._safe_parse_where_clause()
            
        return WithClause(items, where_clause)
        
    def _validate_query_semantics(self, query: Query) -> bool:
        """Validate semantic correctness of parsed query"""
        valid = True
        
        # Check that variables used in WHERE/RETURN are defined in MATCH
        defined_vars = set()
        
        # Collect variables from MATCH clauses
        for match_clause in query.match_clauses:
            for pattern in match_clause.patterns:
                for element in pattern.elements:
                    if isinstance(element, NodePattern) and element.variable:
                        defined_vars.add(element.variable)
                    elif isinstance(element, RelationshipPattern) and element.variable:
                        defined_vars.add(element.variable)
                        
        # Check WHERE clause references
        if query.where_clause:
            where_vars = self._extract_variables_from_expression(query.where_clause.condition)
            for var in where_vars:
                if var not in defined_vars:
                    set_error(ErrorCode.UNDEFINED_VARIABLE,
                             f"Variable '{var}' in WHERE clause is not defined in MATCH")
                    valid = False
                    
        # Check RETURN clause references  
        if query.return_clause:
            for item in query.return_clause.items:
                return_vars = self._extract_variables_from_expression(item.expression)
                for var in return_vars:
                    if var not in defined_vars:
                        set_error(ErrorCode.UNDEFINED_VARIABLE,
                                 f"Variable '{var}' in RETURN clause is not defined in MATCH")
                        valid = False
                        
        return valid
        
    def _extract_variables_from_expression(self, expr: Expression) -> Set[str]:
        """Extract all variable references from an expression"""
        variables = set()
        
        if isinstance(expr, VariableExpression):
            variables.add(expr.name)
        elif isinstance(expr, PropertyExpression):
            variables.add(expr.variable)
        elif isinstance(expr, BinaryExpression):
            variables.update(self._extract_variables_from_expression(expr.left))
            variables.update(self._extract_variables_from_expression(expr.right))
        elif isinstance(expr, FunctionCall):
            for arg in expr.arguments:
                variables.update(self._extract_variables_from_expression(arg))
                
        return variables
        
    # Utility methods
    def _current_token(self) -> Optional[str]:
        """Get current token"""
        return self.tokens[self.position] if self.position < len(self.tokens) else None
        
    def _consume_token(self) -> Optional[str]:
        """Consume and return current token"""
        token = self._current_token()
        if token:
            self.position += 1
            get_error_context().current_position = self.position
        return token
        
    def _peek_token(self, offset: int = 1) -> Optional[str]:
        """Peek at future token"""
        pos = self.position + offset
        return self.tokens[pos] if pos < len(self.tokens) else None
        
    def _expect_token(self, expected: str) -> bool:
        """Expect a specific token"""
        token = self._consume_token()
        if not token or token.upper() != expected.upper():
            set_error(ErrorCode.MISSING_TOKEN,
                     f"Expected '{expected}', got '{token or 'end of input'}'",
                     suggestion=f"Add '{expected}' to your query")
            return False
        return True
        
    def _is_valid_identifier(self, token: str) -> bool:
        """Check if token is a valid identifier"""
        if not token:
            return False
        return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*, token) is not None
        
    def _skip_to_next_clause(self):
        """Skip tokens until next clause for error recovery"""
        clause_keywords = {
            'MATCH', 'WHERE', 'RETURN', 'WITH', 'CREATE', 'MERGE', 
            'DELETE', 'SET', 'REMOVE', 'UNWIND', 'FOREACH', 'CALL'
        }
        
        while self._current_token():
            if self._current_token().upper() in clause_keywords:
                break
            self._consume_token()


# Create wrapper to maintain compatibility with existing code
class CypherParser(EnhancedCypherParser):
    """Backward-compatible wrapper for enhanced parser"""
    
    def parse(self, query: str) -> Query:
        """Parse query and raise exception on error for backward compatibility"""
        result = super().parse(query)
        if result is None:
            error_msg = get_error_context().format_errors()
            raise ValueError(f"Parse error:\n{error_msg}")
        return result, token):
            value = self._consume_token()
            try:
                return float(value) if '.' in value or 'e' in value.lower() else int(value)
            except ValueError:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         f"Invalid number: '{value}'")
                return None
                
        # Boolean and null
        if token.upper() in ("TRUE", "FALSE", "NULL"):
            value = self._consume_token().upper()
            if value == "TRUE":
                return True
            elif value == "FALSE":
                return False
            else:  # NULL
                return None
                
        # Default to string
        return self._consume_token()
        
    def _safe_parse_where_clause(self) -> Optional[WhereClause]:
        """Parse WHERE clause with error handling"""
        if not self._expect_token("WHERE"):
            return None
            
        condition = self._safe_parse_expression()
        return WhereClause(condition) if condition else None
        
    def _safe_parse_expression(self) -> Optional[Expression]:
        """Parse expression with error handling"""
        try:
            return self._parse_or_expression()
        except Exception as e:
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Error parsing expression: {str(e)}")
            return None
            
    def _parse_or_expression(self) -> Optional[Expression]:
        """Parse OR expression"""
        left = self._parse_and_expression()
        if not left:
            return None
            
        while self._current_token() and self._current_token().upper() == "OR":
            op = self._consume_token()
            right = self._parse_and_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected expression after 'OR'")
                return None
            left = BinaryExpression(left, op, right)
            
        return left
        
    def _parse_and_expression(self) -> Optional[Expression]:
        """Parse AND expression"""
        left = self._parse_comparison_expression()
        if not left:
            return None
            
        while self._current_token() and self._current_token().upper() == "AND":
            op = self._consume_token()
            right = self._parse_comparison_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected expression after 'AND'")
                return None
            left = BinaryExpression(left, op, right)
            
        return left
        
    def _parse_comparison_expression(self) -> Optional[Expression]:
        """Parse comparison expression"""
        left = self._parse_primary_expression()
        if not left:
            return None
            
        if self._current_token() in ["=", "<>", "!=", "<", ">", "<=", ">=", "=~", "CONTAINS"]:
            op = self._consume_token()
            right = self._parse_primary_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         f"Expected expression after '{op}'")
                return None
            return BinaryExpression(left, op, right)
            
        return left
        
    def _parse_primary_expression(self) -> Optional[Expression]:
        """Parse primary expression"""
        token = self._current_token()
        if not token:
            return None
            
        # Handle EXISTS function specially
        if token.upper() == "EXISTS":
            return self._parse_exists_function()
            
        # Property access (variable.property)
        if self._peek_token() == ".":
            variable = self._consume_token()
            if not self._is_valid_identifier(variable):
                set_error(ErrorCode.INVALID_PROPERTY,
                         f"Invalid variable name: '{variable}'")
                return None
                
            # Check if variable is in scope
            if variable not in self.current_scope_variables:
                set_warning(ErrorCode.UNDEFINED_VARIABLE,
                           f"Variable '{variable}' may not be defined")
                           
            self._consume_token()  # consume '.'
            property_name = self._consume_token()
            if not property_name or not self._is_valid_identifier(property_name):
                set_error(ErrorCode.INVALID_PROPERTY,
                         "Expected property name after '.'")
                return None
                
            return PropertyExpression(variable, property_name)
            
        # Function call
        if self._peek_token() == "(":
            func_name = self._consume_token()
            if not self._is_valid_identifier(func_name):
                set_error(ErrorCode.UNKNOWN_FUNCTION,
                         f"Invalid function name: '{func_name}'")
                return None
                
            return self._parse_function_call(func_name)
            
        # Literal
        if (token.startswith(("'", '"')) or 
            re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?, token) or
            token.upper() in ("TRUE", "FALSE", "NULL")):
            literal_value = self._safe_parse_literal()
            return LiteralExpression(literal_value) if literal_value is not None else None
            
        # Variable
        if self._is_valid_identifier(token):
            variable = self._consume_token()
            if variable not in self.current_scope_variables:
                set_warning(ErrorCode.UNDEFINED_VARIABLE,
                           f"Variable '{variable}' may not be defined")
            return VariableExpression(variable)
            
        set_error(ErrorCode.UNEXPECTED_TOKEN,
                 f"Unexpected token in expression: '{token}'")
        return None
        
    def _parse_exists_function(self) -> Optional[FunctionCall]:
        """Parse EXISTS function with pattern"""
        func_name = self._consume_token()  # consume 'EXISTS'
        
        if not self._expect_token("("):
            return None
            
        # Parse the pattern inside EXISTS
        pattern_tokens = []
        paren_count = 1
        
        while self._current_token() and paren_count > 0:
            token = self._consume_token()
            if token == "(":
                paren_count += 1
            elif token == ")":
                paren_count -= 1
                
            if paren_count > 0:
                pattern_tokens.append(token)
                
        if paren_count > 0:
            set_error(ErrorCode.UNBALANCED_PARENTHESES,
                     "Unmatched parentheses in EXISTS function")
            return None
            
        pattern_str = " ".join(pattern_tokens)
        args = [LiteralExpression(pattern_str)]
        
        return FunctionCall(func_name, args)
        
    def _parse_function_call(self, func_name: str) -> Optional[FunctionCall]:
        """Parse function call"""
        if not self._expect_token("("):
            return None
            
        args = []
        
        if self._current_token() != ")":
            arg = self._safe_parse_expression()
            if arg:
                args.append(arg)
                
                while self._current_token() == ",":
                    self._consume_token()
                    next_arg = self._safe_parse_expression()
                    if next_arg:
                        args.append(next_arg)
                    else:
                        set_error(ErrorCode.MISSING_TOKEN,
                                 "Expected argument after ','")
                        return None
                        
        if not self._expect_token(")"):
            return None
            
        return FunctionCall(func_name, args)
        
    def _safe_parse_return_clause(self) -> Optional[ReturnClause]:
        """Parse RETURN clause with error handling"""
        if not self._expect_token("RETURN"):
            return None
            
        distinct = False
        if self._current_token() and self._current_token().upper() == "DISTINCT":
            distinct = True
            self._consume_token()
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "RETURN clause must specify what to return")
            return None
            
        order_by = None
        skip = None
        limit = None
        
        # Parse optional clauses
        while self._current_token():
            token = self._current_token().upper()
            if token == "ORDER":
                order_by = self._safe_parse_order_by()
                if not order_by:
                    break
            elif token == "SKIP":
                skip = self._safe_parse_skip()
                if skip is None:
                    break
            elif token == "LIMIT":
                limit = self._safe_parse_limit()
                if limit is None:
                    break
            else:
                break
                
        return ReturnClause(items, distinct, order_by, skip, limit)
        
    def _safe_parse_return_items(self) -> List[ReturnItem]:
        """Parse return items with error handling"""
        items = []
        
        item = self._safe_parse_return_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_return_item()
                if next_item:
                    items.append(next_item)
                else:
                    set_error(ErrorCode.MISSING_TOKEN,
                             "Expected return item after ','")
                    break
                    
        return items
        
    def _safe_parse_return_item(self) -> Optional[ReturnItem]:
        """Parse return item with error handling"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        alias = None
        if self._current_token() and self._current_token().upper() == "AS":
            self._consume_token()
            alias = self._consume_token()
            if not alias or not self._is_valid_identifier(alias):
                set_error(ErrorCode.INVALID_LABEL,
                         "Expected valid alias after 'AS'")
                return None
                
        return ReturnItem(expression, alias)
        
    def _safe_parse_order_by(self) -> Optional[OrderByClause]:
        """Parse ORDER BY clause"""
        if not self._expect_token("ORDER"):
            return None
        if not self._expect_token("BY"):
            return None
            
        items = []
        item = self._safe_parse_order_by_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_order_by_item()
                if next_item:
                    items.append(next_item)
                else:
                    break
                    
        return OrderByClause(items) if items else None
        
    def _safe_parse_order_by_item(self) -> Optional[OrderByItem]:
        """Parse ORDER BY item"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        ascending = True
        if self._current_token() and self._current_token().upper() in ["ASC", "DESC"]:
            ascending = self._consume_token().upper() == "ASC"
            
        return OrderByItem(expression, ascending)
        
    def _safe_parse_skip(self) -> Optional[int]:
        """Parse SKIP value"""
        self._consume_token()  # consume 'SKIP'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value < 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "SKIP value cannot be negative")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected integer after SKIP, got '{token}'")
            return None
            
    def _safe_parse_limit(self) -> Optional[int]:
        """Parse LIMIT value"""
        self._consume_token()  # consume 'LIMIT'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value <= 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "LIMIT value must be positive")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected positive integer after LIMIT, got '{token}'")
            return None
            
    def _safe_parse_with_clause(self) -> Optional[WithClause]:
        """Parse WITH clause"""
        if not self._expect_token("WITH"):
            return None
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "WITH clause must specify items")
            return None
            
        where_clause = None
        if self._current_token() and self._current_token().upper() == "WHERE":
            where_clause = self._safe_parse_where_clause()
            
        return WithClause(items, where_clause)
        
    def _validate_query_semantics(self, query: Query) -> bool:
        """Validate semantic correctness of parsed query"""
        valid = True
        
        # Check that variables used in WHERE/RETURN are defined in MATCH
        defined_vars = set()
        
        # Collect variables from MATCH clauses
        for match_clause in query.match_clauses:
            for pattern in match_clause.patterns:
                for element in pattern.elements:
                    if isinstance(element, NodePattern) and element.variable:
                        defined_vars.add(element.variable)
                    elif isinstance(element, RelationshipPattern) and element.variable:
                        defined_vars.add(element.variable)
                        
        # Check WHERE clause references
        if query.where_clause:
            where_vars = self._extract_variables_from_expression(query.where_clause.condition)
            for var in where_vars:
                if var not in defined_vars:
                    set_error(ErrorCode.UNDEFINED_VARIABLE,
                             f"Variable '{var}' in WHERE clause is not defined in MATCH")
                    valid = False
                    
        # Check RETURN clause references  
        if query.return_clause:
            for item in query.return_clause.items:
                return_vars = self._extract_variables_from_expression(item.expression)
                for var in return_vars:
                    if var not in defined_vars:
                        set_error(ErrorCode.UNDEFINED_VARIABLE,
                                 f"Variable '{var}' in RETURN clause is not defined in MATCH")
                        valid = False
                        
        return valid
        
    def _extract_variables_from_expression(self, expr: Expression) -> Set[str]:
        """Extract all variable references from an expression"""
        variables = set()
        
        if isinstance(expr, VariableExpression):
            variables.add(expr.name)
        elif isinstance(expr, PropertyExpression):
            variables.add(expr.variable)
        elif isinstance(expr, BinaryExpression):
            variables.update(self._extract_variables_from_expression(expr.left))
            variables.update(self._extract_variables_from_expression(expr.right))
        elif isinstance(expr, FunctionCall):
            for arg in expr.arguments:
                variables.update(self._extract_variables_from_expression(arg))
                
        return variables
        
    # Utility methods
    def _current_token(self) -> Optional[str]:
        """Get current token"""
        return self.tokens[self.position] if self.position < len(self.tokens) else None
        
    def _consume_token(self) -> Optional[str]:
        """Consume and return current token"""
        token = self._current_token()
        if token:
            self.position += 1
            get_error_context().current_position = self.position
        return token
        
    def _peek_token(self, offset: int = 1) -> Optional[str]:
        """Peek at future token"""
        pos = self.position + offset
        return self.tokens[pos] if pos < len(self.tokens) else None
        
    def _expect_token(self, expected: str) -> bool:
        """Expect a specific token"""
        token = self._consume_token()
        if not token or token.upper() != expected.upper():
            set_error(ErrorCode.MISSING_TOKEN,
                     f"Expected '{expected}', got '{token or 'end of input'}'",
                     suggestion=f"Add '{expected}' to your query")
            return False
        return True
        
    def _is_valid_identifier(self, token: str) -> bool:
        """Check if token is a valid identifier"""
        if not token:
            return False
        return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*, token) is not None
        
    def _skip_to_next_clause(self):
        """Skip tokens until next clause for error recovery"""
        clause_keywords = {
            'MATCH', 'WHERE', 'RETURN', 'WITH', 'CREATE', 'MERGE', 
            'DELETE', 'SET', 'REMOVE', 'UNWIND', 'FOREACH', 'CALL'
        }
        
        while self._current_token():
            if self._current_token().upper() in clause_keywords:
                break
            self._consume_token()


# Create wrapper to maintain compatibility with existing code
class CypherParser(EnhancedCypherParser):
    """Backward-compatible wrapper for enhanced parser"""
    
    def parse(self, query: str) -> Query:
        """Parse query and raise exception on error for backward compatibility"""
        result = super().parse(query)
        if result is None:
            error_msg = get_error_context().format_errors()
            raise ValueError(f"Parse error:\n{error_msg}")
        return result, token) is not None
        
    def _skip_to_next_clause(self):
        """Skip tokens until next clause for error recovery"""
        clause_keywords = {
            'MATCH', 'WHERE', 'RETURN', 'WITH', 'CREATE', 'MERGE', 
            'DELETE', 'SET', 'REMOVE', 'UNWIND', 'FOREACH', 'CALL'
        }
        
        while self._current_token():
            if self._current_token().upper() in clause_keywords:
                break
            self._consume_token()


# Create wrapper to maintain compatibility with existing code
class CypherParser(EnhancedCypherParser):
    """Backward-compatible wrapper for enhanced parser"""
    
    def parse(self, query: str) -> Query:
        """Parse query and raise exception on error for backward compatibility"""
        result = super().parse(query)
        if result is None:
            error_msg = get_error_context().format_errors()
            raise ValueError(f"Parse error:\n{error_msg}")
        return result, token):
            value = self._consume_token()
            try:
                return float(value) if '.' in value or 'e' in value.lower() else int(value)
            except ValueError:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         f"Invalid number: '{value}'")
                return None
                
        # Boolean and null
        if token.upper() in ("TRUE", "FALSE", "NULL"):
            value = self._consume_token().upper()
            if value == "TRUE":
                return True
            elif value == "FALSE":
                return False
            else:  # NULL
                return None
                
        # Default to string
        return self._consume_token()
        
    def _safe_parse_where_clause(self) -> Optional[WhereClause]:
        """Parse WHERE clause with error handling"""
        if not self._expect_token("WHERE"):
            return None
            
        condition = self._safe_parse_expression()
        return WhereClause(condition) if condition else None
        
    def _safe_parse_expression(self) -> Optional[Expression]:
        """Parse expression with error handling"""
        try:
            return self._parse_or_expression()
        except Exception as e:
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Error parsing expression: {str(e)}")
            return None
            
    def _parse_or_expression(self) -> Optional[Expression]:
        """Parse OR expression"""
        left = self._parse_and_expression()
        if not left:
            return None
            
        while self._current_token() and self._current_token().upper() == "OR":
            op = self._consume_token()
            right = self._parse_and_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected expression after 'OR'")
                return None
            left = BinaryExpression(left, op, right)
            
        return left
        
    def _parse_and_expression(self) -> Optional[Expression]:
        """Parse AND expression"""
        left = self._parse_comparison_expression()
        if not left:
            return None
            
        while self._current_token() and self._current_token().upper() == "AND":
            op = self._consume_token()
            right = self._parse_comparison_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected expression after 'AND'")
                return None
            left = BinaryExpression(left, op, right)
            
        return left
        
    def _parse_comparison_expression(self) -> Optional[Expression]:
        """Parse comparison expression"""
        left = self._parse_primary_expression()
        if not left:
            return None
            
        if self._current_token() in ["=", "<>", "!=", "<", ">", "<=", ">=", "=~", "CONTAINS"]:
            op = self._consume_token()
            right = self._parse_primary_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         f"Expected expression after '{op}'")
                return None
            return BinaryExpression(left, op, right)
            
        return left
        
    def _parse_primary_expression(self) -> Optional[Expression]:
        """Parse primary expression"""
        token = self._current_token()
        if not token:
            return None
            
        # Handle EXISTS function specially
        if token.upper() == "EXISTS":
            return self._parse_exists_function()
            
        # Property access (variable.property)
        if self._peek_token() == ".":
            variable = self._consume_token()
            if not self._is_valid_identifier(variable):
                set_error(ErrorCode.INVALID_PROPERTY,
                         f"Invalid variable name: '{variable}'")
                return None
                
            # Check if variable is in scope
            if variable not in self.current_scope_variables:
                set_warning(ErrorCode.UNDEFINED_VARIABLE,
                           f"Variable '{variable}' may not be defined")
                           
            self._consume_token()  # consume '.'
            property_name = self._consume_token()
            if not property_name or not self._is_valid_identifier(property_name):
                set_error(ErrorCode.INVALID_PROPERTY,
                         "Expected property name after '.'")
                return None
                
            return PropertyExpression(variable, property_name)
            
        # Function call
        if self._peek_token() == "(":
            func_name = self._consume_token()
            if not self._is_valid_identifier(func_name):
                set_error(ErrorCode.UNKNOWN_FUNCTION,
                         f"Invalid function name: '{func_name}'")
                return None
                
            return self._parse_function_call(func_name)
            
        # Literal
        if (token.startswith(("'", '"')) or 
            re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?, token) or
            token.upper() in ("TRUE", "FALSE", "NULL")):
            literal_value = self._safe_parse_literal()
            return LiteralExpression(literal_value) if literal_value is not None else None
            
        # Variable
        if self._is_valid_identifier(token):
            variable = self._consume_token()
            if variable not in self.current_scope_variables:
                set_warning(ErrorCode.UNDEFINED_VARIABLE,
                           f"Variable '{variable}' may not be defined")
            return VariableExpression(variable)
            
        set_error(ErrorCode.UNEXPECTED_TOKEN,
                 f"Unexpected token in expression: '{token}'")
        return None
        
    def _parse_exists_function(self) -> Optional[FunctionCall]:
        """Parse EXISTS function with pattern"""
        func_name = self._consume_token()  # consume 'EXISTS'
        
        if not self._expect_token("("):
            return None
            
        # Parse the pattern inside EXISTS
        pattern_tokens = []
        paren_count = 1
        
        while self._current_token() and paren_count > 0:
            token = self._consume_token()
            if token == "(":
                paren_count += 1
            elif token == ")":
                paren_count -= 1
                
            if paren_count > 0:
                pattern_tokens.append(token)
                
        if paren_count > 0:
            set_error(ErrorCode.UNBALANCED_PARENTHESES,
                     "Unmatched parentheses in EXISTS function")
            return None
            
        pattern_str = " ".join(pattern_tokens)
        args = [LiteralExpression(pattern_str)]
        
        return FunctionCall(func_name, args)
        
    def _parse_function_call(self, func_name: str) -> Optional[FunctionCall]:
        """Parse function call"""
        if not self._expect_token("("):
            return None
            
        args = []
        
        if self._current_token() != ")":
            arg = self._safe_parse_expression()
            if arg:
                args.append(arg)
                
                while self._current_token() == ",":
                    self._consume_token()
                    next_arg = self._safe_parse_expression()
                    if next_arg:
                        args.append(next_arg)
                    else:
                        set_error(ErrorCode.MISSING_TOKEN,
                                 "Expected argument after ','")
                        return None
                        
        if not self._expect_token(")"):
            return None
            
        return FunctionCall(func_name, args)
        
    def _safe_parse_return_clause(self) -> Optional[ReturnClause]:
        """Parse RETURN clause with error handling"""
        if not self._expect_token("RETURN"):
            return None
            
        distinct = False
        if self._current_token() and self._current_token().upper() == "DISTINCT":
            distinct = True
            self._consume_token()
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "RETURN clause must specify what to return")
            return None
            
        order_by = None
        skip = None
        limit = None
        
        # Parse optional clauses
        while self._current_token():
            token = self._current_token().upper()
            if token == "ORDER":
                order_by = self._safe_parse_order_by()
                if not order_by:
                    break
            elif token == "SKIP":
                skip = self._safe_parse_skip()
                if skip is None:
                    break
            elif token == "LIMIT":
                limit = self._safe_parse_limit()
                if limit is None:
                    break
            else:
                break
                
        return ReturnClause(items, distinct, order_by, skip, limit)
        
    def _safe_parse_return_items(self) -> List[ReturnItem]:
        """Parse return items with error handling"""
        items = []
        
        item = self._safe_parse_return_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_return_item()
                if next_item:
                    items.append(next_item)
                else:
                    set_error(ErrorCode.MISSING_TOKEN,
                             "Expected return item after ','")
                    break
                    
        return items
        
    def _safe_parse_return_item(self) -> Optional[ReturnItem]:
        """Parse return item with error handling"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        alias = None
        if self._current_token() and self._current_token().upper() == "AS":
            self._consume_token()
            alias = self._consume_token()
            if not alias or not self._is_valid_identifier(alias):
                set_error(ErrorCode.INVALID_LABEL,
                         "Expected valid alias after 'AS'")
                return None
                
        return ReturnItem(expression, alias)
        
    def _safe_parse_order_by(self) -> Optional[OrderByClause]:
        """Parse ORDER BY clause"""
        if not self._expect_token("ORDER"):
            return None
        if not self._expect_token("BY"):
            return None
            
        items = []
        item = self._safe_parse_order_by_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_order_by_item()
                if next_item:
                    items.append(next_item)
                else:
                    break
                    
        return OrderByClause(items) if items else None
        
    def _safe_parse_order_by_item(self) -> Optional[OrderByItem]:
        """Parse ORDER BY item"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        ascending = True
        if self._current_token() and self._current_token().upper() in ["ASC", "DESC"]:
            ascending = self._consume_token().upper() == "ASC"
            
        return OrderByItem(expression, ascending)
        
    def _safe_parse_skip(self) -> Optional[int]:
        """Parse SKIP value"""
        self._consume_token()  # consume 'SKIP'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value < 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "SKIP value cannot be negative")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected integer after SKIP, got '{token}'")
            return None
            
    def _safe_parse_limit(self) -> Optional[int]:
        """Parse LIMIT value"""
        self._consume_token()  # consume 'LIMIT'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value <= 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "LIMIT value must be positive")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected positive integer after LIMIT, got '{token}'")
            return None
            
    def _safe_parse_with_clause(self) -> Optional[WithClause]:
        """Parse WITH clause"""
        if not self._expect_token("WITH"):
            return None
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "WITH clause must specify items")
            return None
            
        where_clause = None
        if self._current_token() and self._current_token().upper() == "WHERE":
            where_clause = self._safe_parse_where_clause()
            
        return WithClause(items, where_clause)
        
    def _validate_query_semantics(self, query: Query) -> bool:
        """Validate semantic correctness of parsed query"""
        valid = True
        
        # Check that variables used in WHERE/RETURN are defined in MATCH
        defined_vars = set()
        
        # Collect variables from MATCH clauses
        for match_clause in query.match_clauses:
            for pattern in match_clause.patterns:
                for element in pattern.elements:
                    if isinstance(element, NodePattern) and element.variable:
                        defined_vars.add(element.variable)
                    elif isinstance(element, RelationshipPattern) and element.variable:
                        defined_vars.add(element.variable)
                        
        # Check WHERE clause references
        if query.where_clause:
            where_vars = self._extract_variables_from_expression(query.where_clause.condition)
            for var in where_vars:
                if var not in defined_vars:
                    set_error(ErrorCode.UNDEFINED_VARIABLE,
                             f"Variable '{var}' in WHERE clause is not defined in MATCH")
                    valid = False
                    
        # Check RETURN clause references  
        if query.return_clause:
            for item in query.return_clause.items:
                return_vars = self._extract_variables_from_expression(item.expression)
                for var in return_vars:
                    if var not in defined_vars:
                        set_error(ErrorCode.UNDEFINED_VARIABLE,
                                 f"Variable '{var}' in RETURN clause is not defined in MATCH")
                        valid = False
                        
        return valid
        
    def _extract_variables_from_expression(self, expr: Expression) -> Set[str]:
        """Extract all variable references from an expression"""
        variables = set()
        
        if isinstance(expr, VariableExpression):
            variables.add(expr.name)
        elif isinstance(expr, PropertyExpression):
            variables.add(expr.variable)
        elif isinstance(expr, BinaryExpression):
            variables.update(self._extract_variables_from_expression(expr.left))
            variables.update(self._extract_variables_from_expression(expr.right))
        elif isinstance(expr, FunctionCall):
            for arg in expr.arguments:
                variables.update(self._extract_variables_from_expression(arg))
                
        return variables
        
    # Utility methods
    def _current_token(self) -> Optional[str]:
        """Get current token"""
        return self.tokens[self.position] if self.position < len(self.tokens) else None
        
    def _consume_token(self) -> Optional[str]:
        """Consume and return current token"""
        token = self._current_token()
        if token:
            self.position += 1
            get_error_context().current_position = self.position
        return token
        
    def _peek_token(self, offset: int = 1) -> Optional[str]:
        """Peek at future token"""
        pos = self.position + offset
        return self.tokens[pos] if pos < len(self.tokens) else None
        
    def _expect_token(self, expected: str) -> bool:
        """Expect a specific token"""
        token = self._consume_token()
        if not token or token.upper() != expected.upper():
            set_error(ErrorCode.MISSING_TOKEN,
                     f"Expected '{expected}', got '{token or 'end of input'}'",
                     suggestion=f"Add '{expected}' to your query")
            return False
        return True
        
    def _is_valid_identifier(self, token: str) -> bool:
        """Check if token is a valid identifier"""
        if not token:
            return False
        return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*, token) is not None
        
    def _skip_to_next_clause(self):
        """Skip tokens until next clause for error recovery"""
        clause_keywords = {
            'MATCH', 'WHERE', 'RETURN', 'WITH', 'CREATE', 'MERGE', 
            'DELETE', 'SET', 'REMOVE', 'UNWIND', 'FOREACH', 'CALL'
        }
        
        while self._current_token():
            if self._current_token().upper() in clause_keywords:
                break
            self._consume_token()


# Create wrapper to maintain compatibility with existing code
class CypherParser(EnhancedCypherParser):
    """Backward-compatible wrapper for enhanced parser"""
    
    def parse(self, query: str) -> Query:
        """Parse query and raise exception on error for backward compatibility"""
        result = super().parse(query)
        if result is None:
            error_msg = get_error_context().format_errors()
            raise ValueError(f"Parse error:\n{error_msg}")
        return result, token) or
            token.upper() in ("TRUE", "FALSE", "NULL")):
            literal_value = self._safe_parse_literal()
            return LiteralExpression(literal_value) if literal_value is not None else None
            
        # Variable
        if self._is_valid_identifier(token):
            variable = self._consume_token()
            if variable not in self.current_scope_variables:
                set_warning(ErrorCode.UNDEFINED_VARIABLE,
                           f"Variable '{variable}' may not be defined")
            return VariableExpression(variable)
            
        set_error(ErrorCode.UNEXPECTED_TOKEN,
                 f"Unexpected token in expression: '{token}'")
        return None
        
    def _parse_exists_function(self) -> Optional[FunctionCall]:
        """Parse EXISTS function with pattern"""
        func_name = self._consume_token()  # consume 'EXISTS'
        
        if not self._expect_token("("):
            return None
            
        # Parse the pattern inside EXISTS
        pattern_tokens = []
        paren_count = 1
        
        while self._current_token() and paren_count > 0:
            token = self._consume_token()
            if token == "(":
                paren_count += 1
            elif token == ")":
                paren_count -= 1
                
            if paren_count > 0:
                pattern_tokens.append(token)
                
        if paren_count > 0:
            set_error(ErrorCode.UNBALANCED_PARENTHESES,
                     "Unmatched parentheses in EXISTS function")
            return None
            
        pattern_str = " ".join(pattern_tokens)
        args = [LiteralExpression(pattern_str)]
        
        return FunctionCall(func_name, args)
        
    def _parse_function_call(self, func_name: str) -> Optional[FunctionCall]:
        """Parse function call"""
        if not self._expect_token("("):
            return None
            
        args = []
        
        if self._current_token() != ")":
            arg = self._safe_parse_expression()
            if arg:
                args.append(arg)
                
                while self._current_token() == ",":
                    self._consume_token()
                    next_arg = self._safe_parse_expression()
                    if next_arg:
                        args.append(next_arg)
                    else:
                        set_error(ErrorCode.MISSING_TOKEN,
                                 "Expected argument after ','")
                        return None
                        
        if not self._expect_token(")"):
            return None
            
        return FunctionCall(func_name, args)
        
    def _safe_parse_return_clause(self) -> Optional[ReturnClause]:
        """Parse RETURN clause with error handling"""
        if not self._expect_token("RETURN"):
            return None
            
        distinct = False
        if self._current_token() and self._current_token().upper() == "DISTINCT":
            distinct = True
            self._consume_token()
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "RETURN clause must specify what to return")
            return None
            
        order_by = None
        skip = None
        limit = None
        
        # Parse optional clauses
        while self._current_token():
            token = self._current_token().upper()
            if token == "ORDER":
                order_by = self._safe_parse_order_by()
                if not order_by:
                    break
            elif token == "SKIP":
                skip = self._safe_parse_skip()
                if skip is None:
                    break
            elif token == "LIMIT":
                limit = self._safe_parse_limit()
                if limit is None:
                    break
            else:
                break
                
        return ReturnClause(items, distinct, order_by, skip, limit)
        
    def _safe_parse_return_items(self) -> List[ReturnItem]:
        """Parse return items with error handling"""
        items = []
        
        item = self._safe_parse_return_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_return_item()
                if next_item:
                    items.append(next_item)
                else:
                    set_error(ErrorCode.MISSING_TOKEN,
                             "Expected return item after ','")
                    break
                    
        return items
        
    def _safe_parse_return_item(self) -> Optional[ReturnItem]:
        """Parse return item with error handling"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        alias = None
        if self._current_token() and self._current_token().upper() == "AS":
            self._consume_token()
            alias = self._consume_token()
            if not alias or not self._is_valid_identifier(alias):
                set_error(ErrorCode.INVALID_LABEL,
                         "Expected valid alias after 'AS'")
                return None
                
        return ReturnItem(expression, alias)
        
    def _safe_parse_order_by(self) -> Optional[OrderByClause]:
        """Parse ORDER BY clause"""
        if not self._expect_token("ORDER"):
            return None
        if not self._expect_token("BY"):
            return None
            
        items = []
        item = self._safe_parse_order_by_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_order_by_item()
                if next_item:
                    items.append(next_item)
                else:
                    break
                    
        return OrderByClause(items) if items else None
        
    def _safe_parse_order_by_item(self) -> Optional[OrderByItem]:
        """Parse ORDER BY item"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        ascending = True
        if self._current_token() and self._current_token().upper() in ["ASC", "DESC"]:
            ascending = self._consume_token().upper() == "ASC"
            
        return OrderByItem(expression, ascending)
        
    def _safe_parse_skip(self) -> Optional[int]:
        """Parse SKIP value"""
        self._consume_token()  # consume 'SKIP'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value < 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "SKIP value cannot be negative")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected integer after SKIP, got '{token}'")
            return None
            
    def _safe_parse_limit(self) -> Optional[int]:
        """Parse LIMIT value"""
        self._consume_token()  # consume 'LIMIT'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value <= 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "LIMIT value must be positive")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected positive integer after LIMIT, got '{token}'")
            return None
            
    def _safe_parse_with_clause(self) -> Optional[WithClause]:
        """Parse WITH clause"""
        if not self._expect_token("WITH"):
            return None
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "WITH clause must specify items")
            return None
            
        where_clause = None
        if self._current_token() and self._current_token().upper() == "WHERE":
            where_clause = self._safe_parse_where_clause()
            
        return WithClause(items, where_clause)
        
    def _validate_query_semantics(self, query: Query) -> bool:
        """Validate semantic correctness of parsed query"""
        valid = True
        
        # Check that variables used in WHERE/RETURN are defined in MATCH
        defined_vars = set()
        
        # Collect variables from MATCH clauses
        for match_clause in query.match_clauses:
            for pattern in match_clause.patterns:
                for element in pattern.elements:
                    if isinstance(element, NodePattern) and element.variable:
                        defined_vars.add(element.variable)
                    elif isinstance(element, RelationshipPattern) and element.variable:
                        defined_vars.add(element.variable)
                        
        # Check WHERE clause references
        if query.where_clause:
            where_vars = self._extract_variables_from_expression(query.where_clause.condition)
            for var in where_vars:
                if var not in defined_vars:
                    set_error(ErrorCode.UNDEFINED_VARIABLE,
                             f"Variable '{var}' in WHERE clause is not defined in MATCH")
                    valid = False
                    
        # Check RETURN clause references  
        if query.return_clause:
            for item in query.return_clause.items:
                return_vars = self._extract_variables_from_expression(item.expression)
                for var in return_vars:
                    if var not in defined_vars:
                        set_error(ErrorCode.UNDEFINED_VARIABLE,
                                 f"Variable '{var}' in RETURN clause is not defined in MATCH")
                        valid = False
                        
        return valid
        
    def _extract_variables_from_expression(self, expr: Expression) -> Set[str]:
        """Extract all variable references from an expression"""
        variables = set()
        
        if isinstance(expr, VariableExpression):
            variables.add(expr.name)
        elif isinstance(expr, PropertyExpression):
            variables.add(expr.variable)
        elif isinstance(expr, BinaryExpression):
            variables.update(self._extract_variables_from_expression(expr.left))
            variables.update(self._extract_variables_from_expression(expr.right))
        elif isinstance(expr, FunctionCall):
            for arg in expr.arguments:
                variables.update(self._extract_variables_from_expression(arg))
                
        return variables
        
    # Utility methods
    def _current_token(self) -> Optional[str]:
        """Get current token"""
        return self.tokens[self.position] if self.position < len(self.tokens) else None
        
    def _consume_token(self) -> Optional[str]:
        """Consume and return current token"""
        token = self._current_token()
        if token:
            self.position += 1
            get_error_context().current_position = self.position
        return token
        
    def _peek_token(self, offset: int = 1) -> Optional[str]:
        """Peek at future token"""
        pos = self.position + offset
        return self.tokens[pos] if pos < len(self.tokens) else None
        
    def _expect_token(self, expected: str) -> bool:
        """Expect a specific token"""
        token = self._consume_token()
        if not token or token.upper() != expected.upper():
            set_error(ErrorCode.MISSING_TOKEN,
                     f"Expected '{expected}', got '{token or 'end of input'}'",
                     suggestion=f"Add '{expected}' to your query")
            return False
        return True
        
    def _is_valid_identifier(self, token: str) -> bool:
        """Check if token is a valid identifier"""
        if not token:
            return False
        return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*, token) is not None
        
    def _skip_to_next_clause(self):
        """Skip tokens until next clause for error recovery"""
        clause_keywords = {
            'MATCH', 'WHERE', 'RETURN', 'WITH', 'CREATE', 'MERGE', 
            'DELETE', 'SET', 'REMOVE', 'UNWIND', 'FOREACH', 'CALL'
        }
        
        while self._current_token():
            if self._current_token().upper() in clause_keywords:
                break
            self._consume_token()


# Create wrapper to maintain compatibility with existing code
class CypherParser(EnhancedCypherParser):
    """Backward-compatible wrapper for enhanced parser"""
    
    def parse(self, query: str) -> Query:
        """Parse query and raise exception on error for backward compatibility"""
        result = super().parse(query)
        if result is None:
            error_msg = get_error_context().format_errors()
            raise ValueError(f"Parse error:\n{error_msg}")
        return result, token):
            value = self._consume_token()
            try:
                return float(value) if '.' in value or 'e' in value.lower() else int(value)
            except ValueError:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         f"Invalid number: '{value}'")
                return None
                
        # Boolean and null
        if token.upper() in ("TRUE", "FALSE", "NULL"):
            value = self._consume_token().upper()
            if value == "TRUE":
                return True
            elif value == "FALSE":
                return False
            else:  # NULL
                return None
                
        # Default to string
        return self._consume_token()
        
    def _safe_parse_where_clause(self) -> Optional[WhereClause]:
        """Parse WHERE clause with error handling"""
        if not self._expect_token("WHERE"):
            return None
            
        condition = self._safe_parse_expression()
        return WhereClause(condition) if condition else None
        
    def _safe_parse_expression(self) -> Optional[Expression]:
        """Parse expression with error handling"""
        try:
            return self._parse_or_expression()
        except Exception as e:
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Error parsing expression: {str(e)}")
            return None
            
    def _parse_or_expression(self) -> Optional[Expression]:
        """Parse OR expression"""
        left = self._parse_and_expression()
        if not left:
            return None
            
        while self._current_token() and self._current_token().upper() == "OR":
            op = self._consume_token()
            right = self._parse_and_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected expression after 'OR'")
                return None
            left = BinaryExpression(left, op, right)
            
        return left
        
    def _parse_and_expression(self) -> Optional[Expression]:
        """Parse AND expression"""
        left = self._parse_comparison_expression()
        if not left:
            return None
            
        while self._current_token() and self._current_token().upper() == "AND":
            op = self._consume_token()
            right = self._parse_comparison_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         "Expected expression after 'AND'")
                return None
            left = BinaryExpression(left, op, right)
            
        return left
        
    def _parse_comparison_expression(self) -> Optional[Expression]:
        """Parse comparison expression"""
        left = self._parse_primary_expression()
        if not left:
            return None
            
        if self._current_token() in ["=", "<>", "!=", "<", ">", "<=", ">=", "=~", "CONTAINS"]:
            op = self._consume_token()
            right = self._parse_primary_expression()
            if not right:
                set_error(ErrorCode.MISSING_TOKEN,
                         f"Expected expression after '{op}'")
                return None
            return BinaryExpression(left, op, right)
            
        return left
        
    def _parse_primary_expression(self) -> Optional[Expression]:
        """Parse primary expression"""
        token = self._current_token()
        if not token:
            return None
            
        # Handle EXISTS function specially
        if token.upper() == "EXISTS":
            return self._parse_exists_function()
            
        # Property access (variable.property)
        if self._peek_token() == ".":
            variable = self._consume_token()
            if not self._is_valid_identifier(variable):
                set_error(ErrorCode.INVALID_PROPERTY,
                         f"Invalid variable name: '{variable}'")
                return None
                
            # Check if variable is in scope
            if variable not in self.current_scope_variables:
                set_warning(ErrorCode.UNDEFINED_VARIABLE,
                           f"Variable '{variable}' may not be defined")
                           
            self._consume_token()  # consume '.'
            property_name = self._consume_token()
            if not property_name or not self._is_valid_identifier(property_name):
                set_error(ErrorCode.INVALID_PROPERTY,
                         "Expected property name after '.'")
                return None
                
            return PropertyExpression(variable, property_name)
            
        # Function call
        if self._peek_token() == "(":
            func_name = self._consume_token()
            if not self._is_valid_identifier(func_name):
                set_error(ErrorCode.UNKNOWN_FUNCTION,
                         f"Invalid function name: '{func_name}'")
                return None
                
            return self._parse_function_call(func_name)
            
        # Literal
        if (token.startswith(("'", '"')) or 
            re.match(r'^-?\d+(\.\d+)?([eE][+-]?\d+)?, token) or
            token.upper() in ("TRUE", "FALSE", "NULL")):
            literal_value = self._safe_parse_literal()
            return LiteralExpression(literal_value) if literal_value is not None else None
            
        # Variable
        if self._is_valid_identifier(token):
            variable = self._consume_token()
            if variable not in self.current_scope_variables:
                set_warning(ErrorCode.UNDEFINED_VARIABLE,
                           f"Variable '{variable}' may not be defined")
            return VariableExpression(variable)
            
        set_error(ErrorCode.UNEXPECTED_TOKEN,
                 f"Unexpected token in expression: '{token}'")
        return None
        
    def _parse_exists_function(self) -> Optional[FunctionCall]:
        """Parse EXISTS function with pattern"""
        func_name = self._consume_token()  # consume 'EXISTS'
        
        if not self._expect_token("("):
            return None
            
        # Parse the pattern inside EXISTS
        pattern_tokens = []
        paren_count = 1
        
        while self._current_token() and paren_count > 0:
            token = self._consume_token()
            if token == "(":
                paren_count += 1
            elif token == ")":
                paren_count -= 1
                
            if paren_count > 0:
                pattern_tokens.append(token)
                
        if paren_count > 0:
            set_error(ErrorCode.UNBALANCED_PARENTHESES,
                     "Unmatched parentheses in EXISTS function")
            return None
            
        pattern_str = " ".join(pattern_tokens)
        args = [LiteralExpression(pattern_str)]
        
        return FunctionCall(func_name, args)
        
    def _parse_function_call(self, func_name: str) -> Optional[FunctionCall]:
        """Parse function call"""
        if not self._expect_token("("):
            return None
            
        args = []
        
        if self._current_token() != ")":
            arg = self._safe_parse_expression()
            if arg:
                args.append(arg)
                
                while self._current_token() == ",":
                    self._consume_token()
                    next_arg = self._safe_parse_expression()
                    if next_arg:
                        args.append(next_arg)
                    else:
                        set_error(ErrorCode.MISSING_TOKEN,
                                 "Expected argument after ','")
                        return None
                        
        if not self._expect_token(")"):
            return None
            
        return FunctionCall(func_name, args)
        
    def _safe_parse_return_clause(self) -> Optional[ReturnClause]:
        """Parse RETURN clause with error handling"""
        if not self._expect_token("RETURN"):
            return None
            
        distinct = False
        if self._current_token() and self._current_token().upper() == "DISTINCT":
            distinct = True
            self._consume_token()
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "RETURN clause must specify what to return")
            return None
            
        order_by = None
        skip = None
        limit = None
        
        # Parse optional clauses
        while self._current_token():
            token = self._current_token().upper()
            if token == "ORDER":
                order_by = self._safe_parse_order_by()
                if not order_by:
                    break
            elif token == "SKIP":
                skip = self._safe_parse_skip()
                if skip is None:
                    break
            elif token == "LIMIT":
                limit = self._safe_parse_limit()
                if limit is None:
                    break
            else:
                break
                
        return ReturnClause(items, distinct, order_by, skip, limit)
        
    def _safe_parse_return_items(self) -> List[ReturnItem]:
        """Parse return items with error handling"""
        items = []
        
        item = self._safe_parse_return_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_return_item()
                if next_item:
                    items.append(next_item)
                else:
                    set_error(ErrorCode.MISSING_TOKEN,
                             "Expected return item after ','")
                    break
                    
        return items
        
    def _safe_parse_return_item(self) -> Optional[ReturnItem]:
        """Parse return item with error handling"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        alias = None
        if self._current_token() and self._current_token().upper() == "AS":
            self._consume_token()
            alias = self._consume_token()
            if not alias or not self._is_valid_identifier(alias):
                set_error(ErrorCode.INVALID_LABEL,
                         "Expected valid alias after 'AS'")
                return None
                
        return ReturnItem(expression, alias)
        
    def _safe_parse_order_by(self) -> Optional[OrderByClause]:
        """Parse ORDER BY clause"""
        if not self._expect_token("ORDER"):
            return None
        if not self._expect_token("BY"):
            return None
            
        items = []
        item = self._safe_parse_order_by_item()
        if item:
            items.append(item)
            
            while self._current_token() == ",":
                self._consume_token()
                next_item = self._safe_parse_order_by_item()
                if next_item:
                    items.append(next_item)
                else:
                    break
                    
        return OrderByClause(items) if items else None
        
    def _safe_parse_order_by_item(self) -> Optional[OrderByItem]:
        """Parse ORDER BY item"""
        expression = self._safe_parse_expression()
        if not expression:
            return None
            
        ascending = True
        if self._current_token() and self._current_token().upper() in ["ASC", "DESC"]:
            ascending = self._consume_token().upper() == "ASC"
            
        return OrderByItem(expression, ascending)
        
    def _safe_parse_skip(self) -> Optional[int]:
        """Parse SKIP value"""
        self._consume_token()  # consume 'SKIP'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value < 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "SKIP value cannot be negative")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected integer after SKIP, got '{token}'")
            return None
            
    def _safe_parse_limit(self) -> Optional[int]:
        """Parse LIMIT value"""
        self._consume_token()  # consume 'LIMIT'
        token = self._consume_token()
        
        try:
            value = int(token)
            if value <= 0:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                         "LIMIT value must be positive")
                return None
            return value
        except (ValueError, TypeError):
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Expected positive integer after LIMIT, got '{token}'")
            return None
            
    def _safe_parse_with_clause(self) -> Optional[WithClause]:
        """Parse WITH clause"""
        if not self._expect_token("WITH"):
            return None
            
        items = self._safe_parse_return_items()
        if not items:
            set_error(ErrorCode.MISSING_TOKEN,
                     "WITH clause must specify items")
            return None
            
        where_clause = None
        if self._current_token() and self._current_token().upper() == "WHERE":
            where_clause = self._safe_parse_where_clause()
            
        return WithClause(items, where_clause)
        
    def _validate_query_semantics(self, query: Query) -> bool:
        """Validate semantic correctness of parsed query"""
        valid = True
        
        # Check that variables used in WHERE/RETURN are defined in MATCH
        defined_vars = set()
        
        # Collect variables from MATCH clauses
        for match_clause in query.match_clauses:
            for pattern in match_clause.patterns:
                for element in pattern.elements:
                    if isinstance(element, NodePattern) and element.variable:
                        defined_vars.add(element.variable)
                    elif isinstance(element, RelationshipPattern) and element.variable:
                        defined_vars.add(element.variable)
                        
        # Check WHERE clause references
        if query.where_clause:
            where_vars = self._extract_variables_from_expression(query.where_clause.condition)
            for var in where_vars:
                if var not in defined_vars:
                    set_error(ErrorCode.UNDEFINED_VARIABLE,
                             f"Variable '{var}' in WHERE clause is not defined in MATCH")
                    valid = False
                    
        # Check RETURN clause references  
        if query.return_clause:
            for item in query.return_clause.items:
                return_vars = self._extract_variables_from_expression(item.expression)
                for var in return_vars:
                    if var not in defined_vars:
                        set_error(ErrorCode.UNDEFINED_VARIABLE,
                                 f"Variable '{var}' in RETURN clause is not defined in MATCH")
                        valid = False
                        
        return valid
        
    def _extract_variables_from_expression(self, expr: Expression) -> Set[str]:
        """Extract all variable references from an expression"""
        variables = set()
        
        if isinstance(expr, VariableExpression):
            variables.add(expr.name)
        elif isinstance(expr, PropertyExpression):
            variables.add(expr.variable)
        elif isinstance(expr, BinaryExpression):
            variables.update(self._extract_variables_from_expression(expr.left))
            variables.update(self._extract_variables_from_expression(expr.right))
        elif isinstance(expr, FunctionCall):
            for arg in expr.arguments:
                variables.update(self._extract_variables_from_expression(arg))
                
        return variables
        
    # Utility methods
    def _current_token(self) -> Optional[str]:
        """Get current token"""
        return self.tokens[self.position] if self.position < len(self.tokens) else None
        
    def _consume_token(self) -> Optional[str]:
        """Consume and return current token"""
        token = self._current_token()
        if token:
            self.position += 1
            get_error_context().current_position = self.position
        return token
        
    def _peek_token(self, offset: int = 1) -> Optional[str]:
        """Peek at future token"""
        pos = self.position + offset
        return self.tokens[pos] if pos < len(self.tokens) else None
        
    def _expect_token(self, expected: str) -> bool:
        """Expect a specific token"""
        token = self._consume_token()
        if not token or token.upper() != expected.upper():
            set_error(ErrorCode.MISSING_TOKEN,
                     f"Expected '{expected}', got '{token or 'end of input'}'",
                     suggestion=f"Add '{expected}' to your query")
            return False
        return True
        
    def _is_valid_identifier(self, token: str) -> bool:
        """Check if token is a valid identifier"""
        if not token:
            return False
        return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*, token) is not None
        
    def _skip_to_next_clause(self):
        """Skip tokens until next clause for error recovery"""
        clause_keywords = {
            'MATCH', 'WHERE', 'RETURN', 'WITH', 'CREATE', 'MERGE', 
            'DELETE', 'SET', 'REMOVE', 'UNWIND', 'FOREACH', 'CALL'
        }
        
        while self._current_token():
            if self._current_token().upper() in clause_keywords:
                break
            self._consume_token()


# Create wrapper to maintain compatibility with existing code
class CypherParser(EnhancedCypherParser):
    """Backward-compatible wrapper for enhanced parser"""
    
    def parse(self, query: str) -> Query:
        """Parse query and raise exception on error for backward compatibility"""
        result = super().parse(query)
        if result is None:
            error_msg = get_error_context().format_errors()
            raise ValueError(f"Parse error:\n{error_msg}")
        return result