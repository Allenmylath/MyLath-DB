# cypher_planner/parser.py

"""
Production Cypher Parser - Complete Implementation
"""

from typing import List, Optional, Dict, Any, Set
from .tokenizer import CypherTokenizer, Token, TokenType, LexerError, get_token_value
from .ast_nodes import *


class ParseError(Exception):
    """Parser error with comprehensive details"""
    def __init__(self, message: str, token: Optional[Token] = None, 
                 suggestions: List[str] = None, context: str = None):
        self.message = message
        self.token = token
        self.suggestions = suggestions or []
        self.context = context
        
        if token:
            location = f"Line {token.line}, column {token.column}"
            super().__init__(f"{location}: {message}")
        else:
            super().__init__(message)


class CypherParser:
    """Production-ready Cypher parser"""
    
    def __init__(self, enable_recovery: bool = True, max_errors: int = 10):
        self.tokenizer = CypherTokenizer()
        self.enable_recovery = enable_recovery
        self.max_errors = max_errors
        
        # Parser state
        self.tokens: List[Token] = []
        self.position = 0
        self.errors: List[ParseError] = []
        self.panic_mode = False
        
        # Performance tracking
        self.parse_depth = 0
        self.max_depth = 100
        
        # Synchronization points for error recovery
        self.statement_keywords = {
            'MATCH', 'OPTIONAL', 'WHERE', 'RETURN', 'WITH', 'UNWIND',
            'CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE', 'CALL'
        }
    
    def parse(self, query: str) -> Query:
        """Parse Cypher query with comprehensive error handling"""
        # Reset parser state
        self._reset_state()
        
        # Validate input
        if not query or not query.strip():
            raise ParseError("Empty query provided")
        
        if len(query) > 50000:  # 50KB limit
            raise ParseError("Query too large (>50KB)", 
                           suggestions=["Break query into smaller parts"])
        
        try:
            # Tokenize
            self.tokens = self.tokenizer.tokenize(query)
            
            # Parse with depth tracking
            self.parse_depth = 0
            ast = self._parse_query()
            
            # Handle accumulated errors
            if self.errors:
                if len(self.errors) == 1:
                    raise self.errors[0]
                else:
                    error_msgs = [str(e) for e in self.errors]
                    raise ParseError(f"Multiple parse errors:\n" + "\n".join(error_msgs))
            
            return ast
            
        except LexerError:
            raise
        except ParseError:
            raise
        except RecursionError:
            raise ParseError("Query too complex (maximum nesting depth exceeded)",
                           suggestions=["Simplify query structure"])
        except Exception as e:
            raise ParseError(f"Internal parser error: {str(e)}")
    
    def _reset_state(self):
        """Reset parser state for new query"""
        self.tokens = []
        self.position = 0
        self.errors = []
        self.panic_mode = False
        self.parse_depth = 0
    
    def _check_depth(self):
        """Check parsing depth to prevent stack overflow"""
        self.parse_depth += 1
        if self.parse_depth > self.max_depth:
            raise ParseError("Maximum parsing depth exceeded")
    
    # Token navigation methods
    def _current_token(self) -> Optional[Token]:
        if self.position >= len(self.tokens):
            return None
        return self.tokens[self.position]
    
    def _peek_token(self, offset: int = 1) -> Optional[Token]:
        pos = self.position + offset
        if pos >= len(self.tokens):
            return None
        return self.tokens[pos]
    
    def _advance(self) -> Optional[Token]:
        token = self._current_token()
        if token and token.type != TokenType.EOF:
            self.position += 1
        return token
    
    def _at_end(self) -> bool:
        token = self._current_token()
        return token is None or token.type == TokenType.EOF
    
    def _match(self, *token_types: TokenType) -> bool:
        token = self._current_token()
        return token is not None and token.type in token_types
    
    def _match_keyword(self, *keywords: str) -> bool:
        token = self._current_token()
        return (token is not None and 
                token.type == TokenType.KEYWORD and 
                token.value.upper() in {k.upper() for k in keywords})
    
    def _consume(self, token_type: TokenType, message: str = None) -> Optional[Token]:
        token = self._current_token()
        
        if token and token.type == token_type:
            return self._advance()
        
        if message is None:
            message = f"Expected {token_type.name.lower()}"
        
        self._error(message, token)
        return None
    
    def _consume_keyword(self, keyword: str) -> Optional[Token]:
        token = self._current_token()
        
        if token and token.type == TokenType.KEYWORD and token.value.upper() == keyword.upper():
            return self._advance()
        
        self._error(f"Expected keyword '{keyword}'", token, [f"Use '{keyword}' here"])
        return None
    
    # Error handling methods
    def _error(self, message: str, token: Optional[Token] = None, 
               suggestions: List[str] = None):
        if token is None:
            token = self._current_token()
        
        error = ParseError(message, token, suggestions)
        self.errors.append(error)
        
        if len(self.errors) >= self.max_errors:
            raise ParseError(f"Too many errors ({self.max_errors}), stopping parse")
        
        if self.enable_recovery and not self.panic_mode:
            self._enter_panic_mode()
    
    def _enter_panic_mode(self):
        self.panic_mode = True
        self._synchronize()
    
    def _synchronize(self):
        self._advance()
        
        while not self._at_end():
            token = self._current_token()
            
            if (token.type == TokenType.KEYWORD and 
                token.value.upper() in self.statement_keywords):
                break
            
            if token.type in (TokenType.SEMICOLON, TokenType.EOF):
                break
            
            self._advance()
        
        self.panic_mode = False
    
    # Main parsing methods
    def _parse_query(self) -> Query:
        self._check_depth()
        query = Query()
        
        while not self._at_end():
            if self.panic_mode:
                continue
            
            if self._match_keyword('MATCH'):
                clause = self._parse_match_clause()
                if clause:
                    query.match_clauses.append(clause)
            elif self._match_keyword('OPTIONAL'):
                clause = self._parse_optional_match()
                if clause:
                    query.optional_match_clauses.append(clause)
            elif self._match_keyword('WHERE'):
                clause = self._parse_where_clause()
                if clause:
                    query.where_clause = clause
            elif self._match_keyword('RETURN'):
                clause = self._parse_return_clause()
                if clause:
                    query.return_clause = clause
                break
            elif self._match_keyword('WITH'):
                clause = self._parse_with_clause()
                if clause:
                    query.with_clauses.append(clause)
            else:
                token = self._current_token()
                if token:
                    self._error(f"Unexpected token: '{token.value}'", token)
                    if not self.enable_recovery:
                        break
                else:
                    break
        
        self._validate_query_structure(query)
        self.parse_depth -= 1
        return query
    
    def _validate_query_structure(self, query: Query):
        if not query.match_clauses and not query.return_clause:
            self._error("Query must contain at least MATCH or RETURN clause")
        
        if query.where_clause and not query.match_clauses:
            self._error("WHERE clause without MATCH clause",
                       suggestions=["Add MATCH clause before WHERE"])
    
    def _parse_match_clause(self) -> Optional[MatchClause]:
        """Parse MATCH clause"""
        self._check_depth()
        
        if not self._consume_keyword('MATCH'):
            return None
        
        patterns = self._parse_patterns()
        if not patterns:
            self._error("MATCH clause must contain at least one pattern")
            return None
        
        self.parse_depth -= 1
        return MatchClause(patterns)
    
    def _parse_optional_match(self) -> Optional[OptionalMatchClause]:
        """Parse OPTIONAL MATCH clause"""
        self._check_depth()
        
        if not self._consume_keyword('OPTIONAL'):
            return None
        
        if not self._consume_keyword('MATCH'):
            return None
        
        patterns = self._parse_patterns()
        if not patterns:
            self._error("OPTIONAL MATCH clause must contain at least one pattern")
            return None
        
        self.parse_depth -= 1
        return OptionalMatchClause(patterns)
    
    def _parse_patterns(self) -> List[Pattern]:
        """Parse comma-separated patterns"""
        patterns = []
        
        pattern = self._parse_pattern()
        if pattern:
            patterns.append(pattern)
        
        while self._match(TokenType.COMMA):
            self._advance()  # consume comma
            pattern = self._parse_pattern()
            if pattern:
                patterns.append(pattern)
            elif not self.panic_mode:
                self._error("Expected pattern after comma")
                break
        
        return patterns
    
    def _parse_pattern(self) -> Optional[Pattern]:
        """Parse a single pattern"""
        self._check_depth()
        
        elements = []
        
        # Must start with a node
        if not self._match(TokenType.LPAREN):
            self._error("Pattern must start with a node '('", 
                       suggestions=["Start with (variable:Label)"])
            self.parse_depth -= 1
            return None
        
        node = self._parse_node_pattern()
        if node:
            elements.append(node)
        else:
            self.parse_depth -= 1
            return None
        
        # Parse relationship chains
        while self._match(TokenType.DASH, TokenType.ARROW_LEFT, TokenType.ARROW_RIGHT, 
                          TokenType.DOUBLE_DASH, TokenType.LBRACKET):
            
            # Parse relationship
            rel = self._parse_relationship_pattern()
            if rel:
                elements.append(rel)
            else:
                break
            
            # Parse target node
            if self._match(TokenType.LPAREN):
                target_node = self._parse_node_pattern()
                if target_node:
                    elements.append(target_node)
                else:
                    # Create anonymous node for recovery
                    elements.append(NodePattern())
            else:
                # Create anonymous target node
                elements.append(NodePattern())
        
        self.parse_depth -= 1
        return Pattern(elements)
    
    def _parse_node_pattern(self) -> Optional[NodePattern]:
        """Parse node pattern: (variable:Label {prop: value})"""
        if not self._consume(TokenType.LPAREN):
            return None
        
        variable = None
        labels = []
        properties = {}
        
        # Parse variable
        if self._match(TokenType.IDENTIFIER):
            variable = get_token_value(self._advance())
        
        # Parse labels
        while self._match(TokenType.COLON):
            self._advance()  # consume :
            
            if self._match(TokenType.IDENTIFIER):
                label = get_token_value(self._advance())
                labels.append(label)
            else:
                self._error("Expected label name after ':'",
                           suggestions=["Provide a valid label name"])
        
        # Parse properties
        if self._match(TokenType.LBRACE):
            properties = self._parse_properties()
            if properties is None:
                properties = {}  # Recovery
        
        if not self._consume(TokenType.RPAREN):
            return None
        
        return NodePattern(variable, labels, properties)
    
    def _parse_relationship_pattern(self) -> Optional[RelationshipPattern]:
        """Parse relationship pattern"""
        direction = "outgoing"
        rel_variable = None
        rel_types = []
        properties = {}
        min_length = None
        max_length = None
        
        # Parse direction and relationship details
        if self._match(TokenType.ARROW_LEFT):
            direction = "incoming"
            self._advance()
        elif self._match(TokenType.ARROW_RIGHT):
            direction = "outgoing"
            self._advance()
        elif self._match(TokenType.DOUBLE_DASH):
            direction = "bidirectional"
            self._advance()
        elif self._match(TokenType.DASH):
            self._advance()  # consume -
            
            # Check for relationship details in brackets
            if self._match(TokenType.LBRACKET):
                rel_details = self._parse_relationship_brackets()
                if rel_details:
                    rel_variable, rel_types, properties, min_length, max_length = rel_details
            
            # Determine final direction
            if self._match(TokenType.DASH):
                self._advance()
                if self._match(TokenType.GREATER_THAN):
                    self._advance()
                    direction = "outgoing"
                else:
                    direction = "bidirectional"
            elif self._match(TokenType.ARROW_RIGHT):
                self._advance()
                direction = "outgoing"
            else:
                direction = "bidirectional"
        
        elif self._match(TokenType.LBRACKET):
            # Direct bracket notation
            rel_details = self._parse_relationship_brackets()
            if rel_details:
                rel_variable, rel_types, properties, min_length, max_length = rel_details
        
        else:
            self._error("Expected relationship pattern",
                       suggestions=["Use -[]-, -->, <--, etc."])
            return None
        
        return RelationshipPattern(
            variable=rel_variable,
            types=rel_types,
            properties=properties,
            direction=direction,
            min_length=min_length,
            max_length=max_length
        )
    
    def _parse_relationship_brackets(self) -> Optional[tuple]:
        """Parse relationship details within brackets"""
        if not self._consume(TokenType.LBRACKET):
            return None
        
        rel_variable = None
        rel_types = []
        properties = {}
        min_length = None
        max_length = None
        
        # Parse variable
        if self._match(TokenType.IDENTIFIER):
            rel_variable = get_token_value(self._advance())
        
        # Parse types
        while self._match(TokenType.COLON):
            self._advance()  # consume :
            
            if self._match(TokenType.IDENTIFIER):
                rel_type = get_token_value(self._advance())
                rel_types.append(rel_type)
            else:
                self._error("Expected relationship type after ':'")
        
        # Parse variable length
        if self._match(TokenType.VARIABLE_LENGTH):
            var_length_token = self._advance()
            min_length, max_length = self._parse_variable_length_spec(var_length_token.value)
        
        # Parse properties
        if self._match(TokenType.LBRACE):
            properties = self._parse_properties()
            if properties is None:
                properties = {}  # Recovery
        
        if not self._consume(TokenType.RBRACKET):
            return None
        
        return rel_variable, rel_types, properties, min_length, max_length
    
    def _parse_variable_length_spec(self, spec: str) -> tuple:
        """Parse variable length specification like *1..3"""
        # Remove the *
        range_part = spec[1:] if spec.startswith('*') else spec
        
        if not range_part:  # Just *
            return 1, float('inf')
        
        if '..' not in range_part:
            # Single number like *3
            try:
                max_val = int(range_part)
                if max_val < 0:
                    self._error("Variable length cannot be negative")
                    return 1, 1
                return 1, max_val
            except ValueError:
                self._error(f"Invalid variable length: {spec}")
                return 1, 1
        
        # Range like *1..3
        parts = range_part.split('..')
        try:
            min_val = int(parts[0]) if parts[0] else 1
            max_val = int(parts[1]) if parts[1] else float('inf')
            
            if min_val < 0:
                self._error("Variable length minimum cannot be negative")
                return 1, 1
            
            if max_val != float('inf') and min_val > max_val:
                self._error("Variable length minimum cannot exceed maximum")
                return min_val, min_val
            
            return min_val, max_val
        except ValueError:
            self._error(f"Invalid variable length range: {spec}")
            return 1, 1
    
    def _parse_properties(self) -> Optional[Dict[str, Any]]:
        """Parse property map: {key: value, ...}"""
        if not self._consume(TokenType.LBRACE):
            return None
        
        properties = {}
        
        if self._match(TokenType.RBRACE):
            self._advance()
            return properties
        
        while True:
            # Parse property key
            if self._match(TokenType.IDENTIFIER):
                key = get_token_value(self._advance())
            elif self._match(TokenType.STRING):
                key = get_token_value(self._advance())
            elif self._match(TokenType.BACKTICK_IDENTIFIER):
                key = get_token_value(self._advance())
            else:
                self._error("Expected property name",
                           suggestions=["Use valid identifier or string"])
                break
            
            if not self._consume(TokenType.COLON):
                break
            
            # Parse property value
            value = self._parse_literal()
            if value is not None:
                properties[key] = value
            else:
                self._error("Expected property value after ':'")
            
            if self._match(TokenType.COMMA):
                self._advance()
                if self._match(TokenType.RBRACE):
                    # Trailing comma - acceptable
                    break
            elif self._match(TokenType.RBRACE):
                break
            else:
                self._error("Expected ',' or '}' in properties")
                break
        
        self._consume(TokenType.RBRACE)
        return properties
    
    def _parse_literal(self) -> Any:
        """Parse literal value"""
        token = self._current_token()
        if not token:
            return None
        
        if token.type == TokenType.STRING:
            self._advance()
            return get_token_value(token)
        
        elif token.type == TokenType.INTEGER:
            self._advance()
            try:
                return int(token.value)
            except ValueError:
                self._error(f"Invalid integer: {token.value}")
                return 0
        
        elif token.type == TokenType.FLOAT:
            self._advance()
            try:
                return float(token.value)
            except ValueError:
                self._error(f"Invalid float: {token.value}")
                return 0.0
        
        elif token.type == TokenType.BOOLEAN:
            self._advance()
            return token.value.upper() == 'TRUE'
        
        elif token.type == TokenType.NULL:
            self._advance()
            return None
        
        elif token.type == TokenType.PARAMETER:
            self._advance()
            return {"type": "parameter", "name": get_token_value(token)}
        
        else:
            self._error(f"Expected literal value, got {token.type.name}")
            return None
    
    def _parse_where_clause(self) -> Optional[WhereClause]:
        """Parse WHERE clause"""
        if not self._consume_keyword('WHERE'):
            return None
        
        condition = self._parse_expression()
        if condition is None:
            self._error("WHERE clause must contain a condition")
            return None
        
        return WhereClause(condition)
    
    def _parse_expression(self) -> Optional[Expression]:
        """Parse expression with proper precedence"""
        self._check_depth()
        result = self._parse_or_expression()
        self.parse_depth -= 1
        return result
    
    def _parse_or_expression(self) -> Optional[Expression]:
        """Parse OR expression (lowest precedence)"""
        left = self._parse_and_expression()
        if left is None:
            return None
        
        while self._match_keyword('OR'):
            op_token = self._advance()
            right = self._parse_and_expression()
            if right is None:
                self._error("Expected expression after 'OR'")
                return left  # Recovery
            left = BinaryExpression(left, op_token.value, right)
        
        return left
    
    def _parse_and_expression(self) -> Optional[Expression]:
        """Parse AND expression"""
        left = self._parse_comparison_expression()
        if left is None:
            return None
        
        while self._match_keyword('AND'):
            op_token = self._advance()
            right = self._parse_comparison_expression()
            if right is None:
                self._error("Expected expression after 'AND'")
                return left  # Recovery
            left = BinaryExpression(left, op_token.value, right)
        
        return left
    
    def _parse_comparison_expression(self) -> Optional[Expression]:
        """Parse comparison expression"""
        left = self._parse_primary_expression()
        if left is None:
            return None
        
        if self._match(TokenType.EQUALS, TokenType.NOT_EQUALS, 
                      TokenType.LESS_THAN, TokenType.LESS_EQUAL,
                      TokenType.GREATER_THAN, TokenType.GREATER_EQUAL,
                      TokenType.REGEX_MATCH):
            op_token = self._advance()
            right = self._parse_primary_expression()
            if right is None:
                self._error(f"Expected expression after '{op_token.value}'")
                return left  # Recovery
            return BinaryExpression(left, op_token.value, right)
        
        return left
    
    def _parse_primary_expression(self) -> Optional[Expression]:
        """Parse primary expression"""
        token = self._current_token()
        if not token:
            return None
        
        # Property access (variable.property)
        if token.type == TokenType.IDENTIFIER:
            if self._peek_token() and self._peek_token().type == TokenType.DOT:
                variable = get_token_value(self._advance())
                self._advance()  # consume .
                
                prop_token = self._current_token()
                if prop_token and prop_token.type in (TokenType.IDENTIFIER, TokenType.BACKTICK_IDENTIFIER):
                    property_name = get_token_value(self._advance())
                    return PropertyExpression(variable, property_name)
                else:
                    self._error("Expected property name after '.'")
                    return None
            
            # Function call
            elif self._peek_token() and self._peek_token().type == TokenType.LPAREN:
                func_name = get_token_value(self._advance())
                return self._parse_function_call(func_name)
            
            # Simple variable
            else:
                variable = get_token_value(self._advance())
                return VariableExpression(variable)
        
        # Backtick identifier
        elif token.type == TokenType.BACKTICK_IDENTIFIER:
            variable = get_token_value(self._advance())
            return VariableExpression(variable)
        
        # Literals
        elif token.type in (TokenType.STRING, TokenType.INTEGER, TokenType.FLOAT, 
                           TokenType.BOOLEAN, TokenType.NULL, TokenType.PARAMETER):
            literal_value = self._parse_literal()
            return LiteralExpression(literal_value)
        
        # Parenthesized expression
        elif token.type == TokenType.LPAREN:
            self._advance()  # consume (
            expr = self._parse_expression()
            if not self._consume(TokenType.RPAREN):
                return None
            return expr
        
        else:
            self._error(f"Unexpected token in expression: '{token.value}'",
                       suggestions=["Expected variable, literal, or function call"])
            return None
    
    def _parse_function_call(self, func_name: str) -> Optional[FunctionCall]:
        """Parse function call"""
        if not self._consume(TokenType.LPAREN):
            return None
        
        args = []
        
        if not self._match(TokenType.RPAREN):
            arg = self._parse_expression()
            if arg:
                args.append(arg)
            
            while self._match(TokenType.COMMA):
                self._advance()  # consume comma
                arg = self._parse_expression()
                if arg:
                    args.append(arg)
                else:
                    self._error("Expected argument after ','")
                    break
        
        if not self._consume(TokenType.RPAREN):
            return None
        
        return FunctionCall(func_name, args)
    
    def _parse_return_clause(self) -> Optional[ReturnClause]:
        """Parse RETURN clause"""
        if not self._consume_keyword('RETURN'):
            return None
        
        distinct = False
        if self._match_keyword('DISTINCT'):
            distinct = True
            self._advance()
        
        items = self._parse_return_items()
        if not items:
            self._error("RETURN clause must specify what to return")
            return None
        
        order_by = None
        skip = None
        limit = None
        
        # Parse optional clauses
        while not self._at_end():
            if self._match_keyword('ORDER'):
                order_by = self._parse_order_by()
            elif self._match_keyword('SKIP'):
                skip = self._parse_skip()
            elif self._match_keyword('LIMIT'):
                limit = self._parse_limit()
            else:
                break
        
        return ReturnClause(items, distinct, order_by, skip, limit)
    
    def _parse_return_items(self) -> List[ReturnItem]:
        """Parse return items"""
        items = []
        
        item = self._parse_return_item()
        if item:
            items.append(item)
        
        while self._match(TokenType.COMMA):
            self._advance()  # consume comma
            item = self._parse_return_item()
            if item:
                items.append(item)
            else:
                self._error("Expected return item after ','")
                break
        
        return items
    
    def _parse_return_item(self) -> Optional[ReturnItem]:
        """Parse single return item"""
        expression = self._parse_expression()
        if expression is None:
            return None
        
        alias = None
        if self._match_keyword('AS'):
            self._advance()  # consume AS
            if self._match(TokenType.IDENTIFIER, TokenType.BACKTICK_IDENTIFIER):
                alias = get_token_value(self._advance())
            else:
                self._error("Expected alias after 'AS'")
        
        return ReturnItem(expression, alias)
    
    def _parse_order_by(self) -> Optional[OrderByClause]:
        """Parse ORDER BY clause"""
        if not self._consume_keyword('ORDER'):
            return None
        if not self._consume_keyword('BY'):
            return None
        
        items = []
        item = self._parse_order_by_item()
        if item:
            items.append(item)
        
        while self._match(TokenType.COMMA):
            self._advance()
            item = self._parse_order_by_item()
            if item:
                items.append(item)
        
        return OrderByClause(items) if items else None
    
    def _parse_order_by_item(self) -> Optional[OrderByItem]:
        """Parse ORDER BY item"""
        expression = self._parse_expression()
        if expression is None:
            return None
        
        ascending = True
        if self._match_keyword('ASC', 'DESC'):
            direction = self._advance()
            ascending = direction.value.upper() == 'ASC'
        
        return OrderByItem(expression, ascending)
    
    def _parse_skip(self) -> Optional[int]:
        """Parse SKIP value"""
        if not self._consume_keyword('SKIP'):
            return None
        
        if self._match(TokenType.INTEGER):
            value = int(self._advance().value)
            if value < 0:
                self._error("SKIP value cannot be negative")
                return 0
            return value
        else:
            self._error("Expected integer after SKIP")
            return None
    
    def _parse_limit(self) -> Optional[int]:
        """Parse LIMIT value"""
        if not self._consume_keyword('LIMIT'):
            return None
        
        if self._match(TokenType.INTEGER):
            value = int(self._advance().value)
            if value <= 0:
                self._error("LIMIT value must be positive")
                return 1
            return value
        else:
            self._error("Expected positive integer after LIMIT")
            return None
    
    def _parse_with_clause(self) -> Optional[WithClause]:
        """Parse WITH clause"""
        if not self._consume_keyword('WITH'):
            return None
        
        items = self._parse_return_items()
        if not items:
            self._error("WITH clause must specify items")
            return None
        
        where_clause = None
        if self._match_keyword('WHERE'):
            where_clause = self._parse_where_clause()
        
        return WithClause(items, where_clause)


# Convenience functions
def parse_cypher_query(query: str, enable_recovery: bool = True) -> Query:
    """Convenience function to parse Cypher query"""
    parser = CypherParser(enable_recovery=enable_recovery)
    return parser.parse(query)


def validate_cypher_syntax(query: str) -> bool:
    """Quick syntax validation without full parsing"""
    try:
        parse_cypher_query(query)
        return True
    except (ParseError, LexerError):
        return False


def get_parse_errors(query: str) -> List[str]:
    """Get detailed parse errors for a query"""
    try:
        parse_cypher_query(query)
        return []
    except ParseError as e:
        return [str(e)]
    except LexerError as e:
        return [f"Tokenization error: {str(e)}"]