# cypher_planner/tokenizer.py - FIXED VERSION

"""
Fixed Production Tokenizer with proper relationship pattern support
"""

import re
from typing import List, Optional, NamedTuple
from enum import Enum, auto
from dataclasses import dataclass
import unicodedata


class TokenType(Enum):
    # Literals
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    NULL = auto()
    
    # Identifiers and Keywords
    IDENTIFIER = auto()
    KEYWORD = auto()
    LABEL = auto()
    PROPERTY = auto()
    
    # Operators
    EQUALS = auto()
    NOT_EQUALS = auto()
    LESS_THAN = auto()
    LESS_EQUAL = auto()
    GREATER_THAN = auto()
    GREATER_EQUAL = auto()
    REGEX_MATCH = auto()
    
    # Arithmetic
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    POWER = auto()
    
    # Logical
    AND = auto()
    OR = auto()
    XOR = auto()
    NOT = auto()
    
    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    SEMICOLON = auto()
    DOT = auto()
    COLON = auto()
    
    # Relationship patterns
    ARROW_LEFT = auto()      # <-
    ARROW_RIGHT = auto()     # ->
    ARROW_BOTH = auto()      # <>
    DASH = auto()            # -
    DOUBLE_DASH = auto()     # --
    
    # Special
    VARIABLE_LENGTH = auto()  # *1..3, *, *2.., *..5
    BACKTICK_IDENTIFIER = auto()  # `weird name`
    PARAMETER = auto()       # $param
    
    # Control
    NEWLINE = auto()
    EOF = auto()
    WHITESPACE = auto()
    COMMENT = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    line: int
    column: int
    position: int
    length: int = 1
    
    def __post_init__(self):
        object.__setattr__(self, 'length', len(self.value))


class LexerError(Exception):
    """Lexical analysis error"""
    def __init__(self, message: str, line: int, column: int, position: int):
        self.message = message
        self.line = line
        self.column = column
        self.position = position
        super().__init__(f"Line {line}, column {column}: {message}")


class CypherTokenizer:
    """Production-ready Cypher tokenizer with proper lexical analysis"""
    
    # Cypher keywords (case-insensitive)
    KEYWORDS = {
        # Query clauses
        'MATCH', 'OPTIONAL', 'WHERE', 'RETURN', 'WITH', 'UNWIND',
        'CREATE', 'MERGE', 'DELETE', 'DETACH', 'SET', 'REMOVE',
        'FOREACH', 'CALL', 'YIELD', 'UNION', 'ALL',
        
        # Ordering and pagination
        'ORDER', 'BY', 'ASC', 'DESC', 'SKIP', 'LIMIT',
        
        # Logical operators
        'AND', 'OR', 'XOR', 'NOT',
        
        # Comparison and membership
        'IN', 'STARTS', 'ENDS', 'CONTAINS', 'IS',
        
        # Literals
        'TRUE', 'FALSE', 'NULL',
        
        # Special
        'AS', 'DISTINCT', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
        'EXISTS', 'UNIQUE'
    }
    
    def __init__(self):
        self.text = ""
        self.position = 0
        self.line = 1
        self.column = 1
        self.length = 0
        
    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenize Cypher query text into tokens
        
        Args:
            text: Cypher query string
            
        Returns:
            List of tokens
            
        Raises:
            LexerError: If lexical analysis fails
        """
        self.text = text
        self.position = 0
        self.line = 1
        self.column = 1
        self.length = len(text)
        
        tokens = []
        
        try:
            while not self._at_end():
                token = self._next_token()
                if token:
                    # Skip whitespace and comments unless explicitly requested
                    if token.type not in (TokenType.WHITESPACE, TokenType.COMMENT):
                        tokens.append(token)
                        
            # Add EOF token
            tokens.append(Token(TokenType.EOF, "", self.line, self.column, self.position))
            
        except Exception as e:
            if isinstance(e, LexerError):
                raise
            else:
                raise LexerError(f"Unexpected error: {str(e)}", 
                               self.line, self.column, self.position)
        
        return tokens
    
    def _at_end(self) -> bool:
        """Check if we've reached end of input"""
        return self.position >= self.length
    
    def _current_char(self) -> str:
        """Get current character"""
        if self._at_end():
            return '\0'
        return self.text[self.position]
    
    def _peek_char(self, offset: int = 1) -> str:
        """Peek at character ahead"""
        pos = self.position + offset
        if pos >= self.length:
            return '\0'
        return self.text[pos]
    
    def _advance(self) -> str:
        """Advance position and return current character"""
        if self._at_end():
            return '\0'
            
        char = self.text[self.position]
        self.position += 1
        
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
            
        return char
    
    def _next_token(self) -> Optional[Token]:
        """Get next token from input"""
        start_pos = self.position
        start_line = self.line
        start_column = self.column
        
        char = self._current_char()
        
        # Skip whitespace
        if char.isspace():
            return self._read_whitespace(start_line, start_column, start_pos)
        
        # Comments
        if char == '/' and self._peek_char() == '/':
            return self._read_line_comment(start_line, start_column, start_pos)
        
        if char == '/' and self._peek_char() == '*':
            return self._read_block_comment(start_line, start_column, start_pos)
        
        # String literals
        if char in ('"', "'"):
            return self._read_string(start_line, start_column, start_pos)
        
        # Backtick identifiers
        if char == '`':
            return self._read_backtick_identifier(start_line, start_column, start_pos)
        
        # Parameters
        if char == '$':
            return self._read_parameter(start_line, start_column, start_pos)
        
        # Numbers
        if char.isdigit() or (char == '.' and self._peek_char().isdigit()):
            return self._read_number(start_line, start_column, start_pos)
        
        # Variable length patterns
        if char == '*':
            return self._read_variable_length(start_line, start_column, start_pos)
        
        # FIXED: Multi-character operators and relationship patterns
        # Handle relationship patterns FIRST before single operators
        two_char = char + self._peek_char()
        three_char = two_char + self._peek_char(2) if len(self.text) > self.position + 2 else ""
        
        # Three character patterns
        if three_char == '<--':
            self._advance()  # <
            self._advance()  # -
            self._advance()  # -
            return Token(TokenType.ARROW_LEFT, '<--', start_line, start_column, start_pos)
        elif three_char == '-->':
            self._advance()  # -
            self._advance()  # -
            self._advance()  # >
            return Token(TokenType.ARROW_RIGHT, '-->', start_line, start_column, start_pos)
        
        # Two character patterns
        elif two_char == '<-':
            self._advance()
            self._advance()
            return Token(TokenType.ARROW_LEFT, '<-', start_line, start_column, start_pos)
        elif two_char == '->':
            self._advance()
            self._advance()
            return Token(TokenType.ARROW_RIGHT, '->', start_line, start_column, start_pos)
        elif two_char == '--':
            self._advance()
            self._advance()
            return Token(TokenType.DOUBLE_DASH, '--', start_line, start_column, start_pos)
        elif two_char == '<>':
            self._advance()
            self._advance()
            return Token(TokenType.NOT_EQUALS, '<>', start_line, start_column, start_pos)
        elif two_char == '<=':
            self._advance()
            self._advance()
            return Token(TokenType.LESS_EQUAL, '<=', start_line, start_column, start_pos)
        elif two_char == '>=':
            self._advance()
            self._advance()
            return Token(TokenType.GREATER_EQUAL, '>=', start_line, start_column, start_pos)
        elif two_char == '!=':
            self._advance()
            self._advance()
            return Token(TokenType.NOT_EQUALS, '!=', start_line, start_column, start_pos)
        elif two_char == '=~':
            self._advance()
            self._advance()
            return Token(TokenType.REGEX_MATCH, '=~', start_line, start_column, start_pos)
        
        # Single character operators
        elif char == '=':
            self._advance()
            return Token(TokenType.EQUALS, '=', start_line, start_column, start_pos)
        elif char == '<':
            self._advance()
            return Token(TokenType.LESS_THAN, '<', start_line, start_column, start_pos)
        elif char == '>':
            self._advance()
            return Token(TokenType.GREATER_THAN, '>', start_line, start_column, start_pos)
        elif char == '+':
            self._advance()
            return Token(TokenType.PLUS, '+', start_line, start_column, start_pos)
        elif char == '-':
            # IMPORTANT: Single dash for relationship patterns
            self._advance()
            return Token(TokenType.DASH, '-', start_line, start_column, start_pos)
        elif char == '*':
            # This will be handled by variable length parser above if it's *1..3
            self._advance()
            return Token(TokenType.MULTIPLY, '*', start_line, start_column, start_pos)
        elif char == '/':
            self._advance()
            return Token(TokenType.DIVIDE, '/', start_line, start_column, start_pos)
        elif char == '%':
            self._advance()
            return Token(TokenType.MODULO, '%', start_line, start_column, start_pos)
        elif char == '^':
            self._advance()
            return Token(TokenType.POWER, '^', start_line, start_column, start_pos)
        
        # Punctuation
        elif char == '(':
            self._advance()
            return Token(TokenType.LPAREN, '(', start_line, start_column, start_pos)
        elif char == ')':
            self._advance()
            return Token(TokenType.RPAREN, ')', start_line, start_column, start_pos)
        elif char == '[':
            self._advance()
            return Token(TokenType.LBRACKET, '[', start_line, start_column, start_pos)
        elif char == ']':
            self._advance()
            return Token(TokenType.RBRACKET, ']', start_line, start_column, start_pos)
        elif char == '{':
            self._advance()
            return Token(TokenType.LBRACE, '{', start_line, start_column, start_pos)
        elif char == '}':
            self._advance()
            return Token(TokenType.RBRACE, '}', start_line, start_column, start_pos)
        elif char == ',':
            self._advance()
            return Token(TokenType.COMMA, ',', start_line, start_column, start_pos)
        elif char == ';':
            self._advance()
            return Token(TokenType.SEMICOLON, ';', start_line, start_column, start_pos)
        elif char == '.':
            self._advance()
            return Token(TokenType.DOT, '.', start_line, start_column, start_pos)
        elif char == ':':
            self._advance()
            return Token(TokenType.COLON, ':', start_line, start_column, start_pos)
        
        # Identifiers and keywords
        elif self._is_identifier_start(char):
            return self._read_identifier(start_line, start_column, start_pos)
        
        # Unknown character
        else:
            raise LexerError(f"Unexpected character: '{char}' (U+{ord(char):04X})",
                            self.line, self.column, self.position)
    
    def _read_whitespace(self, line: int, column: int, position: int) -> Token:
        """Read whitespace characters"""
        start = self.position
        
        while not self._at_end() and self._current_char().isspace():
            self._advance()
        
        value = self.text[start:self.position]
        return Token(TokenType.WHITESPACE, value, line, column, position)
    
    def _read_line_comment(self, line: int, column: int, position: int) -> Token:
        """Read single-line comment (// ...)"""
        start = self.position
        
        # Skip //
        self._advance()
        self._advance()
        
        # Read until end of line
        while not self._at_end() and self._current_char() != '\n':
            self._advance()
        
        value = self.text[start:self.position]
        return Token(TokenType.COMMENT, value, line, column, position)
    
    def _read_block_comment(self, line: int, column: int, position: int) -> Token:
        """Read block comment (/* ... */)"""
        start = self.position
        
        # Skip /*
        self._advance()
        self._advance()
        
        # Read until */
        while not self._at_end():
            if self._current_char() == '*' and self._peek_char() == '/':
                self._advance()  # consume *
                self._advance()  # consume /
                break
            self._advance()
        else:
            raise LexerError("Unterminated block comment", line, column, position)
        
        value = self.text[start:self.position]
        return Token(TokenType.COMMENT, value, line, column, position)
    
    def _read_string(self, line: int, column: int, position: int) -> Token:
        """Read string literal with proper escape handling"""
        start = self.position
        quote_char = self._advance()  # consume opening quote
        
        value_chars = []
        
        while not self._at_end():
            char = self._current_char()
            
            # End of string
            if char == quote_char:
                self._advance()  # consume closing quote
                break
            
            # Escape sequences
            elif char == '\\':
                self._advance()  # consume backslash
                if self._at_end():
                    raise LexerError("Unterminated string literal", line, column, position)
                
                escaped = self._advance()
                
                # Handle escape sequences
                if escaped == 'n':
                    value_chars.append('\n')
                elif escaped == 't':
                    value_chars.append('\t')
                elif escaped == 'r':
                    value_chars.append('\r')
                elif escaped == 'b':
                    value_chars.append('\b')
                elif escaped == 'f':
                    value_chars.append('\f')
                elif escaped == '\\':
                    value_chars.append('\\')
                elif escaped == quote_char:
                    value_chars.append(quote_char)
                elif escaped == 'u':
                    # Unicode escape \uXXXX
                    unicode_chars = ""
                    for _ in range(4):
                        if self._at_end() or not self._current_char().isalnum():
                            raise LexerError("Invalid Unicode escape sequence", 
                                           self.line, self.column, self.position)
                        unicode_chars += self._advance()
                    
                    try:
                        code_point = int(unicode_chars, 16)
                        value_chars.append(chr(code_point))
                    except ValueError:
                        raise LexerError(f"Invalid Unicode code point: \\u{unicode_chars}",
                                       self.line, self.column, self.position)
                else:
                    # Unknown escape - include literally
                    value_chars.append('\\')
                    value_chars.append(escaped)
            
            # Regular character
            else:
                value_chars.append(self._advance())
        else:
            raise LexerError(f"Unterminated string literal starting with {quote_char}",
                           line, column, position)
        
        # Return the full string including quotes
        full_value = self.text[start:self.position]
        return Token(TokenType.STRING, full_value, line, column, position)
    
    def _read_backtick_identifier(self, line: int, column: int, position: int) -> Token:
        """Read backtick-quoted identifier"""
        start = self.position
        self._advance()  # consume opening backtick
        
        while not self._at_end() and self._current_char() != '`':
            if self._current_char() == '\n':
                raise LexerError("Backtick identifier cannot span multiple lines",
                               self.line, self.column, self.position)
            self._advance()
        
        if self._at_end():
            raise LexerError("Unterminated backtick identifier", line, column, position)
        
        self._advance()  # consume closing backtick
        
        value = self.text[start:self.position]
        return Token(TokenType.BACKTICK_IDENTIFIER, value, line, column, position)
    
    def _read_parameter(self, line: int, column: int, position: int) -> Token:
        """Read parameter reference ($param)"""
        start = self.position
        self._advance()  # consume $
        
        if self._at_end() or not self._is_identifier_start(self._current_char()):
            raise LexerError("Invalid parameter name after $", line, column, position)
        
        # Read identifier part
        while not self._at_end() and self._is_identifier_part(self._current_char()):
            self._advance()
        
        value = self.text[start:self.position]
        return Token(TokenType.PARAMETER, value, line, column, position)
    
    def _read_number(self, line: int, column: int, position: int) -> Token:
        """Read numeric literal with proper validation"""
        start = self.position
        has_dot = False
        has_exponent = False
        
        # Handle leading decimal point
        if self._current_char() == '.':
            has_dot = True
            self._advance()
        
        # Read integer part or digits after decimal
        if not self._current_char().isdigit():
            raise LexerError("Invalid number format", line, column, position)
        
        while not self._at_end() and self._current_char().isdigit():
            self._advance()
        
        # Decimal point (if not already seen)
        if not has_dot and self._current_char() == '.':
            # Check if it's actually a range operator (..)
            if self._peek_char() == '.':
                # Don't consume the decimal point - it's part of range
                pass
            else:
                has_dot = True
                self._advance()
                
                # Read fractional part
                while not self._at_end() and self._current_char().isdigit():
                    self._advance()
        
        # Scientific notation
        if self._current_char().lower() == 'e':
            has_exponent = True
            self._advance()
            
            # Optional + or -
            if self._current_char() in ('+', '-'):
                self._advance()
            
            # Exponent digits (required)
            if not self._current_char().isdigit():
                raise LexerError("Invalid exponent in number", line, column, position)
            
            while not self._at_end() and self._current_char().isdigit():
                self._advance()
        
        value = self.text[start:self.position]
        
        # Validate the number
        try:
            if has_dot or has_exponent:
                float(value)
                token_type = TokenType.FLOAT
            else:
                int(value)
                token_type = TokenType.INTEGER
        except ValueError:
            raise LexerError(f"Invalid number format: {value}", line, column, position)
        
        return Token(token_type, value, line, column, position)
    
    def _read_variable_length(self, line: int, column: int, position: int) -> Token:
        """Read variable length pattern (*1..3, *, etc.)"""
        start = self.position
        self._advance()  # consume *
        
        # Just * means unlimited
        if self._at_end() or not self._current_char().isdigit():
            # Check for range patterns
            if self._current_char() == '.' and self._peek_char() == '.':
                self._advance()  # consume first .
                self._advance()  # consume second .
                
                # Read optional end number
                while not self._at_end() and self._current_char().isdigit():
                    self._advance()
        else:
            # Read start number
            while not self._at_end() and self._current_char().isdigit():
                self._advance()
            
            # Check for range
            if self._current_char() == '.' and self._peek_char() == '.':
                self._advance()  # consume first .
                self._advance()  # consume second .
                
                # Read optional end number
                while not self._at_end() and self._current_char().isdigit():
                    self._advance()
        
        value = self.text[start:self.position]
        
        # Validate variable length syntax
        if not self._validate_variable_length(value):
            raise LexerError(f"Invalid variable length pattern: {value}", 
                           line, column, position)
        
        return Token(TokenType.VARIABLE_LENGTH, value, line, column, position)
    
    def _read_identifier(self, line: int, column: int, position: int) -> Token:
        """Read identifier or keyword"""
        start = self.position
        
        # Read identifier characters
        while not self._at_end() and self._is_identifier_part(self._current_char()):
            self._advance()
        
        value = self.text[start:self.position]
        
        # Check if it's a keyword
        if value.upper() in self.KEYWORDS:
            # Special handling for boolean and null literals
            upper_value = value.upper()
            if upper_value == 'TRUE':
                return Token(TokenType.BOOLEAN, value, line, column, position)
            elif upper_value == 'FALSE':
                return Token(TokenType.BOOLEAN, value, line, column, position)
            elif upper_value == 'NULL':
                return Token(TokenType.NULL, value, line, column, position)
            else:
                return Token(TokenType.KEYWORD, value, line, column, position)
        
        return Token(TokenType.IDENTIFIER, value, line, column, position)
    
    def _is_identifier_start(self, char: str) -> bool:
        """Check if character can start an identifier"""
        if not char:
            return False
        
        # Basic ASCII letters and underscore
        if char.isalpha() or char == '_':
            return True
        
        # Unicode letters
        category = unicodedata.category(char)
        return category.startswith('L')  # Letter categories
    
    def _is_identifier_part(self, char: str) -> bool:
        """Check if character can be part of an identifier"""
        if not char:
            return False
        
        # Basic ASCII letters, digits, underscore
        if char.isalnum() or char == '_':
            return True
        
        # Unicode letters, marks, numbers
        category = unicodedata.category(char)
        return (category.startswith('L') or    # Letters
                category.startswith('M') or    # Marks
                category.startswith('N'))      # Numbers
    
    def _validate_variable_length(self, pattern: str) -> bool:
        """Validate variable length pattern syntax"""
        # Remove the *
        if not pattern.startswith('*'):
            return False
        
        range_part = pattern[1:]
        
        if not range_part:  # Just *
            return True
        
        # Pattern: number, number.., ..number, number..number
        if '..' not in range_part:
            # Just a number
            try:
                num = int(range_part)
                return num >= 0
            except ValueError:
                return False
        
        # Range pattern
        parts = range_part.split('..')
        if len(parts) != 2:
            return False
        
        start_str, end_str = parts
        
        # Validate start
        if start_str:
            try:
                start = int(start_str)
                if start < 0:
                    return False
            except ValueError:
                return False
        
        # Validate end
        if end_str:
            try:
                end = int(end_str)
                if end < 0:
                    return False
                
                # Check range validity
                if start_str:
                    start = int(start_str)
                    if start > end:
                        return False
            except ValueError:
                return False
        
        return True


# Utility functions for easy integration
def tokenize_cypher(query: str) -> List[Token]:
    """
    Convenience function to tokenize a Cypher query
    
    Args:
        query: Cypher query string
        
    Returns:
        List of tokens
        
    Raises:
        LexerError: If tokenization fails
    """
    tokenizer = CypherTokenizer()
    return tokenizer.tokenize(query)


def get_token_value(token: Token) -> str:
    """Extract the actual value from a token, handling quotes, etc."""
    if token.type == TokenType.STRING:
        # Remove quotes and handle escape sequences
        content = token.value[1:-1]  # Remove surrounding quotes
        return content.encode().decode('unicode_escape')
    elif token.type == TokenType.BACKTICK_IDENTIFIER:
        return token.value[1:-1]  # Remove surrounding backticks
    elif token.type == TokenType.PARAMETER:
        return token.value[1:]  # Remove $ prefix
    else:
        return token.value


def print_tokens(tokens: List[Token], show_positions: bool = False) -> None:
    """Debug helper to print tokens"""
    for token in tokens:
        if show_positions:
            print(f"{token.type.name:<20} '{token.value}' @ {token.line}:{token.column}")
        else:
            print(f"{token.type.name:<20} '{token.value}'")


