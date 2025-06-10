# cypher_planner/error_context.py

"""
Error Context System for Cypher Parser
Inspired by FalkorDB's error handling mechanisms
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass

class ErrorCode(Enum):
    # Syntax Errors
    UNEXPECTED_TOKEN = "E001"
    MISSING_TOKEN = "E002"
    INVALID_PATTERN = "E003"
    UNBALANCED_PARENTHESES = "E004"
    UNBALANCED_BRACKETS = "E005"
    UNBALANCED_BRACES = "E006"
    
    # Semantic Errors
    UNDEFINED_VARIABLE = "E101"
    DUPLICATE_VARIABLE = "E102"
    INVALID_LABEL = "E103"
    INVALID_PROPERTY = "E104"
    INVALID_RELATIONSHIP = "E105"
    
    # Pattern Errors
    DANGLING_RELATIONSHIP = "E201"
    INVALID_VARIABLE_LENGTH = "E202"
    CIRCULAR_REFERENCE = "E203"
    
    # Filter Errors
    INVALID_FILTER_PLACEMENT = "E301"
    UNRESOLVABLE_REFERENCE = "E302"
    TYPE_MISMATCH = "E303"
    
    # Function Errors
    UNKNOWN_FUNCTION = "E401"
    INVALID_ARGUMENT_COUNT = "E402"
    INVALID_ARGUMENT_TYPE = "E403"

@dataclass
class ErrorPosition:
    line: int
    column: int
    position: int
    length: int = 1

@dataclass
class ParseError:
    code: ErrorCode
    message: str
    position: Optional[ErrorPosition] = None
    context: Optional[str] = None
    suggestion: Optional[str] = None

class ErrorContext:
    """Global error context for tracking parse errors"""
    
    def __init__(self):
        self.errors: List[ParseError] = []
        self.warnings: List[ParseError] = []
        self.has_errors = False
        self.query_text = ""
        self.current_position = 0
        
    def set_query(self, query: str):
        """Set the query text for error reporting"""
        self.query_text = query
        self.current_position = 0
        self.clear()
        
    def clear(self):
        """Clear all errors and warnings"""
        self.errors.clear()
        self.warnings.clear()
        self.has_errors = False
        
    def add_error(self, code: ErrorCode, message: str, 
                  position: Optional[ErrorPosition] = None,
                  context: Optional[str] = None,
                  suggestion: Optional[str] = None):
        """Add a parse error"""
        error = ParseError(
            code=code,
            message=message,
            position=position or self._current_position(),
            context=context,
            suggestion=suggestion
        )
        self.errors.append(error)
        self.has_errors = True
        
    def add_warning(self, code: ErrorCode, message: str,
                   position: Optional[ErrorPosition] = None):
        """Add a parse warning"""
        warning = ParseError(
            code=code,
            message=message,
            position=position or self._current_position()
        )
        self.warnings.append(warning)
        
    def _current_position(self) -> ErrorPosition:
        """Get current position in query"""
        line = 1
        column = 1
        for i in range(min(self.current_position, len(self.query_text))):
            if self.query_text[i] == '\n':
                line += 1
                column = 1
            else:
                column += 1
        return ErrorPosition(line, column, self.current_position)
        
    def get_context_snippet(self, position: ErrorPosition, window: int = 20) -> str:
        """Get context snippet around error position"""
        start = max(0, position.position - window)
        end = min(len(self.query_text), position.position + window)
        snippet = self.query_text[start:end]
        
        # Add markers
        marker_pos = position.position - start
        if marker_pos >= 0 and marker_pos < len(snippet):
            snippet = (snippet[:marker_pos] + 
                      ">>>" + snippet[marker_pos:marker_pos+position.length] + "<<<" +
                      snippet[marker_pos+position.length:])
        
        return snippet
        
    def format_errors(self) -> str:
        """Format all errors for display"""
        if not self.has_errors:
            return ""
            
        result = []
        result.append(f"Parse errors found in query:")
        result.append(f"Query: {self.query_text}")
        result.append("-" * 50)
        
        for error in self.errors:
            result.append(f"Error {error.code.value}: {error.message}")
            if error.position:
                result.append(f"  at line {error.position.line}, column {error.position.column}")
                context = self.get_context_snippet(error.position)
                if context:
                    result.append(f"  Context: {context}")
            if error.suggestion:
                result.append(f"  Suggestion: {error.suggestion}")
            result.append("")
            
        return "\n".join(result)

# Global error context instance
_error_context = ErrorContext()

def get_error_context() -> ErrorContext:
    """Get the global error context"""
    return _error_context

def set_error(code: ErrorCode, message: str, **kwargs):
    """Convenience function to set an error"""
    _error_context.add_error(code, message, **kwargs)

def set_warning(code: ErrorCode, message: str, **kwargs):
    """Convenience function to set a warning"""
    _error_context.add_warning(code, message, **kwargs)

def has_errors() -> bool:
    """Check if there are any errors"""
    return _error_context.has_errors

def clear_errors():
    """Clear all errors"""
    _error_context.clear()

def format_errors() -> str:
    """Format all errors for display"""
    return _error_context.format_errors()