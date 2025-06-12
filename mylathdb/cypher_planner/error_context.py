# cypher_planner/error_context.py (SIMPLIFIED)

"""Simplified Error Context - Core functionality moved to parser"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional

class ErrorCode(Enum):
    """Shared error codes"""
    SYNTAX_ERROR = "E001"
    SEMANTIC_ERROR = "E002" 
    VALIDATION_ERROR = "E003"
    PERFORMANCE_WARNING = "W001"

@dataclass
class ErrorPosition:
    line: int
    column: int
    position: int
    length: int = 1

def format_error_message(code: ErrorCode, message: str, position: Optional[ErrorPosition] = None) -> str:
    """Format error message with position info"""
    if position:
        return f"{code.value} at line {position.line}, column {position.column}: {message}"
    return f"{code.value}: {message}"