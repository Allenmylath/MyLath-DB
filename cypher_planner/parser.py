# cypher_planner/parser.py
"""
Cypher Parser with Enhanced Error Handling
Backward-compatible wrapper that maintains existing API
"""

# Import the enhanced error-handling parser
from .integrated_parser import (
    CypherParser as EnhancedCypherParser,
    CypherParserError,
    parse_cypher_query,
    validate_cypher_query,
    get_cypher_errors
)

# For backward compatibility, export the enhanced parser as CypherParser
class CypherParser(EnhancedCypherParser):
    """
    Backward-compatible Cypher parser with enhanced error handling
    
    This maintains the same API as the original parser while adding
    comprehensive error handling capabilities.
    """
    
    def __init__(self):
        # Initialize with default settings for backward compatibility
        super().__init__(strict_mode=True, enable_warnings=False)
    
    def parse(self, query: str):
        """
        Parse method that maintains backward compatibility
        Raises ValueError on parse errors (same as original)
        """
        try:
            return super().parse(query)
        except CypherParserError as e:
            # Convert to ValueError for backward compatibility
            raise ValueError(str(e))

# Export additional utilities for advanced users
__all__ = [
    'CypherParser',
    'CypherParserError', 
    'parse_cypher_query',
    'validate_cypher_query', 
    'get_cypher_errors'
]