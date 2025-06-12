# cypher_planner/query_validator.py

"""Query Validator - Updated to work with new tokenizer"""

from .tokenizer import CypherTokenizer, LexerError

class QueryValidator:
    """Pre-parse validation using new tokenizer"""
    
    def __init__(self):
        self.tokenizer = CypherTokenizer()
    
    def validate_query(self, query: str) -> bool:
        """Validate query using tokenizer"""
        if not query or not query.strip():
            return False
        
        try:
            tokens = self.tokenizer.tokenize(query)
            return True
        except LexerError:
            return False
    
    def get_validation_errors(self, query: str) -> list:
        """Get detailed validation errors"""
        try:
            self.tokenizer.tokenize(query)
            return []
        except LexerError as e:
            return [str(e)]