# cypher_planner/integrated_parser.py

"""
Complete integration of error handling components
This replaces the original parser.py with comprehensive error handling
"""

from typing import Optional, Dict, Any, List
from .enhanced_parser import EnhancedCypherParser
from .semantic_validator import SemanticValidator
from .query_validator import QueryValidator
from .error_context import (
    get_error_context, clear_errors, has_errors, format_errors,
    ErrorCode, set_error, set_warning
)
from .ast_nodes import Query

class CypherParserError(Exception):
    """Exception raised when parsing fails"""
    
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []

class CypherParser:
    """
    Complete Cypher parser with comprehensive error handling
    Drop-in replacement for the original parser
    """
    
    def __init__(self, strict_mode: bool = True, enable_warnings: bool = True):
        """
        Initialize parser with error handling options
        
        Args:
            strict_mode: If True, warnings are treated as errors
            enable_warnings: If False, warnings are suppressed
        """
        self.enhanced_parser = EnhancedCypherParser()
        self.semantic_validator = SemanticValidator()
        self.query_validator = QueryValidator()
        self.strict_mode = strict_mode
        self.enable_warnings = enable_warnings
        
        # Statistics for debugging
        self.parse_stats = {
            'total_queries': 0,
            'successful_parses': 0,
            'syntax_errors': 0,
            'semantic_errors': 0,
            'validation_errors': 0
        }
        
    def parse(self, query: str) -> Query:
        """
        Parse a Cypher query with comprehensive error handling
        
        Args:
            query: The Cypher query string to parse
            
        Returns:
            Query: The parsed AST
            
        Raises:
            CypherParserError: If parsing fails
        """
        self.parse_stats['total_queries'] += 1
        
        # Clear previous errors
        clear_errors()
        
        if not query or not query.strip():
            set_error(ErrorCode.MISSING_TOKEN, "Empty query provided")
            self._handle_parse_failure("Empty query")
            
        try:
            # Phase 1: Pre-validation
            if not self._pre_validate_query(query):
                self.parse_stats['validation_errors'] += 1
                self._handle_parse_failure("Pre-validation failed")
                
            # Phase 2: Syntax parsing
            ast = self._parse_syntax(query)
            if ast is None:
                self.parse_stats['syntax_errors'] += 1
                self._handle_parse_failure("Syntax parsing failed")
                
            # Phase 3: Semantic validation
            if not self._validate_semantics(ast):
                self.parse_stats['semantic_errors'] += 1
                self._handle_parse_failure("Semantic validation failed")
                
            # Phase 4: Final validation
            if not self._post_validate_ast(ast):
                self._handle_parse_failure("Post-validation failed")
                
            # Success
            self.parse_stats['successful_parses'] += 1
            return ast
            
        except CypherParserError:
            raise
        except Exception as e:
            set_error(ErrorCode.UNEXPECTED_TOKEN, f"Internal parser error: {str(e)}")
            self._handle_parse_failure("Internal error")
            
    def _pre_validate_query(self, query: str) -> bool:
        """Pre-validation phase"""
        try:
            return self.query_validator.validate_query(query)
        except Exception as e:
            set_error(ErrorCode.UNEXPECTED_TOKEN, 
                     f"Pre-validation error: {str(e)}")
            return False
            
    def _parse_syntax(self, query: str) -> Optional[Query]:
        """Syntax parsing phase"""
        try:
            # Use the enhanced parser with error recovery
            return self.enhanced_parser.parse(query)
        except Exception as e:
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Syntax parsing error: {str(e)}")
            return None
            
    def _validate_semantics(self, ast: Query) -> bool:
        """Semantic validation phase"""
        try:
            return self.semantic_validator.validate_ast(ast)
        except Exception as e:
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Semantic validation error: {str(e)}")
            return False
            
    def _post_validate_ast(self, ast: Query) -> bool:
        """Final AST validation"""
        # Additional validation checks
        valid = True
        
        # Check for common performance issues
        if not self._check_performance_issues(ast):
            valid = False
            
        # Check for deprecated patterns
        if not self._check_deprecated_patterns(ast):
            valid = False
            
        return valid
        
    def _check_performance_issues(self, ast: Query) -> bool:
        """Check for potential performance issues"""
        valid = True
        
        # Warn about queries without MATCH
        if not ast.match_clauses and ast.return_clause:
            # This might be a query like "RETURN 1" which is valid but unusual
            set_warning(ErrorCode.INVALID_PATTERN,
                       "Query without MATCH clause may have performance implications")
                       
        # Check for Cartesian products
        if len(ast.match_clauses) > 1:
            # Multiple MATCH clauses might create Cartesian products
            set_warning(ErrorCode.INVALID_PATTERN,
                       "Multiple MATCH clauses may create Cartesian product - consider using single MATCH with comma-separated patterns")
                       
        # Check for complex WHERE conditions that might not use indexes
        if ast.where_clause:
            self._analyze_where_performance(ast.where_clause)
            
        return valid
        
    def _analyze_where_performance(self, where_clause):
        """Analyze WHERE clause for performance issues"""
        # This is a simplified analysis - a full implementation would be more sophisticated
        condition_str = str(where_clause.condition)
        
        # Look for patterns that prevent index usage
        if 'substring' in condition_str.lower() or 'tolower' in condition_str.lower():
            set_warning(ErrorCode.INVALID_FILTER_PLACEMENT,
                       "String functions in WHERE clause may prevent index usage")
                       
        # Look for OR conditions
        if ' OR ' in condition_str.upper():
            set_warning(ErrorCode.INVALID_FILTER_PLACEMENT,
                       "OR conditions may prevent optimal index usage")
                       
    def _check_deprecated_patterns(self, ast: Query) -> bool:
        """Check for deprecated or discouraged patterns"""
        valid = True
        
        # This would check for patterns that are deprecated in newer Cypher versions
        # For now, just placeholder checks
        
        return valid
        
    def _handle_parse_failure(self, phase: str):
        """Handle parsing failure and raise appropriate exception"""
        error_ctx = get_error_context()
        
        # Convert warnings to errors in strict mode
        if self.strict_mode and error_ctx.warnings:
            for warning in error_ctx.warnings:
                set_error(warning.code, f"[STRICT] {warning.message}")
                
        # Check if we should fail
        if has_errors() or (self.strict_mode and error_ctx.warnings):
            error_msg = format_errors()
            errors = [f"{err.code.value}: {err.message}" for err in error_ctx.errors]
            raise CypherParserError(f"Parse failed in {phase}:\n{error_msg}", errors)
            
    def get_parse_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics for debugging"""
        stats = self.parse_stats.copy()
        stats['success_rate'] = (
            stats['successful_parses'] / max(stats['total_queries'], 1) * 100
        )
        return stats
        
    def reset_statistics(self):
        """Reset parsing statistics"""
        for key in self.parse_stats:
            self.parse_stats[key] = 0
            
    def validate_query_only(self, query: str) -> bool:
        """
        Validate query without full parsing (for quick checks)
        
        Returns:
            bool: True if query appears valid
        """
        clear_errors()
        
        # Quick validation
        if not self.query_validator.validate_query(query):
            return False
            
        # Try basic tokenization
        try:
            tokens = self.enhanced_parser._safe_tokenize(query)
            return not has_errors()
        except:
            return False
            
    def get_suggestions_for_query(self, query: str) -> List[str]:
        """
        Get suggestions for fixing a malformed query
        
        Returns:
            List[str]: List of suggestions
        """
        clear_errors()
        suggestions = []
        
        try:
            # Try to parse and collect suggestions from errors
            self.parse(query)
        except CypherParserError:
            error_ctx = get_error_context()
            for error in error_ctx.errors:
                if error.suggestion:
                    suggestions.append(error.suggestion)
                    
        return list(set(suggestions))  # Remove duplicates
        
    def explain_error(self, query: str) -> Dict[str, Any]:
        """
        Provide detailed error explanation for a query
        
        Returns:
            Dict containing error details and context
        """
        clear_errors()
        
        try:
            self.parse(query)
            return {"success": True, "errors": []}
        except CypherParserError:
            error_ctx = get_error_context()
            
            return {
                "success": False,
                "query": query,
                "errors": [
                    {
                        "code": err.code.value,
                        "message": err.message,
                        "position": {
                            "line": err.position.line if err.position else None,
                            "column": err.position.column if err.position else None,
                            "position": err.position.position if err.position else None
                        },
                        "context": error_ctx.get_context_snippet(err.position) if err.position else None,
                        "suggestion": err.suggestion
                    }
                    for err in error_ctx.errors
                ],
                "warnings": [
                    {
                        "code": warn.code.value,
                        "message": warn.message,
                        "position": {
                            "line": warn.position.line if warn.position else None,
                            "column": warn.position.column if warn.position else None
                        }
                    }
                    for warn in error_ctx.warnings
                ]
            }

# Convenience functions for backward compatibility
def parse_cypher_query(query: str, strict: bool = True) -> Query:
    """
    Convenience function for parsing Cypher queries
    
    Args:
        query: Cypher query string
        strict: Whether to use strict mode
        
    Returns:
        Query: Parsed AST
        
    Raises:
        CypherParserError: If parsing fails
    """
    parser = CypherParser(strict_mode=strict)
    return parser.parse(query)

def validate_cypher_query(query: str) -> bool:
    """
    Quick validation of Cypher query syntax
    
    Args:
        query: Cypher query string
        
    Returns:
        bool: True if query is valid
    """
    parser = CypherParser(strict_mode=False)
    return parser.validate_query_only(query)

def get_cypher_errors(query: str) -> Dict[str, Any]:
    """
    Get detailed error information for a query
    
    Args:
        query: Cypher query string
        
    Returns:
        Dict: Error details and suggestions
    """
    parser = CypherParser(strict_mode=False)
    return parser.explain_error(query)

# Example usage and testing functions
def demonstrate_error_handling():
    """Demonstrate the enhanced error handling capabilities"""
    
    test_queries = [
        # Valid query
        "MATCH (n:Person) RETURN n.name",
        
        # Syntax errors
        "MATCH (n:Person RETURN n.name",  # Missing closing parenthesis
        "MATCH n:Person) RETURN n.name",   # Missing opening parenthesis
        "MATCH (n:Person) WHERE RETURN n.name",  # Empty WHERE
        
        # Semantic errors
        "MATCH (n:Person) RETURN m.name",  # Undefined variable
        "RETURN count(n) + n.name",        # Mixed aggregation
        
        # Performance warnings
        "MATCH (n) MATCH (m) RETURN n, m",  # Cartesian product
        "MATCH (n:Person) WHERE substring(n.name, 0, 3) = 'Tom' RETURN n",  # Non-index-friendly
        
        # Complex valid query
        """
        MATCH (p:Person {country: 'USA'})-[:KNOWS*1..3]->(friend:Person)
        WHERE p.age > 21 AND friend.age < p.age
        WITH p, collect(friend) as friends
        WHERE size(friends) > 2
        RETURN p.name, size(friends) as friend_count
        ORDER BY friend_count DESC
        LIMIT 10
        """
    ]
    
    parser = CypherParser(strict_mode=False, enable_warnings=True)
    
    print("=== Enhanced Cypher Parser Error Handling Demo ===\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query[:50]}{'...' if len(query) > 50 else ''}")
        print("-" * 60)
        
        try:
            ast = parser.parse(query)
            print("✅ Successfully parsed!")
            print(f"   AST type: {type(ast).__name__}")
            
            # Show basic structure
            if ast.match_clauses:
                print(f"   MATCH clauses: {len(ast.match_clauses)}")
            if ast.where_clause:
                print(f"   WHERE clause: Yes")
            if ast.return_clause:
                print(f"   RETURN clause: Yes")
                
        except CypherParserError as e:
            print("❌ Parse failed!")
            print(f"   Error: {str(e)}")
            
            # Show detailed error information
            error_details = parser.explain_error(query)
            if error_details.get('errors'):
                print("   Detailed errors:")
                for error in error_details['errors']:
                    print(f"     - {error['code']}: {error['message']}")
                    if error['suggestion']:
                        print(f"       Suggestion: {error['suggestion']}")
                        
            if error_details.get('warnings'):
                print("   Warnings:")
                for warning in error_details['warnings']:
                    print(f"     - {warning['code']}: {warning['message']}")
                    
        print()
        
    # Show parser statistics
    print("=== Parser Statistics ===")
    stats = parser.get_parse_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()

if __name__ == "__main__":
    demonstrate_error_handling()