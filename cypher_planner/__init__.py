# cypher_planner/__init__.py

"""
Cypher Planner - A comprehensive Cypher query parser and planner with enhanced error handling

This package provides tools for parsing, validating, and planning Cypher queries with
comprehensive error handling inspired by FalkorDB.
"""

from .ast_nodes import (
   # Core AST nodes
   Query,
   MatchClause,
   OptionalMatchClause,
   WhereClause,
   ReturnClause,
   WithClause,
   OrderByClause,
   OrderByItem,
   
   # Pattern nodes
   Pattern,
   NodePattern,
   RelationshipPattern,
   
   # Expression nodes
   Expression,
   VariableExpression,
   PropertyExpression,
   LiteralExpression,
   BinaryExpression,
   FunctionCall,
   
   # Return items
   ReturnItem
)

from .parser import CypherParser

from .planner import (
   QueryPlanner,
   ExecutionPlan,
   PlanNode,
   ScanNode,
   FilterNode,
   ProjectNode,
   JoinNode,
   AggregateNode,
   SortNode,
   LimitNode
)

# Enhanced error handling components
from .error_context import (
   ErrorCode,
   ErrorPosition, 
   ParseError,
   ErrorContext,
   get_error_context,
   set_error,
   set_warning,
   has_errors,
   clear_errors,
   format_errors
)

from .query_validator import QueryValidator
from .semantic_validator import SemanticValidator

from .integrated_parser import (
   CypherParserError,
   parse_cypher_query,
   validate_cypher_query,
   get_cypher_errors
)

# Utility functions
from .utils import (
   format_query,
   optimize_plan,
   estimate_cost,
   MockGraphStatistics
)

# Version information
__version__ = "1.2.0"
__author__ = "Cypher Planner Team"
__description__ = "Comprehensive Cypher query parser and planner with enhanced error handling"

# Public API exports
__all__ = [
   # Core classes
   "CypherParser",
   "QueryPlanner",
   
   # AST nodes
   "Query",
   "MatchClause", 
   "OptionalMatchClause",
   "WhereClause",
   "ReturnClause",
   "WithClause",
   "OrderByClause",
   "OrderByItem",
   "Pattern",
   "NodePattern",
   "RelationshipPattern",
   "Expression",
   "VariableExpression",
   "PropertyExpression", 
   "LiteralExpression",
   "BinaryExpression",
   "FunctionCall",
   "ReturnItem",
   
   # Execution plan nodes
   "ExecutionPlan",
   "PlanNode",
   "ScanNode",
   "FilterNode", 
   "ProjectNode",
   "JoinNode",
   "AggregateNode",
   "SortNode",
   "LimitNode",
   
   # Enhanced error handling
   "ErrorCode",
   "ErrorPosition",
   "ParseError", 
   "ErrorContext",
   "CypherParserError",
   "QueryValidator",
   "SemanticValidator",
   
   # Error handling functions
   "get_error_context",
   "set_error",
   "set_warning", 
   "has_errors",
   "clear_errors",
   "format_errors",
   
   # Convenience functions
   "parse_cypher_query",
   "validate_cypher_query",
   "get_cypher_errors",
   
   # Utility functions
   "format_query",
   "optimize_plan",
   "estimate_cost",
   "MockGraphStatistics",
   
   # Version info
   "__version__",
   "__author__",
   "__description__"
]

# Package-level configuration
def configure_error_handling(strict_mode=True, enable_warnings=True):
   """
   Configure global error handling behavior
   
   Args:
       strict_mode (bool): If True, warnings are treated as errors
       enable_warnings (bool): If False, warnings are suppressed
   """
   global _strict_mode, _enable_warnings
   _strict_mode = strict_mode
   _enable_warnings = enable_warnings

# Global configuration variables
_strict_mode = True
_enable_warnings = True

def get_package_info():
   """
   Get information about the cypher_planner package
   
   Returns:
       dict: Package information including version, features, and capabilities
   """
   return {
       "name": "cypher_planner",
       "version": __version__,
       "description": __description__,
       "author": __author__,
       "features": {
           "cypher_parsing": True,
           "query_planning": True,
           "error_handling": True,
           "semantic_validation": True,
           "performance_analysis": True,
           "query_optimization": True
       },
       "supported_cypher_constructs": {
           "match_clauses": True,
           "optional_match": True,
           "where_clauses": True,
           "return_clauses": True,
           "with_clauses": True,
           "order_by": True,
           "skip_limit": True,
           "patterns": True,
           "relationships": True,
           "variable_length_paths": True,
           "property_access": True,
           "functions": "partial",
           "create_clauses": False,
           "merge_clauses": False,
           "delete_clauses": False,
           "set_clauses": False,
           "procedures": False,
           "constraints": False,
           "indexes": False
       },
       "error_handling_features": {
           "syntax_validation": True,
           "semantic_validation": True,
           "performance_warnings": True,
           "detailed_error_messages": True,
           "error_recovery": True,
           "suggestions": True,
           "context_highlighting": True
       }
   }

# Convenience function for quick demos
def demo_enhanced_features():
   """
   Demonstrate enhanced error handling features
   """
   print("🔍 Cypher Planner Enhanced Features Demo")
   print("=" * 50)
   
   # Test cases
   test_queries = [
       ("Valid Query", "MATCH (n:Person) RETURN n.name"),
       ("Syntax Error", "MATCH (n:Person RETURN n.name"),
       ("Semantic Error", "MATCH (n:Person) RETURN m.name"),
       ("Performance Warning", "MATCH (n) MATCH (m) RETURN n, m")
   ]
   
   for name, query in test_queries:
       print(f"\n📝 {name}:")
       print(f"Query: {query}")
       print("-" * 30)
       
       try:
           # Try parsing
           ast = parse_cypher_query(query, strict=False)
           print("✅ Parse successful!")
           
       except CypherParserError as e:
           print("❌ Parse failed!")
           
           # Get detailed error info
           error_details = get_cypher_errors(query)
           if error_details.get('errors'):
               for error in error_details['errors'][:2]:  # Show first 2
                   print(f"   • {error['code']}: {error['message']}")
                   if error['suggestion']:
                       print(f"     💡 {error['suggestion']}")

# Module initialization
def _initialize_package():
   """Initialize package-level settings"""
   # Clear any existing errors from previous sessions
   clear_errors()
   
   # Set up default configuration
   configure_error_handling(strict_mode=True, enable_warnings=True)
   
   # Validate that all components are properly loaded
   try:
       # Test that core components can be imported
       parser = CypherParser()
       validator = QueryValidator()
       
       # Basic smoke test
       test_query = "MATCH (n) RETURN n"
       if validate_cypher_query(test_query):
           pass  # All good
       else:
           print("⚠️  Warning: Package initialization validation failed")
           
   except Exception as e:
       print(f"⚠️  Warning: Package initialization error: {e}")

# Run initialization when module is imported
_initialize_package()