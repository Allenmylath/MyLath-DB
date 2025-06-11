# cypher_planner/semantic_validator.py

"""Semantic Validator - Simplified"""

from .ast_nodes import Query

class SemanticValidator:
    """Basic semantic validation"""
    
    def validate_ast(self, ast: Query) -> bool:
        """Basic semantic validation"""
        try:
            # Variable scope validation
            defined_vars = self._collect_defined_variables(ast)
            used_vars = self._collect_used_variables(ast)
            
            # Check for undefined variables
            undefined = used_vars - defined_vars
            return len(undefined) == 0
        except Exception:
            return False
    
    def _collect_defined_variables(self, ast: Query) -> set:
        """Collect variables defined in MATCH clauses"""
        variables = set()
        for match_clause in ast.match_clauses:
            for pattern in match_clause.patterns:
                for element in pattern.elements:
                    if hasattr(element, 'variable') and element.variable:
                        variables.add(element.variable)
        return variables
    
    def _collect_used_variables(self, ast: Query) -> set:
        """Collect variables used in WHERE and RETURN"""
        # Simplified implementation
        return set()