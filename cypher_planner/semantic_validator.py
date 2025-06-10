# cypher_planner/semantic_validator.py

"""
Semantic validation for parsed AST
Performs deep semantic analysis before execution plan generation
"""

from typing import Set, Dict, List, Optional, Any
from .ast_nodes import *
from .error_context import ErrorCode, set_error, set_warning

class SemanticValidator:
    """Performs semantic validation on parsed AST"""
    
    def __init__(self):
        self.variable_scopes = []  # Stack of variable scopes
        self.current_scope = set()  # Current scope variables
        self.global_scope = set()   # Global scope variables
        self.path_variables = set()  # Variables representing paths
        self.aggregate_context = False  # Whether we're in aggregation context
        
    def validate_ast(self, ast: Query) -> bool:
        """
        Validate the entire AST for semantic correctness
        Returns True if valid, False otherwise
        """
        self._reset_validation_state()
        
        # Validation phases
        validations = [
            self._validate_variable_scoping,
            self._validate_clause_dependencies,
            self._validate_aggregation_semantics,
            self._validate_path_semantics,
            self._validate_function_usage,
            self._validate_type_consistency
        ]
        
        all_valid = True
        for validation in validations:
            if not validation(ast):
                all_valid = False
                
        return all_valid
        
    def _reset_validation_state(self):
        """Reset validation state for new query"""
        self.variable_scopes.clear()
        self.current_scope.clear()
        self.global_scope.clear()
        self.path_variables.clear()
        self.aggregate_context = False
        
    def _validate_variable_scoping(self, ast: Query) -> bool:
        """Validate variable scoping and usage"""
        valid = True
        
        # Build variable scope from MATCH clauses
        for match_clause in ast.match_clauses:
            if not self._analyze_match_variables(match_clause):
                valid = False
                
        # Validate optional match variables
        for optional_clause in ast.optional_match_clauses:
            if not self._analyze_optional_match_variables(optional_clause):
                valid = False
                
        # Validate WHERE clause variable usage
        if ast.where_clause:
            if not self._validate_expression_variables(ast.where_clause.condition):
                valid = False
                
        # Validate RETURN clause variable usage
        if ast.return_clause:
            if not self._validate_return_variables(ast.return_clause):
                valid = False
                
        # Validate WITH clause variable usage
        for with_clause in ast.with_clauses:
            if not self._validate_with_variables(with_clause):
                valid = False
                
        return valid
        
    def _analyze_match_variables(self, match_clause: MatchClause) -> bool:
        """Analyze variables introduced in MATCH clause"""
        valid = True
        
        for pattern in match_clause.patterns:
            for element in pattern.elements:
                if isinstance(element, NodePattern):
                    if element.variable:
                        if element.variable in self.current_scope:
                            set_warning(ErrorCode.DUPLICATE_VARIABLE,
                                       f"Variable '{element.variable}' redefined in pattern")
                        self.current_scope.add(element.variable)
                        self.global_scope.add(element.variable)
                        
                elif isinstance(element, RelationshipPattern):
                    if element.variable:
                        if element.variable in self.current_scope:
                            set_warning(ErrorCode.DUPLICATE_VARIABLE,
                                       f"Relationship variable '{element.variable}' redefined")
                        self.current_scope.add(element.variable)
                        self.global_scope.add(element.variable)
                        
                        # Check variable length relationships
                        if element.max_length and element.max_length > 1:
                            if not self._validate_variable_length_semantics(element):
                                valid = False
                                
        return valid
        
    def _analyze_optional_match_variables(self, optional_clause: OptionalMatchClause) -> bool:
        """Analyze variables in OPTIONAL MATCH"""
        # Variables from OPTIONAL MATCH can be NULL
        # They should be tracked separately for NULL-safety validation
        valid = True
        optional_vars = set()
        
        for pattern in optional_clause.patterns:
            for element in pattern.elements:
                if isinstance(element, NodePattern) and element.variable:
                    optional_vars.add(element.variable)
                    self.current_scope.add(element.variable)
                    self.global_scope.add(element.variable)
                elif isinstance(element, RelationshipPattern) and element.variable:
                    optional_vars.add(element.variable)
                    self.current_scope.add(element.variable)
                    self.global_scope.add(element.variable)
                    
        # Mark these variables as potentially NULL
        for var in optional_vars:
            set_warning(ErrorCode.UNDEFINED_VARIABLE,
                       f"Variable '{var}' from OPTIONAL MATCH may be NULL")
                       
        return valid
        
    def _validate_variable_length_semantics(self, rel: RelationshipPattern) -> bool:
        """Validate variable length relationship semantics"""
        if rel.min_length is not None and rel.max_length is not None:
            if rel.min_length < 0:
                set_error(ErrorCode.INVALID_VARIABLE_LENGTH,
                         "Variable length minimum cannot be negative")
                return False
                
            if rel.max_length != float('inf') and rel.min_length > rel.max_length:
                set_error(ErrorCode.INVALID_VARIABLE_LENGTH,
                         f"Variable length minimum ({rel.min_length}) cannot exceed maximum ({rel.max_length})")
                return False
                
            if rel.min_length == 0:
                set_warning(ErrorCode.INVALID_VARIABLE_LENGTH,
                           "Variable length with minimum 0 may cause performance issues")
                           
        return True
        
    def _validate_expression_variables(self, expr: Expression) -> bool:
        """Validate variables used in expressions"""
        undefined_vars = self._find_undefined_variables(expr)
        
        valid = True
        for var in undefined_vars:
            set_error(ErrorCode.UNDEFINED_VARIABLE,
                     f"Variable '{var}' is not defined",
                     suggestion="Define variable in MATCH clause")
            valid = False
            
        return valid
        
    def _find_undefined_variables(self, expr: Expression) -> Set[str]:
        """Find undefined variables in expression"""
        used_vars = self._extract_expression_variables(expr)
        return used_vars - self.global_scope
        
    def _extract_expression_variables(self, expr: Expression) -> Set[str]:
        """Extract all variables from expression"""
        variables = set()
        
        if isinstance(expr, VariableExpression):
            variables.add(expr.name)
        elif isinstance(expr, PropertyExpression):
            variables.add(expr.variable)
        elif isinstance(expr, BinaryExpression):
            variables.update(self._extract_expression_variables(expr.left))
            variables.update(self._extract_expression_variables(expr.right))
        elif isinstance(expr, FunctionCall):
            for arg in expr.arguments:
                variables.update(self._extract_expression_variables(arg))
                
        return variables
        
    def _validate_return_variables(self, return_clause: ReturnClause) -> bool:
        """Validate RETURN clause variables"""
        valid = True
        
        for item in return_clause.items:
            if not self._validate_expression_variables(item.expression):
                valid = False
                
            # Check for aggregation functions
            if self._contains_aggregation(item.expression):
                self.aggregate_context = True
                
        # Validate ORDER BY if present
        if return_clause.order_by:
            for order_item in return_clause.order_by.items:
                if not self._validate_expression_variables(order_item.expression):
                    valid = False
                    
        return valid
        
    def _validate_with_variables(self, with_clause: WithClause) -> bool:
        """Validate WITH clause variables"""
        valid = True
        new_scope = set()
        
        # WITH creates a new scope with only the projected variables
        for item in with_clause.items:
            if not self._validate_expression_variables(item.expression):
                valid = False
                
            # Add projected variable to new scope
            if item.alias:
                new_scope.add(item.alias)
            elif isinstance(item.expression, VariableExpression):
                new_scope.add(item.expression.name)
            elif isinstance(item.expression, PropertyExpression):
                # Property expressions don't create new variables
                pass
                
        # Update scope for subsequent clauses
        self.current_scope = new_scope
        
        # Validate WHERE clause with new scope
        if with_clause.where_clause:
            if not self._validate_expression_variables(with_clause.where_clause.condition):
                valid = False
                
        return valid
        
    def _validate_clause_dependencies(self, ast: Query) -> bool:
        """Validate clause ordering and dependencies"""
        valid = True
        
        # Check that MATCH comes before WHERE (if WHERE references MATCH variables)
        if ast.where_clause and not ast.match_clauses:
            # WHERE without MATCH is unusual but might be valid in some contexts
            set_warning(ErrorCode.INVALID_PATTERN,
                       "WHERE clause without preceding MATCH")
                       
        # Check RETURN is present (unless it's a write-only query)
        has_write_ops = any([
            # This would be extended to check for CREATE, MERGE, DELETE, etc.
            # For now, just check basic structure
        ])
        
        if not ast.return_clause and not has_write_ops:
            set_warning(ErrorCode.MISSING_TOKEN,
                       "Query has no RETURN clause - results will not be returned")
                       
        return valid
        
    def _validate_aggregation_semantics(self, ast: Query) -> bool:
        """Validate aggregation function usage"""
        valid = True
        
        if ast.return_clause:
            has_aggregation = False
            has_non_aggregation = False
            
            for item in ast.return_clause.items:
                if self._contains_aggregation(item.expression):
                    has_aggregation = True
                else:
                    # Check if this is a simple variable (allowed with aggregation)
                    if not isinstance(item.expression, VariableExpression):
                        has_non_aggregation = True
                        
            # In Cypher, you can't mix aggregation with non-aggregated expressions
            # unless the non-aggregated expressions are grouping keys
            if has_aggregation and has_non_aggregation:
                set_error(ErrorCode.TYPE_MISMATCH,
                         "Cannot mix aggregation functions with non-aggregated expressions",
                         suggestion="Use only aggregation functions or group by other expressions")
                valid = False
                
        return valid
        
    def _contains_aggregation(self, expr: Expression) -> bool:
        """Check if expression contains aggregation functions"""
        if isinstance(expr, FunctionCall):
            aggregation_functions = {
                'count', 'sum', 'avg', 'min', 'max', 'collect', 
                'stdev', 'stdevp', 'percentilecount', 'percentiledisc'
            }
            if expr.name.lower() in aggregation_functions:
                return True
                
            # Check arguments recursively
            for arg in expr.arguments:
                if self._contains_aggregation(arg):
                    return True
                    
        elif isinstance(expr, BinaryExpression):
            return (self._contains_aggregation(expr.left) or 
                   self._contains_aggregation(expr.right))
                   
        return False
        
    def _validate_path_semantics(self, ast: Query) -> bool:
        """Validate path-related semantics"""
        valid = True
        
        # Check for disconnected patterns
        for match_clause in ast.match_clauses:
            for pattern in match_clause.patterns:
                if not self._validate_pattern_connectivity(pattern):
                    valid = False
                    
        return valid
        
    def _validate_pattern_connectivity(self, pattern: Pattern) -> bool:
        """Validate that pattern elements are properly connected"""
        if len(pattern.elements) < 2:
            return True  # Single node patterns are valid
            
        # Check alternating node-relationship-node pattern
        for i, element in enumerate(pattern.elements):
            if i % 2 == 0:  # Even indices should be nodes
                if not isinstance(element, NodePattern):
                    set_error(ErrorCode.DANGLING_RELATIONSHIP,
                             "Pattern elements must alternate between nodes and relationships")
                    return False
            else:  # Odd indices should be relationships
                if not isinstance(element, RelationshipPattern):
                    set_error(ErrorCode.DANGLING_RELATIONSHIP,
                             "Pattern elements must alternate between nodes and relationships")
                    return False
                    
        return True
        
    def _validate_function_usage(self, ast: Query) -> bool:
        """Validate function calls and their arguments"""
        valid = True
        
        # Validate functions in WHERE clause
        if ast.where_clause:
            if not self._validate_expression_functions(ast.where_clause.condition):
                valid = False
                
        # Validate functions in RETURN clause
        if ast.return_clause:
            for item in ast.return_clause.items:
                if not self._validate_expression_functions(item.expression):
                    valid = False
                    
        return valid
        
    def _validate_expression_functions(self, expr: Expression) -> bool:
        """Validate function calls in expressions"""
        valid = True
        
        if isinstance(expr, FunctionCall):
            if not self._validate_function_call(expr):
                valid = False
        elif isinstance(expr, BinaryExpression):
            if not self._validate_expression_functions(expr.left):
                valid = False
            if not self._validate_expression_functions(expr.right):
                valid = False
                
        return valid
        
    def _validate_function_call(self, func_call: FunctionCall) -> bool:
        """Validate specific function call"""
        func_name = func_call.name.lower()
        arg_count = len(func_call.arguments)
        
        # Known function signatures
        function_signatures = {
            'count': (0, 1),  # COUNT() or COUNT(expr)
            'sum': (1, 1),
            'avg': (1, 1), 
            'min': (1, 1),
            'max': (1, 1),
            'collect': (1, 1),
            'size': (1, 1),
            'length': (1, 1),
            'exists': (1, 1),
            'type': (1, 1),
            'id': (1, 1),
            'labels': (1, 1),
            'keys': (1, 1),
            'properties': (1, 1),
            'startnode': (1, 1),
            'endnode': (1, 1),
            'nodes': (1, 1),
            'relationships': (1, 1),
            'tostring': (1, 1),
            'tointeger': (1, 1),
            'tofloat': (1, 1),
            'toboolean': (1, 1),
            'split': (2, 2),
            'substring': (2, 3),
            'replace': (3, 3),
            'trim': (1, 1),
            'upper': (1, 1),
            'lower': (1, 1)
        }
        
        if func_name in function_signatures:
            min_args, max_args = function_signatures[func_name]
            if arg_count < min_args or arg_count > max_args:
                if min_args == max_args:
                    set_error(ErrorCode.INVALID_ARGUMENT_COUNT,
                             f"Function '{func_name}' expects {min_args} arguments, got {arg_count}")
                else:
                    set_error(ErrorCode.INVALID_ARGUMENT_COUNT,
                             f"Function '{func_name}' expects {min_args}-{max_args} arguments, got {arg_count}")
                return False
        else:
            set_warning(ErrorCode.UNKNOWN_FUNCTION,
                       f"Unknown function: '{func_name}'")
                       
        return True
        
    def _validate_type_consistency(self, ast: Query) -> bool:
        """Validate type consistency in expressions"""
        valid = True
        
        # This is a simplified type checking - a full implementation would
        # require more sophisticated type inference
        
        if ast.where_clause:
            if not self._validate_expression_types(ast.where_clause.condition):
                valid = False
                
        return valid
        
    def _validate_expression_types(self, expr: Expression) -> bool:
        """Validate types in expressions"""
        valid = True
        
        if isinstance(expr, BinaryExpression):
            # Check operator compatibility
            if expr.operator in ['=', '<>', '!=', '<', '>', '<=', '>=']:
                # Comparison operators
                left_type = self._infer_expression_type(expr.left)
                right_type = self._infer_expression_type(expr.right)
                
                if left_type and right_type and not self._types_compatible(left_type, right_type):
                    set_warning(ErrorCode.TYPE_MISMATCH,
                               f"Type mismatch in comparison: {left_type} {expr.operator} {right_type}")
                               
            elif expr.operator in ['+', '-', '*', '/', '%']:
                # Arithmetic operators
                left_type = self._infer_expression_type(expr.left)
                right_type = self._infer_expression_type(expr.right)
                
                if left_type and right_type:
                    if expr.operator == '+':
                        # Addition can work with numbers or strings
                        if not ((left_type in ['int', 'float'] and right_type in ['int', 'float']) or
                               (left_type == 'string' and right_type == 'string')):
                            set_warning(ErrorCode.TYPE_MISMATCH,
                                       "Addition requires numeric or string operands")
                    else:
                        # Other arithmetic requires numbers
                        if not (left_type in ['int', 'float'] and right_type in ['int', 'float']):
                            set_warning(ErrorCode.TYPE_MISMATCH,
                                       f"Operator '{expr.operator}' requires numeric operands")
                                       
            # Recursively validate sub-expressions
            if not self._validate_expression_types(expr.left):
                valid = False
            if not self._validate_expression_types(expr.right):
                valid = False
                
        return valid
        
    def _infer_expression_type(self, expr: Expression) -> Optional[str]:
        """Infer the type of an expression"""
        if isinstance(expr, LiteralExpression):
            if isinstance(expr.value, int):
                return 'int'
            elif isinstance(expr.value, float):
                return 'float'
            elif isinstance(expr.value, str):
                return 'string'
            elif isinstance(expr.value, bool):
                return 'boolean'
            elif expr.value is None:
                return 'null'
        elif isinstance(expr, VariableExpression):
            # In a full implementation, we'd track variable types
            return 'unknown'
        elif isinstance(expr, PropertyExpression):
            # Properties can be any type
            return 'unknown'
        elif isinstance(expr, FunctionCall):
            # Return type depends on function
            return self._get_function_return_type(expr.name)
            
        return None
        
    def _get_function_return_type(self, func_name: str) -> str:
        """Get the return type of a function"""
        func_name = func_name.lower()
        
        type_map = {
            'count': 'int',
            'sum': 'number',
            'avg': 'float',
            'min': 'unknown',  # Depends on input type
            'max': 'unknown',  # Depends on input type
            'collect': 'list',
            'size': 'int',
            'length': 'int',
            'exists': 'boolean',
            'type': 'string',
            'id': 'int',
            'labels': 'list',
            'keys': 'list',
            'tostring': 'string',
            'tointeger': 'int',
            'tofloat': 'float',
            'toboolean': 'boolean'
        }
        
        return type_map.get(func_name, 'unknown')
        
    def _types_compatible(self, type1: str, type2: str) -> bool:
        """Check if two types are compatible for comparison"""
        if type1 == type2:
            return True
            
        # Numbers are compatible with each other
        if type1 in ['int', 'float'] and type2 in ['int', 'float']:
            return True
            
        # NULL is compatible with anything
        if type1 == 'null' or type2 == 'null':
            return True
            
        return False