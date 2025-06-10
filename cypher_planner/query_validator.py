# cypher_planner/query_validator.py

"""
Query Validation System
Pre-validation checks before parsing
"""

import re
from typing import List, Set, Dict, Tuple, Optional
from .error_context import ErrorCode, set_error, set_warning, ErrorPosition

class QueryValidator:
    """Pre-parse validation for Cypher queries"""
    
    def __init__(self):
        self.reserved_keywords = {
            'MATCH', 'WHERE', 'RETURN', 'CREATE', 'MERGE', 'DELETE', 'SET', 
            'REMOVE', 'WITH', 'UNWIND', 'FOREACH', 'CALL', 'YIELD', 'UNION',
            'ORDER', 'BY', 'SKIP', 'LIMIT', 'ASC', 'DESC', 'DISTINCT',
            'OPTIONAL', 'AND', 'OR', 'NOT', 'XOR', 'IN', 'STARTS', 'ENDS',
            'CONTAINS', 'EXISTS', 'IS', 'NULL', 'TRUE', 'FALSE', 'AS'
        }
        
        self.valid_operators = {
            '=', '<>', '!=', '<', '>', '<=', '>=', '+', '-', '*', '/', '%',
            '^', '=~', 'AND', 'OR', 'NOT', 'XOR', 'IN'
        }
        
    def validate_query(self, query: str) -> bool:
        """
        Perform comprehensive pre-parse validation
        Returns True if query passes all checks
        """
        if not query or not query.strip():
            set_error(ErrorCode.MISSING_TOKEN, "Empty query provided")
            return False
            
        # Clean query for analysis
        cleaned_query = self._clean_query(query)
        
        # Run all validation checks
        checks = [
            self._validate_basic_structure,
            self._validate_balanced_delimiters,
            self._validate_clause_sequence,
            self._validate_pattern_syntax,
            self._validate_string_literals,
            self._validate_numeric_literals,
            self._validate_identifier_syntax,
            self._check_common_mistakes
        ]
        
        all_valid = True
        for check in checks:
            if not check(query, cleaned_query):
                all_valid = False
                
        return all_valid
        
    def _clean_query(self, query: str) -> str:
        """Remove comments and normalize whitespace"""
        # Remove single-line comments
        query = re.sub(r'//.*$', '', query, flags=re.MULTILINE)
        
        # Remove multi-line comments  
        query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        return query
        
    def _validate_basic_structure(self, original: str, cleaned: str) -> bool:
        """Validate basic query structure"""
        if len(cleaned) == 0:
            set_error(ErrorCode.MISSING_TOKEN, "Query is empty after removing comments")
            return False
            
        # Check for at least one main clause
        main_clauses = re.findall(r'\b(MATCH|RETURN|CREATE|MERGE|DELETE|WITH|UNWIND|CALL)\b', 
                                 cleaned, re.IGNORECASE)
        if not main_clauses:
            set_error(ErrorCode.INVALID_PATTERN, 
                     "Query must contain at least one main clause (MATCH, RETURN, CREATE, etc.)",
                     suggestion="Add a MATCH or RETURN clause")
            return False
            
        return True
        
    def _validate_balanced_delimiters(self, original: str, cleaned: str) -> bool:
        """Check for balanced parentheses, brackets, and braces"""
        delimiters = {
            '(': (')', ErrorCode.UNBALANCED_PARENTHESES, "parentheses"),
            '[': (']', ErrorCode.UNBALANCED_BRACKETS, "brackets"), 
            '{': ('}', ErrorCode.UNBALANCED_BRACES, "braces")
        }
        
        all_balanced = True
        
        for open_char, (close_char, error_code, name) in delimiters.items():
            stack = []
            in_string = False
            escape_next = False
            
            for i, char in enumerate(original):
                if escape_next:
                    escape_next = False
                    continue
                    
                if char == '\\':
                    escape_next = True
                    continue
                    
                if char in ('"', "'"):
                    in_string = not in_string
                    continue
                    
                if in_string:
                    continue
                    
                if char == open_char:
                    stack.append(i)
                elif char == close_char:
                    if not stack:
                        set_error(error_code, 
                                f"Unmatched closing {name}: '{close_char}'",
                                position=ErrorPosition(1, i+1, i))
                        all_balanced = False
                    else:
                        stack.pop()
                        
            if stack:
                pos = stack[-1]
                set_error(error_code,
                        f"Unmatched opening {name}: '{open_char}'", 
                        position=ErrorPosition(1, pos+1, pos))
                all_balanced = False
                
        return all_balanced
        
    def _validate_clause_sequence(self, original: str, cleaned: str) -> bool:
        """Validate clause ordering and dependencies"""
        # Extract clauses in order
        clause_pattern = r'\b(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|CALL|CREATE|MERGE|SET|REMOVE|DELETE|RETURN|UNION|ORDER\s+BY|SKIP|LIMIT)\b'
        clauses = re.findall(clause_pattern, cleaned, re.IGNORECASE)
        
        if not clauses:
            return True
            
        # Normalize clause names
        normalized_clauses = []
        for clause in clauses:
            clause_upper = clause.upper().strip()
            if clause_upper == 'OPTIONAL MATCH':
                normalized_clauses.append('OPTIONAL_MATCH')
            elif clause_upper == 'ORDER BY':
                normalized_clauses.append('ORDER_BY')
            else:
                normalized_clauses.append(clause_upper)
                
        # Validate sequences
        valid = True
        
        # RETURN should be last main clause (before ORDER BY, SKIP, LIMIT)
        if 'RETURN' in normalized_clauses:
            return_idx = normalized_clauses.index('RETURN')
            for i in range(return_idx + 1, len(normalized_clauses)):
                clause = normalized_clauses[i]
                if clause not in ('ORDER_BY', 'SKIP', 'LIMIT', 'UNION'):
                    set_error(ErrorCode.INVALID_PATTERN,
                            f"Invalid clause '{clause}' after RETURN",
                            suggestion="RETURN should be the last main clause")
                    valid = False
                    
        # ORDER BY should come before SKIP and LIMIT
        if 'ORDER_BY' in normalized_clauses and ('SKIP' in normalized_clauses or 'LIMIT' in normalized_clauses):
            order_by_idx = normalized_clauses.index('ORDER_BY')
            for clause in ['SKIP', 'LIMIT']:
                if clause in normalized_clauses:
                    clause_idx = normalized_clauses.index(clause)
                    if clause_idx < order_by_idx:
                        set_error(ErrorCode.INVALID_PATTERN,
                                f"ORDER BY should come before {clause}")
                        valid = False
                        
        return valid
        
    def _validate_pattern_syntax(self, original: str, cleaned: str) -> bool:
        """Validate basic pattern syntax"""
        # Check for dangling relationships
        relationship_pattern = r'(?:^|[^-])-\[[^\]]*\]-(?:[^>-]|$)'
        if re.search(relationship_pattern, cleaned):
            set_error(ErrorCode.DANGLING_RELATIONSHIP,
                     "Relationship pattern must connect two nodes",
                     suggestion="Use pattern like (a)-[r]->(b)")
            return False
            
        # Check for invalid variable length syntax
        varlen_pattern = r'\[\*[^\]]*\]'
        matches = re.finditer(varlen_pattern, cleaned)
        for match in matches:
            varlen = match.group()
            # Extract range if present
            range_match = re.search(r'\*(\d+)?\.\.(\d+)?', varlen)
            if range_match:
                start_str, end_str = range_match.groups()
                try:
                    start = int(start_str) if start_str else 1
                    end = int(end_str) if end_str else float('inf')
                    if start < 0:
                        set_error(ErrorCode.INVALID_VARIABLE_LENGTH,
                                "Variable length start cannot be negative")
                        return False
                    if end != float('inf') and start > end:
                        set_error(ErrorCode.INVALID_VARIABLE_LENGTH,
                                "Variable length start cannot be greater than end")
                        return False
                except ValueError:
                    set_error(ErrorCode.INVALID_VARIABLE_LENGTH,
                            "Invalid variable length range syntax")
                    return False
                    
        return True
        
    def _validate_string_literals(self, original: str, cleaned: str) -> bool:
        """Validate string literal syntax"""
        # Check for unterminated strings
        in_single_quote = False
        in_double_quote = False
        escape_next = False
        
        for i, char in enumerate(original):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
                
        if in_single_quote or in_double_quote:
            quote_type = "single" if in_single_quote else "double"
            set_error(ErrorCode.UNEXPECTED_TOKEN,
                     f"Unterminated {quote_type} quote in string literal")
            return False
            
        return True
        
    def _validate_numeric_literals(self, original: str, cleaned: str) -> bool:
        """Validate numeric literal syntax"""
        # Find potential numeric literals
        number_pattern = r'\b\d+\.?\d*(?:[eE][+-]?\d+)?\b'
        numbers = re.finditer(number_pattern, cleaned)
        
        for match in numbers:
            num_str = match.group()
            try:
                # Try to parse as float
                float(num_str)
            except ValueError:
                set_error(ErrorCode.UNEXPECTED_TOKEN,
                        f"Invalid numeric literal: '{num_str}'",
                        position=ErrorPosition(1, match.start()+1, match.start()))
                return False
                
        return True
        
    def _validate_identifier_syntax(self, original: str, cleaned: str) -> bool:
        """Validate identifier syntax"""
        # Find potential identifiers (simplified)
        identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        identifiers = re.finditer(identifier_pattern, cleaned)
        
        for match in identifiers:
            identifier = match.group().upper()
            
            # Check if it's a reserved keyword being used as identifier in wrong context
            if identifier in self.reserved_keywords:
                # This is a simplified check - more sophisticated analysis needed
                # for proper context checking
                context_start = max(0, match.start() - 10)
                context_end = min(len(cleaned), match.end() + 10)
                context = cleaned[context_start:context_end]
                
                # Simple heuristic: if keyword appears after : or . it might be misused
                if re.search(r'[:\.]' + re.escape(identifier.lower()), context, re.IGNORECASE):
                    set_warning(ErrorCode.INVALID_LABEL,
                              f"Reserved keyword '{identifier}' used as identifier")
                              
        return True
        
    def _check_common_mistakes(self, original: str, cleaned: str) -> bool:
        """Check for common syntax mistakes"""
        valid = True
        
        # Check for missing RETURN in queries that need it
        if re.search(r'\bMATCH\b', cleaned, re.IGNORECASE) and not re.search(r'\bRETURN\b', cleaned, re.IGNORECASE):
            # Only warn if query doesn't have other ending clauses
            if not re.search(r'\b(CREATE|MERGE|DELETE|SET|REMOVE)\b', cleaned, re.IGNORECASE):
                set_warning(ErrorCode.MISSING_TOKEN,
                          "MATCH clause without RETURN - did you forget to add RETURN?")
                          
        # Check for empty WHERE clauses
        if re.search(r'\bWHERE\s*$', cleaned, re.IGNORECASE):
            set_error(ErrorCode.MISSING_TOKEN,
                     "WHERE clause is empty",
                     suggestion="Add a condition after WHERE")
            valid = False
            
        # Check for empty RETURN clauses  
        if re.search(r'\bRETURN\s*$', cleaned, re.IGNORECASE):
            set_error(ErrorCode.MISSING_TOKEN,
                     "RETURN clause is empty", 
                     suggestion="Specify what to return")
            valid = False
            
        return valid