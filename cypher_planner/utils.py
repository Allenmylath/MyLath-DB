# ==========================================
# cypher_planner/utils.py
# ==========================================

"""
Utility functions for the Cypher Planner
"""

from typing import Dict, Any, List
from .logical_operators import LogicalOperator


def validate_query_syntax(query: str) -> bool:
    """Basic validation of Cypher query syntax"""

    # Check for required keywords
    query_upper = query.upper()

    # Must have at least MATCH or RETURN
    if "MATCH" not in query_upper and "RETURN" not in query_upper:
        return False

    # Check for balanced parentheses and brackets
    paren_count = query.count("(") - query.count(")")
    bracket_count = query.count("[") - query.count("]")
    brace_count = query.count("{") - query.count("}")

    return paren_count == 0 and bracket_count == 0 and brace_count == 0


def extract_variables_from_plan(plan: LogicalOperator) -> List[str]:
    """Extract all variable names used in a logical plan"""

    variables = set()

    def collect_variables(op: LogicalOperator):
        # Extract variables based on operator type
        if hasattr(op, "variable"):
            variables.add(op.variable)
        if hasattr(op, "from_var"):
            variables.add(op.from_var)
        if hasattr(op, "to_var"):
            variables.add(op.to_var)
        if hasattr(op, "rel_var") and op.rel_var:
            variables.add(op.rel_var)

        # Recursively process children
        for child in op.children:
            collect_variables(child)

    collect_variables(plan)
    return list(variables)


def estimate_plan_complexity(plan: LogicalOperator) -> Dict[str, int]:
    """Estimate the complexity of a logical plan"""

    complexity = {
        "total_operators": 0,
        "scan_operations": 0,
        "expand_operations": 0,
        "filter_operations": 0,
        "join_operations": 0,
        "max_depth": 0,
    }

    def analyze_operator(op: LogicalOperator, depth: int = 0):
        complexity["total_operators"] += 1
        complexity["max_depth"] = max(complexity["max_depth"], depth)

        # Count by operator type
        op_name = type(op).__name__
        if "Scan" in op_name:
            complexity["scan_operations"] += 1
        elif "Expand" in op_name:
            complexity["expand_operations"] += 1
        elif "Filter" in op_name:
            complexity["filter_operations"] += 1
        elif "Join" in op_name:
            complexity["join_operations"] += 1

        # Recursively analyze children
        for child in op.children:
            analyze_operator(child, depth + 1)

    analyze_operator(plan)
    return complexity


def format_plan_summary(plan: LogicalOperator) -> str:
    """Generate a human-readable summary of the execution plan"""

    variables = extract_variables_from_plan(plan)
    complexity = estimate_plan_complexity(plan)

    summary = f"""
Execution Plan Summary:
=====================
Variables: {', '.join(variables)}
Total Operators: {complexity['total_operators']}
Plan Depth: {complexity['max_depth']}
Scans: {complexity['scan_operations']}
Expansions: {complexity['expand_operations']}
Filters: {complexity['filter_operations']}
Joins: {complexity['join_operations']}
"""

    return summary.strip()
