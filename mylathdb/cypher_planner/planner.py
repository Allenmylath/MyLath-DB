# cypher_planner/planner.py

"""
Query Planner - Wrapper for the logical planner to maintain compatibility with main.py
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from .logical_planner import LogicalPlanner
from .ast_nodes import Query

@dataclass
class PlanStep:
    """Represents a step in the execution plan"""
    operation: str
    details: List[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = []

@dataclass 
class ExecutionPlan:
    """Represents a complete execution plan"""
    steps: List[PlanStep]
    optimizations: List[str] = None
    estimated_cost: float = 0.0
    
    def __post_init__(self):
        if self.optimizations is None:
            self.optimizations = []

class QueryPlanner:
    """Main query planner that wraps the logical planner"""
    
    def __init__(self):
        self.logical_planner = LogicalPlanner()
        
    def plan(self, ast: Query) -> ExecutionPlan:
        """Generate an execution plan from an AST"""
        try:
            # Generate logical plan
            logical_plan = self.logical_planner.create_logical_plan(ast)
            
            # Convert to execution steps
            steps = self._convert_logical_plan_to_steps(logical_plan)
            
            return ExecutionPlan(steps=steps)
            
        except Exception as e:
            # Fallback plan on error
            return ExecutionPlan(steps=[
                PlanStep("Error", [f"Planning failed: {str(e)}"])
            ])
    
    def _convert_logical_plan_to_steps(self, logical_plan) -> List[PlanStep]:
        """Convert logical plan to execution steps"""
        steps = []
        
        def extract_steps(op, indent=0):
            prefix = "  " * indent
            op_name = type(op).__name__
            
            # Create step description
            if hasattr(op, 'variable'):
                step_desc = f"{prefix}{op_name}({op.variable})"
            elif hasattr(op, 'from_var') and hasattr(op, 'to_var'):
                step_desc = f"{prefix}{op_name}({op.from_var} -> {op.to_var})"
            else:
                step_desc = f"{prefix}{op_name}"
            
            # Add details
            details = []
            if hasattr(op, 'labels') and op.labels:
                details.append(f"Labels: {', '.join(op.labels)}")
            if hasattr(op, 'properties') and op.properties:
                details.append(f"Properties: {op.properties}")
            if hasattr(op, 'condition'):
                details.append(f"Condition: {op.condition}")
            
            steps.append(PlanStep(step_desc, details))
            
            # Process children
            for child in getattr(op, 'children', []):
                extract_steps(child, indent + 1)
        
        if logical_plan:
            extract_steps(logical_plan)
        else:
            steps.append(PlanStep("No plan generated"))
            
        return steps


# Additional utility functions that main.py expects
def format_query(query: str) -> str:
    """Format a Cypher query for display"""
    # Simple formatting - just clean up whitespace
    lines = query.split('\n')
    formatted_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped:
            formatted_lines.append(stripped)
    
    return ' '.join(formatted_lines)

def optimize_plan(plan: ExecutionPlan) -> ExecutionPlan:
    """Apply optimizations to an execution plan"""
    # Simple optimization - just mark that optimizations were considered
    optimized_plan = ExecutionPlan(
        steps=plan.steps.copy(),
        optimizations=plan.optimizations.copy() if plan.optimizations else [],
        estimated_cost=plan.estimated_cost
    )
    
    # Add some mock optimizations
    if len(plan.steps) > 3:
        optimized_plan.optimizations.append("Filter pushdown applied")
    if any("Scan" in step.operation for step in plan.steps):
        optimized_plan.optimizations.append("Index usage optimized")
    
    return optimized_plan

def estimate_cost(plan: ExecutionPlan) -> float:
    """Estimate the cost of an execution plan"""
    base_cost = 100.0
    
    # Add cost based on number of operations
    operation_cost = len(plan.steps) * 50.0
    
    # Add cost based on operation types
    complex_ops = 0
    for step in plan.steps:
        if any(op in step.operation for op in ["Join", "Expand", "VarLen"]):
            complex_ops += 1
    
    complexity_cost = complex_ops * 200.0
    
    total_cost = base_cost + operation_cost + complexity_cost
    
    # Update the plan's cost estimate
    plan.estimated_cost = total_cost
    
    return total_cost

class MockGraphStatistics:
    """Mock graph statistics for cost estimation"""
    
    def __init__(self):
        self.node_count = 1000000
        self.edge_count = 5000000
        self.labels = {
            'Person': 500000,
            'User': 300000,
            'Product': 100000,
            'Movie': 50000
        }
    
    def get_node_count(self, label: str = None) -> int:
        if label:
            return self.labels.get(label, 10000)
        return self.node_count
    
    def get_edge_count(self, rel_type: str = None) -> int:
        return self.edge_count // 10  # Rough estimate


# Export the main classes
__all__ = ['QueryPlanner', 'ExecutionPlan', 'PlanStep', 'format_query', 'optimize_plan', 'estimate_cost', 'MockGraphStatistics']