# STEP 8: Update cypher_planner/__init__.py

"""
Enhanced Cypher Planner with FalkorDB-inspired improvements
"""

# Core components
from .parser import CypherParser
from .logical_planner import LogicalPlanner
from .optimizer import RuleBasedOptimizer
from .physical_planner import PhysicalPlanner, print_physical_plan

# Enhanced components
from .logical_operators import (
    # Base classes
    LogicalOperator,
    
    # Enhanced scan operations
    NodeByLabelScan,
    AllNodeScan,
    PropertyScan,
    
    # Enhanced traversal operations
    ConditionalTraverse,
    ConditionalVarLenTraverse,
    
    # Apply family operations
    Apply,
    SemiApply,
    ApplyMultiplexer,
    Optional,
    
    # Enhanced filter operations
    PropertyFilter,
    StructuralFilter,
    PathFilter,
    
    # Legacy operations (backward compatibility)
    NodeScan,
    Expand,
    Filter,
    Project,
    Aggregate,
    OrderBy,
    Limit,
    Distinct,
    Join,
    
    # Utility functions
    print_plan,
    analyze_plan_execution_targets,
    is_aware_of_variables,
    find_earliest_aware_operator
)

from .filter_placement import (
    FilterOptimizer,
    FilterPlacementEngine,
    FilterNode,
    FilterType
)

from .execution_statistics import (
    ExecutionStatistics,
    StatisticsCollector
)

# AST nodes
from .ast_nodes import *

# Utilities
from .utils import *

__version__ = "2.0.0"
__all__ = [
    # Core components
    'CypherParser',
    'LogicalPlanner', 
    'RuleBasedOptimizer',
    'PhysicalPlanner',
    'print_physical_plan',
    
    # Enhanced operators
    'LogicalOperator',
    'NodeByLabelScan',
    'AllNodeScan',
    'PropertyScan',
    'ConditionalTraverse',
    'ConditionalVarLenTraverse',
    'Apply',
    'SemiApply',
    'ApplyMultiplexer',
    'Optional',
    'PropertyFilter',
    'StructuralFilter',
    'PathFilter',
    
    # Legacy operators
    'NodeScan',
    'Expand', 
    'Filter',
    'Project',
    'Aggregate',
    'OrderBy',
    'Limit',
    'Distinct',
    'Join',
    
    # Enhanced components
    'FilterOptimizer',
    'FilterPlacementEngine',
    'FilterNode',
    'FilterType',
    'ExecutionStatistics',
    'StatisticsCollector',
    
    # Utility functions
    'print_plan',
    'analyze_plan_execution_targets',
    'is_aware_of_variables',
    'find_earliest_aware_operator'
]