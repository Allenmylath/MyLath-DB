# Add at the top of physical_planner.py  
from __future__ import annotations
from typing import List, Dict, Any, Optional, Set, Union
from .logical_operators import *
from .execution_statistics import ExecutionStatistics

class PhysicalOperation:
    """Enhanced base class for physical operations"""
    def __init__(self, operation_type: str, target: str, logical_op: LogicalOperator = None):
        self.operation_type = operation_type
        self.target = target  # "redis", "graphblas", "coordinator"
        self.logical_op = logical_op
        self.estimated_cost = 0.0
        self.estimated_cardinality = 0
        self.children: List['PhysicalOperation'] = []
        self.parallelizable = False
        self.memory_intensive = False

class RedisOperation(PhysicalOperation):
    """Enhanced Redis operation with specific command patterns"""
    def __init__(self, operation_type: str, redis_commands: List[str], 
                 index_usage: List[str] = None, logical_op: LogicalOperator = None):
        super().__init__(operation_type, "redis", logical_op)
        self.redis_commands = redis_commands
        self.index_usage = index_usage or []
        self.batch_size = 1000
        self.pipeline_enabled = True

class GraphBLASOperation(PhysicalOperation):
    """Enhanced GraphBLAS operation with matrix specifics"""
    def __init__(self, operation_type: str, matrix_ops: List[str],
                 matrix_properties: Dict[str, Any] = None, logical_op: LogicalOperator = None):
        super().__init__(operation_type, "graphblas", logical_op)
        self.matrix_operations = matrix_ops
        self.matrix_properties = matrix_properties or {}
        self.parallelizable = True
        self.memory_intensive = True
        self.sparsity_ratio = 0.01

class CoordinatorOperation(PhysicalOperation):
    """Operation that coordinates between Redis and GraphBLAS"""
    def __init__(self, operation_type: str, coordination_pattern: str,
                 data_transfer_ops: List[str] = None, logical_op: LogicalOperator = None):
        super().__init__(operation_type, "coordinator", logical_op)
        self.coordination_pattern = coordination_pattern
        self.data_transfer_ops = data_transfer_ops or []

class PhysicalPlanner:
    """ENHANCED: Physical planner with advanced optimization"""
    
    def __init__(self, statistics: ExecutionStatistics = None):
        self.statistics = statistics or ExecutionStatistics()
        self.redis_indexes = {
            'label_indexes': {'Person', 'User', 'Actor', 'Movie', 'Product'},
            'property_indexes': {'age', 'country', 'name', 'price', 'make'},
            'composite_indexes': {'Person.age', 'User.country', 'Product.price'}
        }
    
    def create_physical_plan(self, logical_plan: LogicalOperator) -> PhysicalOperation:
        """Create optimized physical execution plan"""
        return self._convert_operator(logical_plan)
    
    def _convert_operator(self, logical_op: LogicalOperator) -> PhysicalOperation:
        """Convert a single logical operator to physical operation(s)"""
        
        # NEW: Handle enhanced operators
        if isinstance(logical_op, NodeByLabelScan):
            return self._convert_node_by_label_scan(logical_op)
        elif isinstance(logical_op, AllNodeScan):
            return self._convert_all_node_scan(logical_op)
        elif isinstance(logical_op, PropertyScan):
            return self._convert_property_scan(logical_op)
        elif isinstance(logical_op, ConditionalTraverse):
            return self._convert_conditional_traverse(logical_op)
        elif isinstance(logical_op, ConditionalVarLenTraverse):
            return self._convert_var_len_traverse(logical_op)
        elif isinstance(logical_op, PropertyFilter):
            return self._convert_property_filter(logical_op)
        elif isinstance(logical_op, StructuralFilter):
            return self._convert_structural_filter(logical_op)
        elif isinstance(logical_op, PathFilter):
            return self._convert_path_filter(logical_op)
        elif isinstance(logical_op, Apply):
            return self._convert_apply(logical_op)
        elif isinstance(logical_op, SemiApply):
            return self._convert_semi_apply(logical_op)
        elif isinstance(logical_op, Optional):
            return self._convert_optional(logical_op)
        
        # EXISTING: Handle original operators
        elif isinstance(logical_op, NodeScan):
            return self._convert_node_scan(logical_op)
        elif isinstance(logical_op, Expand):
            return self._convert_expand(logical_op)
        elif isinstance(logical_op, Filter):
            return self._convert_filter(logical_op)
        elif isinstance(logical_op, Project):
            return self._convert_project(logical_op)
        elif isinstance(logical_op, OrderBy):
            return self._convert_order_by(logical_op)
        elif isinstance(logical_op, Limit):
            return self._convert_limit(logical_op)
        elif isinstance(logical_op, Join):
            return self._convert_join(logical_op)
        else:
            # Default conversion
            return self._convert_generic(logical_op)

    # NEW CONVERSION METHODS:
    
    def _convert_node_by_label_scan(self, op: NodeByLabelScan) -> RedisOperation:
        """Convert NodeByLabelScan with index optimization"""
        redis_commands = []
        index_usage = []
        
        # Use label index
        if op.label in self.redis_indexes['label_indexes']:
            redis_commands.append(f"SMEMBERS label:{op.label}")
            index_usage.append(f"label_index:{op.label}")
        else:
            redis_commands.append(f"SCAN 0 MATCH node:{op.label}:*")
        
        # Add property filters if present
        if op.properties:
            for prop_key, prop_value in op.properties.items():
                composite_key = f"{op.label}.{prop_key}"
                if composite_key in self.redis_indexes['composite_indexes']:
                    redis_commands.append(f"SMEMBERS idx:{composite_key}:{prop_value}")
                    index_usage.append(f"composite_index:{composite_key}")
                else:
                    redis_commands.append(f"HGET node:{{id}} {prop_key}")
        
        redis_op = RedisOperation("NodeByLabelScan", redis_commands, index_usage, op)
        cardinality = self.statistics.get_label_cardinality(op.label)
        redis_op.estimated_cardinality = cardinality
        
        # Add children
        for child in op.children:
            redis_op.children.append(self._convert_operator(child))
        
        return redis_op
    
    def _convert_conditional_traverse(self, op: ConditionalTraverse) -> GraphBLASOperation:
        """Convert ConditionalTraverse to optimized GraphBLAS operations"""
        matrix_ops = []
        matrix_props = {}
        
        rel_type = op.rel_types[0] if op.rel_types else "ANY"
        
        # Generate matrix operation based on direction
        if op.direction == "outgoing":
            matrix_ops.append(f"v_{op.to_var} = v_{op.from_var} @ A_{rel_type}")
        elif op.direction == "incoming":
            matrix_ops.append(f"v_{op.to_var} = v_{op.from_var} @ A_{rel_type}.T")
        else:  # bidirectional
            matrix_ops.append(f"v_{op.to_var} = v_{op.from_var} @ (A_{rel_type} + A_{rel_type}.T)")
        
        # Add optimization hints
        edge_cardinality = self.statistics.get_edge_cardinality(rel_type)
        if edge_cardinality > 100000:
            matrix_ops.append("# Use sparse matrix multiplication with threading")
            matrix_props['threading'] = True
        
        gb_op = GraphBLASOperation("ConditionalTraverse", matrix_ops, matrix_props, op)
        gb_op.estimated_cardinality = edge_cardinality // 1000
        
        # Add children
        for child in op.children:
            gb_op.children.append(self._convert_operator(child))
        
        return gb_op
    
    def _convert_var_len_traverse(self, op: ConditionalVarLenTraverse) -> GraphBLASOperation:
        """Convert variable-length traversal to GraphBLAS operations"""
        matrix_ops = []
        matrix_props = {}
        
        rel_type = op.rel_types[0] if op.rel_types else "ANY"
        
        if op.max_length == float('inf'):
            # Transitive closure
            matrix_ops.extend([
                f"# Compute transitive closure for {op.from_var} -> {op.to_var}",
                f"A = adjacency_matrix['{rel_type}']",
                f"if {op.direction} == 'incoming': A = A.T",
                f"result = compute_transitive_closure(v_{op.from_var}, A, min_length={op.min_length})"
            ])
            matrix_props['algorithm'] = 'transitive_closure'
        else:
            # Bounded variable-length path
            matrix_ops.extend([
                f"# Variable-length path {op.min_length}..{op.max_length}",
                f"A = adjacency_matrix['{rel_type}']",
                f"current = v_{op.from_var}",
                f"result = create_empty_vector()",
                f"for length in range({op.min_length}, {op.max_length + 1}):",
                f"    if length > 1: current = current @ A",
                f"    if length >= {op.min_length}: result = result + current"
            ])
            matrix_props['algorithm'] = 'bounded_varlen'
        
        gb_op = GraphBLASOperation("VarLenTraverse", matrix_ops, matrix_props, op)
        gb_op.estimated_cardinality = 1000 * op.max_length if op.max_length != float('inf') else 5000
        gb_op.memory_intensive = True
        
        # Add children
        for child in op.children:
            gb_op.children.append(self._convert_operator(child))
        
        return gb_op
    
    def _convert_property_filter(self, op: PropertyFilter) -> RedisOperation:
        """Convert PropertyFilter with index optimization"""
        redis_commands = []
        index_usage = []
        
        # Check for property index
        if op.property_key in self.redis_indexes['property_indexes']:
            if op.operator == "=":
                redis_commands.append(f"SMEMBERS prop:{op.property_key}:{op.value}")
                index_usage.append(f"property_index:{op.property_key}")
            elif op.operator in [">", ">=", "<", "<="]:
                redis_commands.append(f"ZRANGEBYSCORE prop_sorted:{op.property_key} {self._format_range(op.operator, op.value)}")
                index_usage.append(f"sorted_index:{op.property_key}")
        else:
            # Property scan without index
            redis_commands.extend([
                f"# Filter property {op.property_key} on {op.variable}",
                f"HGET node:{{id}} {op.property_key}"
            ])
        
        redis_op = RedisOperation("PropertyFilter", redis_commands, index_usage, op)
        
        # Estimate selectivity
        selectivity = self.statistics.get_property_selectivity(op.variable, op.property_key)
        redis_op.estimated_cardinality = int(1000 * selectivity)
        
        # Add children
        for child in op.children:
            redis_op.children.append(self._convert_operator(child))
        
        return redis_op
    
    def _convert_semi_apply(self, op: SemiApply) -> CoordinatorOperation:
        """Convert SemiApply to coordinated execution"""
        coordination_pattern = "semi_apply_exists_check"
        data_transfer_ops = [
            "# Semi-apply pattern for EXISTS-style filtering",
            "left_results = execute_left_branch()",
            "filtered_results = []",
            "for record in left_results:",
            "    exists = execute_right_branch_with_context(record)",
            f"    if {'not ' if op.anti else ''}exists:",
            "        filtered_results.append(record)",
            "return filtered_results"
        ]
        
        coord_op = CoordinatorOperation("SemiApply", coordination_pattern, data_transfer_ops, op)
        coord_op.estimated_cardinality = int(1000 * (0.3 if not op.anti else 0.7))
        
        # Add children
        for child in op.children:
            coord_op.children.append(self._convert_operator(child))
        
        return coord_op
    
    # NEW HELPER METHODS:
    
    def _convert_all_node_scan(self, op: AllNodeScan) -> RedisOperation:
        """Convert AllNodeScan"""
        redis_commands = ["SCAN 0 MATCH node:*"]
        redis_op = RedisOperation("AllNodeScan", redis_commands, [], op)
        redis_op.estimated_cardinality = self.statistics.node_count
        
        for child in op.children:
            redis_op.children.append(self._convert_operator(child))
        
        return redis_op
    
    def _convert_property_scan(self, op: PropertyScan) -> RedisOperation:
        """Convert PropertyScan"""
        redis_commands = [f"SMEMBERS prop:{op.property_key}:{op.property_value}"]
        index_usage = [f"property_index:{op.property_key}"]
        redis_op = RedisOperation("PropertyScan", redis_commands, index_usage, op)
        
        selectivity = self.statistics.get_property_selectivity("", op.property_key)
        redis_op.estimated_cardinality = int(self.statistics.node_count * selectivity)
        
        for child in op.children:
            redis_op.children.append(self._convert_operator(child))
        
        return redis_op
    
    def _convert_structural_filter(self, op: StructuralFilter) -> GraphBLASOperation:
        """Convert StructuralFilter to GraphBLAS operations"""
        matrix_ops = [f"# Structural filter: {op.condition}"]
        gb_op = GraphBLASOperation("StructuralFilter", matrix_ops, {}, op)
        
        for child in op.children:
            gb_op.children.append(self._convert_operator(child))
        
        return gb_op
    
    def _convert_path_filter(self, op: PathFilter) -> GraphBLASOperation:
        """Convert PathFilter to GraphBLAS path matching"""
        matrix_ops = [
            f"# Path filter: {'NOT ' if op.anti else ''}{op.path_pattern}",
            "# Use path pattern matching algorithms"
        ]
        gb_op = GraphBLASOperation("PathFilter", matrix_ops, {'path_pattern': op.path_pattern}, op)
        
        for child in op.children:
            gb_op.children.append(self._convert_operator(child))
        
        return gb_op
    
    def _convert_apply(self, op: Apply) -> CoordinatorOperation:
        """Convert Apply operation"""
        coordination_pattern = "correlated_subquery"
        data_transfer_ops = ["# Apply correlated subquery execution"]
        coord_op = CoordinatorOperation("Apply", coordination_pattern, data_transfer_ops, op)
        
        for child in op.children:
            coord_op.children.append(self._convert_operator(child))
        
        return coord_op
    
    def _convert_optional(self, op: Optional) -> CoordinatorOperation:
        """Convert Optional operation"""
        coordination_pattern = "left_outer_join"
        data_transfer_ops = ["# Optional match with NULL handling"]
        coord_op = CoordinatorOperation("Optional", coordination_pattern, data_transfer_ops, op)
        
        for child in op.children:
            coord_op.children.append(self._convert_operator(child))
        
        return coord_op
    
    def _format_range(self, operator: str, value: Any) -> str:
        """Format range for Redis ZRANGEBYSCORE"""
        if operator == '>':
            return f"({value} +inf"
        elif operator == '>=':
            return f"{value} +inf"
        elif operator == '<':
            return f"-inf ({value}"
        elif operator == '<=':
            return f"-inf {value}"
        return f"{value} {value}"
    
    # EXISTING METHODS (keep all your existing conversion methods):
    
    def _convert_node_scan(self, node_scan: NodeScan) -> RedisOperation:
        """Convert NodeScan to Redis operations"""
        redis_commands = []
        
        if node_scan.labels:
            for label in node_scan.labels:
                redis_commands.append(f"SMEMBERS label:{label}")
        
        if node_scan.properties:
            for prop_key, prop_value in node_scan.properties.items():
                redis_commands.append(f"SMEMBERS prop:{prop_key}:{prop_value}")
        
        if node_scan.labels and node_scan.properties:
            sets_to_intersect = [f"label:{label}" for label in node_scan.labels]
            sets_to_intersect.extend([f"prop:{k}:{v}" for k, v in node_scan.properties.items()])
            redis_commands.append(f"SINTER {' '.join(sets_to_intersect)}")
        
        if not redis_commands:
            redis_commands.append("SCAN 0 MATCH node:*")
        
        redis_op = RedisOperation("NodeScan", redis_commands)
        
        for child in node_scan.children:
            redis_op.children.append(self._convert_operator(child))
        
        return redis_op
    
    def _convert_expand(self, expand: Expand) -> PhysicalOperation:
        """Convert Expand to GraphBLAS matrix operations"""
        matrix_ops = []
        
        if expand.max_length == 1:
            rel_type = expand.rel_types[0] if expand.rel_types else "*"
            
            if expand.direction == "outgoing":
                matrix_ops.append(f"v_{expand.to_var} = v_{expand.from_var} @ A_{rel_type}")
            elif expand.direction == "incoming":
                matrix_ops.append(f"v_{expand.to_var} = v_{expand.from_var} @ A_{rel_type}.T")
            else:  # bidirectional
                matrix_ops.append(f"v_{expand.to_var} = v_{expand.from_var} @ (A_{rel_type} + A_{rel_type}.T)")
        else:
            rel_type = expand.rel_types[0] if expand.rel_types else "*"
            matrix_ops.append(f"# Variable length path {expand.min_length}..{expand.max_length}")
            
            if expand.max_length == float("inf"):
                matrix_ops.append(f"result = compute_transitive_closure(v_{expand.from_var}, A_{rel_type}, {expand.min_length})")
            else:
                matrix_ops.append(f"result = compute_variable_path(v_{expand.from_var}, A_{rel_type}, {expand.min_length}, {expand.max_length})")
        
        graphblas_op = GraphBLASOperation("Expand", matrix_ops)
        
        for child in expand.children:
            graphblas_op.children.append(self._convert_operator(child))
        
        return graphblas_op
    
    def _convert_filter(self, filter_op: Filter) -> PhysicalOperation:
        """Convert Filter based on filter type"""
        if filter_op.filter_type == "property":
            redis_commands = [f"# Apply property filter: {filter_op.condition}"]
            
            if isinstance(filter_op.condition, BinaryExpression):
                if isinstance(filter_op.condition.left, PropertyExpression):
                    prop_expr = filter_op.condition.left
                    redis_commands.append(f"HGET node:{{id}} {prop_expr.property_name}")
            
            redis_op = RedisOperation("PropertyFilter", redis_commands)
        else:
            redis_op = PhysicalOperation("Filter", "mixed")
        
        for child in filter_op.children:
            redis_op.children.append(self._convert_operator(child))
        
        return redis_op
    
    def _convert_project(self, project: Project) -> RedisOperation:
        """Convert Project operation"""
        redis_commands = []
        
        for expr, alias in project.projections:
            if hasattr(expr, 'property_name'):  # PropertyExpression
                redis_commands.append(f"HGET node:{{id}} {expr.property_name}")
            elif hasattr(expr, 'name'):  # VariableExpression
                redis_commands.append(f"# Return variable: {expr.name}")
        
        if not redis_commands:
            redis_commands.append("# Project selected columns")
        
        redis_op = RedisOperation("Project", redis_commands)
        
        for child in project.children:
            redis_op.children.append(self._convert_operator(child))
        
        return redis_op
    
    def _convert_order_by(self, order_by: OrderBy) -> RedisOperation:
        """Convert OrderBy operation"""
        redis_commands = ["# Sort results"]
        for expr, ascending in order_by.sort_items:
            direction = "ASC" if ascending else "DESC"
            if hasattr(expr, 'variable') and hasattr(expr, 'property_name'):
                redis_commands.append(f"SORT BY {expr.variable}.{expr.property_name} {direction}")
        
        redis_op = RedisOperation("OrderBy", redis_commands)
        
        for child in order_by.children:
            redis_op.children.append(self._convert_operator(child))
        
        return redis_op
    
    def _convert_limit(self, limit: Limit) -> RedisOperation:
        """Convert Limit operation"""
        redis_commands = []
        if limit.skip > 0:
            redis_commands.append(f"SKIP {limit.skip}")
        if limit.count != float("inf"):
            redis_commands.append(f"LIMIT {limit.count}")
        
        if not redis_commands:
            redis_commands.append("# No limit applied")
        
        redis_op = RedisOperation("Limit", redis_commands)
        
        for child in limit.children:
            redis_op.children.append(self._convert_operator(child))
        
        return redis_op
    
    def _convert_join(self, join: Join) -> PhysicalOperation:
        """Convert Join operation"""
        physical_op = PhysicalOperation("Join", "mixed")
        
        for child in join.children:
            physical_op.children.append(self._convert_operator(child))
        
        return physical_op
    
    def _convert_generic(self, op: LogicalOperator) -> PhysicalOperation:
        """Generic conversion for unknown operators"""
        physical_op = PhysicalOperation(type(op).__name__, "mixed", op)
        physical_op.estimated_cardinality = 1000
        
        for child in op.children:
            physical_op.children.append(self._convert_operator(child))
        
        return physical_op

# Enhanced printing function
def print_physical_plan(physical_op: PhysicalOperation, indent: int = 0) -> None:
    """Enhanced pretty printing for physical plans"""
    prefix = "  " * indent
    
    if isinstance(physical_op, RedisOperation):
        print(f"{prefix}[Redis] {physical_op.operation_type} (card: ~{physical_op.estimated_cardinality})")
        if physical_op.index_usage:
            print(f"{prefix}  📊 Indexes: {', '.join(physical_op.index_usage)}")
        for cmd in physical_op.redis_commands[:3]:
            print(f"{prefix}  > {cmd}")
        if len(physical_op.redis_commands) > 3:
            print(f"{prefix}  > ... ({len(physical_op.redis_commands)-3} more commands)")
            
    elif isinstance(physical_op, GraphBLASOperation):
        print(f"{prefix}[GraphBLAS] {physical_op.operation_type} (card: ~{physical_op.estimated_cardinality})")
        if physical_op.matrix_properties:
            props = [f"{k}={v}" for k, v in physical_op.matrix_properties.items()]
            print(f"{prefix}  ⚙️  Properties: {', '.join(props[:3])}")
        for op in physical_op.matrix_operations[:3]:
            print(f"{prefix}  > {op}")
        if len(physical_op.matrix_operations) > 3:
            print(f"{prefix}  > ... ({len(physical_op.matrix_operations)-3} more operations)")
            
    elif isinstance(physical_op, CoordinatorOperation):
        print(f"{prefix}[Coordinator] {physical_op.operation_type} (card: ~{physical_op.estimated_cardinality})")
        print(f"{prefix}  🔄 Pattern: {physical_op.coordination_pattern}")
        for transfer_op in physical_op.data_transfer_ops[:2]:
            print(f"{prefix}  > {transfer_op}")
        if len(physical_op.data_transfer_ops) > 2:
            print(f"{prefix}  > ... ({len(physical_op.data_transfer_ops)-2} more operations)")
    else:
        print(f"{prefix}[{physical_op.target}] {physical_op.operation_type} (card: ~{physical_op.estimated_cardinality})")

    for child in physical_op.children:
        print_physical_plan(child, indent + 1)