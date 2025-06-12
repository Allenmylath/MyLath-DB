# ==========================================
# cypher_planner/cost_model.py
# ==========================================
from .physical_planner import RedisOperation, GraphBLASOperation

"""
Cost Model for Redis + GraphBLAS Operations
"""


class CostModel:
    """Cost model for estimating execution costs of Redis and GraphBLAS operations"""

    def __init__(self):
        # Cost constants (these would be calibrated based on actual system performance)
        self.redis_costs = {
            "SMEMBERS": 1.0,  # Cost per set member operation
            "HGET": 0.1,  # Cost per hash get
            "SCAN": 10.0,  # Cost for scanning operations
            "SINTER": 2.0,  # Cost per set intersection
        }

        self.graphblas_costs = {
            "mxv": 1.0,  # Matrix-vector multiplication cost per nnz
            "mxm": 2.0,  # Matrix-matrix multiplication cost per nnz
            "element_wise": 0.1,  # Element-wise operations
        }

    def estimate_redis_cost(
        self, operation: "RedisOperation", cardinality: int = 1000
    ) -> float:
        """Estimate cost of Redis operations"""
        total_cost = 0.0

        for command in operation.redis_commands:
            if "SMEMBERS" in command:
                total_cost += self.redis_costs["SMEMBERS"] * cardinality
            elif "HGET" in command:
                total_cost += self.redis_costs["HGET"] * cardinality
            elif "SCAN" in command:
                total_cost += self.redis_costs["SCAN"] * cardinality
            elif "SINTER" in command:
                total_cost += self.redis_costs["SINTER"] * cardinality

        return total_cost

    def estimate_graphblas_cost(
        self,
        operation: "GraphBLASOperation",
        matrix_nnz: int = 1000000,
        vector_nnz: int = 1000,
    ) -> float:
        """Estimate cost of GraphBLAS operations"""
        total_cost = 0.0

        for matrix_op in operation.matrix_operations:
            if "@" in matrix_op and "v_" in matrix_op:
                # Matrix-vector multiplication
                total_cost += self.graphblas_costs["mxv"] * matrix_nnz
            elif "@" in matrix_op:
                # Matrix-matrix multiplication
                total_cost += self.graphblas_costs["mxm"] * matrix_nnz
            else:
                # Element-wise operations
                total_cost += self.graphblas_costs["element_wise"] * vector_nnz

        return total_cost
