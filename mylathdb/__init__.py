# mylathdb/__init__.py

"""
MyLathDB - Graph Database with Cypher Support
A complete graph database implementation with query planning and execution
"""

# Core Cypher Parser and Planner
from .cypher_planner import (
    # Parser
    CypherParser, CypherTokenizer, parse_cypher_query,
    validate_cypher_syntax, get_parse_errors,
    
    # AST Nodes
    Query, MatchClause, WhereClause, ReturnClause,
    Pattern, NodePattern, RelationshipPattern,
    Expression, ReturnItem,
    
    # Logical Planning
    LogicalPlanner, RuleBasedOptimizer, PhysicalPlanner,
    
    # Query Planning
    QueryPlanner, ExecutionPlan, PlanStep,
    
    # Logical Operators
    NodeScan, Expand, Filter, Project, Join,
    NodeByLabelScan, PropertyScan, ConditionalTraverse,
    
    # Errors
    ParseError, LexerError
)

# Execution Engine
from .execution_engine import (
    # Core Engine
    ExecutionEngine, ExecutionResult, ExecutionContext,
    
    # Executors
    RedisExecutor, GraphBLASExecutor, ExecutionCoordinator,
    
    # Data Management
    DataBridge, ResultFormatter, ResultSet,
    
    # Convenience Functions
    execute_physical_plan, create_mylathdb_engine,
    
    # Configuration and Exceptions
    MyLathDBExecutionConfig, MyLathDBExecutionError
)

# Version information
__version__ = "1.0.0"
__author__ = "MyLathDB Team"
__description__ = "Graph Database with Cypher Support and Execution Engine"

# Main API exports
__all__ = [
    # === CORE DATABASE API ===
    "MyLathDB",           # Main database class
    "execute_query",      # High-level query execution
    "create_database",    # Database creation
    
    # === QUERY EXECUTION ===
    "ExecutionEngine",
    "ExecutionResult", 
    "execute_physical_plan",
    "create_mylathdb_engine",
    
    # === QUERY PARSING ===
    "parse_cypher_query",
    "validate_cypher_syntax",
    "CypherParser",
    
    # === QUERY PLANNING ===
    "LogicalPlanner",
    "RuleBasedOptimizer", 
    "PhysicalPlanner",
    "QueryPlanner",
    
    # === AST NODES ===
    "Query", "MatchClause", "WhereClause", "ReturnClause",
    "Pattern", "NodePattern", "RelationshipPattern",
    
    # === RESULTS ===
    "ExecutionResult", "ExecutionPlan", "ResultSet",
    
    # === CONFIGURATION ===
    "MyLathDBExecutionConfig",
    
    # === ERRORS ===
    "ParseError", "MyLathDBExecutionError",
    
    # === VERSION ===
    "__version__"
]


# =============================================================================
# HIGH-LEVEL MYLATHDB API
# =============================================================================

class MyLathDB:
    """
    Main MyLathDB Database Class
    
    Provides a high-level interface for graph database operations
    with Cypher query support and execution.
    """
    
    def __init__(self, redis_host="localhost", redis_port=6379, redis_db=0, 
                 enable_caching=True, **kwargs):
        """
        Initialize MyLathDB instance
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port  
            redis_db: Redis database number
            enable_caching: Enable query result caching
            **kwargs: Additional configuration options
        """
        # Initialize execution engine
        self.engine = create_mylathdb_engine(
            redis_host=redis_host,
            redis_port=redis_port, 
            redis_db=redis_db,
            enable_caching=enable_caching,
            **kwargs
        )
        
        # Initialize planners
        self.logical_planner = LogicalPlanner()
        self.optimizer = RuleBasedOptimizer()
        self.physical_planner = PhysicalPlanner()
        
        # Configuration
        self.config = MyLathDBExecutionConfig()
        
        # Statistics
        self.query_count = 0
        self.total_execution_time = 0.0
    
    def execute_query(self, cypher_query: str, parameters: dict = None, 
                     graph_data: dict = None, **kwargs) -> ExecutionResult:
        """
        Execute a Cypher query
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters (e.g., {"name": "Alice"})
            graph_data: Optional graph data for GraphBLAS operations
            **kwargs: Additional execution options
            
        Returns:
            ExecutionResult with query results and statistics
            
        Example:
            >>> db = MyLathDB()
            >>> result = db.execute_query("MATCH (n:Person) WHERE n.age > $age RETURN n", 
            ...                          parameters={"age": 25})
            >>> print(f"Found {len(result.data)} people")
        """
        try:
            # Update statistics
            self.query_count += 1
            
            # Parse query
            ast = parse_cypher_query(cypher_query)
            
            # Create and optimize logical plan
            logical_plan = self.logical_planner.create_logical_plan(ast)
            optimized_plan = self.optimizer.optimize(logical_plan)
            
            # Create physical plan
            physical_plan = self.physical_planner.create_physical_plan(optimized_plan)
            
            # Execute
            result = self.engine.execute(
                physical_plan, 
                parameters=parameters,
                graph_data=graph_data,
                **kwargs
            )
            
            # Update statistics
            self.total_execution_time += result.execution_time
            
            return result
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Query execution failed: {str(e)}",
                execution_time=0.0
            )
    
    def load_graph_data(self, nodes: list = None, edges: list = None, 
                       adjacency_matrices: dict = None):
        """
        Load graph data into the database
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge tuples (source, relation, target)  
            adjacency_matrices: Pre-computed adjacency matrices
            
        Example:
            >>> db = MyLathDB()
            >>> nodes = [
            ...     {"id": "1", "name": "Alice", "age": 30, "_labels": ["Person"]},
            ...     {"id": "2", "name": "Bob", "age": 25, "_labels": ["Person"]}
            ... ]
            >>> edges = [("1", "KNOWS", "2")]
            >>> db.load_graph_data(nodes=nodes, edges=edges)
        """
        # Load data into Redis
        if nodes:
            self._load_nodes_to_redis(nodes)
        
        if edges:
            self._load_edges_to_redis(edges)
        
        # Load data into GraphBLAS
        if adjacency_matrices or edges:
            graph_data = {}
            if adjacency_matrices:
                graph_data['adjacency_matrices'] = adjacency_matrices
            if edges:
                graph_data['edges'] = self._group_edges_by_type(edges)
            
            self.engine.graphblas_executor.load_graph_data(graph_data)
    
    def get_statistics(self) -> dict:
        """Get database and execution statistics"""
        engine_stats = self.engine.get_execution_statistics()
        
        return {
            'database': {
                'queries_executed': self.query_count,
                'total_execution_time': self.total_execution_time,
                'avg_execution_time': self.total_execution_time / max(1, self.query_count)
            },
            'engine': engine_stats
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.engine.clear_cache()
    
    def shutdown(self):
        """Shutdown the database"""
        self.engine.shutdown()
    
    # Private helper methods
    def _load_nodes_to_redis(self, nodes: list):
        """Load nodes into Redis"""
        if not self.engine.redis_executor.redis:
            return
        
        redis_client = self.engine.redis_executor.redis
        
        for node in nodes:
            node_id = node.get('id')
            if not node_id:
                continue
            
            # Store node properties
            node_key = f"node:{node_id}"
            for key, value in node.items():
                if key != 'id':
                    if isinstance(value, list):
                        redis_client.hset(node_key, key, ",".join(map(str, value)))
                    else:
                        redis_client.hset(node_key, key, str(value))
            
            # Create label indexes
            labels = node.get('_labels', [])
            for label in labels:
                redis_client.sadd(f"label:{label}", node_id)
            
            # Create property indexes
            for key, value in node.items():
                if key not in ['id', '_labels'] and not key.startswith('_'):
                    redis_client.sadd(f"prop:{key}:{value}", node_id)
    
    def _load_edges_to_redis(self, edges: list):
        """Load edges into Redis"""
        if not self.engine.redis_executor.redis:
            return
        
        redis_client = self.engine.redis_executor.redis
        
        for edge in