# mylathdb/__init__.py - COMPLETE FIXED VERSION

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
    Main MyLathDB Database Class - COMPLETE FIXED VERSION
    
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
        Load graph data into the database - FIXED VERSION
        
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
            self._load_nodes_to_redis_fixed(nodes)
        
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
    
    # =============================================================================
    # FIXED PRIVATE HELPER METHODS
    # =============================================================================
    
    def _load_nodes_to_redis_fixed(self, nodes: list):
        """FIXED: Load nodes into Redis with proper label storage and indexing"""
        if not self.engine.redis_executor.redis:
            return
        
        redis_client = self.engine.redis_executor.redis
        
        for node in nodes:
            node_id = node.get('id')
            if not node_id:
                continue
            
            # Store node properties (excluding id and _labels)
            node_key = f"node:{node_id}"
            properties = {}
            for key, value in node.items():
                if key not in ['id', '_id', '_labels']:
                    if isinstance(value, list):
                        properties[key] = ",".join(map(str, value))
                    else:
                        properties[key] = str(value)
            
            # Store properties if any exist
            if properties:
                redis_client.hset(node_key, mapping=properties)
            
            # FIXED: Store labels properly with indexes
            labels = node.get('_labels', [])
            if labels:
                # Store labels in a separate set for this node
                labels_key = f"node_labels:{node_id}"
                redis_client.sadd(labels_key, *labels)
                
                # FIXED: Create label indexes for efficient label-based queries
                for label in labels:
                    redis_client.sadd(f"label:{label}", node_id)
            
            # FIXED: Create property indexes for efficient property-based queries
            for key, value in properties.items():
                # Create property value index
                redis_client.sadd(f"prop:{key}:{value}", node_id)
                
                # Create sorted property index for numeric values (for range queries)
                try:
                    numeric_value = float(value)
                    redis_client.zadd(f"sorted_prop:{key}", {node_id: numeric_value})
                except (ValueError, TypeError):
                    # Not a numeric value, skip sorted index
                    pass
    
    def _load_edges_to_redis(self, edges: list):
        """Load edges into Redis storage"""
        if not self.engine.redis_executor.redis:
            return
        
        redis_client = self.engine.redis_executor.redis
        
        for edge in edges:
            if len(edge) >= 3:
                src_id, rel_type, dest_id = edge[:3]
                
                # Generate edge ID
                edge_id = f"{src_id}_{rel_type}_{dest_id}"
                
                # Store edge endpoints
                redis_client.set(f"edge_endpoints:{edge_id}", f"{src_id}|{dest_id}|{rel_type}")
                
                # Create relationship indexes
                redis_client.sadd(f"out:{src_id}:{rel_type}", edge_id)
                redis_client.sadd(f"in:{dest_id}:{rel_type}", edge_id)
                redis_client.sadd(f"rel:{rel_type}", edge_id)
    
    def _group_edges_by_type(self, edges: list) -> dict:
        """Group edges by relationship type"""
        grouped = {}
        for edge in edges:
            if len(edge) >= 3:
                src_id, rel_type, dest_id = edge[:3]
                if rel_type not in grouped:
                    grouped[rel_type] = []
                grouped[rel_type].append((src_id, dest_id))
        return grouped
    
    # =============================================================================
    # DEBUG AND UTILITY METHODS
    # =============================================================================
    
    def debug_redis_state(self):
        """FIXED: Debug method to check Redis state and data loading"""
        if not self.engine.redis_executor.redis:
            print("❌ Redis not available")
            return
        
        redis_client = self.engine.redis_executor.redis
        
        print("🔍 Redis State Debug:")
        
        # Check all keys
        all_keys = list(redis_client.scan_iter())
        print(f"📋 All Redis keys ({len(all_keys)}): {all_keys}")
        
        # Check node data
        node_count = 0
        for key in all_keys:
            if key.startswith('node:') and key != 'next_node_id':
                node_count += 1
                node_id = key.split(':')[1]
                node_data = redis_client.hgetall(key)
                print(f"📝 {key}: {node_data}")
                
                # Check labels for this node
                labels_key = f"node_labels:{node_id}"
                labels = list(redis_client.smembers(labels_key))
                print(f"🏷️  {labels_key}: {labels}")
        
        print(f"📊 Total nodes found: {node_count}")
        
        # Check label indexes
        label_indexes = [key for key in all_keys if key.startswith('label:')]
        print(f"🏷️  Label indexes ({len(label_indexes)}):")
        for key in label_indexes:
            label_name = key.split(':')[1]
            node_ids = list(redis_client.smembers(key))
            print(f"   {key}: {node_ids}")
        
        # Check property indexes
        prop_indexes = [key for key in all_keys if key.startswith('prop:')]
        print(f"🔧 Property indexes ({len(prop_indexes)}):")
        for key in prop_indexes[:5]:  # Show first 5
            node_ids = list(redis_client.smembers(key))
            print(f"   {key}: {node_ids}")
        if len(prop_indexes) > 5:
            print(f"   ... and {len(prop_indexes) - 5} more property indexes")
    
    def debug_query_execution(self, query: str):
        """Debug method to trace query execution step by step"""
        print(f"🔍 Debugging Query: {query}")
        
        try:
            # Step 1: Parse AST
            ast = parse_cypher_query(query)
            print(f"✅ AST: {type(ast).__name__}")
            
            if ast.return_clause:
                print(f"   Return items: {len(ast.return_clause.items)}")
                for i, item in enumerate(ast.return_clause.items):
                    print(f"     {i}: {item.expression} (alias: {item.alias})")
            
            # Step 2: Create logical plan
            logical_plan = self.logical_planner.create_logical_plan(ast)
            print(f"✅ Logical Plan: {type(logical_plan).__name__}")
            
            if hasattr(logical_plan, 'projections'):
                print(f"   Projections: {logical_plan.projections}")
            if hasattr(logical_plan, 'logical_op'):
                print(f"   Has logical_op: {logical_plan.logical_op is not None}")
            
            # Step 3: Create physical plan
            physical_plan = self.physical_planner.create_physical_plan(logical_plan)
            print(f"✅ Physical Plan: {type(physical_plan).__name__}")
            
            if hasattr(physical_plan, 'logical_op'):
                logical_op = physical_plan.logical_op
                print(f"   Physical logical_op: {logical_op is not None}")
                if logical_op and hasattr(logical_op, 'projections'):
                    print(f"   Physical projections: {logical_op.projections}")
            
            # Step 4: Execute
            result = self.engine.execute(physical_plan)
            print(f"✅ Execution: Success={result.success}")
            print(f"   Results: {len(result.data)} records")
            if result.data:
                print(f"   Sample: {result.data[0]}")
            
            return result
            
        except Exception as e:
            print(f"❌ Debug failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_projection_fix(self):
        """Test if the projection fix is working"""
        print("🧪 Testing Projection Fix...")
        
        # Clear and load minimal test data
        if hasattr(self.engine.redis_executor, 'redis') and self.engine.redis_executor.redis:
            self.engine.redis_executor.redis.flushdb()
        
        self.load_graph_data(nodes=[{
            'id': '1', 
            'name': 'Alice', 
            'age': 30, 
            '_labels': ['Person']
        }])
        
        # Test projection query
        result = self.execute_query("MATCH (n:Person) RETURN n.name")
        
        print(f"Success: {result.success}")
        print(f"Data: {result.data}")
        
        if result.data:
            first_result = result.data[0]
            print(f"Result keys: {list(first_result.keys())}")
            
            # Check if projection worked
            if 'n.name' in first_result:
                print("✅ PROJECTION SUCCESSFUL - Got 'n.name'")
                return True
            elif 'name' in first_result and len(first_result) == 1:
                print("✅ PROJECTION SUCCESSFUL - Got 'name'")
                return True
            elif 'n' in first_result:
                print("❌ PROJECTION FAILED - Still got full node 'n'")
                return False
            else:
                print("⚠️  UNEXPECTED RESULT FORMAT")
                return False
        else:
            print("❌ No results returned")
            return False


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def execute_query(cypher_query: str, database: MyLathDB = None, **kwargs):
    """
    Convenience function to execute a Cypher query
    
    Args:
        cypher_query: Cypher query string
        database: MyLathDB instance (creates new one if None)
        **kwargs: Additional options
        
    Returns:
        ExecutionResult
    """
    if database is None:
        database = MyLathDB()
    
    return database.execute_query(cypher_query, **kwargs)


def create_database(**kwargs) -> MyLathDB:
    """
    Create a new MyLathDB database instance
    
    Args:
        **kwargs: Configuration options
        
    Returns:
        MyLathDB instance
    """
    return MyLathDB(**kwargs)