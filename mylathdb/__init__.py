# mylathdb/__init__.py - COMPLETE FIXED VERSION WITH ALL DEBUG METHODS

"""
MyLathDB - Graph Database with Cypher Support
A complete graph database implementation with PROPER EXECUTION COORDINATION
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
__description__ = "Graph Database with Cypher Support and PROPER Execution Coordination"

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
# COMPLETE MYLATHDB API WITH PROPER COORDINATION + ALL DEBUG METHODS
# =============================================================================

class MyLathDB:
    """
    COMPLETE MyLathDB Database Class - WITH COORDINATION + DEBUG METHODS
    
    Key Fix: All operations now go through ExecutionCoordinator for proper
    Redis + GraphBLAS coordination, fixing traversal and filtering issues.
    """
    
    def __init__(self, redis_host="localhost", redis_port=6379, redis_db=0, 
                 enable_caching=True, **kwargs):
        """
        Initialize MyLathDB instance with PROPER COORDINATION
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port  
            redis_db: Redis database number
            enable_caching: Enable query result caching
            **kwargs: Additional configuration options
        """
        print("🚀 Initializing MyLathDB with PROPER COORDINATION...")
        
        # Initialize execution engine with coordinator-first architecture
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
        
        print("✅ MyLathDB initialized with ExecutionCoordinator-first architecture")
    
    def execute_query(self, cypher_query: str, parameters: dict = None, 
                     graph_data: dict = None, **kwargs) -> ExecutionResult:
        """
        FIXED: Execute a Cypher query using PROPER COORDINATION
        
        Key Fix: Now uses ExecutionCoordinator as primary orchestrator
        for all complex operations involving Redis + GraphBLAS coordination.
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters (e.g., {"name": "Alice"})
            graph_data: Optional graph data for GraphBLAS operations
            **kwargs: Additional execution options
            
        Returns:
            ExecutionResult with properly coordinated query results
            
        Example:
            >>> db = MyLathDB()
            >>> result = db.execute_query("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b")
            >>> print(f"Found {len(result.data)} relationships")
        """
        try:
            print(f"🔍 === EXECUTING QUERY WITH PROPER COORDINATION ===")
            print(f"📝 Query: {cypher_query}")
            
            # Update statistics
            self.query_count += 1
            
            # Parse query
            ast = parse_cypher_query(cypher_query)
            print(f"✅ Parsed AST: {type(ast).__name__}")
            
            # Create and optimize logical plan
            logical_plan = self.logical_planner.create_logical_plan(ast)
            optimized_plan = self.optimizer.optimize(logical_plan)
            print(f"✅ Created logical plan: {type(optimized_plan).__name__}")
            
            # Create physical plan
            physical_plan = self.physical_planner.create_physical_plan(optimized_plan)
            print(f"✅ Created physical plan: {type(physical_plan).__name__}")
            
            # THE KEY FIX: Use coordinator-first execution strategy
            result = self._execute_with_coordination(
                physical_plan, 
                parameters=parameters,
                graph_data=graph_data,
                **kwargs
            )
            
            # Update statistics
            self.total_execution_time += result.execution_time
            
            print(f"🎉 Query executed successfully: {result.success}")
            print(f"📊 Results: {len(result.data)} records in {result.execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=f"Query execution failed: {str(e)}",
                execution_time=0.0
            )
    
    def _execute_with_coordination(self, physical_plan, parameters=None, graph_data=None, **kwargs):
        """
        FIXED: Execute physical plan using ExecutionCoordinator as PRIMARY orchestrator
        
        This is the key fix - instead of trying to route operations ourselves,
        we let the ExecutionCoordinator handle ALL complex coordination patterns.
        """
        print("🎯 === COORDINATOR-FIRST EXECUTION STRATEGY ===")
        
        # Create execution context with FIXED configuration access
        context = ExecutionContext(
            parameters=parameters or {},
            graph_data=graph_data,
            max_execution_time=kwargs.get('max_execution_time', 300.0),  # FIXED: Default value
            enable_parallel=kwargs.get('enable_parallel', True),
            cache_results=kwargs.get('cache_results', getattr(self.config, 'ENABLE_CACHING', True))  # FIXED: Safe access
        )
        
        # Load graph data if provided
        if graph_data:
            self._load_graph_data(graph_data)
        
        # THE KEY FIX: Use ExecutionCoordinator for ALL operations
        print("🔧 Using ExecutionCoordinator as primary orchestrator...")
        
        # Check if this is a complex operation that needs coordination
        if self._requires_coordination(physical_plan):
            print("✅ Complex operation detected - using ExecutionCoordinator")
            result_data = self.engine.coordinator.execute_operation(physical_plan, context)
        else:
            print("📝 Simple operation - using direct execution")
            result_data = self.engine._execute_physical_plan_fixed(physical_plan, context, None)
        
        # Build execution result
        execution_result = ExecutionResult(
            success=True,
            data=result_data,
            execution_time=0.0,  # Will be set by the engine
            execution_plan_summary=f"Executed: {type(physical_plan).__name__}"
        )
        
        return execution_result
    
    def _requires_coordination(self, physical_plan) -> bool:
        """
        Determine if a physical plan requires ExecutionCoordinator
        
        Complex operations that need coordination:
        - Graph traversals (ConditionalTraverse, Expand)  
        - Mixed Redis + GraphBLAS operations
        - OPTIONAL MATCH, EXISTS patterns
        - Multi-step filtering with property access
        """
        from .cypher_planner.physical_planner import (
            CoordinatorOperation, GraphBLASOperation
        )
        
        # Always use coordinator for coordinator operations
        if isinstance(physical_plan, CoordinatorOperation):
            return True
        
        # Use coordinator for GraphBLAS operations that need property access
        if isinstance(physical_plan, GraphBLASOperation):
            operation_type = getattr(physical_plan, 'operation_type', '')
            if operation_type in ['ConditionalTraverse', 'Expand', 'VarLenTraverse']:
                return True
        
        # Check logical operation type
        logical_op = getattr(physical_plan, 'logical_op', None)
        if logical_op:
            op_name = type(logical_op).__name__
            complex_ops = [
                'ConditionalTraverse', 'ConditionalVarLenTraverse', 
                'Expand', 'Optional', 'SemiApply', 'Apply'
            ]
            if op_name in complex_ops:
                return True
        
        # Check for filter chains that need coordination
        if self._has_filter_chain(physical_plan):
            return True
        
        # Default to simple execution
        return False
    
    def _has_filter_chain(self, physical_plan) -> bool:
        """Check if physical plan has a filter chain requiring coordination"""
        
        # Look for patterns like: Scan -> Filter -> Project
        # These often need coordination between Redis scans and filtering
        
        def has_filters(op):
            if hasattr(op, 'logical_op') and op.logical_op:
                op_name = type(op.logical_op).__name__
                if 'Filter' in op_name:
                    return True
            
            # Check children
            for child in getattr(op, 'children', []):
                if has_filters(child):
                    return True
            
            return False
        
        return has_filters(physical_plan)
    
    def load_graph_data(self, nodes: list = None, edges: list = None, 
                       adjacency_matrices: dict = None):
        """
        FIXED: Load graph data with proper Redis + GraphBLAS synchronization
        
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
        print("📦 Loading graph data with coordination...")
        
        # Load data into Redis
        if nodes:
            self._load_nodes_to_redis_fixed(nodes)
            print(f"✅ Loaded {len(nodes)} nodes into Redis")
        
        if edges:
            self._load_edges_to_redis(edges)
            print(f"✅ Loaded {len(edges)} edges into Redis")
        
        # Load data into GraphBLAS with proper synchronization
        if adjacency_matrices or edges:
            graph_data = {}
            if adjacency_matrices:
                graph_data['adjacency_matrices'] = adjacency_matrices
            if edges:
                graph_data['edges'] = self._group_edges_by_type(edges)
            
            # THE FIX: Use data bridge for proper synchronization
            if hasattr(self.engine, 'data_bridge') and self.engine.data_bridge:
                print("🔄 Synchronizing data to GraphBLAS via DataBridge...")
                try:
                    self.engine.data_bridge.sync_redis_to_graphblas(force=True)
                    print("✅ Redis -> GraphBLAS synchronization complete")
                except Exception as e:
                    print(f"⚠️  GraphBLAS sync failed: {e}")
            
            # Also load directly into GraphBLAS executor
            self.engine.graphblas_executor.load_graph_data(graph_data)
            print("✅ Graph data loaded into both Redis and GraphBLAS")
    
    def _load_graph_data(self, graph_data):
        """Internal method to load graph data during query execution"""
        print("📥 Loading graph data for query execution...")
        
        # Load into Redis if node/edge data provided
        if 'nodes' in graph_data:
            self.engine.redis_executor.load_nodes(graph_data['nodes'])
        
        if 'edges' in graph_data:
            self.engine.redis_executor.load_edges(graph_data['edges'])
        
        # Load into GraphBLAS if matrix data provided
        if 'adjacency_matrices' in graph_data:
            self.engine.graphblas_executor.load_adjacency_matrices(graph_data['adjacency_matrices'])
        
        if 'edges' in graph_data:
            # Also create matrices from edge data
            self.engine.graphblas_executor.load_edges_as_matrices(graph_data['edges'])
        
        # Sync data between systems
        if hasattr(self.engine, 'data_bridge'):
            try:
                self.engine.data_bridge.sync_redis_to_graphblas(incremental=True)
            except Exception as e:
                print(f"⚠️  Data sync failed: {e}")
    
    def get_statistics(self) -> dict:
        """Get database and execution statistics"""
        engine_stats = self.engine.get_execution_statistics()
        
        return {
            'database': {
                'queries_executed': self.query_count,
                'total_execution_time': self.total_execution_time,
                'avg_execution_time': self.total_execution_time / max(1, self.query_count)
            },
            'engine': engine_stats,
            'coordination': {
                'coordinator_available': hasattr(self.engine, 'coordinator'),
                'data_bridge_available': hasattr(self.engine, 'data_bridge'),
                'redis_available': self.engine.redis_executor.test_connection() if hasattr(self.engine, 'redis_executor') else False,
                'graphblas_available': self.engine.graphblas_executor.is_available() if hasattr(self.engine, 'graphblas_executor') else False
            }
        }
    
    def clear_cache(self):
        """Clear all caches"""
        self.engine.clear_cache()
    
    def shutdown(self):
        """Shutdown the database"""
        self.engine.shutdown()
    
    # =============================================================================
    # FIXED PRIVATE HELPER METHODS (SAME AS BEFORE)
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
            
            # Store labels properly with indexes
            labels = node.get('_labels', [])
            if labels:
                # Store labels in a separate set for this node
                labels_key = f"node_labels:{node_id}"
                redis_client.sadd(labels_key, *labels)
                
                # Create label indexes for efficient label-based queries
                for label in labels:
                    redis_client.sadd(f"label:{label}", node_id)
            
            # Create property indexes for efficient property-based queries
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
    # ALL DEBUG METHODS THAT TESTS EXPECT
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
        """FIXED: Debug method to trace query execution step by step"""
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
            
            # Step 4: Execute with coordination
            result = self.execute_query(query)
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
        """FIXED: Test if the projection fix is working"""
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
    
    def debug_coordination_state(self):
        """FIXED: Debug method to check coordination system state"""
        print("🔍 === COORDINATION SYSTEM DEBUG ===")
        
        # Check coordinator status
        if hasattr(self.engine, 'coordinator'):
            print("✅ ExecutionCoordinator available")
            print(f"   Redis executor: {self.engine.coordinator.redis_executor is not None}")
            print(f"   GraphBLAS executor: {self.engine.coordinator.graphblas_executor is not None}")
        else:
            print("❌ ExecutionCoordinator NOT available")
        
        # Check data bridge status
        if hasattr(self.engine, 'data_bridge'):
            print("✅ DataBridge available")
            print(f"   Pending updates: {self.engine.data_bridge.has_pending_updates()}")
            if hasattr(self.engine.data_bridge, 'get_pending_count'):
                pending = self.engine.data_bridge.get_pending_count()
                print(f"   Pending counts: {pending}")
        else:
            print("❌ DataBridge NOT available")
        
        # Check executor status
        redis_status = self.engine.redis_executor.get_status() if hasattr(self.engine, 'redis_executor') else {}
        graphblas_status = self.engine.graphblas_executor.get_status() if hasattr(self.engine, 'graphblas_executor') else {}
        
        print(f"📊 Redis status: {redis_status.get('connected', False)}")
        print(f"📊 GraphBLAS status: {graphblas_status.get('available', False)}")
    
    def test_coordination_fix(self):
        """FIXED: Test if the coordination fix is working"""
        print("🧪 === TESTING COORDINATION FIX ===")
        
        # Clear and load test data
        if hasattr(self.engine.redis_executor, 'redis') and self.engine.redis_executor.redis:
            self.engine.redis_executor.redis.flushdb()
        
        # Load test graph with relationships
        test_nodes = [
            {'id': '1', 'name': 'Alice', 'age': 30, '_labels': ['Person']},
            {'id': '2', 'name': 'Bob', 'age': 25, '_labels': ['Person']}
        ]
        test_edges = [('1', 'KNOWS', '2')]
        
        self.load_graph_data(nodes=test_nodes, edges=test_edges)
        
        # Test simple query
        result1 = self.execute_query("MATCH (n:Person) RETURN n.name")
        print(f"✅ Simple query: {len(result1.data)} results")
        
        # Test traversal query (this should now work with coordination)
        result2 = self.execute_query("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name")
        print(f"{'✅' if len(result2.data) > 0 else '❌'} Traversal query: {len(result2.data)} results")
        
        # Test filtered query
        result3 = self.execute_query("MATCH (n:Person) WHERE n.age > 25 RETURN n.name")
        print(f"✅ Filtered query: {len(result3.data)} results")
        
        if result2.data and len(result2.data) > 0:
            print("🎉 COORDINATION FIX SUCCESSFUL!")
            print(f"   Sample traversal result: {result2.data[0]}")
            return True
        else:
            print("❌ Coordination fix needs more work")
            return False


# =============================================================================
# CONVENIENCE FUNCTIONS (UPDATED)
# =============================================================================

def execute_query(cypher_query: str, database: MyLathDB = None, **kwargs):
    """
    Convenience function to execute a Cypher query with PROPER COORDINATION
    
    Args:
        cypher_query: Cypher query string
        database: MyLathDB instance (creates new one if None)
        **kwargs: Additional options
        
    Returns:
        ExecutionResult with coordinated results
    """
    if database is None:
        database = MyLathDB()
    
    return database.execute_query(cypher_query, **kwargs)


def create_database(**kwargs) -> MyLathDB:
    """
    Create a new MyLathDB database instance with PROPER COORDINATION
    
    Args:
        **kwargs: Configuration options
        
    Returns:
        MyLathDB instance with ExecutionCoordinator-first architecture
    """
    return MyLathDB(**kwargs)