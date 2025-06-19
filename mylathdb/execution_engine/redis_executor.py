# mylathdb/execution_engine/redis_executor.py

"""
MyLathDB Redis Executor - COMPLETE FIXED VERSION
Handles Redis operations for node/edge storage and property access
Based on FalkorDB's Redis integration patterns
"""

import time
import logging
import subprocess
import json
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass

try:
    import redis
except ImportError:
    redis = None

from .config import MyLathDBExecutionConfig
from .exceptions import MyLathDBExecutionError, MyLathDBRedisError
from .utils import mylathdb_measure_time

logger = logging.getLogger(__name__)

@dataclass
class RedisEntityStorage:
    """Redis storage schema for graph entities (FalkorDB-inspired)"""
    
    # Node storage patterns
    NODE_KEY_PATTERN = "node:{node_id}"               # node:123 -> hash of properties
    NODE_LABELS_KEY = "node_labels:{node_id}"         # node_labels:123 -> set of labels
    LABEL_NODES_KEY = "label:{label}"                 # label:Person -> set of node IDs
    
    # Edge storage patterns  
    EDGE_KEY_PATTERN = "edge:{edge_id}"               # edge:456 -> hash of properties
    EDGE_ENDPOINTS_KEY = "edge_endpoints:{edge_id}"   # edge_endpoints:456 -> "src_id|dest_id|type"
    
    # Property indexes
    PROPERTY_INDEX_KEY = "prop:{property}:{value}"    # prop:age:25 -> set of node IDs
    SORTED_PROPERTY_KEY = "sorted_prop:{property}"    # sorted_prop:age -> sorted set for range queries
    
    # Relationship indexes
    OUTGOING_EDGES_KEY = "out:{node_id}:{rel_type}"   # out:123:KNOWS -> set of edge IDs
    INCOMING_EDGES_KEY = "in:{node_id}:{rel_type}"    # in:123:KNOWS -> set of edge IDs
    RELATIONSHIP_EDGES_KEY = "rel:{rel_type}"         # rel:KNOWS -> set of edge IDs
    
    # System keys
    NEXT_NODE_ID_KEY = "next_node_id"                 # auto-increment for node IDs
    NEXT_EDGE_ID_KEY = "next_edge_id"                 # auto-increment for edge IDs
    GRAPH_METADATA_KEY = "graph_metadata"             # graph-level metadata

class RedisExecutor:
    """
    Redis executor for MyLathDB based on FalkorDB's entity storage model
    
    Handles:
    - Node/Edge storage and retrieval
    - Property indexing and filtering
    - Label-based operations
    - Relationship indexing
    """
    
    def __init__(self, config: MyLathDBExecutionConfig):
        """Initialize Redis executor with FalkorDB-style configuration"""
        self.config = config
        self.redis: Optional[redis.Redis] = None
        self.storage = RedisEntityStorage()
        self._redis_process = None
        
        # Connection settings
        self.connection_retries = 3
        self.connection_timeout = 5.0
        
        # Performance settings
        self.batch_size = 1000
        self.pipeline_enabled = True
        
    def initialize(self):
        """Initialize Redis connection with auto-start capability"""
        logger.info("Initializing Redis executor")
        
        if redis is None:
            raise MyLathDBRedisError("Redis package not installed. Install with: pip install redis")
        
        # Auto-start Redis if requested and not already running
        if self.config.AUTO_START_REDIS:
            self._ensure_redis_running()
        
        # Establish connection
        self._connect_to_redis()
        
        # Setup indexes and metadata
        self._setup_redis_structures()
        
        logger.info("Redis executor initialized successfully")
    
    def _ensure_redis_running(self):
        """Ensure Redis server is running, start if needed"""
        try:
            # Try to connect first
            test_redis = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,
                db=self.config.REDIS_DB,
                socket_connect_timeout=1,
                decode_responses=True
            )
            test_redis.ping()
            logger.info("Redis server already running")
            return
            
        except redis.ConnectionError:
            logger.info("Redis server not running, attempting to start...")
            
        # Try to start Redis server
        try:
            # Start Redis in background
            self._redis_process = subprocess.Popen(
                ['redis-server', '--port', str(self.config.REDIS_PORT), '--daemonize', 'yes'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for startup
            time.sleep(2)
            
            # Test connection
            test_redis = redis.Redis(
                host=self.config.REDIS_HOST,
                port=self.config.REDIS_PORT,  
                db=self.config.REDIS_DB,
                decode_responses=True
            )
            test_redis.ping()
            logger.info("Redis server started successfully")
            
        except FileNotFoundError:
            logger.warning("Redis server not found in PATH, assuming external Redis")
        except Exception as e:
            logger.warning(f"Could not auto-start Redis: {e}")
    
    def _connect_to_redis(self):
        """Establish Redis connection with retries"""
        for attempt in range(self.connection_retries):
            try:
                self.redis = redis.Redis(
                    host=self.config.REDIS_HOST,
                    port=self.config.REDIS_PORT,
                    db=self.config.REDIS_DB,
                    socket_connect_timeout=self.connection_timeout,
                    decode_responses=True,
                    retry_on_timeout=True
                )
                
                # Test connection
                self.redis.ping()
                logger.info(f"Connected to Redis at {self.config.REDIS_HOST}:{self.config.REDIS_PORT}")
                return
                
            except redis.ConnectionError as e:
                if attempt < self.connection_retries - 1:
                    logger.warning(f"Redis connection attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)
                else:
                    raise MyLathDBRedisError(f"Failed to connect to Redis after {self.connection_retries} attempts: {e}")
    
    def _setup_redis_structures(self):
        """Setup Redis data structures and indexes"""
        try:
            # Initialize auto-increment counters if they don't exist
            if not self.redis.exists(self.storage.NEXT_NODE_ID_KEY):
                self.redis.set(self.storage.NEXT_NODE_ID_KEY, 1)
            
            if not self.redis.exists(self.storage.NEXT_EDGE_ID_KEY):
                self.redis.set(self.storage.NEXT_EDGE_ID_KEY, 1)
            
            # Initialize graph metadata
            if not self.redis.exists(self.storage.GRAPH_METADATA_KEY):
                metadata = {
                    'created_at': time.time(),
                    'version': '1.0.0',
                    'node_count': 0,
                    'edge_count': 0
                }
                self.redis.hset(self.storage.GRAPH_METADATA_KEY, mapping=metadata)
            
            logger.info("Redis data structures initialized")
            
        except Exception as e:
            raise MyLathDBRedisError(f"Failed to setup Redis structures: {e}")
    
    @mylathdb_measure_time
    def execute_operation(self, redis_operation, context) -> List[Dict[str, Any]]:
        """
        Execute Redis operation from physical plan - COMPLETE FIXED VERSION
        
        Args:
            redis_operation: RedisOperation from physical planner
            context: ExecutionContext
            
        Returns:
            List of result dictionaries
        """
        from ..cypher_planner.physical_planner import RedisOperation
        
        if not isinstance(redis_operation, RedisOperation):
            raise MyLathDBRedisError(f"Expected RedisOperation, got {type(redis_operation)}")
        
        logger.debug(f"Executing Redis operation: {redis_operation.operation_type}")
        
        # Route to appropriate handler based on operation type
        operation_type = redis_operation.operation_type
        logical_op = getattr(redis_operation, 'logical_op', None)
        
        if operation_type == "NodeByLabelScan" and logical_op:
            return self._execute_node_by_label_scan_fixed(logical_op, context)
        elif operation_type == "AllNodeScan" and logical_op:
            return self._execute_all_node_scan_fixed(logical_op, context)
        elif operation_type == "PropertyScan" and logical_op:
            return self._execute_property_scan_fixed(logical_op, context)
        elif operation_type == "NodeScan" and logical_op:
            return self._execute_node_scan_fixed(logical_op, context)
        elif operation_type == "PropertyFilter" and logical_op:
            return self._execute_property_filter_fixed(logical_op, context)
        elif operation_type == "Filter":
            return self._execute_generic_filter_operation(redis_operation, context)
        elif operation_type == "Project":
            return self._execute_project_operation_final(redis_operation, context)
        elif operation_type == "OrderBy":
            return self._execute_order_by_operation_final(redis_operation, context)
        elif operation_type == "Limit":
            return self._execute_limit_operation_final(redis_operation, context)
        else:
            # Execute Redis commands directly
            return self._execute_redis_commands_fixed(redis_operation, context)
    
    def _execute_project_operation_final(self, operation, context) -> List[Dict[str, Any]]:
        """FINAL FIX: Execute Project operation without result corruption"""
        
        print("🔍 === FINAL PROJECT OPERATION ===")
        
        # STEP 1: Execute child operations to get base data
        base_results = []
        for child in operation.children:
            child_results = self._execute_child_operation(child, context)
            base_results.extend(child_results)
        
        print(f"📊 Base results: {len(base_results)}")
        if base_results:
            print(f"📋 Sample base result: {base_results[0]}")
        
        # STEP 2: Get projections from logical operation
        logical_op = getattr(operation, 'logical_op', None)
        if not logical_op:
            print("⚠️  No logical operation found, returning base results")
            return base_results
        
        print(f"🎯 Logical operation: {type(logical_op).__name__}")
        
        # STEP 3: Apply projections
        if hasattr(logical_op, 'projections') and logical_op.projections:
            print(f"✅ Found {len(logical_op.projections)} projections")
            
            projected_results = []
            for result in base_results:
                projected_record = {}
                
                for expr, alias in logical_op.projections:
                    try:
                        # Evaluate expression
                        value = self._evaluate_expression_safely(expr, result)
                        
                        # Determine output key
                        key = alias if alias else self._derive_expression_name(expr)
                        projected_record[key] = value
                        
                        print(f"✅ Projection: {key} = {value}")
                        
                    except Exception as e:
                        print(f"❌ Projection failed for {expr}: {e}")
                        key = alias if alias else str(expr)
                        projected_record[key] = None
                
                projected_results.append(projected_record)
            
            print(f"🎉 Final projected results: {projected_results}")
            return projected_results
        
        else:
            print("⚠️  No projections found, returning base results")
            return base_results
    
    def _evaluate_expression_safely(self, expr, result):
        """SAFE: Evaluate projection expression with comprehensive error handling"""
        
        from ..cypher_planner.ast_nodes import (
            PropertyExpression, VariableExpression, LiteralExpression,
            BinaryExpression, FunctionCall
        )
        
        print(f"🔍 Evaluating: {expr} (type: {type(expr).__name__})")
        
        try:
            if isinstance(expr, PropertyExpression):
                # Property access: variable.property
                entity = result.get(expr.variable)
                print(f"📦 Entity for '{expr.variable}': {entity}")
                
                if entity and isinstance(entity, dict):
                    # Try multiple access patterns
                    
                    # Direct property access
                    if expr.property_name in entity:
                        value = entity[expr.property_name]
                        print(f"✅ Direct property access: {expr.property_name} = {value}")
                        return value
                    
                    # Properties sub-dict
                    elif 'properties' in entity and isinstance(entity['properties'], dict):
                        if expr.property_name in entity['properties']:
                            value = entity['properties'][expr.property_name]
                            print(f"✅ Properties dict access: {expr.property_name} = {value}")
                            return value
                    
                    # Search all keys
                    for key, value in entity.items():
                        if key.lower() == expr.property_name.lower():
                            print(f"✅ Case-insensitive match: {key} = {value}")
                            return value
                
                print(f"❌ Property '{expr.property_name}' not found")
                return None
                
            elif isinstance(expr, VariableExpression):
                # Variable reference
                value = result.get(expr.name)
                print(f"✅ Variable '{expr.name}' = {value}")
                return value
                
            elif isinstance(expr, LiteralExpression):
                # Literal value
                print(f"✅ Literal: {expr.value}")
                return expr.value
                
            else:
                print(f"⚠️  Unknown expression type: {type(expr)}")
                return str(expr)
                
        except Exception as e:
            print(f"❌ Expression evaluation failed: {e}")
            return None
    
    def _derive_expression_name(self, expr):
        """Derive a name from an expression for use as a column name"""
        
        from ..cypher_planner.ast_nodes import (
            PropertyExpression, VariableExpression, LiteralExpression
        )
        
        if isinstance(expr, PropertyExpression):
            return f"{expr.variable}.{expr.property_name}"
        elif isinstance(expr, VariableExpression):
            return expr.name
        elif isinstance(expr, LiteralExpression):
            return str(expr.value)
        else:
            return str(expr)
    
    def _execute_order_by_operation_final(self, operation, context) -> List[Dict[str, Any]]:
        """FINAL: Execute OrderBy operation"""
        
        # Execute children first
        base_results = []
        for child in operation.children:
            child_results = self._execute_child_operation(child, context)
            base_results.extend(child_results)
        
        # Apply ordering if we have results and ordering info
        logical_op = getattr(operation, 'logical_op', None)
        if base_results and logical_op and hasattr(logical_op, 'sort_items'):
            try:
                # Simple sorting by first sort item
                sort_expr, ascending = logical_op.sort_items[0]
                
                def sort_key_func(result):
                    try:
                        return self._evaluate_expression_safely(sort_expr, result) or ""
                    except:
                        return ""
                
                base_results = sorted(base_results, key=sort_key_func, reverse=not ascending)
                print(f"✅ Applied ordering: {len(base_results)} results sorted")
                
            except Exception as e:
                print(f"⚠️  Ordering failed: {e}")
        
        return base_results
    
    def _execute_limit_operation_final(self, operation, context) -> List[Dict[str, Any]]:
        """FINAL: Execute Limit operation"""
        
        # Execute children first
        base_results = []
        for child in operation.children:
            child_results = self._execute_child_operation(child, context)
            base_results.extend(child_results)
        
        # Apply limit if we have results and limit info
        logical_op = getattr(operation, 'logical_op', None)
        if base_results and logical_op:
            skip = getattr(logical_op, 'skip', 0)
            limit = getattr(logical_op, 'count', None)
            
            if skip > 0:
                base_results = base_results[skip:]
            if limit is not None and limit < len(base_results):
                base_results = base_results[:limit]
            
            print(f"✅ Applied limit (skip={skip}, limit={limit}): {len(base_results)} results")
        
        return base_results
    def _evaluate_filter_condition(self, actual_value, operator, expected_value):
        """Evaluate filter condition with type conversion and operator handling"""
        
        try:
            # Convert values to comparable types
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, str):
                try:
                    actual_value = float(actual_value)
                except ValueError:
                    return False
            
            # Apply operator
            if operator == "=":
                return actual_value == expected_value
            elif operator == "!=":
                return actual_value != expected_value
            elif operator == ">":
                return actual_value > expected_value
            elif operator == ">=":
                return actual_value >= expected_value
            elif operator == "<":
                return actual_value < expected_value
            elif operator == "<=":
                return actual_value <= expected_value
            elif operator.upper() == "CONTAINS":
                return str(expected_value) in str(actual_value)
            elif operator.upper() == "STARTS WITH":
                return str(actual_value).startswith(str(expected_value))
            elif operator.upper() == "ENDS WITH":
                return str(actual_value).endswith(str(expected_value))
            else:
                return False
                
        except Exception:
            return False
    
    def _execute_child_operation(self, child_operation, context):
        """COMPLETELY FIXED: Execute child operations with proper filter chain handling"""
        
        print(f"🔧 Executing child operation: {type(child_operation).__name__}")
        print(f"   Operation type: {getattr(child_operation, 'operation_type', 'None')}")
        print(f"   Has logical_op: {hasattr(child_operation, 'logical_op') and child_operation.logical_op is not None}")
        print(f"   Children count: {len(getattr(child_operation, 'children', []))}")
        
        # THE CRITICAL FIX: First, execute ALL children operations recursively
        # This ensures we process the ENTIRE operation tree depth-first
        child_results = []
        for grandchild in getattr(child_operation, 'children', []):
            print(f"   🔗 Recursively processing grandchild: {type(grandchild).__name__}")
            grandchild_results = self._execute_child_operation(grandchild, context)
            child_results.extend(grandchild_results)
            print(f"     ↳ Grandchild returned {len(grandchild_results)} results")
        
        # Then execute THIS operation on the results from children
        if hasattr(child_operation, 'logical_op') and child_operation.logical_op:
            logical_op = child_operation.logical_op
            logical_op_name = type(logical_op).__name__
            print(f"   🎯 Processing logical operation: {logical_op_name}")
            
            if logical_op_name == "NodeByLabelScan":
                # This is a leaf operation - no children to consider
                result = self._execute_node_by_label_scan_fixed(logical_op, context)
                print(f"   ✅ NodeByLabelScan returned {len(result)} results")
                return result
                
            elif logical_op_name == "PropertyFilter":
                # CRITICAL: Apply filter to the results from children
                if child_results:
                    print(f"   🔍 Applying PropertyFilter to {len(child_results)} child results")
                    filtered_results = []
                    
                    for result in child_results:
                        entity = result.get(logical_op.variable)
                        if not entity or not isinstance(entity, dict):
                            print(f"     ❌ Skipping result - no entity for variable '{logical_op.variable}'")
                            continue
                        
                        property_value = entity.get(logical_op.property_key)
                        if property_value is None:
                            print(f"     ❌ Skipping {entity.get('name', 'unknown')} - no property '{logical_op.property_key}'")
                            continue
                        
                        if self._evaluate_filter_condition(property_value, logical_op.operator, logical_op.value):
                            filtered_results.append(result)
                            print(f"     ✅ Filter passed: {entity.get('name', 'unknown')} ({logical_op.property_key}={property_value})")
                        else:
                            print(f"     ❌ Filter failed: {entity.get('name', 'unknown')} ({logical_op.property_key}={property_value})")
                    
                    print(f"   🎯 PropertyFilter result: {len(filtered_results)}/{len(child_results)} passed")
                    return filtered_results
                else:
                    # No child results, execute the filter's own scan operation
                    print(f"   🔍 No child results, executing PropertyFilter directly")
                    return self._execute_property_filter_fixed(logical_op, context)
                    
            elif logical_op_name == "Project":
                # Apply projection to the results from children
                print(f"   🎨 Applying projection to {len(child_results)} child results")
                return self._apply_projection_to_results(child_results, logical_op)
                
            elif logical_op_name in ["NodeScan", "AllNodeScan", "PropertyScan"]:
                # Other leaf scan operations
                print(f"   📊 Executing scan operation: {logical_op_name}")
                return self._execute_logical_operation(logical_op, context)
                
            else:
                print(f"   ⚠️  Unknown logical operation: {logical_op_name}, returning child results")
                return child_results
        
        # Handle operations without logical_op
        elif hasattr(child_operation, 'operation_type'):
            op_type = child_operation.operation_type
            print(f"   🔧 Processing operation type: {op_type}")
            
            if op_type == "Project":
                return self._apply_projection_to_results(child_results, child_operation)
            elif op_type in ["NodeByLabelScan", "PropertyFilter"]:
                return self._execute_redis_commands_fixed(child_operation, context)
            else:
                return child_results
        
        print(f"   ⚠️  Fallback: returning {len(child_results)} child results")
        return child_results
    def _apply_projection_to_results(self, base_results, operation):
        """Apply projection to existing results instead of re-executing scans"""
        
        print(f"🔍 === APPLYING PROJECTION TO {len(base_results)} RESULTS ===")
        
        if not base_results:
            print("⚠️  No base results to project")
            return []
        
        # Get projections from logical operation
        logical_op = getattr(operation, 'logical_op', operation)
        if not hasattr(logical_op, 'projections') or not logical_op.projections:
            print("⚠️  No projections found, returning base results")
            return base_results
        
        print(f"✅ Found {len(logical_op.projections)} projections")
        
        projected_results = []
        for result in base_results:
            projected_record = {}
            
            for expr, alias in logical_op.projections:
                try:
                    # Evaluate expression
                    value = self._evaluate_expression_safely(expr, result)
                    
                    # Determine output key
                    key = alias if alias else self._derive_expression_name(expr)
                    projected_record[key] = value
                    
                    print(f"✅ Projection: {key} = {value}")
                    
                except Exception as e:
                    print(f"❌ Projection failed for {expr}: {e}")
                    key = alias if alias else str(expr)
                    projected_record[key] = None
            
            projected_results.append(projected_record)
        
        print(f"🎉 Projection complete: {len(projected_results)} results")
        return projected_results
    def _execute_property_filter_with_children(self, logical_op, physical_op, context):
        """Execute PropertyFilter by getting results from children first, then applying filter"""
        
        # Get base results from child operations
        base_results = []
        for child in physical_op.children:
            child_results = self._execute_child_operation(child, context)
            base_results.extend(child_results)
        
        if not base_results:
            return []
        
        # Apply the property filter to the base results
        filtered_results = []
        
        for result in base_results:
            entity = result.get(logical_op.variable)
            if not entity or not isinstance(entity, dict):
                continue
            
            property_value = entity.get(logical_op.property_key)
            if property_value is None:
                continue
            
            if self._evaluate_filter_condition(property_value, logical_op.operator, logical_op.value):
                filtered_results.append(result)
        
        return filtered_results

    def _evaluate_filter_condition(self, actual_value, operator, expected_value):
        """Evaluate filter condition with type conversion and operator handling"""
        
        try:
            # Convert values to comparable types
            if isinstance(expected_value, (int, float)) and isinstance(actual_value, str):
                try:
                    actual_value = float(actual_value)
                except ValueError:
                    return False
            
            # Apply operator
            if operator == "=":
                return actual_value == expected_value
            elif operator == "!=":
                return actual_value != expected_value
            elif operator == ">":
                return actual_value > expected_value
            elif operator == ">=":
                return actual_value >= expected_value
            elif operator == "<":
                return actual_value < expected_value
            elif operator == "<=":
                return actual_value <= expected_value
            elif operator.upper() == "CONTAINS":
                return str(expected_value) in str(actual_value)
            elif operator.upper() == "STARTS WITH":
                return str(actual_value).startswith(str(expected_value))
            elif operator.upper() == "ENDS WITH":
                return str(actual_value).endswith(str(expected_value))
            else:
                return False
                
        except Exception:
            return False    
    def _execute_logical_operation(self, logical_op, context):
        """Execute logical operation directly"""
        
        op_type = type(logical_op).__name__
        
        if op_type == "NodeByLabelScan":
            return self._execute_node_by_label_scan_fixed(logical_op, context)
        elif op_type == "AllNodeScan":
            return self._execute_all_node_scan_fixed(logical_op, context)
        elif op_type == "PropertyScan":
            return self._execute_property_scan_fixed(logical_op, context)
        elif op_type == "NodeScan":
            return self._execute_node_scan_fixed(logical_op, context)
        elif op_type == "PropertyFilter":
            return self._execute_property_filter_fixed(logical_op, context)
        else:
            logger.warning(f"Unknown logical operation: {op_type}")
            return []
    
    def _execute_node_by_label_scan_fixed(self, logical_op, context) -> List[Dict[str, Any]]:
        """FIXED: Execute NodeByLabelScan using Redis label indexes"""
        
        logger.debug(f"NodeByLabelScan for variable '{logical_op.variable}' with label '{logical_op.label}'")
        
        # Get node IDs with specified label
        label_key = self.storage.LABEL_NODES_KEY.format(label=logical_op.label)
        node_ids = self.redis.smembers(label_key)
        
        logger.debug(f"Found {len(node_ids)} nodes with label '{logical_op.label}': {list(node_ids)}")
        
        # Apply property filters if specified
        if logical_op.properties:
            filtered_ids = self._filter_nodes_by_properties(node_ids, logical_op.properties)
            logger.debug(f"After property filtering: {len(filtered_ids)} nodes")
        else:
            filtered_ids = node_ids
        
        # Fetch node data and format results
        results = []
        for node_id in filtered_ids:
            node_data = self._get_node_data_complete(node_id)
            if node_data:
                # Format result with variable name as key
                result_record = {logical_op.variable: node_data}
                results.append(result_record)
                logger.debug(f"Added result for node {node_id}: {result_record}")
        
        logger.debug(f"NodeByLabelScan returned {len(results)} results")
        return results
    
    def _execute_all_node_scan_fixed(self, logical_op, context) -> List[Dict[str, Any]]:
        """FIXED: Execute AllNodeScan by scanning all nodes"""
        
        logger.debug(f"AllNodeScan for variable '{logical_op.variable}'")
        
        # Get all node IDs using key pattern scan
        node_keys = []
        for key in self.redis.scan_iter(match="node:*"):
            if key != self.storage.NEXT_NODE_ID_KEY:  # Exclude counter
                node_keys.append(key)
        
        # Extract node IDs
        node_ids = [key.split(':')[1] for key in node_keys]
        logger.debug(f"Found {len(node_ids)} total nodes: {node_ids}")
        
        # Fetch node data and format results
        results = []
        for node_id in node_ids:
            node_data = self._get_node_data_complete(node_id)
            if node_data:
                # Format result with variable name as key
                result_record = {logical_op.variable: node_data}
                results.append(result_record)
        
        logger.debug(f"AllNodeScan returned {len(results)} results")
        return results
    
    def _execute_property_scan_fixed(self, logical_op, context) -> List[Dict[str, Any]]:
        """FIXED: Execute PropertyScan using property indexes"""
        
        logger.debug(f"PropertyScan for variable '{logical_op.variable}' where {logical_op.property_key} = {logical_op.property_value}")
        
        # Get node IDs with specified property value
        prop_key = self.storage.PROPERTY_INDEX_KEY.format(
            property=logical_op.property_key,
            value=logical_op.property_value
        )
        node_ids = self.redis.smembers(prop_key)
        logger.debug(f"Found {len(node_ids)} nodes with property {logical_op.property_key}={logical_op.property_value}")
        
        # Fetch node data and format results
        results = []
        for node_id in node_ids:
            node_data = self._get_node_data_complete(node_id)
            if node_data:
                # Format result with variable name as key
                result_record = {logical_op.variable: node_data}
                results.append(result_record)
        
        logger.debug(f"PropertyScan returned {len(results)} results")
        return results
    
    def _execute_node_scan_fixed(self, logical_op, context) -> List[Dict[str, Any]]:
        """FIXED: Execute legacy NodeScan operation"""
        
        logger.debug(f"NodeScan for variable '{logical_op.variable}' with labels {logical_op.labels}")
        
        # If labels are specified, use label scan
        if logical_op.labels:
            # Use first label for scanning (could be improved to intersect multiple labels)
            label_key = self.storage.LABEL_NODES_KEY.format(label=logical_op.labels[0])
            node_ids = self.redis.smembers(label_key)
            logger.debug(f"Found {len(node_ids)} nodes with label '{logical_op.labels[0]}'")
        else:
            # Scan all nodes
            node_keys = []
            for key in self.redis.scan_iter(match="node:*"):
                if key != self.storage.NEXT_NODE_ID_KEY:
                    node_keys.append(key)
            node_ids = [key.split(':')[1] for key in node_keys]
            logger.debug(f"Found {len(node_ids)} total nodes")
        
        # Apply property filters if specified
        if logical_op.properties:
            filtered_ids = self._filter_nodes_by_properties(node_ids, logical_op.properties)
            logger.debug(f"After property filtering: {len(filtered_ids)} nodes")
        else:
            filtered_ids = node_ids
        
        # Fetch node data and format results
        results = []
        for node_id in filtered_ids:
            node_data = self._get_node_data_complete(node_id)
            if node_data:
                # Format result with variable name as key
                result_record = {logical_op.variable: node_data}
                results.append(result_record)
        
        logger.debug(f"NodeScan returned {len(results)} results")
        return results
    
    def _execute_property_filter_fixed(self, logical_op, context) -> List[Dict[str, Any]]:
        """FIXED: Execute PropertyFilter with full range operator support"""
        
        logger.debug(f"PropertyFilter for variable '{logical_op.variable}' where {logical_op.property_key} {logical_op.operator} {logical_op.value}")
        
        # For range operators, use sorted sets
        if logical_op.operator in ['>', '>=', '<', '<=']:
            node_ids = self._execute_range_property_filter_fixed(logical_op)
        elif logical_op.operator == '=':
            # For equality, use property indexes
            prop_key = self.storage.PROPERTY_INDEX_KEY.format(
                property=logical_op.property_key,
                value=logical_op.value
            )
            node_ids = self.redis.smembers(prop_key)
        elif logical_op.operator == '!=':
            # For inequality, get all nodes and subtract equal ones
            node_ids = self._execute_inequality_filter(logical_op)
        else:
            # For other operators, scan all nodes
            node_ids = self._scan_all_nodes_for_property_filter_fixed(logical_op)
        
        # Fetch filtered node data and format results
        results = []
        for node_id in node_ids:
            node_data = self._get_node_data_complete(node_id)
            if node_data:
                result_record = {logical_op.variable: node_data}
                results.append(result_record)
        
        logger.debug(f"PropertyFilter returned {len(results)} results")
        return results
    def _execute_inequality_filter(self, logical_op) -> Set[str]:
        """Execute != filter by getting all nodes except those with the value"""
        
        # Get all nodes with this property
        all_with_prop = set()
        for key in self.redis.scan_iter(match=f"prop:{logical_op.property_key}:*"):
            all_with_prop.update(self.redis.smembers(key))
        
        # Get nodes with the specific value to exclude
        exclude_key = self.storage.PROPERTY_INDEX_KEY.format(
            property=logical_op.property_key,
            value=logical_op.value
        )
        exclude_nodes = self.redis.smembers(exclude_key)
        
        # Return all except the excluded ones
        return all_with_prop - set(exclude_nodes)
    
    def _execute_range_property_filter_fixed(self, logical_op) -> Set[str]:
        """FIXED: Execute range-based property filter using sorted sets"""
        sorted_key = self.storage.SORTED_PROPERTY_KEY.format(property=logical_op.property_key)
        
        # Build range query parameters
        if logical_op.operator == '>':
            node_ids = self.redis.zrangebyscore(sorted_key, f"({logical_op.value}", "+inf")
        elif logical_op.operator == '>=':
            node_ids = self.redis.zrangebyscore(sorted_key, logical_op.value, "+inf")
        elif logical_op.operator == '<':
            node_ids = self.redis.zrangebyscore(sorted_key, "-inf", f"({logical_op.value}")
        elif logical_op.operator == '<=':
            node_ids = self.redis.zrangebyscore(sorted_key, "-inf", logical_op.value)
        else:
            node_ids = set()
        
        return set(node_ids) if node_ids else set()
    
    def _scan_all_nodes_for_property_filter_fixed(self, logical_op) -> Set[str]:
        """FIXED: Scan all nodes for property filter (fallback for unsupported operators)"""
        matching_node_ids = set()
        
        # Scan all node keys
        for key in self.redis.scan_iter(match="node:*"):
            if key == self.storage.NEXT_NODE_ID_KEY:
                continue
                
            node_id = key.split(':')[1]
            node_data = self.redis.hgetall(key)
            
            # Check if property matches condition
            if logical_op.property_key in node_data:
                prop_value = node_data[logical_op.property_key]
                
                # Try to convert to appropriate type for comparison
                try:
                    if isinstance(logical_op.value, (int, float)):
                        prop_value = float(prop_value)
                    
                    # Apply operator
                    if logical_op.operator == '!=':
                        if prop_value != logical_op.value:
                            matching_node_ids.add(node_id)
                    elif logical_op.operator == 'CONTAINS':
                        if str(logical_op.value) in str(prop_value):
                            matching_node_ids.add(node_id)
                    elif logical_op.operator == 'STARTS WITH':
                        if str(prop_value).startswith(str(logical_op.value)):
                            matching_node_ids.add(node_id)
                    elif logical_op.operator == 'ENDS WITH':
                        if str(prop_value).endswith(str(logical_op.value)):
                            matching_node_ids.add(node_id)
                            
                except (ValueError, TypeError):
                    # Skip if type conversion fails
                    continue
        
        return matching_node_ids
    
    def _execute_redis_commands_fixed(self, operation, context) -> List[Dict[str, Any]]:
        """FIXED: Execute raw Redis commands from the operation"""
        results = []
        
        try:
            # Execute each Redis command
            for cmd in operation.redis_commands:
                if cmd.startswith('#'):  # Skip comments
                    continue
                
                # Parse and execute Redis command
                result = self._execute_redis_command_fixed(cmd, context)
                if result:
                    results.extend(result)
        
        except Exception as e:
            logger.error(f"Redis command execution failed: {e}")
            raise MyLathDBRedisError(f"Redis command execution failed: {e}")
        
        return results
    
    def _execute_redis_command_fixed(self, cmd: str, context) -> List[Dict[str, Any]]:
        """FIXED: Execute a single Redis command and return structured results"""
        cmd = cmd.strip()
        logger.debug(f"Executing Redis command: {cmd}")
        
        if cmd.startswith('SMEMBERS'):
            # Extract set key and return members
            parts = cmd.split()
            if len(parts) >= 2:
                key = parts[1]
                members = self.redis.smembers(key)
                return [{'members': list(members), 'key': key}]
        
        elif cmd.startswith('HGET'):
            # Extract hash key and field
            parts = cmd.split()
            if len(parts) >= 3:
                key_pattern = parts[1]
                field = parts[2]
                
                # Handle parameterized keys like "node:{id}"
                if '{id}' in key_pattern:
                    # This would need node IDs from previous operations
                    # For now, return empty
                    return []
                else:
                    value = self.redis.hget(key_pattern, field)
                    return [{'value': value, 'key': key_pattern, 'field': field}] if value else []
        
        elif cmd.startswith('SINTER'):
            # Set intersection
            sets = cmd.split()[1:]
            if sets:
                result = self.redis.sinter(*sets)
                return [{'intersection': list(result), 'sets': sets}]
        
        elif cmd.startswith('SCAN'):
            # Key scanning
            pattern = None
            if 'MATCH' in cmd:
                match_idx = cmd.find('MATCH')
                pattern = cmd[match_idx + 5:].strip()
            
            keys = []
            for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            return [{'keys': keys, 'pattern': pattern}]
        
        # For other commands, log and return empty
        logger.warning(f"Unsupported Redis command: {cmd}")
        return []
    
    def _filter_nodes_by_properties(self, node_ids: Set[str], 
                                   properties: Dict[str, Any]) -> Set[str]:
        """Filter node IDs by property constraints using Redis operations"""
        filtered_ids = set(node_ids)
        
        for prop_key, prop_value in properties.items():
            # Get nodes with this property value
            prop_key_redis = self.storage.PROPERTY_INDEX_KEY.format(
                property=prop_key, value=prop_value
            )
            nodes_with_prop = self.redis.smembers(prop_key_redis)
            
            # Intersect with current filtered set
            filtered_ids = filtered_ids.intersection(nodes_with_prop)
        
        return filtered_ids
    
    def _get_node_data_complete(self, node_id: str) -> Optional[Dict[str, Any]]:
        """FIXED: Get complete node data including properties and labels with proper typing"""
        node_key = self.storage.NODE_KEY_PATTERN.format(node_id=node_id)
        
        # Get node properties
        node_data = self.redis.hgetall(node_key)
        if not node_data:
            logger.debug(f"No data found for node {node_id}")
            return None
        
        # Get node labels
        labels_key = self.storage.NODE_LABELS_KEY.format(node_id=node_id)
        labels = list(self.redis.smembers(labels_key))
        
        # Build complete node data
        result = {}
        result['_id'] = node_id
        result['_labels'] = labels
        
        # Add properties with type conversion
        for key, value in node_data.items():
            if key.startswith('_'):
                continue
            
            # Try to convert to appropriate type
            try:
                # Try integer first
                if str(value).isdigit() or (str(value).startswith('-') and str(value)[1:].isdigit()):
                    result[key] = int(value)
                # Try float
                elif '.' in str(value):
                    result[key] = float(value)
                else:
                    # Keep as string
                    result[key] = str(value)
            except (ValueError, TypeError):
                # Keep as string if conversion fails
                result[key] = str(value)
        
        logger.debug(f"Retrieved complete node data for {node_id}: {result}")
        return result
    def _get_node_data(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        FIXED: Get node data - alias for _get_node_data_complete
        This method is required by DataBridge but was missing
        """
        return self._get_node_data_complete(node_id)
    def execute_generic_operation(self, physical_plan, context) -> List[Dict[str, Any]]:
        """FIXED: Execute generic physical operation using Redis"""
        
        # Try to map logical operation to Redis operations
        logical_op = getattr(physical_plan, 'logical_op', None)
        
        if logical_op:
            op_type = type(logical_op).__name__
            logger.debug(f"Executing generic operation: {op_type}")
            
            if op_type == "NodeScan":
                return self._execute_node_scan_fixed(logical_op, context)
            elif op_type == "NodeByLabelScan":
                return self._execute_node_by_label_scan_fixed(logical_op, context)
            elif op_type == "AllNodeScan":
                return self._execute_all_node_scan_fixed(logical_op, context)
            elif op_type == "PropertyScan":
                return self._execute_property_scan_fixed(logical_op, context)
            elif op_type == "PropertyFilter":
                return self._execute_property_filter_fixed(logical_op, context)
            elif 'Filter' in op_type:
                return self._handle_generic_filter_fixed(logical_op, context)
        
        # Fallback to empty result
        logger.warning(f"Could not execute generic operation: {type(physical_plan)}")
        return []
    
    def _handle_generic_filter_fixed(self, logical_op, context) -> List[Dict[str, Any]]:
        """FIXED: Handle generic filter operations"""
        logger.debug(f"Handling generic filter: {type(logical_op).__name__}")
        
        # For generic filters, we would need to apply them to existing result sets
        # This would typically be coordinated by the ExecutionCoordinator
        # For now, return empty
        return []
    
    # Data loading methods for graph setup
    
    def load_nodes(self, nodes: List[Dict[str, Any]]):
        """Load nodes into Redis storage"""
        logger.info(f"Loading {len(nodes)} nodes into Redis")
        
        pipeline = self.redis.pipeline() if self.pipeline_enabled else None
        
        for node in nodes:
            node_id = node.get('id') or node.get('_id')
            if not node_id:
                # Generate new node ID
                node_id = str(self.redis.incr(self.storage.NEXT_NODE_ID_KEY))
            
            # Store node properties
            node_key = self.storage.NODE_KEY_PATTERN.format(node_id=node_id)
            properties = {k: v for k, v in node.items() 
                         if not k.startswith('_') and k != 'id'}
            
            if properties:
                if pipeline:
                    pipeline.hset(node_key, mapping=properties)
                else:
                    self.redis.hset(node_key, mapping=properties)
            
            # Store node labels
            labels = node.get('_labels', [])
            if labels:
                labels_key = self.storage.NODE_LABELS_KEY.format(node_id=node_id)
                if pipeline:
                    pipeline.sadd(labels_key, *labels)
                else:
                    self.redis.sadd(labels_key, *labels)
                
                # Add to label indexes
                for label in labels:
                    label_key = self.storage.LABEL_NODES_KEY.format(label=label)
                    if pipeline:
                        pipeline.sadd(label_key, node_id)
                    else:
                        self.redis.sadd(label_key, node_id)
            
            # Create property indexes
            for prop_key, prop_value in properties.items():
                # Create property value index
                prop_index_key = self.storage.PROPERTY_INDEX_KEY.format(
                    property=prop_key, value=prop_value
                )
                if pipeline:
                    pipeline.sadd(prop_index_key, node_id)
                else:
                    self.redis.sadd(prop_index_key, node_id)
                
                # Create sorted property index for range queries
                try:
                    numeric_value = float(prop_value)
                    sorted_prop_key = self.storage.SORTED_PROPERTY_KEY.format(property=prop_key)
                    if pipeline:
                        pipeline.zadd(sorted_prop_key, {node_id: numeric_value})
                    else:
                        self.redis.zadd(sorted_prop_key, {node_id: numeric_value})
                except (ValueError, TypeError):
                    # Not a numeric value, skip sorted index
                    pass
        
        if pipeline:
            pipeline.execute()
        
        # Update metadata
        self.redis.hincrby(self.storage.GRAPH_METADATA_KEY, 'node_count', len(nodes))
        
        logger.info(f"Successfully loaded {len(nodes)} nodes")
    
    def load_edges(self, edges: List[tuple]):
        """Load edges into Redis storage"""
        logger.info(f"Loading {len(edges)} edges into Redis")
        
        pipeline = self.redis.pipeline() if self.pipeline_enabled else None
        
        for edge in edges:
            if len(edge) >= 3:
                src_id, rel_type, dest_id = edge[:3]
                properties = edge[3] if len(edge) > 3 else {}
            else:
                continue
            
            # Generate edge ID
            edge_id = str(self.redis.incr(self.storage.NEXT_EDGE_ID_KEY))
            
            # Store edge properties
            if properties:
                edge_key = self.storage.EDGE_KEY_PATTERN.format(edge_id=edge_id)
                if pipeline:
                    pipeline.hset(edge_key, mapping=properties)
                else:
                    self.redis.hset(edge_key, mapping=properties)
            
            # Store edge endpoints
            endpoints_key = self.storage.EDGE_ENDPOINTS_KEY.format(edge_id=edge_id)
            endpoints_value = f"{src_id}|{dest_id}|{rel_type}"
            if pipeline:
                pipeline.set(endpoints_key, endpoints_value)
            else:
                self.redis.set(endpoints_key, endpoints_value)
            
            # Create relationship indexes
            # Outgoing edges from source
            out_key = self.storage.OUTGOING_EDGES_KEY.format(node_id=src_id, rel_type=rel_type)
            if pipeline:
                pipeline.sadd(out_key, edge_id)
            else:
                self.redis.sadd(out_key, edge_id)
            
            # Incoming edges to destination
            in_key = self.storage.INCOMING_EDGES_KEY.format(node_id=dest_id, rel_type=rel_type)
            if pipeline:
                pipeline.sadd(in_key, edge_id)
            else:
                self.redis.sadd(in_key, edge_id)
            
            # All edges of this relationship type
            rel_key = self.storage.RELATIONSHIP_EDGES_KEY.format(rel_type=rel_type)
            if pipeline:
                pipeline.sadd(rel_key, edge_id)
            else:
                self.redis.sadd(rel_key, edge_id)
        
        if pipeline:
            pipeline.execute()
        
        # Update metadata
        self.redis.hincrby(self.storage.GRAPH_METADATA_KEY, 'edge_count', len(edges))
        
        logger.info(f"Successfully loaded {len(edges)} edges")
    
    def test_connection(self) -> bool:
        """Test Redis connection"""
        try:
            if self.redis:
                self.redis.ping()
                return True
        except Exception:
            pass
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get Redis executor status"""
        status = {
            'connected': self.test_connection(),
            'host': self.config.REDIS_HOST,
            'port': self.config.REDIS_PORT,
            'db': self.config.REDIS_DB
        }
        
        if status['connected']:
            try:
                info = self.redis.info()
                metadata = self.redis.hgetall(self.storage.GRAPH_METADATA_KEY)
                
                status.update({
                    'redis_version': info.get('redis_version'),
                    'used_memory': info.get('used_memory_human'),
                    'connected_clients': info.get('connected_clients'),
                    'node_count': metadata.get('node_count', 0),
                    'edge_count': metadata.get('edge_count', 0)
                })
            except Exception as e:
                status['error'] = str(e)
        
        return status
    
    def shutdown(self):
        """Shutdown Redis executor"""
        logger.info("Shutting down Redis executor")
        
        try:
            if self.redis:
                self.redis.close()
            
            # Stop Redis process if we started it
            if self._redis_process:
                try:
                    self._redis_process.terminate()
                    self._redis_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._redis_process.kill()
                except Exception as e:
                    logger.warning(f"Could not stop Redis process: {e}")
            
        except Exception as e:
            logger.error(f"Error during Redis shutdown: {e}")
        
        logger.info("Redis executor shutdown complete")
    
    # DEBUG METHODS
    
    def debug_redis_state(self):
        """Debug method to check Redis state and data loading"""
        if not self.redis:
            print("❌ Redis not available")
            return
        
        print("🔍 Redis State Debug:")
        
        # Check all keys
        all_keys = list(self.redis.scan_iter())
        print(f"📋 All Redis keys ({len(all_keys)}): {all_keys}")
        
        # Check node data
        node_count = 0
        for key in all_keys:
            if key.startswith('node:') and key != 'next_node_id':
                node_count += 1
                node_id = key.split(':')[1]
                node_data = self.redis.hgetall(key)
                print(f"📝 {key}: {node_data}")
                
                # Check labels for this node
                labels_key = f"node_labels:{node_id}"
                labels = list(self.redis.smembers(labels_key))
                print(f"🏷️  {labels_key}: {labels}")
        
        print(f"📊 Total nodes found: {node_count}")
        
        # Check label indexes
        label_indexes = [key for key in all_keys if key.startswith('label:')]
        print(f"🏷️  Label indexes ({len(label_indexes)}):")
        for key in label_indexes:
            label_name = key.split(':')[1]
            node_ids = list(self.redis.smembers(key))
            print(f"   {key}: {node_ids}")
        
        # Check property indexes
        prop_indexes = [key for key in all_keys if key.startswith('prop:')]
        print(f"🔧 Property indexes ({len(prop_indexes)}):")
        for key in prop_indexes[:5]:  # Show first 5
            node_ids = list(self.redis.smembers(key))
            print(f"   {key}: {node_ids}")
        if len(prop_indexes) > 5:
            print(f"   ... and {len(prop_indexes) - 5} more property indexes")
    
    def debug_query_execution(self, query: str):
        """Debug method to trace query execution step by step"""
        print(f"🔍 Debugging Query: {query}")
        
        try:
            from ..cypher_planner import parse_cypher_query, LogicalPlanner, PhysicalPlanner
            
            # Step 1: Parse AST
            ast = parse_cypher_query(query)
            print(f"✅ AST: {type(ast).__name__}")
            
            if ast.return_clause:
                print(f"   Return items: {len(ast.return_clause.items)}")
                for i, item in enumerate(ast.return_clause.items):
                    print(f"     {i}: {item.expression} (alias: {item.alias})")
            
            # Step 2: Create logical plan
            logical_planner = LogicalPlanner()
            logical_plan = logical_planner.create_logical_plan(ast)
            print(f"✅ Logical Plan: {type(logical_plan).__name__}")
            
            if hasattr(logical_plan, 'projections'):
                print(f"   Projections: {logical_plan.projections}")
            if hasattr(logical_plan, 'logical_op'):
                print(f"   Has logical_op: {logical_plan.logical_op is not None}")
            
            # Step 3: Create physical plan
            physical_planner = PhysicalPlanner()
            physical_plan = physical_planner.create_physical_plan(logical_plan)
            print(f"✅ Physical Plan: {type(physical_plan).__name__}")
            
            if hasattr(physical_plan, 'logical_op'):
                logical_op = physical_plan.logical_op
                print(f"   Physical logical_op: {logical_op is not None}")
                if logical_op and hasattr(logical_op, 'projections'):
                    print(f"   Physical projections: {logical_op.projections}")
            
            # Step 4: Execute with context
            from ..execution_engine.engine import ExecutionContext
            context = ExecutionContext()
            result_data = self.execute_operation(physical_plan, context)
            print(f"✅ Execution: {len(result_data)} results")
            if result_data:
                print(f"   Sample: {result_data[0]}")
            
            return result_data
            
        except Exception as e:
            print(f"❌ Debug failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def test_projection_fix(self):
        """Test if the projection fix is working"""
        print("🧪 Testing Projection Fix...")
        
        # Clear and load minimal test data
        if self.redis:
            self.redis.flushdb()
        
        self.load_nodes([{
            'id': '1', 
            'name': 'Alice', 
            'age': 30, 
            '_labels': ['Person']
        }])
        
        # Test projection execution directly
        try:
            from ..cypher_planner import parse_cypher_query, LogicalPlanner, PhysicalPlanner
            from ..execution_engine.engine import ExecutionContext
            
            # Parse and plan
            ast = parse_cypher_query("MATCH (n:Person) RETURN n.name")
            logical_planner = LogicalPlanner()
            logical_plan = logical_planner.create_logical_plan(ast)
            physical_planner = PhysicalPlanner()
            physical_plan = physical_planner.create_physical_plan(logical_plan)
            
            # Execute
            context = ExecutionContext()
            result = self.execute_operation(physical_plan, context)
            
            print(f"Success: {len(result)} results")
            if result:
                first_result = result[0]
                print(f"Result: {first_result}")
                print(f"Result keys: {list(first_result.keys())}")
                
                # Check if projection worked
                if 'n.name' in first_result and first_result['n.name'] == 'Alice':
                    print("✅ PROJECTION SUCCESSFUL!")
                    return True
                else:
                    print("❌ PROJECTION FAILED")
                    return False
            else:
                print("❌ No results returned")
                return False
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    def _execute_generic_filter_operation(self, operation, context) -> List[Dict[str, Any]]:
        """Execute generic filter operation by applying it to child results"""
        
        # Get base results from children
        base_results = []
        for child in operation.children:
            child_results = self._execute_child_operation(child, context)
            base_results.extend(child_results)
        
        # Get filter condition from logical operation
        logical_op = getattr(operation, 'logical_op', None)
        if not logical_op or not hasattr(logical_op, 'condition'):
            return base_results
        
        # Apply filter condition to each result
        filtered_results = []
        for result in base_results:
            if self._evaluate_filter_condition_on_result(logical_op.condition, result):
                filtered_results.append(result)
        
        return filtered_results

    def _evaluate_filter_condition_on_result(self, condition, result) -> bool:
        """Evaluate filter condition against a result record"""
        
        from ..cypher_planner.ast_nodes import (
            BinaryExpression, PropertyExpression, LiteralExpression, VariableExpression
        )
        
        if isinstance(condition, BinaryExpression):
            if condition.operator.upper() == "AND":
                left_result = self._evaluate_filter_condition_on_result(condition.left, result)
                right_result = self._evaluate_filter_condition_on_result(condition.right, result)
                return left_result and right_result
            elif condition.operator.upper() == "OR":
                left_result = self._evaluate_filter_condition_on_result(condition.left, result)
                right_result = self._evaluate_filter_condition_on_result(condition.right, result)
                return left_result or right_result
            else:
                # Property comparison
                left_val = self._evaluate_expression_in_result(condition.left, result)
                right_val = self._evaluate_expression_in_result(condition.right, result)
                return self._evaluate_filter_condition(left_val, condition.operator, right_val)
        
        return True

    def _evaluate_expression_in_result(self, expr, result):
        """Evaluate expression against result record"""
        
        from ..cypher_planner.ast_nodes import PropertyExpression, LiteralExpression, VariableExpression
        
        if isinstance(expr, PropertyExpression):
            entity = result.get(expr.variable)
            if entity and isinstance(entity, dict):
                return entity.get(expr.property_name)
            return None
        elif isinstance(expr, LiteralExpression):
            return expr.value
        elif isinstance(expr, VariableExpression):
            return result.get(expr.name)
        else:
            return None