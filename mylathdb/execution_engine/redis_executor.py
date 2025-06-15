# mylathdb/execution_engine/redis_executor.py

"""
MyLathDB Redis Executor - FIXED VERSION
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
        Execute Redis operation from physical plan - FIXED VERSION
        
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
        
        # FIXED: Route to appropriate handler based on operation type
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
        elif operation_type == "Project":
            return self._execute_project_fixed(redis_operation, context)
        elif operation_type == "OrderBy":
            return self._execute_order_by_fixed(redis_operation, context)
        elif operation_type == "Limit":
            return self._execute_limit_fixed(redis_operation, context)
        else:
            # Execute Redis commands directly
            return self._execute_redis_commands_fixed(redis_operation, context)
    
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
        """FIXED: Execute PropertyFilter operation"""
        
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
        else:
            # For other operators, scan all nodes
            node_ids = self._scan_all_nodes_for_property_filter_fixed(logical_op)
        
        # Fetch filtered node data and format results
        results = []
        for node_id in node_ids:
            node_data = self._get_node_data_complete(node_id)
            if node_data:
                # Format result with variable name as key
                result_record = {logical_op.variable: node_data}
                results.append(result_record)
        
        logger.debug(f"PropertyFilter returned {len(results)} results")
        return results
    
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
    
    def _execute_project_fixed(self, operation, context) -> List[Dict[str, Any]]:
        """FIXED: Execute Project operation - extract specific properties"""
        # Project operations are typically handled by combining with other operations
        # For now, return empty (would be handled by coordinator)
        logger.debug("Project operation executed (pass-through)")
        return []
    
    def _execute_order_by_fixed(self, operation, context) -> List[Dict[str, Any]]:
        """FIXED: Execute OrderBy operation"""
        # OrderBy is typically applied to previous results by coordinator
        logger.debug("OrderBy operation executed (pass-through)")
        return []
    
    def _execute_limit_fixed(self, operation, context) -> List[Dict[str, Any]]:
        """FIXED: Execute Limit operation"""
        # Limit is typically applied to previous results by coordinator
        logger.debug("Limit operation executed (pass-through)")
        return []
    
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