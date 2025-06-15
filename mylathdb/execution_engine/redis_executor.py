# mylathdb/execution_engine/redis_executor.py

"""
MyLathDB Redis Executor
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
        Execute Redis operation from physical plan
        
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
        
        if operation_type == "NodeByLabelScan":
            return self._execute_node_by_label_scan(redis_operation, context)
        elif operation_type == "AllNodeScan":
            return self._execute_all_node_scan(redis_operation, context)
        elif operation_type == "PropertyScan":
            return self._execute_property_scan(redis_operation, context)
        elif operation_type == "PropertyFilter":
            return self._execute_property_filter(redis_operation, context)
        elif operation_type == "Project":
            return self._execute_project(redis_operation, context)
        elif operation_type == "OrderBy":
            return self._execute_order_by(redis_operation, context)
        elif operation_type == "Limit":
            return self._execute_limit(redis_operation, context)
        else:
            # Execute generic Redis commands
            return self._execute_redis_commands(redis_operation, context)
    
    def _execute_node_by_label_scan(self, operation, context) -> List[Dict[str, Any]]:
        """Execute NodeByLabelScan using Redis label indexes"""
        logical_op = operation.logical_op
        
        # Get node IDs with specified label
        label_key = self.storage.LABEL_NODES_KEY.format(label=logical_op.label)
        node_ids = self.redis.smembers(label_key)
        
        # Apply property filters if specified
        if logical_op.properties:
            filtered_ids = self._filter_nodes_by_properties(node_ids, logical_op.properties)
        else:
            filtered_ids = node_ids
        
        # Fetch node data
        results = []
        for node_id in filtered_ids:
            node_data = self._get_node_data(node_id)
            if node_data:
                node_data['_id'] = node_id
                node_data['_labels'] = [logical_op.label]
                results.append({logical_op.variable: node_data})
        
        logger.debug(f"NodeByLabelScan returned {len(results)} nodes")
        return results
    
    def _execute_all_node_scan(self, operation, context) -> List[Dict[str, Any]]:
        """Execute AllNodeScan by scanning all nodes"""
        logical_op = operation.logical_op
        
        # Get all node IDs using key pattern scan
        node_keys = []
        for key in self.redis.scan_iter(match="node:*"):
            if key != self.storage.NEXT_NODE_ID_KEY:  # Exclude counter
                node_keys.append(key)
        
        # Extract node IDs
        node_ids = [key.split(':')[1] for key in node_keys]
        
        # Fetch node data
        results = []
        for node_id in node_ids:
            node_data = self._get_node_data(node_id)
            if node_data:
                node_data['_id'] = node_id
                results.append({logical_op.variable: node_data})
        
        logger.debug(f"AllNodeScan returned {len(results)} nodes")
        return results
    
    def _execute_property_scan(self, operation, context) -> List[Dict[str, Any]]:
        """Execute PropertyScan using property indexes"""
        logical_op = operation.logical_op
        
        # Get node IDs with specified property value
        prop_key = self.storage.PROPERTY_INDEX_KEY.format(
            property=logical_op.property_key,
            value=logical_op.property_value
        )
        node_ids = self.redis.smembers(prop_key)
        
        # Fetch node data
        results = []
        for node_id in node_ids:
            node_data = self._get_node_data(node_id)
            if node_data:
                node_data['_id'] = node_id
                results.append({logical_op.variable: node_data})
        
        logger.debug(f"PropertyScan returned {len(results)} nodes")
        return results
    
    def _execute_property_filter(self, operation, context) -> List[Dict[str, Any]]:
        """Execute PropertyFilter operation"""
        logical_op = operation.logical_op
        
        # For range operators, use sorted sets
        if logical_op.operator in ['>', '>=', '<', '<=']:
            return self._execute_range_property_filter(logical_op)
        
        # For equality, use property indexes
        if logical_op.operator == '=':
            prop_key = self.storage.PROPERTY_INDEX_KEY.format(
                property=logical_op.property_key,
                value=logical_op.value
            )
            node_ids = self.redis.smembers(prop_key)
        else:
            # For other operators, scan all nodes (expensive)
            node_ids = self._scan_all_nodes_for_property_filter(logical_op)
        
        # Fetch filtered node data
        results = []
        for node_id in node_ids:
            node_data = self._get_node_data(node_id)
            if node_data:
                node_data['_id'] = node_id
                results.append({logical_op.variable: node_data})
        
        return results
    
    def _execute_range_property_filter(self, logical_op) -> List[Dict[str, Any]]:
        """Execute range-based property filter using sorted sets"""
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
            node_ids = []
        
        # Fetch filtered node data
        results = []
        for node_id in node_ids:
            node_data = self._get_node_data(node_id)
            if node_data:
                node_data['_id'] = node_id
                results.append({logical_op.variable: node_data})
        
        return results
    
    def _scan_all_nodes_for_property_filter(self, logical_op) -> List[str]:
        """Scan all nodes for property filter (fallback for unsupported operators)"""
        node_ids = []
        
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
                            node_ids.append(node_id)
                    elif logical_op.operator == 'CONTAINS':
                        if str(logical_op.value) in str(prop_value):
                            node_ids.append(node_id)
                    elif logical_op.operator == 'STARTS WITH':
                        if str(prop_value).startswith(str(logical_op.value)):
                            node_ids.append(node_id)
                    elif logical_op.operator == 'ENDS WITH':
                        if str(prop_value).endswith(str(logical_op.value)):
                            node_ids.append(node_id)
                            
                except (ValueError, TypeError):
                    # Skip if type conversion fails
                    continue
        
        return node_ids
    
    def _execute_project(self, operation, context) -> List[Dict[str, Any]]:
        """Execute Project operation - extract specific properties"""
        # For Redis operations, this is typically handled by fetching specific fields
        # This is a pass-through that would be combined with other operations
        return []  # Project is usually combined with scan operations
    
    def _execute_order_by(self, operation, context) -> List[Dict[str, Any]]:
        """Execute OrderBy operation using Redis SORT"""
        # Redis SORT is limited, so this might need to be done in-memory
        # For now, return empty (ordering would be handled by coordinator)
        return []
    
    def _execute_limit(self, operation, context) -> List[Dict[str, Any]]:
        """Execute Limit operation"""
        # Limit is typically applied to previous results
        # This is handled by the coordinator
        return []
    
    def _execute_redis_commands(self, operation, context) -> List[Dict[str, Any]]:
        """Execute raw Redis commands from the operation"""
        results = []
        
        try:
            # Execute each Redis command
            for cmd in operation.redis_commands:
                if cmd.startswith('#'):  # Skip comments
                    continue
                
                # Parse and execute Redis command
                result = self._execute_redis_command(cmd, context)
                if result:
                    results.extend(result)
        
        except Exception as e:
            raise MyLathDBRedisError(f"Redis command execution failed: {e}")
        
        return results
    
    def _execute_redis_command(self, cmd: str, context) -> List[Dict[str, Any]]:
        """Execute a single Redis command and return structured results"""
        # Simple command parsing (would need more sophisticated parsing in production)
        cmd = cmd.strip()
        
        if cmd.startswith('SMEMBERS'):
            # Extract set key and return members
            key = cmd.split()[1]
            members = self.redis.smembers(key)
            return [{'result': list(members)}]
        
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
                    return [{'result': value}] if value else []
        
        elif cmd.startswith('SINTER'):
            # Set intersection
            sets = cmd.split()[1:]
            if sets:
                result = self.redis.sinter(*sets)
                return [{'result': list(result)}]
        
        elif cmd.startswith('SCAN'):
            # Key scanning
            pattern = None
            if 'MATCH' in cmd:
                match_idx = cmd.find('MATCH')
                pattern = cmd[match_idx + 5:].strip()
            
            keys = []
            for key in self.redis.scan_iter(match=pattern):
                keys.append(key)
            
            return [{'result': keys}]
        
        # For other commands, try to execute directly (be careful!)
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
    
    def _get_node_data(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get complete node data including properties and labels"""
        node_key = self.storage.NODE_KEY_PATTERN.format(node_id=node_id)
        
        # Get node properties
        node_data = self.redis.hgetall(node_key)
        if not node_data:
            return None
        
        # Get node labels
        labels_key = self.storage.NODE_LABELS_KEY.format(node_id=node_id)
        labels = list(self.redis.smembers(labels_key))
        
        # Combine data
        result = dict(node_data)
        result['_id'] = node_id
        result['_labels'] = labels
        
        # Type conversion for numeric values
        for key, value in result.items():
            if key.startswith('_'):
                continue
            try:
                # Try to convert to number if possible
                if '.' in str(value):
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except (ValueError, TypeError):
                # Keep as string
                pass
        
        return result
    
    def execute_generic_operation(self, physical_plan, context) -> List[Dict[str, Any]]:
        """Execute generic physical operation using Redis"""
        # Try to map logical operation to Redis operations
        logical_op = getattr(physical_plan, 'logical_op', None)
        
        if logical_op:
            op_type = type(logical_op).__name__
            
            if 'Scan' in op_type:
                return self._handle_generic_scan(logical_op, context)
            elif 'Filter' in op_type:
                return self._handle_generic_filter(logical_op, context)
        
        # Fallback to empty result
        logger.warning(f"Could not execute generic operation: {type(physical_plan)}")
        return []
    
    def _handle_generic_scan(self, logical_op, context) -> List[Dict[str, Any]]:
        """Handle generic scan operations"""
        op_type = type(logical_op).__name__
        
        if op_type == "NodeScan":
            if logical_op.labels:
                # Use first label for scanning
                label_key = self.storage.LABEL_NODES_KEY.format(label=logical_op.labels[0])
                node_ids = self.redis.smembers(label_key)
                
                # Apply property filters
                if logical_op.properties:
                    node_ids = self._filter_nodes_by_properties(node_ids, logical_op.properties)
                
                # Fetch node data
                results = []
                for node_id in node_ids:
                    node_data = self._get_node_data(node_id)
                    if node_data:
                        results.append({logical_op.variable: node_data})
                
                return results
        
        return []
    
    def _handle_generic_filter(self, logical_op, context) -> List[Dict[str, Any]]:
        """Handle generic filter operations"""
        # For generic filters, we would need to apply them to existing result sets
        # This would typically be coordinated by the ExecutionCoordinator
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
