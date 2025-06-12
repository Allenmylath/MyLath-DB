# execution_engine/redis_executor.py

"""
Redis Executor - Executes Redis operations for node/property operations
"""

from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
import redis
import json
import logging
from datetime import datetime, timedelta

from cypher_planner.physical_planner import RedisOperation


@dataclass
class RedisResult:
    """Result of Redis operation execution"""
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    node_ids: Set[str] = field(default_factory=set)
    properties: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_time: float = 0.0
    operations_count: int = 0
    cache_hits: int = 0
    error: Optional[str] = None


class RedisExecutor:
    """Executes Redis operations for graph data access"""
    
    def __init__(self, redis_client=None, enable_caching=True, cache_ttl=3600):
        # Redis connection
        self.redis = redis_client or self._create_default_redis()
        
        # Configuration
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # Statistics
        self.stats = {
            'operations_executed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_execution_time': 0.0
        }
        
        # Cache for expensive operations
        self.operation_cache = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def _create_default_redis(self):
        """Create default Redis connection"""
        try:
            return redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        except Exception as e:
            self.logger.warning(f"Could not connect to Redis: {e}")
            return None
    
    def execute(self, operation: RedisOperation, context, input_data=None) -> RedisResult:
        """Execute a Redis operation"""
        
        start_time = datetime.now()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key(operation, input_data)
            if self.enable_caching and cache_key in self.operation_cache:
                cached_result = self.operation_cache[cache_key]
                if self._is_cache_valid(cached_result):
                    self.stats['cache_hits'] += 1
                    return cached_result['result']
            
            # Execute based on operation type
            if operation.operation_type == "NodeByLabelScan":
                result = self._execute_node_by_label_scan(operation, context, input_data)
            elif operation.operation_type == "AllNodeScan":
                result = self._execute_all_node_scan(operation, context, input_data)
            elif operation.operation_type == "PropertyScan":
                result = self._execute_property_scan(operation, context, input_data)
            elif operation.operation_type == "PropertyFilter":
                result = self._execute_property_filter(operation, context, input_data)
            elif operation.operation_type == "NodeScan":
                result = self._execute_node_scan(operation, context, input_data)
            elif operation.operation_type == "Project":
                result = self._execute_project(operation, context, input_data)
            elif operation.operation_type == "OrderBy":
                result = self._execute_order_by(operation, context, input_data)
            elif operation.operation_type == "Limit":
                result = self._execute_limit(operation, context, input_data)
            else:
                result = self._execute_generic_redis(operation, context, input_data)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Update statistics
            self.stats['operations_executed'] += 1
            self.stats['total_execution_time'] += execution_time
            if not (self.enable_caching and cache_key in self.operation_cache):
                self.stats['cache_misses'] += 1
            
            # Cache result if caching is enabled
            if self.enable_caching and result.success:
                self.operation_cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now(),
                    'ttl': self.cache_ttl
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Redis operation failed: {str(e)}")
            return RedisResult(
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _execute_node_by_label_scan(self, operation: RedisOperation, context, input_data) -> RedisResult:
        """Execute NodeByLabelScan operation"""
        
        result = RedisResult(success=True)
        
        if not self.redis:
            return RedisResult(success=False, error="No Redis connection")
        
        try:
            # Extract label from logical operation if available
            logical_op = operation.logical_op
            if logical_op and hasattr(logical_op, 'label'):
                label = logical_op.label
                variable = logical_op.variable
                properties = getattr(logical_op, 'properties', {})
            else:
                # Parse from Redis commands as fallback
                label = self._extract_label_from_commands(operation.redis_commands)
                variable = "n"  # default
                properties = {}
            
            # Get nodes with label
            if label:
                label_key = f"label:{label}"
                node_ids = set(self.redis.smembers(label_key))
                
                if not node_ids:
                    # If no label index, scan for nodes
                    node_ids = self._scan_nodes_by_label(label)
            else:
                node_ids = set()
            
            # Apply property filters if any
            if properties:
                filtered_node_ids = self._filter_nodes_by_properties(node_ids, properties)
                node_ids = filtered_node_ids
            
            # Load node properties
            node_properties = {}
            for node_id in node_ids:
                props = self._get_node_properties(node_id)
                if props:
                    node_properties[node_id] = props
            
            # Set variable in context
            if variable and context:
                context.set_variable(variable, list(node_ids))
            
            result.node_ids = node_ids
            result.properties = node_properties
            result.data = {
                'nodes': list(node_ids),
                'properties': node_properties,
                'variable': variable
            }
            result.operations_count = len(operation.redis_commands)
            
            return result
            
        except Exception as e:
            return RedisResult(success=False, error=f"NodeByLabelScan failed: {str(e)}")
    
    def _execute_property_scan(self, operation: RedisOperation, context, input_data) -> RedisResult:
        """Execute PropertyScan operation"""
        
        result = RedisResult(success=True)
        
        if not self.redis:
            return RedisResult(success=False, error="No Redis connection")
        
        try:
            logical_op = operation.logical_op
            if logical_op and hasattr(logical_op, 'property_key'):
                property_key = logical_op.property_key
                property_value = logical_op.property_value
                variable = logical_op.variable
            else:
                # Parse from commands
                property_key, property_value = self._extract_property_from_commands(operation.redis_commands)
                variable = "n"
            
            # Get nodes with property value
            prop_key = f"prop:{property_key}:{property_value}"
            node_ids = set(self.redis.smembers(prop_key))
            
            if not node_ids:
                # Fallback to scanning all nodes
                node_ids = self._scan_nodes_by_property(property_key, property_value)
            
            # Load node properties
            node_properties = {}
            for node_id in node_ids:
                props = self._get_node_properties(node_id)
                if props:
                    node_properties[node_id] = props
            
            # Set variable in context
            if variable and context:
                context.set_variable(variable, list(node_ids))
            
            result.node_ids = node_ids
            result.properties = node_properties
            result.data = {
                'nodes': list(node_ids),
                'properties': node_properties,
                'variable': variable
            }
            
            return result
            
        except Exception as e:
            return RedisResult(success=False, error=f"PropertyScan failed: {str(e)}")
    
    def _execute_property_filter(self, operation: RedisOperation, context, input_data) -> RedisResult:
        """Execute PropertyFilter operation"""
        
        result = RedisResult(success=True)
        
        try:
            # Get input nodes from child operations or context
            input_nodes = set()
            if input_data and hasattr(input_data, 'node_ids'):
                input_nodes = input_data.node_ids
            elif input_data and isinstance(input_data, dict) and 'nodes' in input_data:
                input_nodes = set(input_data['nodes'])
            
            logical_op = operation.logical_op
            if logical_op and hasattr(logical_op, 'property_key'):
                property_key = logical_op.property_key
                operator = logical_op.operator
                value = logical_op.value
                variable = logical_op.variable
            else:
                # Parse from commands - simplified
                property_key = "name"  # default
                operator = "="
                value = ""
                variable = "n"
            
            # Filter nodes based on property condition
            filtered_nodes = set()
            node_properties = {}
            
            for node_id in input_nodes:
                props = self._get_node_properties(node_id)
                if props and property_key in props:
                    prop_value = props[property_key]
                    
                    # Apply operator
                    if self._evaluate_condition(prop_value, operator, value):
                        filtered_nodes.add(node_id)
                        node_properties[node_id] = props
            
            # Update context variable
            if variable and context:
                context.set_variable(variable, list(filtered_nodes))
            
            result.node_ids = filtered_nodes
            result.properties = node_properties
            result.data = {
                'nodes': list(filtered_nodes),
                'properties': node_properties,
                'variable': variable
            }
            
            return result
            
        except Exception as e:
            return RedisResult(success=False, error=f"PropertyFilter failed: {str(e)}")
    
    def _execute_all_node_scan(self, operation: RedisOperation, context, input_data) -> RedisResult:
        """Execute AllNodeScan operation"""
        
        result = RedisResult(success=True)
        
        if not self.redis:
            return RedisResult(success=False, error="No Redis connection")
        
        try:
            # Scan all nodes
            node_ids = self._scan_all_nodes()
            
            # Load properties for all nodes (be careful with large datasets)
            node_properties = {}
            for node_id in list(node_ids)[:1000]:  # Limit to first 1000 for safety
                props = self._get_node_properties(node_id)
                if props:
                    node_properties[node_id] = props
            
            logical_op = operation.logical_op
            variable = logical_op.variable if logical_op else "n"
            
            # Set variable in context
            if variable and context:
                context.set_variable(variable, list(node_ids))
            
            result.node_ids = node_ids
            result.properties = node_properties
            result.data = {
                'nodes': list(node_ids),
                'properties': node_properties,
                'variable': variable
            }
            
            return result
            
        except Exception as e:
            return RedisResult(success=False, error=f"AllNodeScan failed: {str(e)}")
    
    def _execute_node_scan(self, operation: RedisOperation, context, input_data) -> RedisResult:
        """Execute legacy NodeScan operation"""
        
        # Delegate to appropriate specific scan based on operation details
        logical_op = operation.logical_op
        
        if logical_op and hasattr(logical_op, 'labels') and logical_op.labels:
            # Has labels - convert to NodeByLabelScan
            return self._execute_node_by_label_scan(operation, context, input_data)
        else:
            # No labels - convert to AllNodeScan
            return self._execute_all_node_scan(operation, context, input_data)
    
    def _execute_project(self, operation: RedisOperation, context, input_data) -> RedisResult:
        """Execute Project operation"""
        
        result = RedisResult(success=True)
        
        try:
            # Get input data
            if input_data and hasattr(input_data, 'data'):
                input_data_dict = input_data.data
            elif isinstance(input_data, dict):
                input_data_dict = input_data
            else:
                input_data_dict = {}
            
            # For now, just pass through the data
            # In a real implementation, you'd apply the projection
            result.data = input_data_dict.copy()
            
            return result
            
        except Exception as e:
            return RedisResult(success=False, error=f"Project failed: {str(e)}")
    
    def _execute_order_by(self, operation: RedisOperation, context, input_data) -> RedisResult:
        """Execute OrderBy operation"""
        
        result = RedisResult(success=True)
        
        try:
            # Get input data
            if input_data and hasattr(input_data, 'data'):
                data = input_data.data.copy()
            elif isinstance(input_data, dict):
                data = input_data.copy()
            else:
                data = {}
            
            # For now, just pass through the data
            # In a real implementation, you'd sort the results
            result.data = data
            
            return result
            
        except Exception as e:
            return RedisResult(success=False, error=f"OrderBy failed: {str(e)}")
    
    def _execute_limit(self, operation: RedisOperation, context, input_data) -> RedisResult:
        """Execute Limit operation"""
        
        result = RedisResult(success=True)
        
        try:
            # Get input data
            if input_data and hasattr(input_data, 'data'):
                data = input_data.data.copy()
            elif isinstance(input_data, dict):
                data = input_data.copy()
            else:
                data = {}
            
            # Apply limit if nodes exist
            if 'nodes' in data and isinstance(data['nodes'], list):
                # Parse limit from commands (simplified)
                limit = 100  # default
                for cmd in operation.redis_commands:
                    if 'LIMIT' in cmd:
                        try:
                            limit = int(cmd.split('LIMIT')[1].strip())
                        except:
                            pass
                
                data['nodes'] = data['nodes'][:limit]
                
                # Also limit properties
                if 'properties' in data:
                    limited_props = {}
                    for node_id in data['nodes']:
                        if node_id in data['properties']:
                            limited_props[node_id] = data['properties'][node_id]
                    data['properties'] = limited_props
            
            result.data = data
            
            return result
            
        except Exception as e:
            return RedisResult(success=False, error=f"Limit failed: {str(e)}")
    
    def _execute_generic_redis(self, operation: RedisOperation, context, input_data) -> RedisResult:
        """Execute generic Redis operation"""
        
        result = RedisResult(success=True)
        
        try:
            # Execute Redis commands directly
            command_results = []
            
            for command in operation.redis_commands:
                if self.redis and not command.startswith('#'):  # Skip comments
                    try:
                        # Parse and execute command (simplified)
                        cmd_result = self._execute_redis_command(command)
                        command_results.append(cmd_result)
                    except Exception as e:
                        self.logger.warning(f"Failed to execute Redis command '{command}': {e}")
            
            result.data = {
                'command_results': command_results,
                'operation_type': operation.operation_type
            }
            result.operations_count = len(operation.redis_commands)
            
            return result
            
        except Exception as e:
            return RedisResult(success=False, error=f"Generic Redis operation failed: {str(e)}")
    
    # Helper methods
    
    def _get_node_properties(self, node_id: str) -> Dict[str, Any]:
        """Get properties for a node"""
        if not self.redis:
            return {}
        
        try:
            node_key = f"node:{node_id}"
            props = self.redis.hgetall(node_key)
            
            # Convert Redis strings back to appropriate types
            typed_props = {}
            for key, value in props.items():
                try:
                    # Try to parse as JSON first (for complex types)
                    typed_props[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # Keep as string if not valid JSON
                    typed_props[key] = value
            
            return typed_props
            
        except Exception as e:
            self.logger.warning(f"Failed to get properties for node {node_id}: {e}")
            return {}
    
    def _scan_nodes_by_label(self, label: str) -> Set[str]:
        """Scan for nodes with specific label"""
        if not self.redis:
            return set()
        
        try:
            node_ids = set()
            cursor = 0
            
            while True:
                cursor, keys = self.redis.scan(cursor, match=f"node:*", count=100)
                
                for key in keys:
                    # Check if node has the label
                    if self.redis.sismember(f"node_labels:{key.split(':')[1]}", label):
                        node_ids.add(key.split(':')[1])
                
                if cursor == 0:
                    break
            
            return node_ids
            
        except Exception as e:
            self.logger.warning(f"Failed to scan nodes by label {label}: {e}")
            return set()
    
    def _scan_nodes_by_property(self, property_key: str, property_value: Any) -> Set[str]:
        """Scan for nodes with specific property value"""
        if not self.redis:
            return set()
        
        try:
            node_ids = set()
            cursor = 0
            
            while True:
                cursor, keys = self.redis.scan(cursor, match="node:*", count=100)
                
                for key in keys:
                    node_id = key.split(':')[1]
                    props = self._get_node_properties(node_id)
                    
                    if property_key in props and props[property_key] == property_value:
                        node_ids.add(node_id)
                
                if cursor == 0:
                    break
            
            return node_ids
            
        except Exception as e:
            self.logger.warning(f"Failed to scan nodes by property {property_key}={property_value}: {e}")
            return set()
    
    def _scan_all_nodes(self) -> Set[str]:
        """Scan all nodes in the database"""
        if not self.redis:
            return set()
        
        try:
            node_ids = set()
            cursor = 0
            
            while True:
                cursor, keys = self.redis.scan(cursor, match="node:*", count=100)
                
                for key in keys:
                    node_ids.add(key.split(':')[1])
                
                if cursor == 0:
                    break
            
            return node_ids
            
        except Exception as e:
            self.logger.warning(f"Failed to scan all nodes: {e}")
            return set()
    
    def _filter_nodes_by_properties(self, node_ids: Set[str], properties: Dict[str, Any]) -> Set[str]:
        """Filter nodes by property conditions"""
        filtered_ids = set()
        
        for node_id in node_ids:
            node_props = self._get_node_properties(node_id)
            
            # Check if all required properties match
            matches = True
            for prop_key, prop_value in properties.items():
                if prop_key not in node_props or node_props[prop_key] != prop_value:
                    matches = False
                    break
            
            if matches:
                filtered_ids.add(node_id)
        
        return filtered_ids
    
    def _evaluate_condition(self, value1: Any, operator: str, value2: Any) -> bool:
        """Evaluate a comparison condition"""
        try:
            if operator == "=":
                return value1 == value2
            elif operator == "!=":
                return value1 != value2
            elif operator == "<":
                return value1 < value2
            elif operator == "<=":
                return value1 <= value2
            elif operator == ">":
                return value1 > value2
            elif operator == ">=":
                return value1 >= value2
            elif operator == "CONTAINS":
                return str(value2).lower() in str(value1).lower()
            elif operator == "STARTS WITH":
                return str(value1).lower().startswith(str(value2).lower())
            elif operator == "ENDS WITH":
                return str(value1).lower().endswith(str(value2).lower())
            else:
                self.logger.warning(f"Unknown operator: {operator}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Failed to evaluate condition {value1} {operator} {value2}: {e}")
            return False
    
    def _execute_redis_command(self, command: str) -> Any:
        """Execute a raw Redis command"""
        if not self.redis:
            return None
        
        try:
            # Parse command (very basic implementation)
            parts = command.strip().split()
            if not parts:
                return None
            
            cmd = parts[0].upper()
            args = parts[1:]
            
            if cmd == "SMEMBERS":
                return list(self.redis.smembers(args[0])) if args else []
            elif cmd == "HGETALL":
                return self.redis.hgetall(args[0]) if args else {}
            elif cmd == "HGET":
                return self.redis.hget(args[0], args[1]) if len(args) >= 2 else None
            elif cmd == "SINTER":
                return list(self.redis.sinter(*args)) if args else []
            elif cmd == "SCAN":
                cursor = int(args[0]) if args else 0
                pattern = args[2] if len(args) > 2 and args[1] == "MATCH" else "*"
                cursor, keys = self.redis.scan(cursor, match=pattern)
                return {"cursor": cursor, "keys": keys}
            else:
                self.logger.warning(f"Unsupported Redis command: {cmd}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to execute Redis command '{command}': {e}")
            return None
    
    def _extract_label_from_commands(self, commands: List[str]) -> str:
        """Extract label from Redis commands"""
        for command in commands:
            if "label:" in command:
                try:
                    label_part = command.split("label:")[1]
                    label = label_part.split()[0].split("}")[0]
                    return label
                except:
                    pass
        return ""
    
    def _extract_property_from_commands(self, commands: List[str]) -> tuple:
        """Extract property key and value from Redis commands"""
        for command in commands:
            if "prop:" in command:
                try:
                    prop_part = command.split("prop:")[1]
                    parts = prop_part.split(":")
                    if len(parts) >= 2:
                        return parts[0], parts[1].split()[0]
                except:
                    pass
        return "", ""
    
    def _get_cache_key(self, operation: RedisOperation, input_data) -> str:
        """Generate cache key for operation"""
        key_parts = [
            operation.operation_type,
            str(hash(tuple(operation.redis_commands))),
            str(hash(str(input_data))) if input_data else "no_input"
        ]
        return "|".join(key_parts)
    
    def _is_cache_valid(self, cached_entry: Dict) -> bool:
        """Check if cached entry is still valid"""
        if 'timestamp' not in cached_entry or 'ttl' not in cached_entry:
            return False
        
        age = (datetime.now() - cached_entry['timestamp']).total_seconds()
        return age < cached_entry['ttl']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics"""
        cache_hit_rate = 0.0
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
        
        return {
            'operations_executed': self.stats['operations_executed'],
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'total_execution_time': self.stats['total_execution_time'],
            'avg_execution_time': self.stats['total_execution_time'] / max(1, self.stats['operations_executed']),
            'cache_size': len(self.operation_cache),
            'redis_connected': self.redis is not None
        }
    
    def clear_cache(self):
        """Clear operation cache"""
        self.operation_cache.clear()
        self.logger.info("Redis executor cache cleared")
    
    def shutdown(self):
        """Shutdown Redis executor"""
        if self.redis:
            try:
                self.redis.close()
            except:
                pass
        self.operation_cache.clear()
        self.logger.info("Redis executor shutdown")