# mylath/advanced/partitioning.py
import hashlib
from typing import Dict, Any, List
import redis


class GraphPartitioner:
    """Partition graph data across multiple Redis instances"""
    
    def __init__(self, shard_configs: List[dict]):
        self.shards = [redis.Redis(**config) for config in shard_configs]
        self.num_shards = len(self.shards)
    
    def _get_shard(self, key: str) -> redis.Redis:
        """Get shard for a given key using consistent hashing"""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        shard_index = hash_value % self.num_shards
        return self.shards[shard_index]
    
    def _get_node_shard(self, node_id: str) -> redis.Redis:
        """Get shard for a node"""
        return self._get_shard(f"node:{node_id}")
    
    def _get_edge_shard(self, edge_id: str) -> redis.Redis:
        """Get shard for an edge"""
        return self._get_shard(f"edge:{edge_id}")
    
    def store_node(self, node_id: str, node_data: Dict[str, Any]):
        """Store node in appropriate shard"""
        shard = self._get_node_shard(node_id)
        shard.hset(f"nodes:{node_id}", mapping=node_data)
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node from appropriate shard"""
        shard = self._get_node_shard(node_id)
        data = shard.hgetall(f"nodes:{node_id}")
        return {k.decode(): v.decode() for k, v in data.items()} if data else None
    
    def store_edge(self, edge_id: str, edge_data: Dict[str, Any]):
        """Store edge in appropriate shard"""
        shard = self._get_edge_shard(edge_id)
        shard.hset(f"edges:{edge_id}", mapping=edge_data)
        
        # Store edge references in both nodes' shards for locality
        from_shard = self._get_node_shard(edge_data['from_node'])
        to_shard = self._get_node_shard(edge_data['to_node'])
        
        from_shard.sadd(f"out:{edge_data['from_node']}:{edge_data['label']}", edge_id)
        to_shard.sadd(f"in:{edge_data['to_node']}:{edge_data['label']}", edge_id)
