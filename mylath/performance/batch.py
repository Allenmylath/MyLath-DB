# mylath/performance/batch.py
from typing import List, Dict, Any, Tuple
from ..storage.redis_storage import RedisStorage, Node, Edge
import json


class BatchOperations:
    """Batch operations for improved performance"""
    
    def __init__(self, storage: RedisStorage):
        self.storage = storage
    
    def create_nodes_batch(self, nodes_data: List[Dict[str, Any]]) -> List[Node]:
        """Create multiple nodes in a single operation"""
        pipe = self.storage.redis.pipeline()
        nodes = []
        
        for node_data in nodes_data:
            label = node_data.get('label')
            properties = node_data.get('properties', {})
            
            node = Node(
                id=self.storage._generate_id(),
                label=label,
                properties=properties
            )
            nodes.append(node)
            
            # Add to pipeline
            node_key = f"nodes:{node.id}"
            pipe.hset(node_key, mapping={
                "id": node.id,
                "label": node.label,
                "properties": json.dumps(node.properties),
                "created_at": str(node.created_at)
            })
            
            # Add to indices
            pipe.sadd(f"idx:label:{label}", node.id)
            for prop_name, prop_value in properties.items():
                pipe.sadd(f"idx:{prop_name}:{prop_value}", node.id)
        
        # Execute pipeline
        pipe.execute()
        return nodes
    
    def create_edges_batch(self, edges_data: List[Dict[str, Any]]) -> List[Edge]:
        """Create multiple edges in a single operation"""
        pipe = self.storage.redis.pipeline()
        edges = []
        
        for edge_data in edges_data:
            label = edge_data.get('label')
            from_node = edge_data.get('from_node')
            to_node = edge_data.get('to_node')
            properties = edge_data.get('properties', {})
            
            edge = Edge(
                id=self.storage._generate_id(),
                label=label,
                from_node=from_node,
                to_node=to_node,
                properties=properties
            )
            edges.append(edge)
            
            # Add to pipeline
            edge_key = f"edges:{edge.id}"
            pipe.hset(edge_key, mapping={
                "id": edge.id,
                "label": edge.label,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "properties": json.dumps(edge.properties),
                "created_at": str(edge.created_at)
            })
            
            # Add to adjacency lists
            label_hash = self.storage._hash_label(label)
            pipe.sadd(f"out:{from_node}:{label_hash}", edge.id)
            pipe.sadd(f"in:{to_node}:{label_hash}", edge.id)
        
        # Execute pipeline
        pipe.execute()
        return edges
    
    def delete_nodes_batch(self, node_ids: List[str]) -> int:
        """Delete multiple nodes in a single operation"""
        deleted_count = 0
        pipe = self.storage.redis.pipeline()
        
        for node_id in node_ids:
            node = self.storage.get_node(node_id)
            if node:
                # Get and delete all edges
                out_edges = self.storage.get_outgoing_edges(node_id)
                in_edges = self.storage.get_incoming_edges(node_id)
                
                for edge in out_edges + in_edges:
                    label_hash = self.storage._hash_label(edge.label)
                    pipe.srem(f"out:{edge.from_node}:{label_hash}", edge.id)
                    pipe.srem(f"in:{edge.to_node}:{label_hash}", edge.id)
                    pipe.delete(f"edges:{edge.id}")
                
                # Remove from indices
                pipe.srem(f"idx:label:{node.label}", node_id)
                for prop_name, prop_value in node.properties.items():
                    pipe.srem(f"idx:{prop_name}:{prop_value}", node_id)
                
                # Delete node
                pipe.delete(f"nodes:{node_id}")
                deleted_count += 1
        
        pipe.execute()
        return deleted_count
