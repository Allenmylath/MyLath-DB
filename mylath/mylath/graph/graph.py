# mylath/graph/graph.py
from typing import Dict, Any, List, Optional
from ..storage.redis_storage import RedisStorage, Node, Edge
from ..graph.traversal import GraphTraversal
from ..vector.vector_core import VectorCore


class Graph:
    """Main graph interface for MyLath"""
    
    def __init__(self, storage: RedisStorage):
        self.storage = storage
        self.vectors = VectorCore(storage)
        
    def create_node(self, label: str, properties: Dict[str, Any] = None) -> Node:
        """Create a new node"""
        return self.storage.create_node(label, properties)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        return self.storage.get_node(node_id)
    
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties"""
        return self.storage.update_node(node_id, properties)
    
    def delete_node(self, node_id: str) -> bool:
        """Delete node"""
        return self.storage.delete_node(node_id)
    
    def create_edge(self, label: str, from_node: str, to_node: str,
                   properties: Dict[str, Any] = None) -> Edge:
        """Create a new edge"""
        return self.storage.create_edge(label, from_node, to_node, properties)
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get edge by ID"""
        return self.storage.get_edge(edge_id)
    
    def delete_edge(self, edge_id: str) -> bool:
        """Delete edge"""
        return self.storage.delete_edge(edge_id)
    
    def traversal(self) -> GraphTraversal:
        """Start a new graph traversal"""
        return GraphTraversal(self.storage)
    
    def V(self, *node_ids) -> GraphTraversal:
        """Start traversal from vertices - shortcut method"""
        return self.traversal().V(list(node_ids) if node_ids else None)
    
    def E(self, *edge_ids) -> GraphTraversal:
        """Start traversal from edges - shortcut method"""
        return self.traversal().E(list(edge_ids) if edge_ids else None)
    
    def find_nodes_by_label(self, label: str) -> List[Node]:
        """Find nodes by label"""
        return self.storage.find_nodes_by_label(label)
    
    def find_nodes_by_property(self, prop_name: str, prop_value: Any) -> List[Node]:
        """Find nodes by property"""
        return self.storage.find_nodes_by_property(prop_name, prop_value)
    
    def get_stats(self) -> Dict[str, int]:
        """Get graph statistics"""
        node_count = len(self.storage.redis.keys("nodes:*"))
        edge_count = len(self.storage.redis.keys("edges:*"))
        vector_count = self.storage.redis.scard("vector_index")
        
        return {
            "nodes": node_count,
            "edges": edge_count,
            "vectors": vector_count
        }
