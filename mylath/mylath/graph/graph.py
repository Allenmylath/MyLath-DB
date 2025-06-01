# mylath/mylath/graph/graph.py - UPDATED ORIGINAL
from typing import Dict, Any, List, Optional, Tuple
from ..storage.redis_storage import RedisStorage, Node, Edge
from ..graph.traversal import GraphTraversal
from ..vector.vector_core import VectorCore


class Graph:
    """Enhanced Graph interface with integrated vector capabilities"""
    
    def __init__(self, storage: RedisStorage):
        self.storage = storage
        self.vectors = VectorCore(storage)
        
    def create_node(self, label: str, properties: Dict[str, Any] = None, 
                   embeddings: Dict[str, List[float]] = None) -> Node:
        """
        Create node with optional embeddings
        
        Args:
            label: Node label
            properties: Node properties
            embeddings: Dict of {property_name: embedding_vector}
                       e.g., {"details_emb": [0.1, 0.2, ...]}
        """
        if properties is None:
            properties = {}
        if embeddings is None:
            embeddings = {}
            
        # Create the node first
        node = self.storage.create_node(label, properties)
        
        # Add embeddings and link them to the node
        for emb_property, embedding_vector in embeddings.items():
            vector = self.vectors.add_vector(
                data=embedding_vector,
                metadata={
                    "type": f"{label}_embedding",
                    "node_id": node.id,
                    "property": emb_property
                },
                properties=properties  # Copy node properties for vector filtering
            )
            
            # Store vector ID reference in node
            self.storage.redis.hset(f"nodes:{node.id}", f"vector_{emb_property}", vector.id)
        
        return node
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        return self.storage.get_node(node_id)
    
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties"""
        return self.storage.update_node(node_id, properties)
    
    def delete_node(self, node_id: str) -> bool:
        """Delete node and associated vectors"""
        node = self.get_node(node_id)
        if not node:
            return False
            
        # Delete associated vectors
        node_data = self.storage.redis.hgetall(f"nodes:{node_id}")
        for key, value in node_data.items():
            if isinstance(key, bytes):
                key = key.decode()
            if key.startswith("vector_"):
                if isinstance(value, bytes):
                    value = value.decode()
                self.vectors.delete_vector(value)
        
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
        """Start a new graph traversal with enhanced capabilities"""
        return GraphTraversal(self.storage, self.vectors)
    
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
        vector_stats = self.vectors.get_stats()
        
        return {
            "nodes": node_count,
            "edges": edge_count,
            **vector_stats
        }


