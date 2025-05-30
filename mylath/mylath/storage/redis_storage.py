# mylath/storage/redis_storage.py
import redis
import json
import uuid
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass, asdict
import pickle
import numpy as np


@dataclass
class Node:
    id: str
    label: str
    properties: Dict[str, Any]
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            import time
            self.created_at = time.time()


@dataclass 
class Edge:
    id: str
    label: str
    from_node: str
    to_node: str
    properties: Dict[str, Any]
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            import time
            self.created_at = time.time()


@dataclass
class Vector:
    id: str
    data: List[float]
    metadata: Dict[str, Any]
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class RedisStorage:
    """Core Redis storage layer for MyLath"""
    
    def __init__(self, host='localhost', port=6379, db=0, **kwargs):
        self.redis = redis.Redis(host=host, port=port, db=db, **kwargs)
        self.pipe = self.redis.pipeline()
        
    def _generate_id(self) -> str:
        """Generate unique ID"""
        return str(uuid.uuid4())
    
    def _hash_label(self, label: str) -> str:
        """Hash label for consistent key generation"""
        import hashlib
        return hashlib.md5(label.encode()).hexdigest()[:8]
    
    # Node Operations
    def create_node(self, label: str, properties: Dict[str, Any] = None) -> Node:
        """Create a new node"""
        if properties is None:
            properties = {}
            
        node = Node(
            id=self._generate_id(),
            label=label,
            properties=properties
        )
        
        # Store node data
        node_key = f"nodes:{node.id}"
        self.redis.hset(node_key, mapping={
            "id": node.id,
            "label": node.label,
            "properties": json.dumps(node.properties),
            "created_at": str(node.created_at)
        })
        
        # Add to label index
        self.redis.sadd(f"idx:label:{label}", node.id)
        
        # Add to property indices
        for prop_name, prop_value in properties.items():
            self.redis.sadd(f"idx:{prop_name}:{prop_value}", node.id)
            
        return node
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID"""
        node_key = f"nodes:{node_id}"
        data = self.redis.hgetall(node_key)
        
        if not data:
            return None
            
        return Node(
            id=data[b'id'].decode(),
            label=data[b'label'].decode(),
            properties=json.loads(data[b'properties'].decode()),
            created_at=float(data[b'created_at'].decode())
        )
    
    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties"""
        node = self.get_node(node_id)
        if not node:
            return False
            
        # Remove from old property indices
        for prop_name, prop_value in node.properties.items():
            self.redis.srem(f"idx:{prop_name}:{prop_value}", node_id)
        
        # Update properties
        node.properties.update(properties)
        
        # Update in Redis
        self.redis.hset(f"nodes:{node_id}", "properties", 
                       json.dumps(node.properties))
        
        # Add to new property indices  
        for prop_name, prop_value in node.properties.items():
            self.redis.sadd(f"idx:{prop_name}:{prop_value}", node_id)
            
        return True
    
    def delete_node(self, node_id: str) -> bool:
        """Delete node and all its edges"""
        node = self.get_node(node_id)
        if not node:
            return False
            
        # Get all edges involving this node
        out_edges = self.get_outgoing_edges(node_id)
        in_edges = self.get_incoming_edges(node_id)
        
        # Delete all edges
        for edge in out_edges + in_edges:
            self.delete_edge(edge.id)
            
        # Remove from indices
        self.redis.srem(f"idx:label:{node.label}", node_id)
        for prop_name, prop_value in node.properties.items():
            self.redis.srem(f"idx:{prop_name}:{prop_value}", node_id)
            
        # Delete node data
        self.redis.delete(f"nodes:{node_id}")
        return True
    
    # Edge Operations  
    def create_edge(self, label: str, from_node: str, to_node: str, 
                   properties: Dict[str, Any] = None) -> Edge:
        """Create a new edge"""
        if properties is None:
            properties = {}
            
        edge = Edge(
            id=self._generate_id(),
            label=label,
            from_node=from_node,
            to_node=to_node,
            properties=properties
        )
        
        # Store edge data
        edge_key = f"edges:{edge.id}"
        self.redis.hset(edge_key, mapping={
            "id": edge.id,
            "label": edge.label,
            "from_node": edge.from_node,
            "to_node": edge.to_node,
            "properties": json.dumps(edge.properties),
            "created_at": str(edge.created_at)
        })
        
        # Add to adjacency lists
        label_hash = self._hash_label(label)
        self.redis.sadd(f"out:{from_node}:{label_hash}", edge.id)
        self.redis.sadd(f"in:{to_node}:{label_hash}", edge.id)
        
        return edge
    
    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Get edge by ID"""
        edge_key = f"edges:{edge_id}"
        data = self.redis.hgetall(edge_key)
        
        if not data:
            return None
            
        return Edge(
            id=data[b'id'].decode(),
            label=data[b'label'].decode(),
            from_node=data[b'from_node'].decode(),
            to_node=data[b'to_node'].decode(),
            properties=json.loads(data[b'properties'].decode()),
            created_at=float(data[b'created_at'].decode())
        )
    
    def delete_edge(self, edge_id: str) -> bool:
        """Delete edge"""
        edge = self.get_edge(edge_id)
        if not edge:
            return False
            
        # Remove from adjacency lists
        label_hash = self._hash_label(edge.label)
        self.redis.srem(f"out:{edge.from_node}:{label_hash}", edge_id)
        self.redis.srem(f"in:{edge.to_node}:{label_hash}", edge_id)
        
        # Delete edge data
        self.redis.delete(f"edges:{edge_id}")
        return True
    
    def get_outgoing_edges(self, node_id: str, label: str = None) -> List[Edge]:
        """Get outgoing edges from a node"""
        if label:
            label_hash = self._hash_label(label)
            edge_ids = self.redis.smembers(f"out:{node_id}:{label_hash}")
        else:
            # Get all outgoing edges
            pattern = f"out:{node_id}:*"
            keys = self.redis.keys(pattern)
            edge_ids = set()
            for key in keys:
                edge_ids.update(self.redis.smembers(key))
                
        edges = []
        for edge_id in edge_ids:
            edge = self.get_edge(edge_id.decode() if isinstance(edge_id, bytes) else edge_id)
            if edge:
                edges.append(edge)
        return edges
    
    def get_incoming_edges(self, node_id: str, label: str = None) -> List[Edge]:
        """Get incoming edges to a node"""
        if label:
            label_hash = self._hash_label(label)
            edge_ids = self.redis.smembers(f"in:{node_id}:{label_hash}")
        else:
            # Get all incoming edges
            pattern = f"in:{node_id}:*"
            keys = self.redis.keys(pattern)
            edge_ids = set()
            for key in keys:
                edge_ids.update(self.redis.smembers(key))
                
        edges = []
        for edge_id in edge_ids:
            edge = self.get_edge(edge_id.decode() if isinstance(edge_id, bytes) else edge_id)
            if edge:
                edges.append(edge)
        return edges
    
    # Query Operations
    def find_nodes_by_label(self, label: str) -> List[Node]:
        """Find all nodes with given label"""
        node_ids = self.redis.smembers(f"idx:label:{label}")
        nodes = []
        for node_id in node_ids:
            node = self.get_node(node_id.decode() if isinstance(node_id, bytes) else node_id)
            if node:
                nodes.append(node)
        return nodes
    
    def find_nodes_by_property(self, prop_name: str, prop_value: Any) -> List[Node]:
        """Find nodes by property value"""
        node_ids = self.redis.smembers(f"idx:{prop_name}:{prop_value}")
        nodes = []
        for node_id in node_ids:
            node = self.get_node(node_id.decode() if isinstance(node_id, bytes) else node_id)
            if node:
                nodes.append(node)
        return nodes
