# mylath/graph/traversal.py
from typing import List, Dict, Any, Callable, Optional, Set
from ..storage.redis_storage import RedisStorage, Node, Edge
import time


class TraversalStep:
    """Represents a single step in graph traversal"""
    def __init__(self, nodes: List[Node] = None, edges: List[Edge] = None):
        self.nodes = nodes or []
        self.edges = edges or []


class GraphTraversal:
    """Graph traversal engine similar to Gremlin"""
    
    def __init__(self, storage: RedisStorage):
        self.storage = storage
        self.current_step = TraversalStep()
        self.steps_history = []
        
    def V(self, node_ids: List[str] = None) -> 'GraphTraversal':
        """Start traversal from vertices"""
        if node_ids:
            nodes = [self.storage.get_node(nid) for nid in node_ids]
            nodes = [n for n in nodes if n is not None]
        else:
            # Get all nodes (expensive operation - should be used carefully)
            nodes = []
            pattern = "nodes:*"
            keys = self.storage.redis.keys(pattern)
            for key in keys:
                node_id = key.decode().split(":")[-1]
                node = self.storage.get_node(node_id)
                if node:
                    nodes.append(node)
                    
        self.current_step = TraversalStep(nodes=nodes)
        return self
    
    def E(self, edge_ids: List[str] = None) -> 'GraphTraversal':
        """Start traversal from edges"""
        if edge_ids:
            edges = [self.storage.get_edge(eid) for eid in edge_ids]
            edges = [e for e in edges if e is not None]
        else:
            # Get all edges
            edges = []
            pattern = "edges:*"
            keys = self.storage.redis.keys(pattern)
            for key in keys:
                edge_id = key.decode().split(":")[-1]
                edge = self.storage.get_edge(edge_id)
                if edge:
                    edges.append(edge)
                    
        self.current_step = TraversalStep(edges=edges)
        return self
    
    def has(self, key: str, value: Any = None) -> 'GraphTraversal':
        """Filter nodes/edges by property"""
        if value is None:
            # Check if property exists
            filtered_nodes = [n for n in self.current_step.nodes 
                            if key in n.properties]
            filtered_edges = [e for e in self.current_step.edges
                            if key in e.properties]
        else:
            # Check property value
            filtered_nodes = [n for n in self.current_step.nodes 
                            if n.properties.get(key) == value]
            filtered_edges = [e for e in self.current_step.edges
                            if e.properties.get(key) == value]
            
        self.current_step = TraversalStep(nodes=filtered_nodes, edges=filtered_edges)
        return self
    
    def out(self, label: str = None) -> 'GraphTraversal':
        """Follow outgoing edges"""
        result_nodes = []
        for node in self.current_step.nodes:
            out_edges = self.storage.get_outgoing_edges(node.id, label)
            for edge in out_edges:
                target_node = self.storage.get_node(edge.to_node)
                if target_node:
                    result_nodes.append(target_node)
                    
        self.current_step = TraversalStep(nodes=result_nodes)
        return self
    
    def in_(self, label: str = None) -> 'GraphTraversal':
        """Follow incoming edges"""
        result_nodes = []
        for node in self.current_step.nodes:
            in_edges = self.storage.get_incoming_edges(node.id, label)
            for edge in in_edges:
                source_node = self.storage.get_node(edge.from_node)
                if source_node:
                    result_nodes.append(source_node)
                    
        self.current_step = TraversalStep(nodes=result_nodes)
        return self
    
    def both(self, label: str = None) -> 'GraphTraversal':
        """Follow both incoming and outgoing edges"""
        result_nodes = []
        for node in self.current_step.nodes:
            # Outgoing
            out_edges = self.storage.get_outgoing_edges(node.id, label)
            for edge in out_edges:
                target_node = self.storage.get_node(edge.to_node)
                if target_node:
                    result_nodes.append(target_node)
            
            # Incoming  
            in_edges = self.storage.get_incoming_edges(node.id, label)
            for edge in in_edges:
                source_node = self.storage.get_node(edge.from_node)
                if source_node:
                    result_nodes.append(source_node)
                    
        # Remove duplicates
        seen_ids = set()
        unique_nodes = []
        for node in result_nodes:
            if node.id not in seen_ids:
                seen_ids.add(node.id)
                unique_nodes.append(node)
                
        self.current_step = TraversalStep(nodes=unique_nodes)
        return self
    
    def outE(self, label: str = None) -> 'GraphTraversal':
        """Get outgoing edges"""
        result_edges = []
        for node in self.current_step.nodes:
            out_edges = self.storage.get_outgoing_edges(node.id, label)
            result_edges.extend(out_edges)
            
        self.current_step = TraversalStep(edges=result_edges)
        return self
    
    def inE(self, label: str = None) -> 'GraphTraversal':
        """Get incoming edges"""
        result_edges = []
        for node in self.current_step.nodes:
            in_edges = self.storage.get_incoming_edges(node.id, label)
            result_edges.extend(in_edges)
            
        self.current_step = TraversalStep(edges=result_edges)
        return self
    
    def outV(self) -> 'GraphTraversal':
        """Get outgoing vertices from edges"""
        result_nodes = []
        for edge in self.current_step.edges:
            node = self.storage.get_node(edge.to_node)
            if node:
                result_nodes.append(node)
                
        self.current_step = TraversalStep(nodes=result_nodes)
        return self
    
    def inV(self) -> 'GraphTraversal':
        """Get incoming vertices from edges"""
        result_nodes = []
        for edge in self.current_step.edges:
            node = self.storage.get_node(edge.from_node)
            if node:
                result_nodes.append(node)
                
        self.current_step = TraversalStep(nodes=result_nodes)
        return self
    
    def limit(self, count: int) -> 'GraphTraversal':
        """Limit results"""
        self.current_step.nodes = self.current_step.nodes[:count]
        self.current_step.edges = self.current_step.edges[:count]
        return self
    
    def dedup(self) -> 'GraphTraversal':
        """Remove duplicates"""
        # Deduplicate nodes
        seen_node_ids = set()
        unique_nodes = []
        for node in self.current_step.nodes:
            if node.id not in seen_node_ids:
                seen_node_ids.add(node.id)
                unique_nodes.append(node)
        
        # Deduplicate edges
        seen_edge_ids = set()
        unique_edges = []
        for edge in self.current_step.edges:
            if edge.id not in seen_edge_ids:
                seen_edge_ids.add(edge.id)
                unique_edges.append(edge)
                
        self.current_step = TraversalStep(nodes=unique_nodes, edges=unique_edges)
        return self
    
    def filter(self, predicate: Callable) -> 'GraphTraversal':
        """Filter with custom predicate"""
        filtered_nodes = [n for n in self.current_step.nodes if predicate(n)]
        filtered_edges = [e for e in self.current_step.edges if predicate(e)]
        
        self.current_step = TraversalStep(nodes=filtered_nodes, edges=filtered_edges)
        return self
    
    def count(self) -> int:
        """Count results"""
        return len(self.current_step.nodes) + len(self.current_step.edges)
    
    def to_list(self) -> List:
        """Convert to list"""
        return self.current_step.nodes + self.current_step.edges
    
    def values(self, *property_names: str) -> List[Any]:
        """Get property values"""
        result = []
        for node in self.current_step.nodes:
            for prop_name in property_names:
                if prop_name in node.properties:
                    result.append(node.properties[prop_name])
                    
        for edge in self.current_step.edges:
            for prop_name in property_names:
                if prop_name in edge.properties:
                    result.append(edge.properties[prop_name])
                    
        return result
    
    def shortest_path(self, target_id: str, edge_label: str = None) -> Optional[List]:
        """Find shortest path to target node using BFS"""
        if not self.current_step.nodes:
            return None
            
        start_node = self.current_step.nodes[0]
        return self._bfs_shortest_path(start_node.id, target_id, edge_label)
    
    def _bfs_shortest_path(self, start_id: str, target_id: str, edge_label: str = None):
        """BFS implementation for shortest path"""
        from collections import deque
        
        queue = deque([(start_id, [start_id])])
        visited = {start_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            if current_id == target_id:
                # Convert IDs to nodes
                return [self.storage.get_node(nid) for nid in path]
            
            # Get neighbors
            current_node = self.storage.get_node(current_id)
            if not current_node:
                continue
                
            out_edges = self.storage.get_outgoing_edges(current_id, edge_label)
            for edge in out_edges:
                neighbor_id = edge.to_node
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
                    
        return None  # No path found
