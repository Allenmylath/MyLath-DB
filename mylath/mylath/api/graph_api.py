# mylath/api/graph_api.py
from flask import Flask, request, jsonify
from typing import Dict, Any
from ..graph.graph import Graph
from ..storage.redis_storage import RedisStorage
import json


class GraphAPI:
    """REST API for MyLath graph operations"""
    
    def __init__(self, storage: RedisStorage):
        self.app = Flask(__name__)
        self.graph = Graph(storage)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Node endpoints
        @self.app.route('/nodes', methods=['POST'])
        def create_node():
            data = request.get_json()
            label = data.get('label')
            properties = data.get('properties', {})
            
            if not label:
                return jsonify({"error": "Label is required"}), 400
                
            node = self.graph.create_node(label, properties)
            return jsonify({
                "id": node.id,
                "label": node.label,
                "properties": node.properties,
                "created_at": node.created_at
            }), 201
        
        @self.app.route('/nodes/<node_id>', methods=['GET'])
        def get_node(node_id):
            node = self.graph.get_node(node_id)
            if not node:
                return jsonify({"error": "Node not found"}), 404
                
            return jsonify({
                "id": node.id,
                "label": node.label,
                "properties": node.properties,
                "created_at": node.created_at
            })
        
        @self.app.route('/nodes/<node_id>', methods=['PUT'])
        def update_node(node_id):
            data = request.get_json()
            properties = data.get('properties', {})
            
            success = self.graph.update_node(node_id, properties)
            if not success:
                return jsonify({"error": "Node not found"}), 404
                
            return jsonify({"message": "Node updated successfully"})
        
        @self.app.route('/nodes/<node_id>', methods=['DELETE'])
        def delete_node(node_id):
            success = self.graph.delete_node(node_id)
            if not success:
                return jsonify({"error": "Node not found"}), 404
                
            return jsonify({"message": "Node deleted successfully"})
        
        # Edge endpoints
        @self.app.route('/edges', methods=['POST'])
        def create_edge():
            data = request.get_json()
            label = data.get('label')
            from_node = data.get('from_node')
            to_node = data.get('to_node')
            properties = data.get('properties', {})
            
            if not all([label, from_node, to_node]):
                return jsonify({"error": "Label, from_node, and to_node are required"}), 400
                
            edge = self.graph.create_edge(label, from_node, to_node, properties)
            return jsonify({
                "id": edge.id,
                "label": edge.label,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "properties": edge.properties,
                "created_at": edge.created_at
            }), 201
        
        @self.app.route('/edges/<edge_id>', methods=['GET'])
        def get_edge(edge_id):
            edge = self.graph.get_edge(edge_id)
            if not edge:
                return jsonify({"error": "Edge not found"}), 404
                
            return jsonify({
                "id": edge.id,
                "label": edge.label,
                "from_node": edge.from_node,
                "to_node": edge.to_node,
                "properties": edge.properties,
                "created_at": edge.created_at
            })
        
        @self.app.route('/edges/<edge_id>', methods=['DELETE'])
        def delete_edge(edge_id):
            success = self.graph.delete_edge(edge_id)
            if not success:
                return jsonify({"error": "Edge not found"}), 404
                
            return jsonify({"message": "Edge deleted successfully"})
        
        # Query endpoints
        @self.app.route('/query', methods=['POST'])
        def execute_query():
            data = request.get_json()
            query_type = data.get('type')
            params = data.get('params', {})
            
            try:
                if query_type == 'find_nodes_by_label':
                    label = params.get('label')
                    nodes = self.graph.find_nodes_by_label(label)
                    return jsonify([{
                        "id": n.id,
                        "label": n.label,
                        "properties": n.properties,
                        "created_at": n.created_at
                    } for n in nodes])
                
                elif query_type == 'find_nodes_by_property':
                    prop_name = params.get('property_name')
                    prop_value = params.get('property_value')
                    nodes = self.graph.find_nodes_by_property(prop_name, prop_value)
                    return jsonify([{
                        "id": n.id,
                        "label": n.label,
                        "properties": n.properties,
                        "created_at": n.created_at
                    } for n in nodes])
                
                elif query_type == 'traversal':
                    # Execute custom traversal
                    steps = params.get('steps', [])
                    traversal = self.graph.traversal()
                    
                    for step in steps:
                        step_type = step.get('type')
                        step_params = step.get('params', {})
                        
                        if step_type == 'V':
                            node_ids = step_params.get('node_ids')
                            traversal = traversal.V(node_ids)
                        elif step_type == 'has':
                            key = step_params.get('key')
                            value = step_params.get('value')
                            traversal = traversal.has(key, value)
                        elif step_type == 'out':
                            label = step_params.get('label')
                            traversal = traversal.out(label)
                        elif step_type == 'in':
                            label = step_params.get('label')
                            traversal = traversal.in_(label)
                        elif step_type == 'limit':
                            count = step_params.get('count')
                            traversal = traversal.limit(count)
                        # Add more step types as needed
                    
                    results = traversal.to_list()
                    return jsonify([{
                        "id": item.id,
                        "label": item.label if hasattr(item, 'label') else None,
                        "properties": item.properties,
                        "type": "node" if hasattr(item, 'created_at') else "edge"
                    } for item in results])
                
                else:
                    return jsonify({"error": "Unknown query type"}), 400
                    
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        # Vector endpoints
        @self.app.route('/vectors', methods=['POST'])
        def add_vector():
            data = request.get_json()
            vector_data = data.get('data')
            metadata = data.get('metadata', {})
            properties = data.get('properties', {})
            
            if not vector_data:
                return jsonify({"error": "Vector data is required"}), 400
                
            vector = self.graph.vectors.add_vector(vector_data, metadata, properties)
            return jsonify({
                "id": vector.id,
                "metadata": vector.metadata,
                "properties": vector.properties,
                "dimension": len(vector.data)
            }), 201
        
        @self.app.route('/vectors/<vector_id>', methods=['GET'])
        def get_vector(vector_id):
            vector = self.graph.vectors.get_vector(vector_id)
            if not vector:
                return jsonify({"error": "Vector not found"}), 404
                
            return jsonify({
                "id": vector.id,
                "data": vector.data,
                "metadata": vector.metadata,
                "properties": vector.properties
            })
        
        @self.app.route('/vectors/search', methods=['POST'])
        def search_vectors():
            data = request.get_json()
            query_vector = data.get('query_vector')
            k = data.get('k', 10)
            filters = data.get('filters', {})
            metric = data.get('metric', 'cosine')
            
            if not query_vector:
                return jsonify({"error": "Query vector is required"}), 400
                
            results = self.graph.vectors.search_vectors(
                query_vector, k, filters, metric
            )
            
            return jsonify([{
                "vector": {
                    "id": vector.id,
                    "metadata": vector.metadata,
                    "properties": vector.properties
                },
                "score": score
            } for vector, score in results])
        
        # Stats endpoint
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            return jsonify(self.graph.get_stats())
    
    def run(self, host='localhost', port=5000, debug=False):
        """Run the API server"""
        self.app.run(host=host, port=port, debug=debug)
